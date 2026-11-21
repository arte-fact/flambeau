//! Single-device session over any backend bundle. `SoloSession<B, A>` runs an
//! `Arch` on one device through `ForwardEngine::<B>::new_solo` inline (no worker
//! thread, no cluster, no AR) and exposes the same slot-aware forward / KV
//! surface the worker-based `Session<A>` does. It is the path the HTTP server
//! drives for `Topology::SingleDevice` on either backend (the only path CUDA
//! supports today — multi-device CUDA needs cluster peer-copy).

use anyhow::{anyhow, Context, Result};
use flambeau_backend::{Backend, DecodeGraphExec};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_quant::GgufFile;

use crate::core::scratch::MAX_SPLITK_CHUNKS;
use crate::core::{DecodeCapture, DecodeRebindPlan, RebindValue, ScratchPool};
use crate::ctx::ForwardCtx;
use crate::engine::ForwardEngine;
use crate::loader::ShardMode;
use crate::runtime::orchestrate::LaunchParams;
use crate::runtime::Arch;

/// KV-tail length from which single-token decode attention takes the split-K
/// path; below it the single-pass kernel (no `_dpos` variant) is not
/// graph-capturable, so the driver runs the plain forward for those tokens.
const DECODE_GRAPH_MIN_KV: usize = flambeau_ops::attention::SPLITK_MIN_TOKENS_KV + 1;

/// Cached captured decode graph + its rebind program, keyed by the inflight
/// slot it was captured against. `DecodeRebindPlan::reusable_at` decides whether
/// a later position can replay it or forces a recapture.
struct DecodeGraphCache<B: Backend> {
    exec: B::DecodeGraph,
    plan: DecodeRebindPlan,
    slot_id: usize,
}

/// Standing refusal after an uncapturable capture attempt. Without it the
/// driver pays a full capture+instantiate+discard (~5-7 ms) EVERY token while
/// the condition persists (e.g. the whole SWA SlideBand). Blocks capture
/// attempts for `slot_id` at positions in `[marked_at, retry_at)`; a slot
/// reset (position dropping below `marked_at`) naturally unblocks, and a
/// config-permanent reason blocks to `usize::MAX`.
struct DecodeGraphBlock {
    slot_id: usize,
    marked_at: usize,
    retry_at: usize,
}


/// Single-device, single-rank session. Owns the device, registry, scratch pool
/// and model; every forward builds a fresh `ForwardEngine::<B>::new_solo` over
/// them. Field order puts `device` last so it drops last (after pool/model
/// release their device memory).
pub struct SoloSession<B: Backend, A: Arch> {
    model: A::Model,
    pool: ScratchPool,
    reg: B::Registry,
    /// Cached single-token decode graph (CUDA-only, `decode_graph` on).
    /// `None` until the first eligible decode token captures one; invalidated
    /// (recaptured) when the split-K grid / SWA regime / slot changes.
    /// Declared BEFORE `device` so the graph exec (which needs the CUDA context
    /// in its destructor) drops before the device releases it.
    graph: Option<DecodeGraphCache<B>>,
    graph_block: Option<DecodeGraphBlock>,
    /// Capture single-token decode into a graph where the backend supports
    /// it (CUDA, non-paged KV); off forces the non-graph path, which gives
    /// byte-identical output.
    decode_graph: bool,
    device: B::Device,
    last_logits: Vec<f32>,
    prefill_ubatch: usize,
    disposed: bool,
}

impl<B: Backend, A: Arch> SoloSession<B, A> {
    /// Open `device_id` on backend `B`, load the model replicated, size the
    /// scratch pool and registry. Mirrors `workers::init_rank` for the
    /// single-device (`Replicated`, no layer slice) case.
    pub fn new(file: GgufFile, device_id: i32, params: LaunchParams) -> Result<Self> {
        let LaunchParams {
            ctx_cap,
            prefill_ubatch,
            max_slots,
            paged_kv_pages,
            kv_layout,
            mtp_max_draft,
            decode_graph,
        } = params;
        if prefill_ubatch == 0 {
            anyhow::bail!("SoloSession::new: prefill_ubatch must be > 0");
        }
        if max_slots == 0 {
            anyhow::bail!("SoloSession::new: max_slots must be > 0");
        }
        let device = B::new_device(device_id)?;
        device.bind().map_err(|e| anyhow!("device.bind: {e:?}"))?;
        let model = A::load(&file, &device, ShardMode::Replicated, None, ctx_cap)?;
        let cfg = A::scratch_config(
            &model,
            ShardMode::Replicated,
            prefill_ubatch,
            max_slots,
            paged_kv_pages,
            kv_layout,
            mtp_max_draft,
        );
        let pool = ScratchPool::new(&device, cfg)?;
        let reg = B::new_registry(&device)?;
        Ok(Self {
            model,
            pool,
            reg,
            device,
            last_logits: Vec::new(),
            prefill_ubatch,
            graph: None,
            graph_block: None,
            decode_graph,
            disposed: false,
        })
    }

    pub fn prefill_ubatch(&self) -> usize {
        self.prefill_ubatch
    }

    /// Core forward dispatcher. Routes the single-token, contiguous-KV CUDA
    /// decode case through the capture/replay graph driver when
    /// `decode_graph` is on; everything else (prefill, batched decode,
    /// HIP, paged KV, graph off) runs the untouched inline forward. With the
    /// gate off the graph path is never reached and behaviour is byte-identical.
    fn forward(&mut self, tokens: &[u32], positions: &[usize], slot_ids: &[usize]) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("SoloSession::forward: empty tokens");
        }
        if positions.len() != tokens.len() || slot_ids.len() != tokens.len() {
            anyhow::bail!(
                "SoloSession::forward: positions.len {} / slot_ids.len {} != tokens.len {}",
                positions.len(),
                slot_ids.len(),
                tokens.len()
            );
        }
        if B::supports_decode_graph()
            && self.decode_graph
            && tokens.len() == 1
            && self.pool.paged_kv_caches.is_none()
        {
            return self.forward_decode_graph(tokens[0], positions[0], slot_ids[0]);
        }
        self.forward_general(tokens, positions, slot_ids)
    }

    /// The inline single-pass forward (one `ForwardEngine::<B>::new_solo`).
    /// Logits for the last token (or N rows under distinct slot_ids) land in
    /// `self.last_logits`. This is the untouched pre-graph path; the driver
    /// falls back to it for uncapturable / early-token decode.
    fn forward_general(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
    ) -> Result<()> {
        let stream = self.device.default_stream();
        let mut ctx = ForwardEngine::<B>::new_solo(&self.device, stream, &self.reg, &mut self.pool);
        A::forward(&self.model, &mut ctx, tokens, positions, slot_ids)?;
        let logits = ctx.logits().to_vec();
        self.last_logits = logits;
        Ok(())
    }

    /// Capture-or-replay driver for CUDA single-stream n=1 decode. Replays the
    /// cached graph when its split-K grid + SWA regime + slot still match this
    /// `position`; otherwise recaptures (the forward runs ONCE inside capture,
    /// which records but does not execute — the subsequent `launch` computes
    /// this token). Early tokens (`position + 1 < DECODE_GRAPH_MIN_KV`) and any
    /// pass a composite flagged uncapturable (SWA SlideBand, Q8, single-pass)
    /// fall back to the plain forward.
    fn forward_decode_graph(&mut self, token: u32, position: usize, slot_id: usize) -> Result<()> {
        let vocab = self.pool.config.vocab;
        let reusable = self.graph.as_ref().is_some_and(|g| {
            g.slot_id == slot_id
                && g.plan
                    .reusable_at(position, |cl| flambeau_model_ops::splitk_chunk_size(cl, MAX_SPLITK_CHUNKS))
        });
        if reusable {
            return self.graph_set_and_launch(token, position, vocab);
        }
        // Below the split-K threshold the attention takes the single-pass
        // (uncapturable) path — run the plain forward and keep any stale graph
        // (it will be revalidated / recaptured once we re-enter the split-K
        // regime).
        if position + 1 < DECODE_GRAPH_MIN_KV {
            return self.forward_general(&[token], &[position], &[slot_id]);
        }
        // A prior capture attempt at this slot failed as uncapturable and its
        // condition still holds — don't pay capture+instantiate+discard again.
        if let Some(b) = &self.graph_block {
            if b.slot_id == slot_id && position >= b.marked_at && position < b.retry_at {
                return self.forward_general(&[token], &[position], &[slot_id]);
            }
        }

        // Recapture. The forward runs once in record-only capture mode, building
        // the rebind plan + grid witnesses. The record pass still runs the
        // model's host-side bookkeeping, so the MTP pending marker is restored
        // if the capture is refused and the plain forward runs instead.
        let mtp_pending_before = self.pool.mtp_pending;
        let mut capture = DecodeCapture::new();
        let exec_res = {
            let device = &self.device;
            let reg = &self.reg;
            let model = &self.model;
            let pool = &mut self.pool;
            let cap_ref = &mut capture;
            let stream = device.default_stream();
            <B::DecodeGraph as DecodeGraphExec<B>>::capture(stream, move |s| {
                let mut engine =
                    ForwardEngine::<B>::new_solo_capturing(device, s, reg, pool, cap_ref);
                A::forward(model, &mut engine, &[token], &[position], &[slot_id])
            })
        };
        let exec = exec_res.context("SoloSession::forward_decode_graph: capture")?;
        if let Some(why) = capture.is_uncapturable() {
            tracing::debug!(position, why, "decode-graph capture refused");
            let retry_at = match capture.uncapturable_retry() {
                Some(crate::core::capture::RetryAt::Position(p)) => p,
                _ => usize::MAX,
            };
            self.graph_block = Some(DecodeGraphBlock { slot_id, marked_at: position, retry_at });
            self.pool.mtp_pending = mtp_pending_before;
            // The captured forward did not execute; run the real one for the
            // correct logits and drop the unusable graph.
            drop(exec);
            self.graph = None;
            return self.forward_general(&[token], &[position], &[slot_id]);
        }
        tracing::debug!(position, slots = capture.entries().len(), "decode-graph captured");
        self.graph_block = None;
        let plan = capture.into_plan();
        self.graph = Some(DecodeGraphCache { exec, plan, slot_id });
        self.graph_set_and_launch(token, position, vocab)
    }

    /// Rebind every slot for `(position, token)`, upload the position (uncaptured
    /// — kept out of the graph), replay the captured graph, then D2H the logits.
    fn graph_set_and_launch(&mut self, token: u32, position: usize, vocab: usize) -> Result<()> {
        // Uncaptured positions upload into the fixed device buffer the captured
        // RoPE kernels read. Same stream as the launch, so it is ordered before
        // the graph; the host staging buffer stays valid until the D2H sync.
        self.pool.position_stage_host[0] = position as i32;
        let pos_src = DevicePtr(self.pool.position_stage_host.as_ptr() as usize);
        let pos_dst = self.pool.position_i32;
        // Decode context: `[position, token_id]` read on-device by the captured
        // `_dpos` kernels. One uncaptured 8-byte HtoD replaces the
        // per-arg rebind loop; ordered before the launch on the same stream,
        // valid until the D2H sync in graph_download_logits.
        self.pool.decode_ctx_host[0] = position as i32;
        self.pool.decode_ctx_host[1] = token as i32;
        let ctx_src = DevicePtr(self.pool.decode_ctx_host.as_ptr() as usize);
        let ctx_dst = self.pool.decode_ctx_i32;
        let values = self
            .graph
            .as_ref()
            .expect("graph_set_and_launch: graph cache present")
            .plan
            .evaluate(position, token);
        {
            let stream = self.device.default_stream();
            // SAFETY: pos_src is the pool's stable host buffer (>= 1 i32);
            // pos_dst is position_i32 (>= max_prefill_tokens i32). Valid until
            // the sync in graph_download_logits.
            unsafe {
                self.device
                    .memcpy_async(stream, CopyDirection::HostToDevice, pos_dst, pos_src, 4)
                    .context("decode-graph: positions HtoD")?;
            }
            // SAFETY: ctx_src is the pool's stable 2-i32 host buffer; ctx_dst is
            // decode_ctx_i32 (2 i32). Valid until the D2H sync.
            unsafe {
                self.device
                    .memcpy_async(stream, CopyDirection::HostToDevice, ctx_dst, ctx_src, 8)
                    .context("decode-graph: decode-ctx HtoD")?;
            }
            let exec = &self
                .graph
                .as_ref()
                .expect("graph_set_and_launch: graph cache present")
                .exec;
            // `values` outlives `launch`, so every scalar/pointer backing a slot
            // stays live across sibling updates on the same node + the launch.
            for v in &values {
                match v {
                    RebindValue::ScalarI32 { slot, value } => exec.set_scalar_i32(*slot, value)?,
                    RebindValue::PointerU64 { slot, value } => exec.set_pointer_u64(*slot, value)?,
                    RebindValue::MemcpySrc { slot, value } => {
                        exec.set_memcpy_src(*slot, DevicePtr(*value))?
                    }
                }
            }
            exec.launch(stream).context("decode-graph: launch")?;
        }
        self.graph_download_logits(vocab)
    }

    /// Uncaptured D2H of the softcapped logits from `pool.logits_f32_dev`
    /// (the graph's split point) into `self.last_logits`, then sync.
    fn graph_download_logits(&mut self, vocab: usize) -> Result<()> {
        if self.last_logits.len() != vocab {
            self.last_logits = vec![0.0_f32; vocab];
        }
        let dst = DevicePtr(self.last_logits.as_mut_ptr() as usize);
        let src = self.pool.logits_f32_dev;
        let stream = self.device.default_stream();
        // SAFETY: dst holds `vocab` f32; src is the logits device buffer
        // (>= max_slots * vocab f32); synced immediately below.
        unsafe {
            self.device
                .memcpy_async(stream, CopyDirection::DeviceToHost, dst, src, vocab * 4)
                .context("decode-graph: logits D2H")?;
        }
        Stream::synchronize(stream).context("decode-graph: logits sync")?;
        Ok(())
    }

    pub fn forward_one_token_logits_slot(
        &mut self,
        token: u32,
        position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward(&[token], &[position], &[slot_id])?;
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    /// One greedy MTP cycle (single-device). `Ok(None)` when the model has
    /// no MTP head. Returns (accepted draft tokens, logits at the last
    /// accepted position).
    pub fn mtp_step(
        &mut self,
        token: u32,
        position: usize,
        slot_id: usize,
        n_draft: usize,
        sample_factory: crate::runtime::workers::SampleFactory,
        seed: u64,
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        if !A::has_mtp(&self.model) {
            return Ok(None);
        }
        let stream = self.device.default_stream();
        let mut ctx = ForwardEngine::<B>::new_solo(&self.device, stream, &self.reg, &mut self.pool);
        let mut out_logits = Vec::new();
        let emitted = A::mtp_cycle(
            &self.model,
            &mut ctx,
            token,
            position,
            slot_id,
            n_draft,
            &mut (sample_factory)(seed),
            seed,
            &mut out_logits,
        )?;
        self.last_logits = out_logits.clone();
        Ok(Some((emitted, out_logits)))
    }

    /// Chunked prefill into `slot_id` (chunk size = `prefill_ubatch`).
    pub fn forward_prefill_logits_slot(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("SoloSession::forward_prefill_logits_slot: empty tokens");
        }
        let chunk_size = self.prefill_ubatch;
        let mut pos = start_position;
        let mut i = 0;
        while i < tokens.len() {
            let end = (i + chunk_size).min(tokens.len());
            let chunk_len = end - i;
            let chunk_positions: Vec<usize> = (0..chunk_len).map(|k| pos + k).collect();
            let chunk_slots = vec![slot_id; chunk_len];
            self.forward(&tokens[i..end], &chunk_positions, &chunk_slots)?;
            pos += chunk_len;
            i = end;
        }
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    /// Batched decode: N (token, position, slot) tuples in one forward; emits
    /// `[N, vocab]` row-major in `self.last_logits`. Slot ids must be distinct.
    pub fn forward_decode_batched(&mut self, slots: &[(u32, usize, usize)]) -> Result<()> {
        if slots.is_empty() {
            anyhow::bail!("SoloSession::forward_decode_batched: empty slots");
        }
        let tokens: Vec<u32> = slots.iter().map(|s| s.0).collect();
        let positions: Vec<usize> = slots.iter().map(|s| s.1).collect();
        let slot_ids: Vec<usize> = slots.iter().map(|s| s.2).collect();
        if tokens.len() > 1 && slot_ids.iter().all(|&s| s == slot_ids[0]) {
            anyhow::bail!(
                "SoloSession::forward_decode_batched: slot_ids all equal — \
                 use forward_prefill_logits for single-slot multi-token"
            );
        }
        self.forward(&tokens, &positions, &slot_ids)
    }

    /// Sarathi-Serve mixed batch: rows `[0..prefill_rows)` prefill into
    /// `slot_ids[0]`, rows `[prefill_rows..n)` decode N distinct slots. Emits
    /// `(N + 1)` logit rows. Arch must override `Arch::forward_mixed`.
    pub fn forward_mixed(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
        prefill_rows: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("SoloSession::forward_mixed: empty tokens");
        }
        if positions.len() != tokens.len() || slot_ids.len() != tokens.len() {
            anyhow::bail!(
                "SoloSession::forward_mixed: positions.len {} / slot_ids.len {} != tokens.len {}",
                positions.len(),
                slot_ids.len(),
                tokens.len()
            );
        }
        if prefill_rows == 0 || prefill_rows >= tokens.len() {
            anyhow::bail!(
                "SoloSession::forward_mixed: prefill_rows must satisfy 0 < K < n (got K={prefill_rows}, n={})",
                tokens.len()
            );
        }
        let stream = self.device.default_stream();
        let mut ctx = ForwardEngine::<B>::new_solo(&self.device, stream, &self.reg, &mut self.pool);
        A::forward_mixed(&self.model, &mut ctx, tokens, positions, slot_ids, prefill_rows)?;
        let logits = ctx.logits().to_vec();
        self.last_logits = logits;
        Ok(())
    }

    /// Slice the i-th logit row from the most recent forward.
    pub fn logits_row(&self, i: usize, vocab: usize) -> &[f32] {
        &self.last_logits[i * vocab..(i + 1) * vocab]
    }

    pub fn reset_kv_slot(&mut self, slot_id: usize) -> Result<()> {
        self.pool.mtp_pending = None;
        self.pool.reset_gdn_state_slot(slot_id, &self.device)
    }

    pub fn snapshot_kv_slot(&self, slot_id: usize, n_tokens: usize) -> Result<Vec<Vec<u8>>> {
        let bytes = self.pool.snapshot_slot_bytes(slot_id, n_tokens, &self.device)?;
        Ok(vec![bytes])
    }

    pub fn restore_kv_slot(
        &mut self,
        slot_id: usize,
        n_tokens: usize,
        snaps: std::sync::Arc<Vec<Vec<u8>>>,
    ) -> Result<()> {
        let bytes = snaps.first().ok_or_else(|| {
            anyhow!("SoloSession::restore_kv_slot: snapshot has no rank-0 buffer")
        })?;
        self.pool.mtp_pending = None;
        self.pool.restore_slot_bytes(slot_id, n_tokens, bytes, &self.device)
    }

    pub fn release_paged_slot(&mut self, slot_id: usize) -> Result<()> {
        for pool in self.pool.page_pools.iter_mut() {
            pool.release_slot(slot_id);
        }
        Ok(())
    }

    pub fn dispose_in_place(&mut self) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        // Destroy the captured graph while the device/context is still live.
        self.graph = None;
        let pool_res = self.pool.dispose(&self.device);
        let model_res = A::dispose(&mut self.model, &self.device);
        pool_res.and(model_res)
    }
}

impl<B: Backend, A: Arch> Drop for SoloSession<B, A> {
    fn drop(&mut self) {
        if self.disposed {
            return;
        }
        self.disposed = true;
        // Destroy the captured graph while the device/context is still live.
        self.graph = None;
        let _ = self.pool.dispose(&self.device);
        let _ = A::dispose(&mut self.model, &self.device);
    }
}
