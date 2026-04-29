//! Pipeline-parallel (Mesh&lt;N&gt; for N > 1) forward entry points.
//!
//! Each rank owns ~`num_layers / N` contiguous layers; the hidden state
//! is passed rank-to-rank via `HipCluster::peer_copy_via_host` (pinned-
//! host bounce on PCIe-only rigs per the V1 CLAUDE.md rationale).
//!
//! Mesh&lt;1&gt; is a degenerate instance — the single-device entry points in
//! `forward::single_device` sidestep the peer-copy path entirely.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;

use super::{
    argmax_token_host, download_logits_host, forward_embed_decode_host, forward_layer_decode,
    forward_layer_prefill, forward_output_head_decode, LayerForwardScratch, LayerPrefillScratch,
    OutputHeadScratch,
};
use crate::session::LayerCache;

// ---------------------------------------------------------------------------
// V1.7.5.C — pipeline-parallel forward_one_token.
// ---------------------------------------------------------------------------

/// Per-rank scratch for a pipeline-parallel single-token decode. Only the
/// last rank in the cluster owns an `OutputHeadScratch` (middle ranks
/// never run the LM head).
pub struct RankForwardScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerForwardScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl RankForwardScratch {
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for RankForwardScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate scratch: one `RankForwardScratch` per rank in the cluster.
pub struct ShardedForwardOneTokenScratch {
    pub per_rank: Vec<RankForwardScratch>,
    /// V2.27.a-i3 — per-rank graph-capture cache for decode. One
    /// `HipGraphExec` per rank populated lazily on the first token
    /// when `FLAMBEAU_DECODE_GRAPH=1`. Stores only the per-rank
    /// layer chain — embed (rank 0), peer-copy, and argmax
    /// (rank N-1) stay uncaptured.
    pub graph_cache_decode: Vec<Option<GraphCacheDecodeEntry>>,
}

/// V2.27.a-i3 — one cached decode exec per rank with per-layer slot
/// bundles for full-attn layers.
pub struct GraphCacheDecodeEntry {
    pub exec: flambeau_backend_hip::HipGraphExec,
    /// Per-layer slot bundle. `None` entries correspond to GDN
    /// layers, which advance state in-place and need no slot
    /// updates at replay.
    pub layer_slots: Vec<Option<super::layer::LayerDecodeSlots>>,
}

impl ShardedForwardOneTokenScratch {
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<Self> {
        let hidden_bytes = model.config.hidden_size * 2;
        // C3: pre-size the cluster's pinned bounce buffers to the
        // stage-boundary payload. Decode hand-offs are one hidden-state
        // F16 vector per hop (`hidden_size * 2`). Doing this once here
        // means every subsequent `peer_copy_via_host` hits the lock-free
        // atomic fast path in `ensure_bounce`.
        cluster.reserve_bounce_capacity(hidden_bytes)?;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerForwardScratch::new(&model.config, device)?);
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            per_rank.push(RankForwardScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                hidden_a,
                hidden_b,
                layer,
                output_head,
                hidden_bytes,
                disposed: false,
            });
        }
        let graph_cache_decode = (0..cluster.ranks()).map(|_| None).collect();
        Ok(Self { per_rank, graph_cache_decode })
    }

    pub fn dispose(
        mut self,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Pipeline-parallel single-token decode across an N-rank cluster.
///
/// Flow:
///   1. Rank 0 gathers the input embedding into its `hidden_a`.
///   2. For r in 0..N:
///        - If r > 0: `peer_copy_via_host` pulls the previous rank's
///          final hidden (in `hidden_a` by convention — see step 3)
///          into this rank's `hidden_a`.
///        - Run `forward_layer_decode` over the layers the shard owns,
///          ping-ponging `hidden_a ↔ hidden_b`.
///        - Normalise the final hidden back into `hidden_a` so the
///          next peer-copy has a known source.
///   3. Last rank runs `forward_output_head_decode` + `argmax_token_host`.
///
/// Single-token in flight — no micro-batching (the bubble is the sum of
/// each rank's compute; V2 can add 2-micro-batch pipelining). Per-hop
/// cost ≈ 30 µs (V1.7.5.B measurement), 3 hops for N=4 ≈ 0.5% of the
/// 16.7 ms/token budget at 60 tok/s.
pub fn forward_one_token_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_one_token_pp: zero-rank cluster");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let hidden_bytes = hidden * 2;

    // 1. Embed on rank 0. Bind first so the embed memcpys land on the
    // right device.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        flambeau_backend_hip::profile::mark("step_start", rank0, rank0.default_stream())?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        forward_embed_decode_host(
            rank0,
            rank0.default_stream(),
            token_embd,
            token_id,
            scratch0.hidden_a,
            hidden,
        )?;
        flambeau_backend_hip::profile::mark("embed_done", rank0, rank0.default_stream())?;
    }

    // V2.27.a-i3 — opt-in decode graph capture gate. Per-rank layer
    // chain captured on first call, replayed with slot updates on
    // subsequent calls. Embed (rank 0), peer-copy, and argmax
    // (rank N-1) stay uncaptured regardless.
    //
    // V2.27.a-i5 — additionally fold the output head
    // (rmsnorm_quant_q8_1 + lm_head mmvq) on the LAST rank into that
    // rank's captured graph. argmax_token_host stays uncaptured
    // (host-side scan).
    let use_decode_graph = std::env::var("FLAMBEAU_DECODE_GRAPH").is_ok();
    let last_idx = n_ranks - 1;
    // Hoist output-head tensor references before the rank loop so the
    // last rank's capture closure can borrow them.
    let last_shard = &model.shards[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are hidden_bytes long on their
            // respective devices; no other stream touches them here.
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    hidden_bytes,
                )?;
            }
        }
        // Bind this rank's device before issuing any kernels through its
        // OpsRegistry — HIP's module-launched kernels use the thread's
        // current device context, not whichever device the module was
        // loaded on.
        device.bind()?;
        flambeau_backend_hip::profile::mark(
            "stage_start",
            device,
            device.default_stream(),
        )?;

        let shard = &model.shards[rank_idx];
        // Split-borrow: graph_cache_decode[rank] and per_rank[rank] are
        // disjoint fields of scratch.
        let graph_slot_ptr: *mut Option<GraphCacheDecodeEntry> = if use_decode_graph {
            &mut scratch.graph_cache_decode[rank_idx]
        } else {
            std::ptr::null_mut()
        };
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;

        // SAFETY: pointer non-null iff use_decode_graph; unique derivation
        // from &mut, no aliasing live here.
        let graph_slot: Option<&mut Option<GraphCacheDecodeEntry>> = if use_decode_graph {
            Some(unsafe { &mut *graph_slot_ptr })
        } else {
            None
        };
        let needs_capture = graph_slot.as_ref().map(|s| s.is_none()).unwrap_or(false);
        let needs_replay = graph_slot.as_ref().map(|s| s.is_some()).unwrap_or(false);

        if needs_replay {
            // === REPLAY branch ===
            let entry = graph_slot.as_ref().unwrap().as_ref().unwrap();
            let n_layers = entry.layer_slots.len();
            // Stable per-layer backing for the updated pos scalar.
            let mut n_tokens_kv_vals: Vec<i32> = vec![0; n_layers];
            for local_idx in 0..n_layers {
                let Some(slots) = entry.layer_slots[local_idx] else { continue };
                let layer_cache = &rank_session.caches[local_idx];
                let LayerCache::FullAttn(kv) = layer_cache else {
                    bail!("graph-capture decode: layer {local_idx} on rank {rank_idx} expected FullAttn cache");
                };
                n_tokens_kv_vals[local_idx] = (position + 1) as i32;
                // F16 → 2 bytes per element.
                let per_token_bytes = kv.n_heads() * kv.head_dim() * 2;
                let k_dst = kv.k_buffer().offset_bytes(position * per_token_bytes);
                let v_dst = kv.v_buffer().offset_bytes(position * per_token_bytes);
                // SAFETY: n_tokens_kv_vals lives through the launch below;
                // k_dst/v_dst are device pointers valid for the exec's lifetime.
                unsafe {
                    entry.exec.set_slot(
                        slots.full_attn.n_tokens_kv_slot,
                        &n_tokens_kv_vals[local_idx],
                    )?;
                    entry.exec.set_memcpy_slot(slots.full_attn.k_append_slot, k_dst)?;
                    entry.exec.set_memcpy_slot(slots.full_attn.v_append_slot, v_dst)?;
                }
                // Update the persistent positions_host in the FullAttnScratch
                // so the captured HtoD memcpy reads the new position at replay.
                layer_scratch
                    .full_attn
                    .as_mut()
                    .context("full_attn scratch missing")?
                    .positions_host[0] = position as i32;
            }
            entry.exec.launch(device.default_stream())?;
            // Manually bump each full-attn layer's tail by 1 (captured
            // kv_cache_append_hip_slot's Rust-side bump only ran at capture).
            for local_idx in 0..n_layers {
                if entry.layer_slots[local_idx].is_none() { continue; }
                if let LayerCache::FullAttn(kv) = &mut rank_session.caches[local_idx] {
                    kv.bump_tail(1).map_err(|e| anyhow::anyhow!(
                        "bump_tail rank={rank_idx} layer={local_idx}: {e}"
                    ))?;
                }
            }
            continue;  // skip uncaptured per-layer loop below
        }

        if needs_capture {
            // === CAPTURE branch (first decode call on this rank) ===
            let layer_slots: Vec<Option<super::layer::LayerDecodeSlots>> = shard
                .layers
                .iter()
                .map(|lw| {
                    if cfg.is_recurrent(lw.layer_idx) {
                        None
                    } else {
                        Some(super::layer::LayerDecodeSlots {
                            full_attn: super::attn::AttnDecodeSlots {
                                n_tokens_kv_slot: flambeau_backend_hip::ScalarSlot::new(),
                                k_append_slot: flambeau_backend_hip::MemcpySlot::new(),
                                v_append_slot: flambeau_backend_hip::MemcpySlot::new(),
                            },
                        })
                    }
                })
                .collect();
            let hidden_a = rank_scratch.hidden_a;
            let hidden_b = rank_scratch.hidden_b;
            // V2.27.a-i5 — take output_head scratch as a separate
            // split-borrow so the capture closure can run the output
            // head on the last rank.
            let output_head_scratch_opt: Option<&mut OutputHeadScratch> =
                rank_scratch.output_head.as_mut();
            let is_last_rank = rank_idx == last_idx;
            let layer_slots_clone = layer_slots.clone();
            let exec = flambeau_backend_hip::HipGraphExec::capture(
                device.default_stream(),
                |capture_s| {
                    let (mut x_in, mut x_out) = (hidden_a, hidden_b);
                    for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
                        let layer_cache = &mut rank_session.caches[local_idx];
                        forward_layer_decode(
                            &shard.ops,
                            capture_s,
                            device,
                            cfg,
                            layer_weights,
                            layer_cache,
                            layer_scratch,
                            x_in,
                            x_out,
                            position,
                            layer_slots_clone[local_idx],
                        )
                        .map_err(|e| flambeau_core::DeviceError::Backend {
                            backend: "hip",
                            code: -1,
                            message: format!(
                                "capture decode rank {} layer {}: {e}",
                                rank_idx, layer_weights.layer_idx
                            ),
                        })?;
                        std::mem::swap(&mut x_in, &mut x_out);
                    }
                    // Tail memcpy to land final hidden in hidden_a for the
                    // peer-copy / output-head consumer. Captured graph bakes
                    // in whichever branch the parity of n_layers selected.
                    if x_in != hidden_a {
                        // SAFETY: both pointers live; hidden_bytes bounded.
                        unsafe {
                            device.memcpy_async(
                                capture_s,
                                CopyDirection::DeviceToDevice,
                                hidden_a,
                                x_in,
                                hidden_bytes,
                            )?;
                        }
                    }
                    // V2.27.a-i5 — fold output head on last rank into
                    // the captured graph. rmsnorm_quant_q8_1 + mmvq.
                    // argmax stays uncaptured (host-side scan after
                    // the graph replay).
                    if is_last_rank {
                        if let Some(output_head_scratch) = output_head_scratch_opt {
                            forward_output_head_decode(
                                &shard.ops,
                                capture_s,
                                cfg,
                                output_norm,
                                lm_head,
                                output_head_scratch,
                                hidden_a,
                            )
                            .map_err(|e| flambeau_core::DeviceError::Backend {
                                backend: "hip",
                                code: -1,
                                message: format!("capture output head: {e}"),
                            })?;
                        }
                    }
                    Ok(())
                },
            )?;
            exec.launch(device.default_stream())?;
            *graph_slot.unwrap() = Some(GraphCacheDecodeEntry { exec, layer_slots });
            continue;
        }

        // === UNCAPTURED branch (legacy sync decode) ===
        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_decode(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                position,
                None,
            )
            .with_context(|| {
                format!(
                    "rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            // V1.7.4.b per-layer activation dump for A/B vs llama.cpp.
            // Env-gated so the hot path pays zero cost when unset. Pair
            // with `llama-eval-callback` + grep `l_out-<il>` to bisect a
            // future forward divergence.
            if std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok() && position == 0 {
                let mut buf = vec![half::f16::from_f32(0.0); hidden];
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToHost,
                        DevicePtr(buf.as_mut_ptr() as usize),
                        x_out,
                        hidden * 2,
                    )?;
                }
                device.default_stream().synchronize()?;
                let vals: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
                let l2 = vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                let (mn, mx) = vals
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                eprintln!(
                    "[layer-dump] l_out-{} ({}): L2={:.6} min={} max={} head={:?}",
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
                    l2, mn, mx, &vals[..4]
                );
            }
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Normalise final hidden into hidden_a for the next hand-off.
        // NO sync: subsequent peer_copy_via_host + downstream kernels all run on
        // the same default_stream, so stream ordering guarantees correctness.
        // Removing this sync saves one ~200 µs CPU-wait per rank per forward.
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    hidden_bytes,
                )?;
            }
        }
        flambeau_backend_hip::profile::mark(
            "stage_end",
            device,
            device.default_stream(),
        )?;
    }

    // 3. Output head on the last rank.
    // V2.27.a-i5 — under FLAMBEAU_DECODE_GRAPH the output head is
    // folded into the last rank's captured graph (runs during the
    // exec.launch() above). Skip the uncaptured dispatch here.
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    flambeau_backend_hip::profile::mark(
        "output_head_start",
        last_device,
        last_device.default_stream(),
    )?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    if !use_decode_graph {
        forward_output_head_decode(
            &last_shard.ops,
            last_device.default_stream(),
            cfg,
            output_norm,
            lm_head,
            output_head_scratch,
            last_scratch.hidden_a,
        )?;
    }

    // 4. Host argmax.
    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
}

/// Variant of [`forward_one_token_pp`] that downloads the F32 logit row
/// into a caller-owned `Vec<f32>` instead of argmax-ing on host. Used by
/// the HTTP server when `temperature > 0` / `top_p < 1`; greedy callers
/// should keep using [`forward_one_token_pp`] to skip the download +
/// server-side softmax.
pub fn forward_one_token_pp_logits(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    forward_one_token_pp_inner(
        model, session, cluster, scratch, token_id, position, /*download_logits=*/ Some(logits_out),
    )
    .map(|_| ())
}

/// Shared body for `forward_one_token_pp` and `forward_one_token_pp_logits`.
/// When `logits_out` is `Some`, downloads the F32 logits into it and returns 0;
/// when `None`, runs host argmax and returns the sampled token id.
fn forward_one_token_pp_inner(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
    logits_out: Option<&mut Vec<f32>>,
) -> Result<u32> {
    // Re-run the same composition as `forward_one_token_pp`, but branch on
    // the final reducer. Copy-paste is deliberate — the body is ~150 lines
    // of tightly-ordered HIP calls and threading a branch through would
    // hurt readability more than a second copy that tracks the original
    // line-for-line.
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_one_token_pp_inner: zero-rank cluster");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let hidden_bytes = hidden * 2;

    // V1-BENCH-CN-80B-5 — section markers. No-op when the thread-local
    // timer in flambeau_backend_hip::profile is disabled. Each mark is
    // recorded on the relevant rank's default stream so the elapsed_ms
    // delta to the next mark on the SAME rank captures the device-side
    // time spent on that section.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        flambeau_backend_hip::profile::mark("step_start", rank0, rank0.default_stream())?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        forward_embed_decode_host(
            rank0,
            rank0.default_stream(),
            token_embd,
            token_id,
            scratch0.hidden_a,
            hidden,
        )?;
        flambeau_backend_hip::profile::mark("embed_done", rank0, rank0.default_stream())?;
    }

    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    hidden_bytes,
                )?;
            }
        }
        device.bind()?;
        flambeau_backend_hip::profile::mark(
            "stage_start",
            device,
            device.default_stream(),
        )?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        let pp_probe = std::env::var("FLAMBEAU_PP_PROBE").is_ok();
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_decode(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                position,
                None,
            )
            .with_context(|| {
                format!(
                    "rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            std::mem::swap(&mut x_in, &mut x_out);
            // FLAMBEAU_PP_PROBE — dump x_in (this layer's output, post-swap)
            // for parity comparison vs the TP path (FLAMBEAU_TP_PROBE). Same
            // format: per-layer min/max/L2/head[0..4]. Disabled by default.
            if pp_probe {
                use flambeau_core::CopyDirection;
                let n = hidden_bytes / 2;
                let mut host = vec![0u16; n];
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToHost,
                        DevicePtr(host.as_mut_ptr() as usize),
                        x_in,
                        hidden_bytes,
                    )?;
                }
                device.default_stream().synchronize()?;
                let mut nan = 0usize;
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                let mut sum = 0.0f64;
                let mut sumsq = 0.0f64;
                for &b in &host {
                    let v = half::f16::from_bits(b).to_f32();
                    if v.is_nan() {
                        nan += 1;
                    } else {
                        if v < min { min = v; }
                        if v > max { max = v; }
                        sum += v as f64;
                        sumsq += (v as f64) * (v as f64);
                    }
                }
                let mean = sum / (n - nan).max(1) as f64;
                let l2 = sumsq.sqrt();
                let head: Vec<f32> = host[..host.len().min(4)]
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect();
                let il = layer_weights.layer_idx;
                eprintln!(
                    "  PP_PROBE after-layer hidden_a rank={rank_idx} il={il:>3} \
                     n={n} nan={nan} L2={l2:.6} min={min:.6} max={max:.6} \
                     mean={mean:.6} head={head:?}"
                );
            }
        }
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    hidden_bytes,
                )?;
            }
        }
        flambeau_backend_hip::profile::mark(
            "stage_end",
            device,
            device.default_stream(),
        )?;
    }

    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    flambeau_backend_hip::profile::mark(
        "output_head_start",
        last_device,
        last_device.default_stream(),
    )?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_scratch.hidden_a,
    )?;

    match logits_out {
        Some(buf) => {
            download_logits_host(
                last_device,
                last_device.default_stream(),
                output_head_scratch.logits_f32,
                cfg.vocab_size,
                buf,
            )?;
            Ok(0)
        }
        None => argmax_token_host(
            last_device,
            last_device.default_stream(),
            output_head_scratch.logits_f32,
            cfg.vocab_size,
        ),
    }
}

// ---------------------------------------------------------------------------
// V1.7.5.D — pipeline-parallel forward_prefill.
// ---------------------------------------------------------------------------

/// V2.25.c — one ubatch lane's ping-pong + per-layer scratch. Each rank
/// holds `u_lanes` of these so V2.25.d can pipeline ubatches across ranks
/// without aliasing intermediate buffers.
pub struct UbatchLane {
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: LayerPrefillScratch,
    hidden_bytes: usize,
}

impl UbatchLane {
    fn new(
        cfg: &crate::Qwen3MoEConfig,
        device: &HipDevice,
        ubatch_size: usize,
    ) -> Result<Self> {
        let hidden_bytes = ubatch_size * cfg.hidden_size * 2;
        let hidden_a = device.alloc(hidden_bytes)?;
        let hidden_b = device.alloc(hidden_bytes)?;
        let layer = LayerPrefillScratch::new(cfg, device, ubatch_size)?;
        Ok(Self { hidden_a, hidden_b, layer, hidden_bytes })
    }

    fn dispose(self, device: &HipDevice) -> Result<()> {
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        self.layer.dispose(device)?;
        Ok(())
    }
}

/// Per-rank scratch for a pipeline-parallel prefill chunk of up to
/// `max_tokens` tokens. Layout mirrors `RankForwardScratch` with the
/// hidden ping-pong buffers and `LayerPrefillScratch` both sized for `L`
/// tokens. Only the last rank owns an `OutputHeadScratch`.
///
/// V2.25.c — `extra_lanes` holds ADDITIONAL `UbatchLane`s beyond the
/// implicit lane 0 (which is the hidden_a/hidden_b/layer fields below).
/// Empty by default; `new_with_lanes(u_lanes > 1)` pre-allocates them so
/// V2.25.d can pipeline ubatches across ranks without aliasing.
pub struct RankForwardPrefillScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    /// V2.25.c — additional ubatch lanes beyond lane 0 (= the above
    /// hidden_a/hidden_b/layer fields). Used by V2.25.d.
    pub extra_lanes: Vec<UbatchLane>,
    /// MTP-5h-1 — one device buffer per local layer, holding a single
    /// hidden row (hidden * 2 bytes). Populated lazily during paired-L=2
    /// verify with the per-layer GDN input at L=2 batch position 0, for
    /// use by `Qwen3MoEShardedSession::redo_gdn_only_pp` on spec-decode
    /// reject. `Some(ptr)` for GDN layers, `None` for full-attn layers.
    pub gdn_input_snapshots: Vec<Option<DevicePtr>>,
    hidden_bytes: usize,
    /// Size of one snapshot row = `hidden_size * 2` (F16). Per-rank
    /// constant; cached for the dispose path.
    snapshot_row_bytes: usize,
    disposed: bool,
}

impl RankForwardPrefillScratch {
    /// V2.25.c — total ubatch lanes (includes lane 0 = the direct fields).
    pub fn u_lanes(&self) -> usize { 1 + self.extra_lanes.len() }

    /// V2.25.c — hidden_a for lane `idx`. Lane 0 = `self.hidden_a`;
    /// lane i>0 = `self.extra_lanes[i-1].hidden_a`.
    pub fn lane_hidden_a(&self, idx: usize) -> DevicePtr {
        if idx == 0 { self.hidden_a } else { self.extra_lanes[idx - 1].hidden_a }
    }

    /// V2.25.c — hidden_b for lane `idx`.
    pub fn lane_hidden_b(&self, idx: usize) -> DevicePtr {
        if idx == 0 { self.hidden_b } else { self.extra_lanes[idx - 1].hidden_b }
    }

    /// V2.25.c — mutable LayerPrefillScratch for lane `idx`.
    pub fn lane_layer_mut(&mut self, idx: usize) -> Option<&mut LayerPrefillScratch> {
        if idx == 0 {
            self.layer.as_mut()
        } else {
            Some(&mut self.extra_lanes[idx - 1].layer)
        }
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        // MTP-5h-1 — release per-GDN-layer snapshot buffers.
        for snap in self.gdn_input_snapshots.drain(..) {
            if let Some(ptr) = snap {
                unsafe {
                    device.dealloc(ptr, self.snapshot_row_bytes)?;
                }
            }
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        for lane in self.extra_lanes.drain(..) {
            lane.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for RankForwardPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardPrefillScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate scratch for PP prefill: one `RankForwardPrefillScratch` per rank.
pub struct ShardedForwardPrefillScratch {
    pub per_rank: Vec<RankForwardPrefillScratch>,
    /// V2.26.a-i5c — per (rank, lane) graph-capture cache used by
    /// `forward_prefill_pp_async` when `FLAMBEAU_ASYNC_GRAPH=1`. The
    /// first ubatch on (rank, lane) captures the whole layer chain;
    /// subsequent ubatches update pos-bearing slots + replay. Empty
    /// Vec (no sub-Vec) means disabled / lazy.
    pub graph_cache: Vec<Vec<Option<GraphCacheEntry>>>,
    /// V2.26.a-i7b — persistent host-side scratch for rank 0's
    /// batched embed (`forward_embed_prefill_batch`). Replaces per-token
    /// DtoH-sync-HtoD-sync pattern that issued 2·u host barriers per
    /// ubatch and starved async-PP's Rust dispatcher.
    pub embed_host: super::io::EmbedPrefillHostScratch,
    /// V2.30.a — per (rank, local_layer_idx) HipEvent that serialises
    /// `gdn_state_step` kernel access across lanes on the same rank.
    /// Ubatch N+1 on the opposite lane waits on this event before
    /// running its state_step; ubatch N records it after its state_step.
    /// Same-lane ubatches are already stream-ordered. Eliminates the
    /// V2.27.d / V2.28.c.1 / V2.28.a-i1 GDN race guards.
    /// Shape: outer vec is n_ranks; inner vec is local layer count on
    /// that rank. Entry is None for non-GDN (full-attn) layers.
    pub gdn_state_events: Vec<Vec<Option<flambeau_backend_hip::HipEvent>>>,
}

/// V2.26.a-i5c — one cached exec per (rank, lane) with the per-layer
/// slot bundles that drive pos updates. Captures only the layer-chain
/// portion (no peer-copy, no embed) — the wrap/unwrap happens outside
/// the capture closure.
pub struct GraphCacheEntry {
    pub exec: flambeau_backend_hip::HipGraphExec,
    pub layer_slots: Vec<super::layer::LayerPrefillSlots>,
}

impl ShardedForwardPrefillScratch {
    /// V2.25.c back-compat constructor — single lane, lane size = max_tokens.
    /// Equivalent to `new_with_lanes(model, cluster, max_tokens, 1)`.
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::new_with_lanes(model, cluster, max_tokens, 1)
    }

    /// V2.25.c — construct per-rank scratch with `u_lanes` lanes, each
    /// sized for `ubatch_size` tokens. `u_lanes = 1` is byte-identical to
    /// the pre-V2.25 layout. `u_lanes >= 2` enables V2.25.d async PP
    /// pipelining (rank k can work on ubatch i+1 while rank k+1 waits
    /// driver-side for ubatch i's peer-copy).
    ///
    /// Memory footprint per rank: `u_lanes × (2 × ubatch_size × hidden × 2
    /// bytes hidden ping-pong + LayerPrefillScratch)`.
    pub fn new_with_lanes(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        ubatch_size: usize,
        u_lanes: usize,
    ) -> Result<Self> {
        assert!(ubatch_size >= 1, "ubatch_size must be >= 1");
        assert!(u_lanes >= 1, "u_lanes must be >= 1");
        let hidden_bytes = ubatch_size * model.config.hidden_size * 2;
        // C3: size the cluster's pinned bounces to the max prefill payload
        // (decode hand-offs only need `hidden_bytes / max_tokens`; prefill
        // dominates). This is a max; `reserve_bounce_capacity` never
        // shrinks.
        cluster.reserve_bounce_capacity(hidden_bytes)?;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            // Lane 0 — the legacy fields.
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerPrefillScratch::new(&model.config, device, ubatch_size)?);
            // Extra lanes — one fresh UbatchLane per additional u_lane.
            let mut extra_lanes = Vec::with_capacity(u_lanes.saturating_sub(1));
            for _ in 1..u_lanes {
                extra_lanes.push(UbatchLane::new(&model.config, device, ubatch_size)?);
            }
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            // MTP-5h-1 — one snapshot row per local layer, allocated only
            // for GDN-bearing layers. Each snapshot holds a single hidden
            // row (position 0 of the L=2 batch) for the spec-decode reject
            // path's GDN-only re-step.
            let snapshot_row_bytes = model.config.hidden_size * 2;
            let shard = &model.shards[rank_idx];
            let mut gdn_input_snapshots: Vec<Option<DevicePtr>> =
                Vec::with_capacity(shard.layers.len());
            for layer_weights in shard.layers.iter() {
                if model.config.is_recurrent(layer_weights.layer_idx) {
                    let ptr = device.alloc(snapshot_row_bytes)?;
                    gdn_input_snapshots.push(Some(ptr));
                } else {
                    gdn_input_snapshots.push(None);
                }
            }
            per_rank.push(RankForwardPrefillScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                max_tokens: ubatch_size,
                hidden_a,
                hidden_b,
                layer,
                output_head,
                extra_lanes,
                gdn_input_snapshots,
                hidden_bytes,
                snapshot_row_bytes,
                disposed: false,
            });
        }
        // V2.26.a-i5c — pre-size the graph cache to [ranks][u_lanes]
        // of None. Populated lazily on the first ubatch that hits
        // (rank, lane) under FLAMBEAU_ASYNC_GRAPH.
        let graph_cache: Vec<Vec<Option<GraphCacheEntry>>> = (0..cluster.ranks())
            .map(|_| (0..u_lanes).map(|_| None).collect())
            .collect();
        // V2.26.a-i7b — rank-0 embed host scratch sized for max ubatch.
        // row_bytes depends on the token_embd dtype which isn't known
        // here; start empty and grow on first use.
        let embed_host = super::io::EmbedPrefillHostScratch { raw: Vec::new(), f16: Vec::new() };
        // V2.30.a — one HipEvent per (rank, GDN layer) for cross-lane
        // state_step serialisation. Allocated eagerly so no hot-path
        // None-check + device-bind branch; unused on sync / u_lanes=1.
        let mut gdn_state_events: Vec<Vec<Option<flambeau_backend_hip::HipEvent>>> =
            Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let shard = &model.shards[rank_idx];
            let mut rank_events = Vec::with_capacity(shard.layers.len());
            for layer_weights in shard.layers.iter() {
                if model.config.is_recurrent(layer_weights.layer_idx) {
                    rank_events.push(Some(flambeau_backend_hip::HipEvent::new(device.id())?));
                } else {
                    rank_events.push(None);
                }
            }
            gdn_state_events.push(rank_events);
        }
        Ok(Self {
            per_rank,
            graph_cache,
            embed_host,
            gdn_state_events,
        })
    }

    pub fn dispose(
        mut self,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Pipeline-parallel prefill over `L` tokens across an N-rank cluster.
///
/// Same per-rank flow as [`forward_one_token_pp`] but each stage processes
/// `L` tokens at once:
///   1. Rank 0 gathers `L` embeddings (one per token) into `hidden_a[0..L*H]`.
///   2. For r in 0..N:
///        - If r > 0: `peer_copy_via_host` moves `L * hidden * 2` bytes of
///          F16 hidden state from r-1's `hidden_a` to r's `hidden_a`.
///        - Bind this rank's device.
///        - Ping-pong through the rank's local layers via `forward_layer_prefill`.
///          KV cache / GDN state / conv-history get `L` tokens of history
///          appended per layer.
///        - If the final swap left the output in `hidden_b`, copy back
///          into `hidden_a` so the next rank's peer-copy has a known source.
///   3. Last rank: output head on the LAST token's hidden row + argmax.
///
/// No microbatching / 1F1B schedule — the pipeline bubble is the sum of
/// each rank's prefill compute. For Qwen3.6-31B at L=512 across 4 ranks,
/// measured prefill is compute-bound enough that microbatching is a V2+
/// perf lever, not a V1 correctness blocker.
pub fn forward_prefill_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
) -> Result<u32> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp called with empty tokens");
    }
    let max_tokens = scratch.per_rank[0].max_tokens;

    // V2.25.d — opt-in async ubatch path. Requires scratch with
    // `u_lanes >= 2` (from `new_with_lanes`) and `FLAMBEAU_ASYNC_UBATCH`
    // set. `FLAMBEAU_UBATCH` controls the ubatch size (defaults to
    // scratch.max_tokens which keeps behaviour equivalent to sync path).
    let async_enabled = std::env::var("FLAMBEAU_ASYNC_UBATCH").is_ok();
    let u_lanes = scratch.per_rank[0].u_lanes();
    if async_enabled && u_lanes >= 2 {
        let ubatch_size: usize = std::env::var("FLAMBEAU_UBATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(max_tokens);
        if l > ubatch_size {
            return forward_prefill_pp_async(
                model, session, cluster, scratch, tokens, start_position, ubatch_size,
            );
        }
    }

    // V1-BENCH-#116 — Q8 KV cache has no batched-prefill kernel
    // (attention_prefill_q8_kv doesn't exist; would need a new flash-tile
    // kernel). For the Q8 path we fall back to driving the prompt one
    // token at a time through the decode path, which already supports
    // L: CacheLayout via `forward_full_attn_decode<L>`. Slow (~10× slower
    // than batched prefill) but correct; lets the chat-slowdown user
    // flow run end-to-end on Q8 KV. Replace with a real prefill_q8_kv
    // kernel under #116 follow-up.
    if session.is_q8_kv() {
        let mut last_argmax = 0u32;
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_position + i;
            let mut decode_scratch = crate::forward::ShardedForwardOneTokenScratch::new(model, cluster)?;
            last_argmax = forward_one_token_pp(model, session, cluster, &mut decode_scratch, tok, pos)?;
            decode_scratch.dispose(cluster).ok();
        }
        return Ok(last_argmax);
    }

    // V1-BENCH-#111 (2026-04-27) — when scratch is sized below L, transparently
    // chunk: loop sequentially over ubatches of `max_tokens` tokens each. The
    // KV cache + GDN state already thread state via `start_position`, so each
    // recursive call writes its slice into the right cache positions. The
    // returned argmax is the LAST ubatch's argmax (output head runs every
    // ubatch — wasteful for non-last, but vocab×hidden is small relative to
    // a layer chain). Async path (FLAMBEAU_ASYNC_UBATCH) already chunks via
    // `forward_prefill_pp_async` and was checked above.
    if l > max_tokens {
        if max_tokens == 0 {
            bail!("forward_prefill_pp: scratch.max_tokens=0 — invalid");
        }
        let mut last_argmax = 0u32;
        let mut chunk_start = 0usize;
        while chunk_start < l {
            let chunk_end = (chunk_start + max_tokens).min(l);
            last_argmax = forward_prefill_pp(
                model, session, cluster, scratch,
                &tokens[chunk_start..chunk_end],
                start_position + chunk_start,
            )?;
            chunk_start = chunk_end;
        }
        return Ok(last_argmax);
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    // 1. Embed all L tokens on rank 0. Row-by-row host dequant + upload —
    //    matches single-device `forward_prefill`'s embedding path.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                token_id,
                scratch0.hidden_a.offset_bytes(t * row_bytes),
                hidden,
            )?;
        }
    }

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are at least `chunk_bytes` long
            // on their respective devices (sized against scratch.max_tokens
            // ≥ L); no other stream touches them here.
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_prefill(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                l,
                start_position,
                None,
                None,
            )
            .with_context(|| {
                format!(
                    "prefill rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            if std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok() {
                // Dump each token's post-layer hidden for A/B vs llama.cpp's
                // per-layer prefill output (`l_out-<il>` covers the full
                // [L, hidden] tensor in the callback trace).
                for t in 0..l {
                    let mut buf = vec![half::f16::from_f32(0.0); hidden];
                    unsafe {
                        device.memcpy_async(
                            device.default_stream(),
                            CopyDirection::DeviceToHost,
                            DevicePtr(buf.as_mut_ptr() as usize),
                            x_out.offset_bytes(t * row_bytes),
                            hidden * 2,
                        )?;
                    }
                    device.default_stream().synchronize()?;
                    let vals: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
                    let l2 = vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                    eprintln!(
                        "[prefill-dump] l_out-{} t={} ({}): L2={:.6} head={:?}",
                        layer_weights.layer_idx,
                        t,
                        if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
                        l2, &vals[..4]
                    );
                }
            }
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Normalise final hidden into hidden_a for the next peer-copy hop
        // (L rows of F16 hidden).
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    chunk_bytes,
                )?;
            }
            device.default_stream().synchronize()?;
        }
    }

    // 3. Output head on the LAST token row on the last rank.
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    let last_token_hidden = last_scratch.hidden_a.offset_bytes((l - 1) * row_bytes);
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
}

/// V2.25.d — async ubatch-pipelined prefill.
///
/// Splits `tokens[0..L]` into ubatches of size `ubatch_size` and drives
/// them across the N ranks using per-rank aux streams (V2.25.a) + async
/// peer-copies (V2.25.b) + per-lane scratch (V2.25.c). Ubatch i lives on
/// lane `i % u_lanes`; same-lane kernels serialize via stream ordering
/// (KV/GDN state coherence), different-lane kernels overlap on the
/// driver's DAG.
///
/// The caller is responsible for:
///   - constructing `scratch` via `new_with_lanes(..., ubatch_size, u_lanes)`
///     so each rank has `u_lanes` hidden+layer scratches sized for ubatch
///   - ensuring `u_lanes >= 2` — the async path has no benefit at u_lanes=1
///   - setting `FLAMBEAU_ASYNC_UBATCH` env opt-in (top-level
///     `forward_prefill_pp` routes here when set)
///
/// Returns argmax of the LAST token of the LAST ubatch.
pub fn forward_prefill_pp_async(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
    ubatch_size: usize,
) -> Result<u32> {
    use flambeau_backend_hip::HipEvent;

    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_async: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp_async called with empty tokens");
    }
    if ubatch_size == 0 {
        bail!("forward_prefill_pp_async: ubatch_size must be >= 1");
    }
    let max_ubatch = scratch.per_rank[0].max_tokens;
    if ubatch_size > max_ubatch {
        bail!(
            "forward_prefill_pp_async: ubatch_size={ubatch_size} > scratch.max_tokens={max_ubatch}"
        );
    }
    let u_lanes = scratch.per_rank[0].u_lanes();
    if u_lanes < 2 {
        bail!(
            "forward_prefill_pp_async: u_lanes={u_lanes} < 2 — no async benefit; construct scratch via new_with_lanes(..., u_lanes >= 2)"
        );
    }
    // V2.30.a — the three GDN cross-lane state-race guards (V2.27.d
    // ubatch<128/tail<128, V2.28.a-i1 gdn_per_rank>10, V2.28.c.1 K>64)
    // are all the same bug: concurrent ubatches on different aux streams
    // read/write the shared per-layer GDN state tensor with no
    // ordering. V2.30.a serialises the GDN `state_step` call itself via
    // a per-(rank, layer) HipEvent recorded in `scratch.gdn_state_events`.
    // Each lane's `stream_wait(event)` before `state_step` and `record`
    // after ensures cross-lane state_step executions are ordered, which
    // is all that is needed for parity — the rest of the GDN chain
    // reads/writes only lane-local ubatch-sized buffers.
    cluster.reserve_aux_streams(u_lanes)?;
    // V2.25.g — per-lane pinned bounces break the single-slab
    // serialisation. Size to a full ubatch worth of F16 hidden.
    let bounce_bytes = ubatch_size * model.config.hidden_size * 2;
    cluster.reserve_lane_bounces(u_lanes, bounce_bytes)?;
    // V2.26.a-i5c — opt-in graph-capture path. When set, rank r>0's
    // per-(rank, lane) layer chain is captured on the first ubatch
    // and replayed (with pos-bearing slot updates) on subsequent
    // ubatches. Rank 0 stays uncaptured (embed loop is token-varying;
    // tackled in V2.26.a-i6). Default behaviour unchanged.
    let use_async_graph = std::env::var("FLAMBEAU_ASYNC_GRAPH").is_ok();

    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let n_ubatches = l.div_ceil(ubatch_size);

    // Pre-allocate bridge events: one per (src_rank, lane) pair. Used to
    // bridge the DtoH on src_rank to the HtoD on dst_rank inside
    // peer_copy_via_host_async. Reused each ubatch using the same lane —
    // hipEventRecord overwrites the prior record.
    let mut bridge_events: Vec<Vec<HipEvent>> = Vec::with_capacity(n_ranks.saturating_sub(1));
    for r in 0..n_ranks.saturating_sub(1) {
        let device = cluster.device(r);
        device.bind()?;
        let mut row = Vec::with_capacity(u_lanes);
        for _ in 0..u_lanes {
            row.push(HipEvent::new(device.id())?);
        }
        bridge_events.push(row);
    }

    let shards = &model.shards;

    // V2.25.h — interleaved 1F1B dispatch. At time step t, rank r
    // processes ubatch (t - r) if in [0, n_ubatches). At steady state
    // (t in [n_ranks-1, n_ubatches-1]), every rank is dispatching a
    // different ubatch concurrently, producing real pipeline fill.
    //
    // Pre-V2.25.h dispatched `for ub in 0..n_ubatches { for rank in
    // 0..n_ranks }` which is serial-across-ubatches. That wasted the
    // aux-stream / per-lane-bounce infrastructure because rank r's
    // lane-k work finished before rank r ever started lane-k+1 work.
    let n_timesteps = n_ranks + n_ubatches - 1;
    for t in 0..n_timesteps {
        for rank_idx in 0..n_ranks {
            let ub_idx_signed = t as isize - rank_idx as isize;
            if ub_idx_signed < 0 || (ub_idx_signed as usize) >= n_ubatches {
                continue;
            }
            let ub_idx = ub_idx_signed as usize;
            let lane = ub_idx % u_lanes;
            let start = ub_idx * ubatch_size;
            let end = (start + ubatch_size).min(l);
            let u = end - start;
            let chunk = &tokens[start..end];
            let pos = start_position + start;
            let chunk_bytes = u * row_bytes;

        // ----- Rank 0: embed + layers -----
        if rank_idx == 0 {
            let rank0 = cluster.device(0);
            rank0.bind()?;
            let shard0 = &shards[0];
            let token_embd = shard0
                .token_embd
                .as_ref()
                .context("rank 0 shard missing token_embd")?;
            // Split-borrow: embed_host and per_rank are disjoint
            // fields of scratch; we need mutable access to both.
            let embed_host: *mut super::io::EmbedPrefillHostScratch = &mut scratch.embed_host;
            let scratch0 = &mut scratch.per_rank[0];
            let lane_hidden_a = scratch0.lane_hidden_a(lane);

            // V2.26.a-i7b — batched embed (one sync per ubatch
            // instead of 2·u). Under 1F1B the Rust driver thread
            // returned here sooner → more time for other-rank
            // dispatches.
            cluster.with_aux_stream(0, lane, |s| -> flambeau_core::DeviceResult<()> {
                // SAFETY: embed_host is a *mut into scratch.embed_host
                // — disjoint from scratch.per_rank we borrowed above,
                // so no aliasing.
                let embed_host_ref = unsafe { &mut *embed_host };
                super::io::forward_embed_prefill_batch(
                    rank0,
                    s,
                    token_embd,
                    chunk,
                    lane_hidden_a,
                    hidden,
                    embed_host_ref,
                )
                .map_err(|e| flambeau_core::DeviceError::Backend {
                    backend: "hip",
                    code: -1,
                    message: format!("embed_batch: {e}"),
                })
            })?;

            // Run rank 0's layers on the same aux stream.
            let rank_session = &mut session.per_rank[0];
            let lane_hidden_b = scratch0.lane_hidden_b(lane);
            let layer_scratch = scratch0
                .lane_layer_mut(lane)
                .context("rank 0 lane_layer_mut")?;
            // V2.30.a — event vector for this rank's GDN layers. Disjoint
            // from scratch.per_rank borrowed above.
            let rank_events: *mut Vec<Option<flambeau_backend_hip::HipEvent>> =
                &mut scratch.gdn_state_events[0];

            cluster.with_aux_stream(0, lane, |s| -> flambeau_core::DeviceResult<()> {
                // SAFETY: rank_events is *mut into scratch.gdn_state_events[0],
                // disjoint from scratch.per_rank[0] borrowed above.
                let rank_events_ref = unsafe { &mut *rank_events };
                let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                for (local_idx, layer_weights) in shard0.layers.iter().enumerate() {
                    let layer_cache = &mut rank_session.caches[local_idx];
                    let gdn_event = rank_events_ref[local_idx].as_ref();
                    forward_layer_prefill(
                        &shard0.ops,
                        s,
                        rank0,
                        cfg,
                        layer_weights,
                        layer_cache,
                        layer_scratch,
                        x_in,
                        x_out,
                        u,
                        pos,
                        None,
                        gdn_event,
                    )
                    .map_err(|e| flambeau_core::DeviceError::Backend {
                        backend: "hip",
                        code: -1,
                        message: format!("prefill rank 0 layer {}: {e}", layer_weights.layer_idx),
                    })?;
                    std::mem::swap(&mut x_in, &mut x_out);
                }
                // Normalise final hidden into lane's hidden_a for peer-copy.
                if x_in != lane_hidden_a {
                    unsafe {
                        rank0.memcpy_async(
                            s,
                            CopyDirection::DeviceToDevice,
                            lane_hidden_a,
                            x_in,
                            chunk_bytes,
                        )?;
                    }
                }
                Ok(())
            })?;

        } else {
            // ----- Rank r > 0: peer-copy + layers -----
            let device = cluster.device(rank_idx);

            // Gather the raw stream handles for the cross-rank peer-copy.
            // We hold both mutexes simultaneously (different rank entries →
            // different Mutex instances; no deadlock).
            let src_dst_copy_result: Result<()> =
                cluster.with_aux_stream(rank_idx - 1, lane, |src_stream| {
                    cluster.with_aux_stream(rank_idx, lane, |dst_stream| {
                        unsafe {
                            // V2.25.g — use per-lane bounce to allow
                            // concurrent DtoH across lanes on the same rank.
                            cluster.peer_copy_via_host_async_laned(
                                scratch.per_rank[rank_idx].lane_hidden_a(lane),
                                rank_idx,
                                scratch.per_rank[rank_idx - 1].lane_hidden_a(lane),
                                rank_idx - 1,
                                chunk_bytes,
                                src_stream,
                                dst_stream,
                                &bridge_events[rank_idx - 1][lane],
                                None,
                                Some(lane),
                            )?;
                        }
                        Ok(())
                    })
                })
                .map_err(|e| anyhow::anyhow!("peer_copy_async r{}->{}: {e}", rank_idx - 1, rank_idx));
            src_dst_copy_result?;

            device.bind()?;
            let shard = &shards[rank_idx];
            let rank_session = &mut session.per_rank[rank_idx];
            // Split-borrow scratch: graph_cache[rank][lane] and per_rank
            // are disjoint fields of scratch, so Rust lets us borrow
            // both mutably at once via explicit field access.
            let graph_slot_ptr: *mut Option<GraphCacheEntry> = if use_async_graph {
                &mut scratch.graph_cache[rank_idx][lane]
            } else {
                std::ptr::null_mut()
            };
            // V2.30.a — pull GDN state events out before borrowing per_rank.
            // Disjoint field of scratch.
            let rank_events_ptr: *mut Vec<Option<flambeau_backend_hip::HipEvent>> =
                &mut scratch.gdn_state_events[rank_idx];
            let rank_scratch = &mut scratch.per_rank[rank_idx];
            let lane_hidden_a = rank_scratch.lane_hidden_a(lane);
            let lane_hidden_b = rank_scratch.lane_hidden_b(lane);
            let layer_scratch = rank_scratch
                .lane_layer_mut(lane)
                .context("lane_layer_mut")?;

            // V2.26.a-i5c — if FLAMBEAU_ASYNC_GRAPH is set and the
            // (rank, lane) cache is empty, capture; if set and cache
            // is populated, replay with slot updates; otherwise fall
            // through to the uncaptured closure.
            // SAFETY: graph_slot_ptr is non-null iff use_async_graph is
            // true; it was derived from a unique &mut, and no other
            // borrow of scratch.graph_cache is live here.
            let graph_slot: Option<&mut Option<GraphCacheEntry>> = if use_async_graph {
                Some(unsafe { &mut *graph_slot_ptr })
            } else {
                None
            };

            let needs_capture = graph_slot
                .as_ref()
                .map(|s| s.is_none())
                .unwrap_or(false);
            let needs_replay = graph_slot
                .as_ref()
                .map(|s| s.is_some())
                .unwrap_or(false);

            if needs_replay {
                // === REPLAY branch ===
                let entry = graph_slot.as_ref().unwrap().as_ref().unwrap();
                // Stable per-layer backing for the updated pos scalars.
                // Must outlive the set_slot call (driver copies values
                // at submit time). We only update slots for full-attn
                // layers — GDN layers' state advances naturally inside
                // the captured graph via in-place state buffer r/w.
                let n_layers = entry.layer_slots.len();
                let mut n_k_vals: Vec<i32> = vec![0; n_layers];
                let mut q_off_vals: Vec<i32> = vec![0; n_layers];
                for local_idx in 0..n_layers {
                    let layer_cache = &rank_session.caches[local_idx];
                    // V1-BENCH-#116 — graph-capture path is F16-only. Q8 KV
                    // bypasses capture (no perf data yet on whether async
                    // graph helps Q8 decode); see forward_full_attn_decode's
                    // bail when slots is Some + L = Q8Contig.
                    let kv = match layer_cache {
                        LayerCache::FullAttn(kv) => kv,
                        LayerCache::FullAttnQ8(_) => continue,  // Q8: no async graph
                        LayerCache::Gdn(_) => continue,  // GDN: no slots
                    };
                    n_k_vals[local_idx] = (pos + u) as i32;
                    q_off_vals[local_idx] = pos as i32;
                    // F16 → 2 bytes per element.
                    let per_token_bytes = kv.n_heads() * kv.head_dim() * 2;
                    let k_dst = kv.k_buffer().offset_bytes(pos * per_token_bytes);
                    let v_dst = kv.v_buffer().offset_bytes(pos * per_token_bytes);
                    let slots = entry.layer_slots[local_idx].full_attn;
                    // SAFETY: n_k_vals / q_off_vals live until end of
                    // this branch (past the launch below). k_dst/v_dst
                    // are device pointers that stay valid for the
                    // exec's lifetime.
                    unsafe {
                        entry.exec.set_slot(slots.n_k_slot, &n_k_vals[local_idx])
                            .with_context(|| format!(
                                "set_slot n_k rank={rank_idx} layer={local_idx}"
                            ))?;
                        entry.exec.set_slot(slots.q_off_slot, &q_off_vals[local_idx])
                            .with_context(|| format!(
                                "set_slot q_off rank={rank_idx} layer={local_idx}"
                            ))?;
                        entry.exec.set_memcpy_slot(slots.k_append_slot, k_dst)
                            .with_context(|| format!(
                                "set_memcpy k_dst rank={rank_idx} layer={local_idx}"
                            ))?;
                        entry.exec.set_memcpy_slot(slots.v_append_slot, v_dst)
                            .with_context(|| format!(
                                "set_memcpy v_dst rank={rank_idx} layer={local_idx}"
                            ))?;
                    }
                }
                // Launch the cached exec on this lane's aux stream.
                cluster.with_aux_stream(rank_idx, lane, |s| entry.exec.launch(s))?;
                // Manually bump each full-attn layer's tail — the
                // captured kv_cache_append_hip_slot's Rust-side bump
                // only ran at capture time (it's not part of the graph
                // replay). GDN layers don't have a tail counter.
                for local_idx in 0..n_layers {
                    let layer_cache = &mut rank_session.caches[local_idx];
                    if let LayerCache::FullAttn(kv) = layer_cache {
                        kv.bump_tail(u).map_err(|e| anyhow::anyhow!(
                            "bump_tail rank={rank_idx} layer={local_idx}: {e}"
                        ))?;
                    }
                }
            } else if needs_capture {
                // === CAPTURE branch (first ubatch on this lane) ===
                let layer_slots: Vec<super::layer::LayerPrefillSlots> = (0..shard.layers.len())
                    .map(|_| super::layer::LayerPrefillSlots {
                        full_attn: super::attn::AttnPrefillSlots {
                            n_k_slot: flambeau_backend_hip::ScalarSlot::new(),
                            q_off_slot: flambeau_backend_hip::ScalarSlot::new(),
                            k_append_slot: flambeau_backend_hip::MemcpySlot::new(),
                            v_append_slot: flambeau_backend_hip::MemcpySlot::new(),
                        },
                    })
                    .collect();

                // Capture under the aux stream. The captured graph
                // includes: layer chain + optional tail DtoD copy to
                // lane_hidden_a. kv_cache_append_hip_slot's Rust-side
                // bump_tail runs ONCE during capture (advancing each
                // layer's tail by u), which matches this first
                // ubatch's correct final state.
                let exec = cluster.with_aux_stream(rank_idx, lane, |aux_s| {
                    flambeau_backend_hip::HipGraphExec::capture(aux_s, |capture_s| {
                        // SAFETY: rank_events_ptr is a *mut into
                        // scratch.gdn_state_events[rank_idx], disjoint from
                        // scratch.per_rank[rank_idx] borrowed above.
                        let rank_events_ref = unsafe { &mut *rank_events_ptr };
                        let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
                            let layer_cache = &mut rank_session.caches[local_idx];
                            let gdn_event = rank_events_ref[local_idx].as_ref();
                            forward_layer_prefill(
                                &shard.ops,
                                capture_s,
                                device,
                                cfg,
                                layer_weights,
                                layer_cache,
                                layer_scratch,
                                x_in,
                                x_out,
                                u,
                                pos,
                                Some(layer_slots[local_idx]),
                                gdn_event,
                            )
                            .map_err(|e| flambeau_core::DeviceError::Backend {
                                backend: "hip",
                                code: -1,
                                message: format!(
                                    "capture prefill rank {} layer {}: {e}",
                                    rank_idx, layer_weights.layer_idx
                                ),
                            })?;
                            std::mem::swap(&mut x_in, &mut x_out);
                        }
                        if x_in != lane_hidden_a {
                            // SAFETY: lane_hidden_a / x_in are live
                            // device ptrs; chunk_bytes is bounded.
                            unsafe {
                                device.memcpy_async(
                                    capture_s,
                                    CopyDirection::DeviceToDevice,
                                    lane_hidden_a,
                                    x_in,
                                    chunk_bytes,
                                )?;
                            }
                        }
                        Ok(())
                    })
                })?;

                // Launch the freshly-instantiated exec to execute the
                // captured work for this first ubatch.
                cluster.with_aux_stream(rank_idx, lane, |s| exec.launch(s))?;

                // Stash in cache for subsequent ubatches.
                *graph_slot.unwrap() = Some(GraphCacheEntry { exec, layer_slots });
            } else {
                // === UNCAPTURED branch (legacy async) ===
                cluster.with_aux_stream(rank_idx, lane, |s| -> flambeau_core::DeviceResult<()> {
                    // SAFETY: rank_events_ptr is a *mut into
                    // scratch.gdn_state_events[rank_idx], disjoint from
                    // scratch.per_rank[rank_idx] borrowed above.
                    let rank_events_ref = unsafe { &mut *rank_events_ptr };
                    let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                    for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
                        let layer_cache = &mut rank_session.caches[local_idx];
                        let gdn_event = rank_events_ref[local_idx].as_ref();
                        forward_layer_prefill(
                            &shard.ops,
                            s,
                            device,
                            cfg,
                            layer_weights,
                            layer_cache,
                            layer_scratch,
                            x_in,
                            x_out,
                            u,
                            pos,
                            None,
                            gdn_event,
                        )
                        .map_err(|e| flambeau_core::DeviceError::Backend {
                            backend: "hip",
                            code: -1,
                            message: format!(
                                "prefill rank {} layer {}: {e}",
                                rank_idx, layer_weights.layer_idx
                            ),
                        })?;
                        std::mem::swap(&mut x_in, &mut x_out);
                    }
                    if x_in != lane_hidden_a {
                        unsafe {
                            device.memcpy_async(
                                s,
                                CopyDirection::DeviceToDevice,
                                lane_hidden_a,
                                x_in,
                                chunk_bytes,
                            )?;
                        }
                    }
                    Ok(())
                })?;
            }
        }
        } // close for rank_idx
    } // close for t (outer 1F1B timestep loop)

    // ----- Output head on the last rank after the FINAL ubatch -----
    let last_idx = n_ranks - 1;
    let last_ub = n_ubatches - 1;
    let last_lane = last_ub % u_lanes;
    let last_start = last_ub * ubatch_size;
    let last_end = (last_start + ubatch_size).min(l);
    let last_u = last_end - last_start;

    let last_device = cluster.device(last_idx);
    last_device.bind()?;

    // Sync the last rank's lane so the output head sees completed hidden.
    cluster.with_aux_stream(last_idx, last_lane, |s| {
        flambeau_core::Stream::synchronize(s)
    })?;

    let last_shard = &shards[last_idx];
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    // Compute the lane hidden ptr before the mutable borrow of output_head.
    let last_lane_hidden_a = last_scratch.lane_hidden_a(last_lane);
    let last_token_hidden = last_lane_hidden_a.offset_bytes((last_u - 1) * row_bytes);
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
}

/// Variant of [`forward_prefill_pp`] that downloads the F32 logit row for
/// the last token into `logits_out` instead of argmax-ing on host. See
/// [`forward_one_token_pp_logits`] for the streaming rationale.
pub fn forward_prefill_pp_logits(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_logits: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp_logits called with empty tokens");
    }
    let max_tokens = scratch.per_rank[0].max_tokens;
    if l > max_tokens {
        bail!(
            "forward_prefill_pp_logits: L={l} > scratch.max_tokens={max_tokens}; caller must chunk"
        );
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                token_id,
                scratch0.hidden_a.offset_bytes(t * row_bytes),
                hidden,
            )?;
        }
    }

    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_prefill(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                l,
                start_position,
                None,
                None,
            )
            .with_context(|| {
                format!(
                    "prefill rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    chunk_bytes,
                )?;
            }
            device.default_stream().synchronize()?;
        }
    }

    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    let last_token_hidden = last_scratch.hidden_a.offset_bytes((l - 1) * row_bytes);
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    download_logits_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
        logits_out,
    )
}

/// MTP-5e — paired-logits L=2 primitive for K=1 spec-decode verify.
///
/// Identical body to [`forward_prefill_pp_logits`] except `tokens.len()`
/// is required to be 2 and the LM-head pass runs **twice** on the last
/// rank: once on `hidden_a[0]` (predicts token at `start_position+1`),
/// once on `hidden_a[1]` (predicts token at `start_position+2` given the
/// draft). On return, `logits_out_pos0` and `logits_out_pos1` each hold
/// `[vocab]` F32 logits.
///
/// This replaces the 2× sequential L=1 fallback in `forward/spec.rs`
/// — the body of the model only runs once, amortising the per-layer
/// launch overhead and (more importantly on hybrid arch) only paying
/// one GDN state-step pair instead of two.
pub fn forward_prefill_pp_logits_paired_l2(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
    logits_out_pos0: &mut Vec<f32>,
    // MTP-5h-Lever-C — when None, skip the pos1 head + download. Caller
    // can fetch pos1 logits later via [`forward_output_head_at_pp`] if
    // accept-path needs them. Eliminates ~3 ms/macro of always-paid head
    // work on reject paths (saves ~0.4 ms/macro avg at 12.5 % rejects).
    logits_out_pos1: Option<&mut Vec<f32>>,
) -> Result<()> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_logits_paired_l2: zero-rank cluster");
    }
    if tokens.len() != 2 {
        bail!(
            "forward_prefill_pp_logits_paired_l2: requires L=2, got L={}",
            tokens.len()
        );
    }
    let l = 2;
    let max_tokens = scratch.per_rank[0].max_tokens;
    if l > max_tokens {
        bail!(
            "forward_prefill_pp_logits_paired_l2: L=2 > scratch.max_tokens={max_tokens}"
        );
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    // Embed both tokens on rank 0.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        flambeau_backend_hip::profile::mark("l2_step_start", rank0, rank0.default_stream())?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                token_id,
                scratch0.hidden_a.offset_bytes(t * row_bytes),
                hidden,
            )?;
        }
        flambeau_backend_hip::profile::mark("l2_embed_done", rank0, rank0.default_stream())?;
    }

    // Per-rank: peer-copy hidden in, run all owned layers at L=2.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }
        device.bind()?;
        flambeau_backend_hip::profile::mark(
            "l2_stage_start",
            device,
            device.default_stream(),
        )?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        // MTP-5h-1 — copy snapshot ptrs out before borrowing the layer
        // scratch mutably. DevicePtr is Copy, so this is a cheap clone.
        let gdn_snapshots: Vec<Option<DevicePtr>> = rank_scratch.gdn_input_snapshots.clone();
        let snapshot_row_bytes = rank_scratch.snapshot_row_bytes;
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            // MTP-5h-1 — for GDN-bearing layers, save x_in (position 0
            // only) to the per-layer snapshot buffer for the spec-decode
            // reject path. DtoD memcpy on this rank's default stream;
            // ordered before forward_layer_prefill's own kernels.
            if let Some(snap_ptr) = gdn_snapshots[local_idx] {
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToDevice,
                        snap_ptr,
                        x_in,
                        snapshot_row_bytes,
                    )?;
                }
            }
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_prefill(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                l,
                start_position,
                None,
                None,
            )
            .with_context(|| {
                format!(
                    "paired-L2 prefill rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    chunk_bytes,
                )?;
            }
            device.default_stream().synchronize()?;
        }
        flambeau_backend_hip::profile::mark(
            "l2_stage_end",
            device,
            device.default_stream(),
        )?;
    }

    // Last rank: run output head twice and download both logit rows.
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    flambeau_backend_hip::profile::mark(
        "l2_output_head_start",
        last_device,
        last_device.default_stream(),
    )?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;

    // Position 0 of the L=2 batch.
    let h_pos0 = last_scratch.hidden_a;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        h_pos0,
    )?;
    download_logits_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
        logits_out_pos0,
    )?;
    flambeau_backend_hip::profile::mark(
        "l2_output_pos0_done",
        last_device,
        last_device.default_stream(),
    )?;

    // Position 1. Output-head scratch is reused; logits buffer is
    // overwritten by the next mmvq, so we had to download pos0 first.
    // MTP-5h-Lever-C — only run pos1 head when caller asks for it; else
    // skip and let caller defer this work to the accept branch.
    if let Some(logits_out_pos1) = logits_out_pos1 {
        let h_pos1 = last_scratch.hidden_a.offset_bytes(row_bytes);
        forward_output_head_decode(
            &last_shard.ops,
            last_device.default_stream(),
            cfg,
            output_norm,
            lm_head,
            output_head_scratch,
            h_pos1,
        )?;
        download_logits_host(
            last_device,
            last_device.default_stream(),
            output_head_scratch.logits_f32,
            cfg.vocab_size,
            logits_out_pos1,
        )?;
        flambeau_backend_hip::profile::mark(
            "l2_output_pos1_done",
            last_device,
            last_device.default_stream(),
        )?;
    }

    Ok(())
}

/// MTP-5h-Lever-C — run the LM head on a single hidden row located at
/// `last_scratch.hidden_a + position * hidden_bytes`. Used by the spec
/// driver to lazily compute pos1 logits only on accept.
pub fn forward_output_head_at_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    position_in_l2: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_output_head_at_pp: zero-rank cluster");
    }
    let cfg = &model.config;
    let row_bytes = cfg.hidden_size * 2;
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    let h_at = last_scratch.hidden_a.offset_bytes(position_in_l2 * row_bytes);
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        h_at,
    )?;
    download_logits_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
        logits_out,
    )?;
    Ok(())
}

