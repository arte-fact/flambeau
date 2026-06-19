//! Single-device session over any backend bundle. `SoloSession<B, A>` runs an
//! `Arch` on one device through `ForwardEngine::<B>::new_solo` inline (no worker
//! thread, no cluster, no AR) and exposes the same slot-aware forward / KV
//! surface the worker-based `Session<A>` does. It is the path the HTTP server
//! drives for `Topology::SingleDevice` on either backend (the only path CUDA
//! supports today — multi-device CUDA needs cluster peer-copy).

use anyhow::{anyhow, Result};
use flambeau_backend::Backend;
use flambeau_core::Device;
use flambeau_quant::GgufFile;

use crate::core::ScratchPool;
use crate::ctx::ForwardCtx;
use crate::engine::ForwardEngine;
use crate::loader::ShardMode;
use crate::runtime::orchestrate::LaunchParams;
use crate::runtime::Arch;

/// Single-device, single-rank session. Owns the device, registry, scratch pool
/// and model; every forward builds a fresh `ForwardEngine::<B>::new_solo` over
/// them. Field order puts `device` last so it drops last (after pool/model
/// release their device memory).
pub struct SoloSession<B: Backend, A: Arch> {
    model: A::Model,
    pool: ScratchPool,
    reg: B::Registry,
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
        let LaunchParams { ctx_cap, prefill_ubatch, max_slots, paged_kv_pages, kv_layout } = params;
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
            disposed: false,
        })
    }

    pub fn prefill_ubatch(&self) -> usize {
        self.prefill_ubatch
    }

    /// Core forward: one `ForwardEngine::<B>::new_solo` pass over `tokens`.
    /// Logits for the last token (or N rows under distinct slot_ids) land in
    /// `self.last_logits`.
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
        let stream = self.device.default_stream();
        let mut ctx = ForwardEngine::<B>::new_solo(&self.device, stream, &self.reg, &mut self.pool);
        A::forward(&self.model, &mut ctx, tokens, positions, slot_ids)?;
        let logits = ctx.logits().to_vec();
        self.last_logits = logits;
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
        let _ = self.pool.dispose(&self.device);
        let _ = A::dispose(&mut self.model, &self.device);
    }
}
