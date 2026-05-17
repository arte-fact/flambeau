//! `PpForwardCtx` — pipeline-parallel forward context.
//!
//! Each rank owns a contiguous slice of layers `[layer_start, layer_end)`.
//! Composites that act locally (rmsnorm, residual_add, standard_attn,
//! dense_ffn, moe_ffn) delegate to `core::composites::*_local`
//! unchanged. Three points are rank-aware:
//!
//! - `embed` — rank 0 does the actual lookup. Ranks > 0 receive the
//!   residual from the previous rank's `output_head` via `peer_buffer`
//!   (a host-side F16 staging vec the caller threads through to both
//!   contexts).
//! - `output_head` — only the last rank does the actual rmsnorm + LM
//!   head + DtoH. Non-last ranks DtoH-stage the residual into
//!   `peer_buffer` for the next rank.
//! - `layer_range` — yields this rank's slice.
//!
//! The `peer_buffer` plumbing is host-roundtrip (DtoH → HtoD) because
//! it works across any HIP topology, including non-P2P pairs. A device
//! peer-copy fast path is a future optimisation behind the same trait
//! signature.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;

use crate::core::{composites, CoreState, ScratchPool, TopologyHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// Topology hooks for PP. No-op for the empty-trait surface today;
/// AR / handoff entrypoints will land here as the hook trait grows.
pub struct PpHooks;

impl TopologyHooks for PpHooks {}

/// Pipeline-parallel forward context.
///
/// `peer_buffer` is a host-side F16 staging slot of length `hidden`,
/// shared with the neighbouring rank's ctx. Convention: rank `r` writes
/// its post-final-layer residual into `peer_buffer` at `output_head` if
/// it isn't the last rank; rank `r+1` reads it at `embed` if it isn't
/// rank 0.
pub struct PpForwardCtx<'a> {
    core: CoreState<'a>,
    hooks: PpHooks,
    rank: usize,
    n_ranks: usize,
    layer_start: usize,
    layer_end: usize,
    peer_buffer: &'a mut Vec<f16>,
}

impl<'a> PpForwardCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
        rank: usize,
        n_ranks: usize,
        layer_start: usize,
        layer_end: usize,
        peer_buffer: &'a mut Vec<f16>,
    ) -> Self {
        let mut core = CoreState::new(device, stream, reg, pool);
        core.layer_idx_offset = layer_start;
        Self {
            core,
            hooks: PpHooks,
            rank,
            n_ranks,
            layer_start,
            layer_end,
            peer_buffer,
        }
    }

    fn is_first_rank(&self) -> bool {
        self.rank == 0
    }

    fn is_last_rank(&self) -> bool {
        self.rank + 1 == self.n_ranks
    }
}

impl ForwardCtx for PpForwardCtx<'_> {
    fn embed(&mut self, weights: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        if self.is_first_rank() {
            composites::embed_local(&mut self.core, &mut self.hooks, weights, token_id)
        } else {
            // Peer-receive: HtoD from the host staging buffer the previous
            // rank's `output_head` populated.
            let hidden = self.core.hidden();
            if self.peer_buffer.len() != hidden {
                anyhow::bail!(
                    "embed: peer_buffer len {} != hidden {hidden} (rank {})",
                    self.peer_buffer.len(),
                    self.rank
                );
            }
            let dst = self.core.pool.next_residual_slot();
            let bytes = hidden * 2;
            // SAFETY: dst is a residual slot sized for `hidden * 2`;
            // peer_buffer holds `hidden` F16 host elems.
            unsafe {
                self.core
                    .device
                    .memcpy_async(
                        self.core.stream,
                        CopyDirection::HostToDevice,
                        dst,
                        DevicePtr(self.peer_buffer.as_ptr() as usize),
                        bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("pp embed peer-receive HtoD: {e}"))?;
            }
            Ok(unsafe { Tensor::<F16>::from_raw(dst, hidden) })
        }
    }

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
    ) -> Result<Tensor<F16>> {
        composites::rmsnorm_local(&mut self.core, &mut self.hooks, input, weight, eps)
    }

    fn residual_add(&mut self, a: Tensor<F16>, b: Tensor<F16>) -> Result<Tensor<F16>> {
        composites::residual_add_local(&mut self.core, &mut self.hooks, a, b)
    }

    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>> {
        composites::standard_attn_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            layer_idx,
            position,
        )
    }

    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>> {
        composites::dense_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn moe_ffn(&mut self, input: &Tensor<F16>, weights: &MoeWeights) -> Result<Tensor<F16>> {
        composites::moe_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn output_head(&mut self, input: &Tensor<F16>, lm_head: &LmHeadWeights) -> Result<()> {
        if self.is_last_rank() {
            composites::output_head_local(&mut self.core, &mut self.hooks, input, lm_head)
        } else {
            // Peer-send: DtoH this rank's post-final-layer residual into
            // the shared host staging buffer. Rank+1's `embed` will HtoD
            // it into its residual slot.
            let hidden = self.core.hidden();
            if self.peer_buffer.len() != hidden {
                *self.peer_buffer = vec![f16::ZERO; hidden];
            }
            let bytes = hidden * 2;
            // SAFETY: input.ptr is a device residual slot of `hidden * 2`
            // bytes; peer_buffer is a `hidden`-element host F16 vec.
            unsafe {
                self.core
                    .device
                    .memcpy_async(
                        self.core.stream,
                        CopyDirection::DeviceToHost,
                        DevicePtr(self.peer_buffer.as_mut_ptr() as usize),
                        input.ptr,
                        bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("pp output_head peer-send DtoH: {e}"))?;
            }
            // Sync so the host buffer is observable by the next rank
            // before its `embed` reads it.
            flambeau_core::Stream::synchronize(self.core.stream)?;
            Ok(())
        }
    }

    fn layer_range<'b>(
        &'b mut self,
        _layout: &'b ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'b> {
        Box::new(self.layer_start..self.layer_end)
    }

    fn logits(&self) -> &[f32] {
        &self.core.logits_host
    }
}
