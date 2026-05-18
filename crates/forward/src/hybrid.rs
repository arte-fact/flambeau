//! PP-of-TP forward context. Outer ring: PP stages handing the
//! residual off via `peer_buffer`. Inner ring: TP within each stage,
//! AR-summing partials after row-parallel matmuls.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;
use std::sync::{Arc, Barrier, Mutex};

use crate::core::{composites, CoreState, ScratchPool, TopologyHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

/// `ar_callback` operates on the rank's stage (sums `tp_size` ranks).
pub struct HybridHooks {
    pub rank_in_stage: usize,
    pub tp_size: usize,
    pub ar_callback: Box<
        dyn FnMut(
                usize,
                usize,
                flambeau_core::DevicePtr,
                usize,
                &HipDevice,
                &HipStream,
            ) -> Result<()>
            + Send,
    >,
}

impl TopologyHooks for HybridHooks {
    fn ar_sum_f32(
        &mut self,
        buf: flambeau_core::DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        if self.tp_size <= 1 {
            return Ok(());
        }
        (self.ar_callback)(self.rank_in_stage, self.tp_size, buf, n_elems, device, stream)
    }
}

pub struct HybridForwardCtx<'a> {
    core: CoreState<'a>,
    hooks: HybridHooks,
    stage_idx: usize,
    n_stages: usize,
    layer_start: usize,
    layer_end: usize,
    /// Host F16 staging slot (length `hidden`). Stage S rank 0
    /// deposits at `output_head`; every rank of stage S+1 reads at
    /// `embed`. The handoff barrier guards exclusive access.
    peer_buffer: Arc<Mutex<Vec<f16>>>,
    /// Total-rank barrier sat at every stage boundary so peer_buffer
    /// is observable across threads before any reader pulls it in.
    handoff_barrier: Arc<Barrier>,
}

impl<'a> HybridForwardCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
        stage_idx: usize,
        n_stages: usize,
        rank_in_stage: usize,
        tp_size: usize,
        layer_start: usize,
        layer_end: usize,
        ar_callback: Box<
            dyn FnMut(
                    usize,
                    usize,
                    flambeau_core::DevicePtr,
                    usize,
                    &HipDevice,
                    &HipStream,
                ) -> Result<()>
                + Send,
        >,
        peer_buffer: Arc<Mutex<Vec<f16>>>,
        handoff_barrier: Arc<Barrier>,
    ) -> Self {
        let mut core = CoreState::new(device, stream, reg, pool);
        core.layer_idx_offset = layer_start;
        Self {
            core,
            hooks: HybridHooks {
                rank_in_stage,
                tp_size,
                ar_callback,
            },
            stage_idx,
            n_stages,
            layer_start,
            layer_end,
            peer_buffer,
            handoff_barrier,
        }
    }

    fn is_first_stage(&self) -> bool {
        self.stage_idx == 0
    }

    fn is_last_stage(&self) -> bool {
        self.stage_idx + 1 == self.n_stages
    }

    fn is_rank_zero_in_stage(&self) -> bool {
        self.hooks.rank_in_stage == 0
    }
}

impl ForwardCtx for HybridForwardCtx<'_> {
    fn embed(&mut self, weights: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        if self.is_first_stage() {
            // Replicated lookup on every TP rank within stage 0.
            composites::embed_local(&mut self.core, &mut self.hooks, weights, token_id)
        } else {
            // Wait for prev stage to populate peer_buffer.
            self.handoff_barrier.wait();
            let hidden = self.core.hidden();
            let dst = self.core.pool.next_residual_slot();
            let bytes = hidden * 2;
            {
                let buf = self
                    .peer_buffer
                    .lock()
                    .map_err(|e| anyhow::anyhow!("peer_buffer poisoned: {e}"))?;
                if buf.len() != hidden {
                    anyhow::bail!(
                        "embed: peer_buffer len {} != hidden {hidden}",
                        buf.len()
                    );
                }
                // SAFETY: dst sized for hidden F16; buf holds `hidden` host F16.
                unsafe {
                    self.core
                        .device
                        .memcpy_async(
                            self.core.stream,
                            CopyDirection::HostToDevice,
                            dst,
                            DevicePtr(buf.as_ptr() as usize),
                            bytes,
                        )
                        .map_err(|e| anyhow::anyhow!("hybrid embed peer-receive HtoD: {e}"))?;
                }
                flambeau_core::Stream::synchronize(self.core.stream)?;
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

    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &crate::ctx::GdnWeights,
        layer_idx: usize,
    ) -> Result<Tensor<F16>> {
        composites::gdn_layer_local(&mut self.core, &mut self.hooks, input, weights, layer_idx)
    }

    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>> {
        composites::dense_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn moe_ffn(&mut self, input: &Tensor<F16>, weights: &MoeWeights) -> Result<Tensor<F16>> {
        composites::moe_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn output_head(&mut self, input: &Tensor<F16>, lm_head: &LmHeadWeights) -> Result<()> {
        if self.is_last_stage() {
            composites::output_head_local(&mut self.core, &mut self.hooks, input, lm_head)
        } else {
            // Only rank 0 deposits — residual is replicated within stage.
            if self.is_rank_zero_in_stage() {
                let hidden = self.core.hidden();
                let mut buf = self
                    .peer_buffer
                    .lock()
                    .map_err(|e| anyhow::anyhow!("peer_buffer poisoned: {e}"))?;
                if buf.len() != hidden {
                    *buf = vec![f16::ZERO; hidden];
                }
                let bytes = hidden * 2;
                // SAFETY: input.ptr sized for hidden F16; buf is hidden host F16.
                unsafe {
                    self.core
                        .device
                        .memcpy_async(
                            self.core.stream,
                            CopyDirection::DeviceToHost,
                            DevicePtr(buf.as_mut_ptr() as usize),
                            input.ptr,
                            bytes,
                        )
                        .map_err(|e| anyhow::anyhow!("hybrid output_head peer-send DtoH: {e}"))?;
                }
                flambeau_core::Stream::synchronize(self.core.stream)?;
            }
            // Every rank meets the barrier so next stage's embed sees the deposit.
            self.handoff_barrier.wait();
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
