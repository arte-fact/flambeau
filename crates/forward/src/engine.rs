//! Generic forward engine. `TopologyHooks` covers intra-stage AR;
//! `StageHooks` covers inter-stage role + peer-copy. SD/TP use
//! `SoloStage` (no peering); PP uses `PpStage`; Hybrid uses `HybStage`
//! (peer-buffer + handoff barrier).

use std::ops::Range;
use std::sync::{Arc, Barrier, Mutex};

use anyhow::{anyhow, bail, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;

use crate::core::{composites, CoreState, NoopHooks, ScratchPool, TopologyHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

pub type ArCallback = Box<
    dyn FnMut(
            usize,
            usize,
            DevicePtr,
            usize,
            &HipDevice,
            &HipStream,
        ) -> Result<()>
        + Send,
>;

pub struct TpHooks {
    pub rank: usize,
    pub n_ranks: usize,
    pub ar_callback: ArCallback,
}

impl TopologyHooks for TpHooks {
    fn ar_sum_f32(
        &mut self,
        buf: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        if self.n_ranks <= 1 {
            return Ok(());
        }
        (self.ar_callback)(self.rank, self.n_ranks, buf, n_elems, device, stream)
    }
}

pub struct HybridHooks {
    pub rank_in_stage: usize,
    pub tp_size: usize,
    pub ar_callback: ArCallback,
}

impl TopologyHooks for HybridHooks {
    fn ar_sum_f32(
        &mut self,
        buf: DevicePtr,
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

pub trait StageHooks {
    fn is_first(&self) -> bool;
    fn is_last(&self) -> bool;
    fn layer_range(&self, layout: &ModelLayout) -> Range<usize>;
    fn peer_recv(
        &mut self,
        core: &mut CoreState<'_>,
        n_tokens: usize,
    ) -> Result<Tensor<F16>>;
    fn peer_send(
        &mut self,
        core: &mut CoreState<'_>,
        input: &Tensor<F16>,
        n_tokens: usize,
    ) -> Result<()>;
}

pub struct SoloStage;

impl StageHooks for SoloStage {
    fn is_first(&self) -> bool {
        true
    }
    fn is_last(&self) -> bool {
        true
    }
    fn layer_range(&self, layout: &ModelLayout) -> Range<usize> {
        0..layout.num_layers
    }
    fn peer_recv(
        &mut self,
        _core: &mut CoreState<'_>,
        _n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        bail!("SoloStage::peer_recv unreachable — is_first() is true")
    }
    fn peer_send(
        &mut self,
        _core: &mut CoreState<'_>,
        _input: &Tensor<F16>,
        _n_tokens: usize,
    ) -> Result<()> {
        bail!("SoloStage::peer_send unreachable — is_last() is true")
    }
}

pub struct PpStage<'a> {
    pub rank: usize,
    pub n_ranks: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub peer_buffer: &'a mut Vec<f16>,
}

impl<'a> StageHooks for PpStage<'a> {
    fn is_first(&self) -> bool {
        self.rank == 0
    }
    fn is_last(&self) -> bool {
        self.rank + 1 == self.n_ranks
    }
    fn layer_range(&self, _layout: &ModelLayout) -> Range<usize> {
        self.layer_start..self.layer_end
    }
    fn peer_recv(
        &mut self,
        core: &mut CoreState<'_>,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        let hidden = core.hidden();
        let need = n_tokens * hidden;
        if self.peer_buffer.len() < need {
            bail!(
                "PpStage::peer_recv: peer_buffer len {} < n_tokens*hidden {need}",
                self.peer_buffer.len()
            );
        }
        let dst = core.pool.next_residual_slot();
        let bytes = need * 2;
        // SAFETY: `dst` is the residual slot (sized `hidden` F16);
        // `peer_buffer` holds `need` host F16 (checked above).
        unsafe {
            core.device
                .memcpy_async(
                    core.stream,
                    CopyDirection::HostToDevice,
                    dst,
                    DevicePtr(self.peer_buffer.as_ptr() as usize),
                    bytes,
                )
                .map_err(|e| anyhow!("PpStage peer_recv HtoD: {e}"))?;
        }
        Ok(unsafe { Tensor::<F16>::from_raw(dst, need) })
    }
    fn peer_send(
        &mut self,
        core: &mut CoreState<'_>,
        input: &Tensor<F16>,
        n_tokens: usize,
    ) -> Result<()> {
        let hidden = core.hidden();
        let need = n_tokens * hidden;
        if self.peer_buffer.len() < need {
            *self.peer_buffer = vec![f16::ZERO; need];
        }
        let bytes = need * 2;
        // SAFETY: `input.ptr` carries `need` F16 (caller invariant);
        // `peer_buffer` was just sized to `need`.
        unsafe {
            core.device
                .memcpy_async(
                    core.stream,
                    CopyDirection::DeviceToHost,
                    DevicePtr(self.peer_buffer.as_mut_ptr() as usize),
                    input.ptr,
                    bytes,
                )
                .map_err(|e| anyhow!("PpStage peer_send DtoH: {e}"))?;
        }
        // Consumer rank runs on a different worker thread; sync the
        // stream so the host buffer is observable when it reads.
        flambeau_core::Stream::synchronize(core.stream)?;
        Ok(())
    }
}

/// Hybrid runs all ranks in parallel — peer_buffer access is guarded
/// by `handoff_barrier`. Only rank 0 of each stage writes/reads.
pub struct HybStage {
    pub stage_idx: usize,
    pub n_stages: usize,
    pub rank_in_stage: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub peer_buffer: Arc<Mutex<Vec<f16>>>,
    pub handoff_barrier: Arc<Barrier>,
}

impl StageHooks for HybStage {
    fn is_first(&self) -> bool {
        self.stage_idx == 0
    }
    fn is_last(&self) -> bool {
        self.stage_idx + 1 == self.n_stages
    }
    fn layer_range(&self, _layout: &ModelLayout) -> Range<usize> {
        self.layer_start..self.layer_end
    }
    fn peer_recv(
        &mut self,
        core: &mut CoreState<'_>,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        self.handoff_barrier.wait();
        let hidden = core.hidden();
        let need = n_tokens * hidden;
        let dst = core.pool.next_residual_slot();
        let bytes = need * 2;
        let buf = self
            .peer_buffer
            .lock()
            .map_err(|e| anyhow!("HybStage peer_buffer poisoned: {e}"))?;
        if buf.len() < need {
            bail!(
                "HybStage::peer_recv: peer_buffer len {} < n_tokens*hidden {need}",
                buf.len()
            );
        }
        // SAFETY: `dst` is the residual slot; `buf` holds `need` host F16.
        unsafe {
            core.device
                .memcpy_async(
                    core.stream,
                    CopyDirection::HostToDevice,
                    dst,
                    DevicePtr(buf.as_ptr() as usize),
                    bytes,
                )
                .map_err(|e| anyhow!("HybStage peer_recv HtoD: {e}"))?;
        }
        flambeau_core::Stream::synchronize(core.stream)?;
        Ok(unsafe { Tensor::<F16>::from_raw(dst, need) })
    }
    fn peer_send(
        &mut self,
        core: &mut CoreState<'_>,
        input: &Tensor<F16>,
        n_tokens: usize,
    ) -> Result<()> {
        if self.rank_in_stage == 0 {
            let hidden = core.hidden();
            let need = n_tokens * hidden;
            let mut buf = self
                .peer_buffer
                .lock()
                .map_err(|e| anyhow!("HybStage peer_buffer poisoned: {e}"))?;
            if buf.len() < need {
                *buf = vec![f16::ZERO; need];
            }
            let bytes = need * 2;
            // SAFETY: `input.ptr` carries `need` F16; `buf` is `need` host F16.
            unsafe {
                core.device
                    .memcpy_async(
                        core.stream,
                        CopyDirection::DeviceToHost,
                        DevicePtr(buf.as_mut_ptr() as usize),
                        input.ptr,
                        bytes,
                    )
                    .map_err(|e| anyhow!("HybStage peer_send DtoH: {e}"))?;
            }
            flambeau_core::Stream::synchronize(core.stream)?;
        }
        self.handoff_barrier.wait();
        Ok(())
    }
}

pub struct ForwardEngine<'a, H: TopologyHooks, S: StageHooks> {
    pub core: CoreState<'a>,
    pub hooks: H,
    pub stage: S,
}

impl<'a, H: TopologyHooks, S: StageHooks> ForwardEngine<'a, H, S> {
    fn build(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
        hooks: H,
        stage: S,
        layer_idx_offset: usize,
    ) -> Self {
        let mut core = CoreState::new(device, stream, reg, pool);
        core.layer_idx_offset = layer_idx_offset;
        Self { core, hooks, stage }
    }
}

impl<'a> ForwardEngine<'a, NoopHooks, SoloStage> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
    ) -> Self {
        Self::build(device, stream, reg, pool, NoopHooks, SoloStage, 0)
    }
}

impl<'a> ForwardEngine<'a, TpHooks, SoloStage> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
        hooks: TpHooks,
    ) -> Self {
        Self::build(device, stream, reg, pool, hooks, SoloStage, 0)
    }
}

impl<'a> ForwardEngine<'a, NoopHooks, PpStage<'a>> {
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
        let stage = PpStage {
            rank,
            n_ranks,
            layer_start,
            layer_end,
            peer_buffer,
        };
        Self::build(device, stream, reg, pool, NoopHooks, stage, layer_start)
    }
}

impl<'a> ForwardEngine<'a, HybridHooks, HybStage> {
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
        ar_callback: ArCallback,
        peer_buffer: Arc<Mutex<Vec<f16>>>,
        handoff_barrier: Arc<Barrier>,
    ) -> Self {
        let hooks = HybridHooks {
            rank_in_stage,
            tp_size,
            ar_callback,
        };
        let stage = HybStage {
            stage_idx,
            n_stages,
            rank_in_stage,
            layer_start,
            layer_end,
            peer_buffer,
            handoff_barrier,
        };
        Self::build(device, stream, reg, pool, hooks, stage, layer_start)
    }
}

impl<H: TopologyHooks, S: StageHooks> ForwardCtx for ForwardEngine<'_, H, S> {
    fn embed(&mut self, weights: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        if self.stage.is_first() {
            composites::embed_local(&mut self.core, &mut self.hooks, weights, token_id)
        } else {
            self.stage.peer_recv(&mut self.core, 1)
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
        if self.stage.is_last() {
            composites::output_head_local(&mut self.core, &mut self.hooks, input, lm_head)
        } else {
            self.stage.peer_send(&mut self.core, input, 1)
        }
    }

    fn layer_range<'b>(
        &'b mut self,
        layout: &'b ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'b> {
        Box::new(self.stage.layer_range(layout))
    }

    fn logits(&self) -> &[f32] {
        &self.core.logits_host
    }
}

pub type SingleDeviceEngine<'a> = ForwardEngine<'a, NoopHooks, SoloStage>;
pub type TpEngine<'a> = ForwardEngine<'a, TpHooks, SoloStage>;
pub type PpEngine<'a> = ForwardEngine<'a, NoopHooks, PpStage<'a>>;
pub type HybridEngine<'a> = ForwardEngine<'a, HybridHooks, HybStage>;

pub type SingleDeviceForwardCtx<'a> = SingleDeviceEngine<'a>;
pub type TpForwardCtx<'a> = TpEngine<'a>;
pub type PpForwardCtx<'a> = PpEngine<'a>;
pub type HybridForwardCtx<'a> = HybridEngine<'a>;
