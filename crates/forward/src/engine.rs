//! Generic forward engine. `TopologyHooks` covers intra-stage AR;
//! `StageHooks` covers inter-stage role + peer-copy. SD/TP use
//! `SoloStage` (no peering); PP uses `PpStage`; Hybrid uses `HybStage`
//! (peer-buffer + handoff barrier).

use std::ops::Range;
use std::sync::{Arc, Barrier, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;

use crate::core::{composites, CoreState, NoopHooks, ScratchPool, TopologyHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout, MoeWeights,
};
use crate::runtime::ar::{bar_ar_residual_f16, bar_ar_residual_rmsnorm_f16, BarArCoordinator};

pub type ArCallback =
    Box<dyn FnMut(usize, usize, DevicePtr, usize, &HipDevice, &HipStream) -> Result<()> + Send>;

pub struct TpHooks {
    pub rank: usize,
    pub n_ranks: usize,
    pub ar_callback: ArCallback,
    pub bar: Option<Arc<BarArCoordinator>>,
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

    fn supports_ar_residual_f16(&self) -> bool {
        self.n_ranks == 2 && self.bar.is_some()
    }

    fn ar_residual_f16(
        &mut self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let bar = self
            .bar
            .as_ref()
            .ok_or_else(|| anyhow!("TpHooks::ar_residual_f16: bar coordinator not configured"))?;
        bar_ar_residual_f16(
            bar,
            self.rank,
            residual_inout,
            partial_f16,
            n_elems,
            device,
            stream,
        )
    }

    fn supports_ar_residual_rmsnorm_f16(&self) -> bool {
        self.n_ranks == 2 && self.bar.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    fn ar_residual_rmsnorm_f16(
        &mut self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        rms_weight: DevicePtr,
        out_norm: DevicePtr,
        n_elems: usize,
        eps: f32,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let bar = self.bar.as_ref().ok_or_else(|| {
            anyhow!("TpHooks::ar_residual_rmsnorm_f16: bar coordinator not configured")
        })?;
        bar_ar_residual_rmsnorm_f16(
            bar,
            self.rank,
            residual_inout,
            partial_f16,
            rms_weight,
            out_norm,
            n_elems,
            eps,
            device,
            stream,
        )
    }
}

pub struct HybridHooks {
    pub rank_in_stage: usize,
    pub tp_size: usize,
    pub ar_callback: ArCallback,
    pub bar: Option<Arc<BarArCoordinator>>,
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
        (self.ar_callback)(
            self.rank_in_stage,
            self.tp_size,
            buf,
            n_elems,
            device,
            stream,
        )
    }

    fn supports_ar_residual_f16(&self) -> bool {
        self.tp_size == 2 && self.bar.is_some()
    }

    fn ar_residual_f16(
        &mut self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let bar = self.bar.as_ref().ok_or_else(|| {
            anyhow!("HybridHooks::ar_residual_f16: bar coordinator not configured")
        })?;
        bar_ar_residual_f16(
            bar,
            self.rank_in_stage,
            residual_inout,
            partial_f16,
            n_elems,
            device,
            stream,
        )
    }

    fn supports_ar_residual_rmsnorm_f16(&self) -> bool {
        self.tp_size == 2 && self.bar.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    fn ar_residual_rmsnorm_f16(
        &mut self,
        residual_inout: DevicePtr,
        partial_f16: DevicePtr,
        rms_weight: DevicePtr,
        out_norm: DevicePtr,
        n_elems: usize,
        eps: f32,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let bar = self.bar.as_ref().ok_or_else(|| {
            anyhow!("HybridHooks::ar_residual_rmsnorm_f16: bar coordinator not configured")
        })?;
        bar_ar_residual_rmsnorm_f16(
            bar,
            self.rank_in_stage,
            residual_inout,
            partial_f16,
            rms_weight,
            out_norm,
            n_elems,
            eps,
            device,
            stream,
        )
    }
}

pub trait StageHooks {
    fn is_first(&self) -> bool;
    fn is_last(&self) -> bool;
    fn layer_range(&self, layout: &ModelLayout) -> Range<usize>;
    fn peer_recv(&mut self, core: &mut CoreState<'_>, n_tokens: usize) -> Result<Tensor<F16>>;
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
    fn peer_recv(&mut self, _core: &mut CoreState<'_>, _n_tokens: usize) -> Result<Tensor<F16>> {
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
    fn peer_recv(&mut self, core: &mut CoreState<'_>, n_tokens: usize) -> Result<Tensor<F16>> {
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
    fn peer_recv(&mut self, core: &mut CoreState<'_>, n_tokens: usize) -> Result<Tensor<F16>> {
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
        bar: Option<Arc<BarArCoordinator>>,
        peer_buffer: Arc<Mutex<Vec<f16>>>,
        handoff_barrier: Arc<Barrier>,
    ) -> Self {
        let hooks = HybridHooks {
            rank_in_stage,
            tp_size,
            ar_callback,
            bar,
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
    fn embed(&mut self, weights: &EmbeddingWeights, tokens: &[u32]) -> Result<Tensor<F16>> {
        if self.stage.is_first() {
            composites::embed_local(&mut self.core, &mut self.hooks, weights, tokens)
        } else {
            self.stage.peer_recv(&mut self.core, tokens.len())
        }
    }

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        composites::rmsnorm_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weight,
            eps,
            n_tokens,
        )
    }

    fn residual_add(
        &mut self,
        a: Tensor<F16>,
        b: Tensor<F16>,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        composites::residual_add_local(&mut self.core, &mut self.hooks, a, b, n_tokens)
    }

    fn scale_inplace_f16(
        &mut self,
        buf: Tensor<F16>,
        scale: f32,
        n_tokens: usize,
    ) -> Result<Tensor<F16>> {
        let hidden = self.core.hidden();
        let ops = self.core.ops();
        let n_elems = n_tokens * hidden;
        let buf_in = unsafe { Tensor::<F16>::from_raw(buf.ptr, n_elems) };
        let mut buf_out = unsafe { Tensor::<F16>::from_raw(buf.ptr, n_elems) };
        flambeau_model_ops::scale_f16(&buf_in, &mut buf_out, n_elems, scale, &ops)?;
        Ok(buf)
    }

    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        positions: &[usize],
        slot_ids: &[usize],
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        composites::standard_attn_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            layer_idx,
            positions,
            slot_ids,
            next_norm,
        )
    }

    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &crate::ctx::GdnWeights,
        layer_idx: usize,
        slot_ids: &[usize],
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        composites::gdn_layer_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            layer_idx,
            slot_ids,
            next_norm,
        )
    }

    fn dense_ffn(
        &mut self,
        input: &Tensor<F16>,
        weights: &FfnWeights,
        n_tokens: usize,
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        composites::dense_ffn_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            n_tokens,
            next_norm,
        )
    }

    fn moe_ffn(
        &mut self,
        input: &Tensor<F16>,
        weights: &MoeWeights,
        n_tokens: usize,
        next_norm: Option<&Tensor<F16>>,
    ) -> Result<Option<Tensor<F16>>> {
        composites::moe_ffn_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            n_tokens,
            next_norm,
        )
    }

    fn per_layer_embd_apply(
        &mut self,
        resid: &mut Tensor<F16>,
        weights: &flambeau_blocks::per_layer_embd::PerLayerEmbedLayerWeights,
        table_dev: flambeau_core::DevicePtr,
        layer_idx: usize,
        pe: usize,
        n_tokens: usize,
        n_tokens_total: usize,
        rms_eps: f32,
    ) -> Result<()> {
        let hidden = self.core.hidden();
        let pool = &self.core.pool;
        let scratch = flambeau_blocks::per_layer_embd::PerLayerEmbedDecodeScratch {
            gate_out_f32: pool.ple_gate_out_f32,
            activated_f32: pool.ple_activated_f32,
            activated_f16: pool.ple_activated_f16,
            proj_out_f32: pool.ple_proj_out_f32,
            proj_out_f16: pool.ple_proj_out_f16,
            normed_f16: pool.ple_normed_f16,
        };
        // Layer-major table layout: [n_layer, n_tokens_total, pe] F32.
        let table_slice = table_dev.offset_bytes(layer_idx * n_tokens_total * pe * 4);
        let block =
            flambeau_blocks::per_layer_embd::PerLayerEmbedBlock::new(*weights, pe, hidden, rms_eps);
        let ops = self.core.ops();
        block.forward_n_tokens(&ops, resid.ptr, table_slice, scratch, resid.ptr, n_tokens)
    }

    fn per_layer_embd_build_table(
        &mut self,
        main_embd: &Tensor<F16>,
        tok_embd_rows_raw: &[u8],
        tok_embd_dtype: flambeau_quant::GgmlDType,
        tok_embd_row_bytes: usize,
        model_proj_f16_dev: flambeau_core::DevicePtr,
        proj_matmul_f32_dev: flambeau_core::DevicePtr,
        proj_norm_raw: &[u8],
        table_dev: flambeau_core::DevicePtr,
        pe: usize,
        n_layer: usize,
        hidden: usize,
        rms_eps: f32,
    ) -> Result<()> {
        use flambeau_core::{CopyDirection, DevicePtr, Stream};
        if main_embd.n_elems < hidden {
            anyhow::bail!(
                "per_layer_embd_build_table: main_embd has {} elems, hidden = {hidden}",
                main_embd.n_elems
            );
        }
        let n_tokens = main_embd.n_elems / hidden;
        if n_tokens * hidden != main_embd.n_elems {
            anyhow::bail!(
                "per_layer_embd_build_table: main_embd elems {} not a multiple of hidden {hidden}",
                main_embd.n_elems
            );
        }
        let device = self.core.device;
        let stream = self.core.stream;

        // GPU-side per_layer_model_proj @ main_embd[n_tokens, hidden]
        // → F32 [n_tokens, pe * n_layer].
        let per_token = pe * n_layer;
        let total = n_tokens * per_token;
        if n_tokens == 1 {
            flambeau_ops::hip::router::dense_gemv_f16_f16(
                self.core.reg,
                stream,
                model_proj_f16_dev,
                main_embd.ptr,
                proj_matmul_f32_dev,
                per_token,
                hidden,
            )
            .context("per_layer_embd build: dense_gemv_f16_f16")?;
        } else {
            flambeau_ops::hip::router::dense_gemv_f16_f16_batched(
                self.core.reg,
                stream,
                model_proj_f16_dev,
                main_embd.ptr,
                proj_matmul_f32_dev,
                per_token,
                hidden,
                n_tokens,
            )
            .context("per_layer_embd build: dense_gemv_f16_f16_batched")?;
        }

        let mut proj_matmul_host = vec![0.0f32; total];
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(proj_matmul_host.as_mut_ptr() as usize),
                proj_matmul_f32_dev,
                total * 4,
            )?;
        }
        Stream::synchronize(stream)
            .context("sync after DtoH proj_matmul for per_layer_embd build")?;

        let table = flambeau_blocks::per_layer_embd::build_inp_per_layer_table_with_proj(
            tok_embd_rows_raw,
            tok_embd_dtype,
            tok_embd_row_bytes,
            &proj_matmul_host,
            proj_norm_raw,
            pe,
            n_layer,
            n_tokens,
            hidden,
            rms_eps,
        )?;

        let bytes = std::mem::size_of_val(table.as_slice());
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                table_dev,
                DevicePtr(table.as_ptr() as usize),
                bytes,
            )?;
        }
        Stream::synchronize(stream).context("sync after HtoD for per_layer_embd table")?;
        Ok(())
    }

    fn output_head(
        &mut self,
        input: &Tensor<F16>,
        lm_head: &LmHeadWeights,
        slot_ids: &[usize],
    ) -> Result<()> {
        if self.stage.is_last() {
            composites::output_head_local(&mut self.core, &mut self.hooks, input, lm_head, slot_ids)
        } else {
            self.stage.peer_send(&mut self.core, input, slot_ids.len())
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
