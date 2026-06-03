//! `HipOps` — HIP backend implementation of `Ops`.
//!
//! Holds a `(registry, stream)` pair and forwards each trait method to
//! the matching free function in `flambeau_ops::hip::*`. Constructed
//! per-rank, per-stream; cheap (two references). Free function API is
//! unchanged; consumers can migrate to the trait incrementally.

use anyhow::Result;
use flambeau_core::device::DevicePtr;
use flambeau_core::op::QDtype;

use super::{HipStream, OpsRegistry};
use crate::ops_trait::Ops;
use crate::MoeShape;

/// `Ops` implementor for the HIP backend.
///
/// `&self` carries the `(OpsRegistry, HipStream)` pair every kernel
/// launch needs. New stream → new `HipOps`.
#[derive(Copy, Clone)]
pub struct HipOps<'a> {
    pub reg: &'a OpsRegistry,
    pub stream: &'a HipStream,
}

impl<'a> HipOps<'a> {
    pub fn new(reg: &'a OpsRegistry, stream: &'a HipStream) -> Self {
        Self { reg, stream }
    }

    #[inline]
    pub fn ctx(&self) -> crate::OpCtx<'a> {
        crate::OpCtx {
            reg: self.reg,
            stream: self.stream,
        }
    }
}

impl<'a> Ops for HipOps<'a> {
    // -- qmatmul --

    fn qmatmul(
        &self,
        buf: crate::QmatmulBuffers,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::qmatmul(self.ctx(), buf, shape, dtype_weight)
    }

    fn mmvq_q4_0_t128(
        &self,
        weights: DevicePtr,
        y_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_t128(self.reg, self.stream, weights, y_q8_1, dst, n_rows, k)
    }

    fn mmvq_q4_0_gate_up_t128(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_gate_up_t128(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q4_0_gate_up_row_tile_batched(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpBatchShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_gate_up_row_tile_batched(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q4_0_row_tile_batched(
        &self,
        buffers: crate::MmvqBuffers,
        shape: crate::MmvqBatchShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_row_tile_batched(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q8_0_row_tile_batched(
        &self,
        buffers: crate::MmvqBuffers,
        shape: crate::MmvqBatchShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q8_0_row_tile_batched(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q4_0_warpcoop64(
        &self,
        weights: DevicePtr,
        y_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_warpcoop64(self.reg, self.stream, weights, y_q8_1, dst, n_rows, k)
    }

    fn mmvq_q4_0_kv_f16dst(
        &self,
        buf: crate::MmvqKvF16Buffers,
        n_rows_kv: usize,
        k: usize,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_kv_f16dst(self.ctx(), buf, n_rows_kv, k)
    }

    fn mmvq_q4_0_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q4_1_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_1_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q8_0_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q8_0_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq_q5_k_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        super::qmatmul::mmvq_q5_k_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn mmvq(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MmvqShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::mmvq(self.ctx(), buf, shape, dtype_weight)
    }

    fn mmvq_f16_direct(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MmvqShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::mmvq_f16_direct(self.ctx(), buf, shape, dtype_weight)
    }

    fn mmq(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::mmq(self.ctx(), buf, shape, dtype_weight)
    }

    // -- attention --

    fn attention_decode_f16(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_decode_f16(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
        )
    }

    fn attention_decode_f16_slots(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnDecodeSlots>,
    ) -> Result<()> {
        super::attention::attention_decode_f16_slots(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
            slots,
        )
    }

    fn attention_decode_f16_batched(
        &self,
        buffers: crate::AttnBatchedBuffers,
        shape: crate::AttnDecodeBatchedShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_decode_f16_batched(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
        )
    }

    fn kv_append_f16_batched_slots(
        &self,
        buf: crate::KvAppendBatchedSlotsBuffers,
        shape: crate::KvAppendBatchedSlotsShape,
    ) -> Result<()> {
        super::attention::kv_append_f16_batched_slots(self.ctx(), buf, shape)
    }

    fn attention_prefill_f16_paged(
        &self,
        buffers: crate::AttnPagedPrefillBuffers,
        shape: crate::AttnPrefillPagedShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_prefill_f16_paged(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
        )
    }

    fn kv_append_f16_paged_prefill(
        &self,
        buf: crate::KvAppendPagedPrefillBuffers,
        shape: crate::KvAppendPagedPrefillShape,
    ) -> Result<()> {
        super::attention::kv_append_f16_paged_prefill(self.ctx(), buf, shape)
    }

    fn kv_append_f16_paged_slots(
        &self,
        buf: crate::KvAppendPagedSlotsBuffers,
        shape: crate::KvAppendPagedSlotsShape,
    ) -> Result<()> {
        super::attention::kv_append_f16_paged_slots(self.ctx(), buf, shape)
    }

    fn attention_decode_f16_paged(
        &self,
        buffers: crate::AttnPagedDecodeBuffers,
        shape: crate::AttnDecodePagedShape,
        scale: f32,
    ) -> Result<()> {
        super::attention::attention_decode_f16_paged(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            scale,
        )
    }

    fn kv_append_v_unit_norm_f16(
        &self,
        buf: crate::KvAppendBuffers,
        shape: crate::KvAppendVUnitShape,
        write_pos: usize,
        eps: f32,
    ) -> Result<()> {
        super::attention::kv_append_v_unit_norm_f16(self.ctx(), buf, shape, write_pos, eps)
    }

    fn attention_decode_f16_splitk(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_decode_f16_splitk(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            partials,
            shape,
            knobs,
        )
    }

    fn attention_decode_f16_splitk_h2(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_decode_f16_splitk_h2(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            partials,
            shape,
            knobs,
        )
    }

    fn attention_decode_q8_kv(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_decode_q8_kv(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
        )
    }

    fn attention_decode_q8_kv_splitk(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_decode_q8_kv_splitk(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            partials,
            shape,
            knobs,
        )
    }

    fn attention_prefill_q8_kv(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_prefill_q8_kv(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
        )
    }

    fn attention_prefill_f16(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        super::attention::attention_prefill_f16(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
        )
    }

    fn attention_prefill_f16_slots(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnPrefillSlots>,
    ) -> Result<()> {
        super::attention::attention_prefill_f16_slots(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
            knobs,
            slots,
        )
    }

    fn split_q_gate_f16(
        &self,
        fused_qg: DevicePtr,
        q_out: DevicePtr,
        gate_out: DevicePtr,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        super::attention::split_q_gate_f16(
            self.reg,
            self.stream,
            fused_qg,
            q_out,
            gate_out,
            n_tokens,
            n_heads,
            head_dim,
        )
    }

    // -- norm --

    fn rmsnorm_f16(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f16(self.ctx(), buf, shape, eps)
    }

    fn rmsnorm_f16_add_residual(
        &self,
        buf: crate::NormFusedAddBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f16_add_residual(self.ctx(), buf, shape, eps)
    }

    fn v_unit_norm_per_head_f16(
        &self,
        v: DevicePtr,
        n_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::v_unit_norm_per_head_f16(
            self.reg,
            self.stream,
            v,
            n_tokens,
            n_kv_heads,
            head_dim,
            eps,
        )
    }

    fn rmsnorm_quant_q8_1(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_quant_q8_1(self.ctx(), buf, shape, eps)
    }

    fn rmsnorm_f32(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f32(self.ctx(), buf, shape, eps)
    }

    fn rmsnorm_f32_to_f16(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f32_to_f16(self.ctx(), buf, shape, eps)
    }

    fn rmsnorm_f32_to_f16_add_residual(
        &self,
        buf: crate::NormResidualBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f32_to_f16_add_residual(self.ctx(), buf, shape, eps)
    }

    fn rmsnorm_f16_to_f16_add_residual(
        &self,
        buf: crate::NormResidualBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f16_to_f16_add_residual(self.ctx(), buf, shape, eps)
    }

    fn l2_norm_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::l2_norm_f32(self.reg, self.stream, x, y, n_rows, k, eps)
    }

    fn quantize_q8_1(&self, x_f32: DevicePtr, y_q8_1: DevicePtr, n_elems: usize) -> Result<()> {
        super::norm::quantize_q8_1(self.reg, self.stream, x_f32, y_q8_1, n_elems)
    }

    fn quantize_q8_1_mmq(
        &self,
        x_f32: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()> {
        super::norm::quantize_q8_1_mmq(self.reg, self.stream, x_f32, y_q8_1_mmq, ncols, total_b)
    }

    fn quantize_f16_q8_1_mmq(
        &self,
        x_f16: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()> {
        super::norm::quantize_f16_q8_1_mmq(self.reg, self.stream, x_f16, y_q8_1_mmq, ncols, total_b)
    }

    fn quantize_f16_q8_1(&self, x_f16: DevicePtr, y_q8_1: DevicePtr, n_elems: usize) -> Result<()> {
        super::norm::quantize_f16_q8_1(self.reg, self.stream, x_f16, y_q8_1, n_elems)
    }

    fn quantize_f16_q8_0(&self, x_f16: DevicePtr, y_q8_0: DevicePtr, n_elems: usize) -> Result<()> {
        super::norm::quantize_f16_q8_0(self.reg, self.stream, x_f16, y_q8_0, n_elems)
    }

    // -- mlp --

    fn silu_f32(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::silu_f32(self.reg, self.stream, x, y, n)
    }

    fn swiglu_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::swiglu_f32(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f32_to_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::swiglu_f32_to_f16(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f32_to_q8_1(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y_q8_1: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::swiglu_f32_to_q8_1(self.reg, self.stream, a, b, y_q8_1, n)
    }

    fn scale_f32(&self, x: DevicePtr, y: DevicePtr, n: usize, scale: f32) -> Result<()> {
        super::mlp::scale_f32(self.reg, self.stream, x, y, n, scale)
    }

    fn scale_f16(&self, x: DevicePtr, y: DevicePtr, n: usize, scale: f32) -> Result<()> {
        super::mlp::scale_f16(self.reg, self.stream, x, y, n, scale)
    }

    fn add_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::add_f16(self.reg, self.stream, a, b, y, n)
    }

    fn add_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::add_f32(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f16(&self, gate: DevicePtr, up: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::swiglu_f16(self.reg, self.stream, gate, up, y, n)
    }

    fn sigmoid_mul_f16(&self, gate: DevicePtr, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::sigmoid_mul_f16(self.reg, self.stream, gate, x, y, n)
    }

    // -- pe --

    fn rope_f16(
        &self,
        buf: crate::RopeBuffers,
        shape: crate::RopeShape,
        theta_base: f32,
    ) -> Result<()> {
        super::pe::rope_f16(self.ctx(), buf, shape, theta_base)
    }

    fn rope_neox_partial_f16(
        &self,
        buf: crate::RopeBuffers,
        shape: crate::RopePartialShape,
        theta_base: f32,
    ) -> Result<()> {
        super::pe::rope_neox_partial_f16(self.ctx(), buf, shape, theta_base)
    }

    fn rmsnorm_rope_neox_partial_f16(
        &self,
        buf: crate::RopeFusedBuffers,
        shape: crate::RopePartialShape,
        theta_base: f32,
        eps: f32,
    ) -> Result<()> {
        super::pe::rmsnorm_rope_neox_partial_f16(self.ctx(), buf, shape, theta_base, eps)
    }

    // -- cast --

    fn cast_f32_to_f16(&self, x_f32: DevicePtr, y_f16: DevicePtr, n: usize) -> Result<()> {
        super::cast::cast_f32_to_f16(self.reg, self.stream, x_f32, y_f16, n)
    }

    fn cast_f16_to_f32(&self, x_f16: DevicePtr, y_f32: DevicePtr, n: usize) -> Result<()> {
        super::cast::cast_f16_to_f32(self.reg, self.stream, x_f16, y_f32, n)
    }

    // -- sampling --

    fn apply_penalties_f32(
        &self,
        bufs: crate::PenaltyBuffers,
        n_pairs: usize,
        vocab: usize,
        knobs: crate::PenaltyKnobs,
    ) -> Result<()> {
        super::sampling::apply_penalties_f32(self.ctx(), bufs, n_pairs, vocab, knobs)
    }

    fn topk_softmax_f32(
        &self,
        logits: DevicePtr,
        out_ids: DevicePtr,
        out_probs: DevicePtr,
        vocab: usize,
        k: usize,
        inv_temp: f32,
    ) -> Result<()> {
        super::sampling::topk_softmax_f32(
            self.reg,
            self.stream,
            logits,
            out_ids,
            out_probs,
            vocab,
            k,
            inv_temp,
        )
    }

    // -- recurrent --

    fn gdn_state_step_f32_s128(
        &self,
        bufs: crate::GdnStepBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()> {
        super::recurrent::gdn_state_step_f32_s128(self.ctx(), bufs, shape)
    }

    fn gdn_alpha_beta_f32(
        &self,
        bufs: crate::GdnAlphaBetaBuffers,
        shape: crate::GdnAlphaBetaShape,
    ) -> Result<()> {
        super::recurrent::gdn_alpha_beta_f32(self.ctx(), bufs, shape)
    }

    fn gdn_state_step_alphabeta_f32_s128(
        &self,
        bufs: crate::GdnStepAlphaBetaBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()> {
        super::recurrent::gdn_state_step_alphabeta_f32_s128(self.ctx(), bufs, shape)
    }

    fn gdn_assemble_conv_input_f32(
        &self,
        history: DevicePtr,
        current: DevicePtr,
        conv_input: DevicePtr,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Result<()> {
        super::recurrent::gdn_assemble_conv_input_f32(
            self.reg,
            self.stream,
            history,
            current,
            conv_input,
            conv_channels,
            conv_kernel,
        )
    }

    fn gdn_state_step_alphabeta_f32_s128_batched_slots(
        &self,
        bufs: crate::GdnStepAlphaBetaBatchedSlotsBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()> {
        super::recurrent::gdn_state_step_alphabeta_f32_s128_batched_slots(
            self.ctx(),
            bufs,
            shape,
        )
    }

    fn gdn_conv_trio_decode_f32_batched_slots(
        &self,
        bufs: crate::GdnConvTrioBatchedSlotsBuffers,
        shape: crate::GdnConvTrioShape,
    ) -> Result<()> {
        super::recurrent::gdn_conv_trio_decode_f32_batched_slots(self.ctx(), bufs, shape)
    }

    fn gdn_split_qkv_f32(
        &self,
        bufs: crate::GdnSplitQkvBuffers,
        shape: crate::GdnSplitQkvShape,
    ) -> Result<()> {
        super::recurrent::gdn_split_qkv_f32(self.ctx(), bufs, shape)
    }

    // -- router --

    fn dense_gemv_f32_f16(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        super::router::dense_gemv_f32_f16(self.reg, self.stream, w, x, y, n_rows, k)
    }

    fn dense_gemv_f16_f16(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        super::router::dense_gemv_f16_f16(self.reg, self.stream, w, x, y, n_rows, k)
    }

    fn dense_gemv_f16_f16_batched(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<()> {
        super::router::dense_gemv_f16_f16_batched(
            self.reg,
            self.stream,
            w,
            x,
            y,
            n_rows,
            k,
            n_tokens,
        )
    }

    fn dense_gemv_f32_f16_batched(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<()> {
        super::router::dense_gemv_f32_f16_batched(
            self.reg,
            self.stream,
            w,
            x,
            y,
            n_rows,
            k,
            n_tokens,
        )
    }

    // -- softmax --

    fn softmax_masked_f16(
        &self,
        scores: DevicePtr,
        mask: DevicePtr,
        out: DevicePtr,
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<()> {
        super::softmax::softmax_masked_f16(self.reg, self.stream, scores, mask, out, m, k, scale)
    }

    // -- conv --

    fn causal_conv1d_f32(
        &self,
        conv_input: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        n_new: usize,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Result<()> {
        super::conv::causal_conv1d_f32(
            self.reg,
            self.stream,
            conv_input,
            weight,
            y,
            n_new,
            conv_channels,
            conv_kernel,
        )
    }

    // -- moe --

    fn topk_f32(
        &self,
        logits: DevicePtr,
        idx: DevicePtr,
        weights: DevicePtr,
        n_tokens: usize,
        n_experts: usize,
        k: usize,
    ) -> Result<()> {
        super::moe::topk_f32(
            self.reg,
            self.stream,
            logits,
            idx,
            weights,
            n_tokens,
            n_experts,
            k,
        )
    }

    fn apply_per_expert_scale_f32(
        &self,
        expert_weights: DevicePtr,
        expert_ids: DevicePtr,
        expert_scales: DevicePtr,
        n_tokens: usize,
        top_k: usize,
    ) -> Result<()> {
        super::moe::apply_per_expert_scale_f32(
            self.reg,
            self.stream,
            expert_weights,
            expert_ids,
            expert_scales,
            n_tokens,
            top_k,
        )
    }

    fn indexed_moe_mmvq_q4_k_r2(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_r2(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q6_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q6_k(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q5_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q5_k(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q3_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q3_k(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq4_xs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq4_xs(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq4_nl(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq4_nl(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq3_xxs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq3_xxs(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq3_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq3_s(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq2_xxs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq2_xxs(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq2_xs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq2_xs(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq2_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq2_s(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq1_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq1_s(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq1_m(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_iq1_m(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_0(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_0(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_1(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_1(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_0_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_0_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmvq_q8_0(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q8_0(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q8_0_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q8_0_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmvq_q4_k_r2_sorted(
        &self,
        buffers: crate::MoeMmvqSortedBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_r2_sorted(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q3_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q3_k_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q3_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q3_k_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q5_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q5_k_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q6_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q6_k_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_0_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_0_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_0_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_0_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_1_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_1_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q8_0_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q8_0_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q8_0_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q8_0_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    // ---- IQ tile8 impls. All follow the same shape: dispatch to the
    // matching free function in `super::moe`.

    fn indexed_moe_mmq_iq4_xs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_xs_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq4_xs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_xs_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq4_nl_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_nl_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq4_nl_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_nl_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq3_xxs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_xxs_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq3_xxs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_xxs_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq3_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_s_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq3_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_s_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq2_xxs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xxs_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq2_xxs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xxs_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq2_xs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xs_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq2_xs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xs_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq2_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_s_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq2_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_s_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq1_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_s_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq1_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_s_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq1_m_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_m_gate_up_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq1_m_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_m_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k_gate_up_turbo(
        &self,
        bufs: crate::MoeMmqQ4KGateUpTurboBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_gate_up_turbo(self.ctx(), bufs, shape)
    }

    fn indexed_moe_mmq_q4_k_down_turbo(
        &self,
        bufs: crate::MoeMmqQ4KDownTurboBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_down_turbo(self.ctx(), bufs, shape)
    }

    fn indexed_moe_mmq_q5_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q5_k_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q6_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q6_k_down_tile8(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_k_gate_up_sorted(
        &self,
        buffers: crate::MoeMmvqGateUpSortedBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_gate_up_sorted(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_k_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_gate_up(
            crate::OpCtx {
                reg: self.reg,
                stream: self.stream,
            },
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k(
        &self,
        bufs: crate::MoeMmqQ4KBuffers,
        shape: crate::MoeMmqQ4KShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k(self.ctx(), bufs, shape)
    }

    fn shared_expert_scale_f32(
        &self,
        shared_out: DevicePtr,
        x: DevicePtr,
        gate_w: DevicePtr,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<()> {
        super::moe::shared_expert_scale_f32(
            self.reg,
            self.stream,
            shared_out,
            x,
            gate_w,
            n_tokens,
            hidden,
        )
    }

    fn moe_combine_f16(
        &self,
        buf: crate::MoeCombineBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        super::moe::moe_combine_f16(self.ctx(), buf, shape)
    }

    fn moe_combine_no_residual_f16(
        &self,
        buf: crate::MoeCombineNoResidualBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        super::moe::moe_combine_no_residual_f16(self.ctx(), buf, shape)
    }

    fn moe_combine_no_residual_f32(
        &self,
        buf: crate::MoeCombineNoResidualBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        super::moe::moe_combine_no_residual_f32(self.ctx(), buf, shape)
    }

    fn moe_combine_two_residuals_f16(
        &self,
        buf: crate::MoeCombineTwoResidualsBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        super::moe::moe_combine_two_residuals_f16(self.ctx(), buf, shape)
    }

    fn moe_sort_by_expert(
        &self,
        buf: crate::MoeSortBuffers,
        shape: crate::MoeSortShape,
    ) -> Result<()> {
        super::moe::moe_sort_by_expert(self.ctx(), buf, shape)
    }

    fn moe_sort_by_expert_padded_16(
        &self,
        buf: crate::MoeSortPaddedBuffers,
        shape: crate::MoeSortPaddedShape,
    ) -> Result<()> {
        super::moe::moe_sort_by_expert_padded_16(self.ctx(), buf, shape)
    }

    fn moe_sort_by_expert_padded(
        &self,
        buf: crate::MoeSortPaddedBuffers,
        shape: crate::MoeSortPaddedShape,
    ) -> Result<()> {
        super::moe::moe_sort_by_expert_padded(self.ctx(), buf, shape)
    }

    fn apply_softcap_f32(&self, x: DevicePtr, y: DevicePtr, n: usize, cap: f32) -> Result<()> {
        super::softcap::apply_softcap_f32(self.reg, self.stream, x, y, n, cap)
    }

    fn gelu_f32_to_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::gelu_f32_to_f16(self.reg, self.stream, a, b, y, n)
    }

    fn gelu_mul_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::gelu_mul_f32(self.reg, self.stream, a, b, y, n)
    }
}
