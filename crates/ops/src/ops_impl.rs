//! `OpsImpl<B>` — the one [`Ops`] implementation, generic over the
//! [`KernelBackend`] launch seam.
//!
//! Holds a `(registry, stream)` pair and forwards each trait method to the
//! matching generic free function in [`crate::launch`]. Constructed per-rank,
//! per-stream; cheap (two references). `HipOps` / `CudaOps` are the per-backend
//! type aliases (`OpsImpl<HipKernels>` / `OpsImpl<CudaKernels>`) — one body
//! serves both, so a new kernel or arch never re-copies the launch layer
//! (CLAUDE.md rule 14).

use anyhow::Result;
use flambeau_core::device::{Device, DevicePtr};
use flambeau_core::op::QDtype;
use flambeau_core::KernelBackend;

use crate::ops_trait::Ops;
use crate::registry::Registry;
use crate::MoeShape;

/// `Ops` implementor for backend `B`.
///
/// `&self` carries the `(Registry<B>, stream)` pair every kernel launch needs.
/// New stream → new `OpsImpl`.
pub struct OpsImpl<'a, B: KernelBackend> {
    pub reg: &'a Registry<B>,
    pub stream: &'a <B::Device as Device>::Stream,
}

// Hand-rolled so the bound is `B: KernelBackend`, not `B: Copy` (the derive
// would add the wrong bound — `B` is an uninhabited launch-plane marker, never
// `Copy`; the two fields are shared references, which are).
impl<B: KernelBackend> Clone for OpsImpl<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: KernelBackend> Copy for OpsImpl<'_, B> {}

impl<'a, B: KernelBackend> OpsImpl<'a, B> {
    pub fn new(reg: &'a Registry<B>, stream: &'a <B::Device as Device>::Stream) -> Self {
        Self { reg, stream }
    }
}

impl<B: KernelBackend> Ops for OpsImpl<'_, B> {
    // -- qmatmul --

    fn qmatmul(
        &self,
        buf: crate::QmatmulBuffers,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        crate::launch::qmatmul::qmatmul(self.reg, self.stream, buf, shape, dtype_weight)
    }

    fn mmvq_q4_0_t128(
        &self,
        weights: DevicePtr,
        y_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_0_t128(self.reg, self.stream, weights, y_q8_1, dst, n_rows, k)
    }

    fn mmvq_q4_0_gate_up_t128(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_0_gate_up_t128(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn mmvq_q4_0_gate_up_row_tile_batched(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpBatchShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_0_gate_up_row_tile_batched(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn mmvq_q4_0_row_tile_batched(
        &self,
        buffers: crate::MmvqBuffers,
        shape: crate::MmvqBatchShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_0_row_tile_batched(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn mmvq_q8_0_row_tile_batched(
        &self,
        buffers: crate::MmvqBuffers,
        shape: crate::MmvqBatchShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q8_0_row_tile_batched(
            self.reg,
            self.stream,
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
        crate::launch::qmatmul::mmvq_q4_0_warpcoop64(self.reg, self.stream, weights, y_q8_1, dst, n_rows, k)
    }

    fn mmvq_q4_0_kv_f16dst(
        &self,
        buf: crate::MmvqKvF16Buffers,
        n_rows_kv: usize,
        k: usize,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_0_kv_f16dst(self.reg, self.stream, buf, n_rows_kv, k)
    }

    fn mmvq_q4_0_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_0_gate_up(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn mmvq_q4_1_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q4_1_gate_up(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn mmvq_q8_0_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q8_0_gate_up(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn mmvq_q5_k_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_q5_k_gate_up(
            self.reg,
            self.stream,
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
        crate::launch::qmatmul::mmvq(self.reg, self.stream, buf, shape, dtype_weight)
    }

    fn mmvq_q4_k_r4(
        &self,
        buf: crate::MmvqBuffers,
        n_rows: usize,
        n_superblocks: usize,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_simple_launch(
            self.reg,
            self.stream,
            crate::launch::qmatmul::KernelEntry {
                stem: "mmvq_q4_k_r4",
                entry: "flambeau_mmvq_q4_k_r4_q8_1",
            },
            buf,
            n_rows,
            n_superblocks,
            crate::launch::qmatmul::MmvqLaunchTune {
                threads: 64,
                rows_per_block: 4,
            },
        )
    }

    fn mmvq_f16_direct(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MmvqShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        crate::launch::qmatmul::mmvq_f16_direct(self.reg, self.stream, buf, shape, dtype_weight)
    }

    fn mmq(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> Result<()> {
        crate::launch::qmatmul::mmq(self.reg, self.stream, buf, shape, dtype_weight)
    }

    // -- attention --

    fn attention_decode_f16(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16(
            self.reg,
            self.stream,
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
        crate::launch::attention::attention_decode_f16_slots(
            self.reg,
            self.stream,
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
        crate::launch::attention::attention_decode_f16_batched(
            self.reg,
            self.stream,
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
        crate::launch::attention::kv_append_f16_batched_slots(self.reg, self.stream, buf, shape)
    }

    fn attention_prefill_f16_paged(
        &self,
        buffers: crate::AttnPagedPrefillBuffers,
        shape: crate::AttnPrefillPagedShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_prefill_f16_paged(
            self.reg,
            self.stream,
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
        crate::launch::attention::kv_append_f16_paged_prefill(self.reg, self.stream, buf, shape)
    }

    fn kv_append_f16_paged_slots(
        &self,
        buf: crate::KvAppendPagedSlotsBuffers,
        shape: crate::KvAppendPagedSlotsShape,
    ) -> Result<()> {
        crate::launch::attention::kv_append_f16_paged_slots(self.reg, self.stream, buf, shape)
    }

    fn attention_decode_f16_paged(
        &self,
        buffers: crate::AttnPagedDecodeBuffers,
        shape: crate::AttnDecodePagedShape,
        scale: f32,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16_paged(
            self.reg,
            self.stream,
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
        crate::launch::attention::kv_append_v_unit_norm_f16(self.reg, self.stream, buf, shape, write_pos, eps)
    }

    fn kv_append_v_unit_norm_f16_slots(
        &self,
        buf: crate::KvAppendBuffers,
        shape: crate::KvAppendVUnitShape,
        write_pos: usize,
        eps: f32,
        slots: Option<crate::KvAppendSlots>,
    ) -> Result<()> {
        crate::launch::attention::kv_append_v_unit_norm_f16_slots(
            self.reg, self.stream, buf, shape, write_pos, eps, slots,
        )
    }

    fn attention_decode_f16_splitk(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16_splitk(
            self.reg,
            self.stream,
            buffers,
            partials,
            shape,
            knobs,
        )
    }

    fn attention_decode_f16_splitk_slots(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnSplitkSlots>,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16_splitk_slots(
            self.reg,
            self.stream,
            buffers,
            partials,
            shape,
            knobs,
            slots,
        )
    }

    fn attention_decode_f16_splitk_h2(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16_splitk_h2(
            self.reg,
            self.stream,
            buffers,
            partials,
            shape,
            knobs,
        )
    }

    fn attention_decode_f16_splitk_dpos(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkDposShape,
        knobs: crate::AttnDposKnobs,
        pos_ptr: DevicePtr,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16_splitk_dpos(
            self.reg, self.stream, buffers, partials, shape, knobs, pos_ptr,
        )
    }

    fn kv_append_v_unit_norm_f16_dpos(
        &self,
        buf: crate::KvAppendBuffers,
        shape: crate::KvAppendVUnitShape,
        eps: f32,
        pos_ptr: DevicePtr,
    ) -> Result<()> {
        crate::launch::attention::kv_append_v_unit_norm_f16_dpos(
            self.reg, self.stream, buf, shape, eps, pos_ptr,
        )
    }

    fn embed_gather_f16_dpos(
        &self,
        embd: DevicePtr,
        dst: DevicePtr,
        hidden: usize,
        ctx_ptr: DevicePtr,
    ) -> Result<()> {
        crate::launch::attention::embed_gather_f16_dpos(
            self.reg, self.stream, embd, dst, hidden, ctx_ptr,
        )
    }

    fn attention_decode_f16_batched_splitk(
        &self,
        buffers: crate::AttnBatchedBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnDecodeBatchedSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_f16_batched_splitk(
            self.reg,
            self.stream,
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
        crate::launch::attention::attention_decode_q8_kv(
            self.reg,
            self.stream,
            buffers,
            shape,
            knobs,
        )
    }

    fn attention_decode_q8_kv_slots(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnDecodeQ8Slots>,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_q8_kv_slots(
            self.reg,
            self.stream,
            buffers,
            shape,
            knobs,
            slots,
        )
    }

    fn attention_decode_q8_kv_splitk(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_q8_kv_splitk(
            self.reg,
            self.stream,
            buffers,
            partials,
            shape,
            knobs,
        )
    }

    fn attention_decode_q8_kv_splitk_slots(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnDecodeQ8SplitkSlots>,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_q8_kv_splitk_slots(
            self.reg,
            self.stream,
            buffers,
            partials,
            shape,
            knobs,
            slots,
        )
    }

    fn attention_decode_q8_kv_splitk_dpos(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkDposShape,
        knobs: crate::AttnDposKnobs,
        pos_ptr: DevicePtr,
    ) -> Result<()> {
        crate::launch::attention::attention_decode_q8_kv_splitk_dpos(
            self.reg, self.stream, buffers, partials, shape, knobs, pos_ptr,
        )
    }

    fn attention_prefill_q8_kv(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()> {
        crate::launch::attention::attention_prefill_q8_kv(
            self.reg,
            self.stream,
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
        crate::launch::attention::attention_prefill_f16(
            self.reg,
            self.stream,
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
        crate::launch::attention::attention_prefill_f16_slots(
            self.reg,
            self.stream,
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
        crate::launch::attention::split_q_gate_f16(
            self.reg,
            self.stream,
            crate::SplitQGateBuffers { fused_qg, q_out, gate_out },
            crate::SplitQGateShape { n_tokens, n_heads, head_dim },
        )
    }

    // -- norm --

    fn rmsnorm_f16(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::rmsnorm_f16(self.reg, self.stream, buf, shape, eps)
    }

    fn rmsnorm_f16_add_residual(
        &self,
        buf: crate::NormFusedAddBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::rmsnorm_f16_add_residual(self.reg, self.stream, buf, shape, eps)
    }

    fn v_unit_norm_per_head_f16(
        &self,
        v: DevicePtr,
        n_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::v_unit_norm_per_head_f16(
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
        crate::launch::norm::rmsnorm_quant_q8_1(self.reg, self.stream, buf, shape, eps)
    }

    fn rmsnorm_f32(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::rmsnorm_f32(self.reg, self.stream, buf, shape, eps)
    }

    fn rmsnorm_f32_to_f16(
        &self,
        buf: crate::NormBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::rmsnorm_f32_to_f16(self.reg, self.stream, buf, shape, eps)
    }

    fn rmsnorm_f32_to_f16_add_residual(
        &self,
        buf: crate::NormResidualBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::rmsnorm_f32_to_f16_add_residual(self.reg, self.stream, buf, shape, eps)
    }

    fn rmsnorm_f16_to_f16_add_residual(
        &self,
        buf: crate::NormResidualBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::rmsnorm_f16_to_f16_add_residual(self.reg, self.stream, buf, shape, eps)
    }

    fn l2_norm_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        crate::launch::norm::l2_norm_f32(self.reg, self.stream, x, y, n_rows, k, eps)
    }

    fn quantize_q8_1(&self, x_f32: DevicePtr, y_q8_1: DevicePtr, n_elems: usize) -> Result<()> {
        crate::launch::norm::quantize_q8_1(self.reg, self.stream, x_f32, y_q8_1, n_elems)
    }

    fn quantize_q8_1_mmq(
        &self,
        x_f32: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()> {
        crate::launch::norm::quantize_q8_1_mmq(self.reg, self.stream, x_f32, y_q8_1_mmq, ncols, total_b)
    }

    fn quantize_f16_q8_1_mmq(
        &self,
        x_f16: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()> {
        crate::launch::norm::quantize_f16_q8_1_mmq(self.reg, self.stream, x_f16, y_q8_1_mmq, ncols, total_b)
    }

    fn quantize_f16_q8_1(&self, x_f16: DevicePtr, y_q8_1: DevicePtr, n_elems: usize) -> Result<()> {
        crate::launch::norm::quantize_f16_q8_1(self.reg, self.stream, x_f16, y_q8_1, n_elems)
    }

    fn quantize_f16_q8_0(&self, x_f16: DevicePtr, y_q8_0: DevicePtr, n_elems: usize) -> Result<()> {
        crate::launch::norm::quantize_f16_q8_0(self.reg, self.stream, x_f16, y_q8_0, n_elems)
    }

    fn quantize_f16_q8_0_slots(
        &self,
        x_f16: DevicePtr,
        y_q8_0: DevicePtr,
        n_elems: usize,
        dst_slot: Option<flambeau_core::ScalarSlot>,
    ) -> Result<()> {
        crate::launch::norm::quantize_f16_q8_0_slots(
            self.reg, self.stream, x_f16, y_q8_0, n_elems, dst_slot,
        )
    }

    fn quantize_f16_q8_0_dpos(
        &self,
        x_f16: DevicePtr,
        y_q8_0_base: DevicePtr,
        n_elems: usize,
        ring_depth: i32,
        pos_ptr: DevicePtr,
    ) -> Result<()> {
        crate::launch::norm::quantize_f16_q8_0_dpos(
            self.reg, self.stream, x_f16, y_q8_0_base, n_elems, ring_depth, pos_ptr,
        )
    }

    // -- mlp --

    fn silu_f32(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::silu_f32(self.reg, self.stream, x, y, n)
    }

    fn swiglu_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::swiglu_f32(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f32_to_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::swiglu_f32_to_f16(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f32_to_q8_1(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y_q8_1: DevicePtr,
        n: usize,
    ) -> Result<()> {
        crate::launch::mlp::swiglu_f32_to_q8_1(self.reg, self.stream, a, b, y_q8_1, n)
    }

    fn gelu_mul_f32_to_q8_1(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y_q8_1: DevicePtr,
        n: usize,
    ) -> Result<()> {
        crate::launch::mlp::gelu_mul_f32_to_q8_1(self.reg, self.stream, a, b, y_q8_1, n)
    }

    fn gelu_mul_f32_to_q8_1_mmq(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()> {
        crate::launch::mlp::gelu_mul_f32_to_q8_1_mmq(
            self.reg, self.stream, a, b, y_q8_1_mmq, ncols, total_b,
        )
    }

    fn qmatmul_reads_mmq_activation(
        &self,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> bool {
        crate::launch::qmatmul::qmatmul_reads_mmq_activation::<B>(shape, dtype_weight)
    }

    fn scale_f32(&self, x: DevicePtr, y: DevicePtr, n: usize, scale: f32) -> Result<()> {
        crate::launch::mlp::scale_f32(self.reg, self.stream, x, y, n, scale)
    }

    fn scale_f16(&self, x: DevicePtr, y: DevicePtr, n: usize, scale: f32) -> Result<()> {
        crate::launch::mlp::scale_f16(self.reg, self.stream, x, y, n, scale)
    }

    fn add_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::add_f16(self.reg, self.stream, a, b, y, n)
    }

    fn add_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::add_f32(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f16(&self, gate: DevicePtr, up: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::swiglu_f16(self.reg, self.stream, gate, up, y, n)
    }

    fn sigmoid_mul_f16(&self, gate: DevicePtr, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::sigmoid_mul_f16(self.reg, self.stream, gate, x, y, n)
    }

    // -- pe --

    fn rope_f16(
        &self,
        buf: crate::RopeBuffers,
        shape: crate::RopeShape,
        theta_base: f32,
    ) -> Result<()> {
        crate::launch::pe::rope_f16(self.reg, self.stream, buf, shape, theta_base)
    }

    fn rope_neox_partial_f16(
        &self,
        buf: crate::RopeBuffers,
        shape: crate::RopePartialShape,
        theta_base: f32,
    ) -> Result<()> {
        crate::launch::pe::rope_neox_partial_f16(self.reg, self.stream, buf, shape, theta_base)
    }

    fn rmsnorm_rope_neox_partial_f16(
        &self,
        buf: crate::RopeFusedBuffers,
        shape: crate::RopePartialShape,
        theta_base: f32,
        eps: f32,
    ) -> Result<()> {
        crate::launch::pe::rmsnorm_rope_neox_partial_f16(self.reg, self.stream, buf, shape, theta_base, eps)
    }

    // -- cast --

    fn cast_f32_to_f16(&self, x_f32: DevicePtr, y_f16: DevicePtr, n: usize) -> Result<()> {
        crate::launch::cast::cast_f32_to_f16(self.reg, self.stream, x_f32, y_f16, n)
    }

    fn cast_f16_to_f32(&self, x_f16: DevicePtr, y_f32: DevicePtr, n: usize) -> Result<()> {
        crate::launch::cast::cast_f16_to_f32(self.reg, self.stream, x_f16, y_f32, n)
    }

    // -- sampling --

    fn apply_penalties_f32(
        &self,
        bufs: crate::PenaltyBuffers,
        n_pairs: usize,
        vocab: usize,
        knobs: crate::PenaltyKnobs,
    ) -> Result<()> {
        crate::launch::sampling::apply_penalties_f32(self.reg, self.stream, bufs, n_pairs, vocab, knobs)
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
        crate::launch::sampling::topk_softmax_f32(
            self.reg,
            self.stream,
            crate::SamplerTopkSoftmaxBuffers { logits, out_ids, out_probs },
            crate::SamplerTopkSoftmaxKnobs { vocab, k, inv_temp },
        )
    }

    // -- recurrent --

    fn gdn_state_step_f32_s128(
        &self,
        bufs: crate::GdnStepBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()> {
        crate::launch::recurrent::gdn_state_step_f32_s128(self.reg, self.stream, bufs, shape)
    }

    fn gdn_alpha_beta_f32(
        &self,
        bufs: crate::GdnAlphaBetaBuffers,
        shape: crate::GdnAlphaBetaShape,
    ) -> Result<()> {
        crate::launch::recurrent::gdn_alpha_beta_f32(self.reg, self.stream, bufs, shape)
    }

    fn gdn_state_step_alphabeta_f32_s128(
        &self,
        bufs: crate::GdnStepAlphaBetaBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()> {
        crate::launch::recurrent::gdn_state_step_alphabeta_f32_s128(self.reg, self.stream, bufs, shape)
    }

    fn gdn_assemble_conv_input_f32(
        &self,
        history: DevicePtr,
        current: DevicePtr,
        conv_input: DevicePtr,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Result<()> {
        crate::launch::recurrent::gdn_assemble_conv_input_f32(
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
        crate::launch::recurrent::gdn_state_step_alphabeta_f32_s128_batched_slots(
            self.reg,
            self.stream,
            bufs,
            shape,
        )
    }

    fn gdn_conv_trio_decode_f32_batched_slots(
        &self,
        bufs: crate::GdnConvTrioBatchedSlotsBuffers,
        shape: crate::GdnConvTrioShape,
    ) -> Result<()> {
        crate::launch::recurrent::gdn_conv_trio_decode_f32_batched_slots(self.reg, self.stream, bufs, shape)
    }

    fn gdn_split_qkv_f32(
        &self,
        bufs: crate::GdnSplitQkvBuffers,
        shape: crate::GdnSplitQkvShape,
    ) -> Result<()> {
        crate::launch::recurrent::gdn_split_qkv_f32(self.reg, self.stream, bufs, shape)
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
        crate::launch::router::dense_gemv_f32_f16(self.reg, self.stream, w, x, y, n_rows, k)
    }

    fn dense_gemv_f16_f16(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        crate::launch::router::dense_gemv_f16_f16(self.reg, self.stream, w, x, y, n_rows, k)
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
        crate::launch::router::dense_gemv_f16_f16_batched(
            self.reg,
            self.stream,
            crate::DenseGemvBatchedBuffers { w, x, y },
            crate::DenseGemvBatchedShape { n_rows, k, n_tokens },
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
        crate::launch::router::dense_gemv_f32_f16_batched(
            self.reg,
            self.stream,
            crate::DenseGemvBatchedBuffers { w, x, y },
            crate::DenseGemvBatchedShape { n_rows, k, n_tokens },
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
        crate::launch::softmax::softmax_masked_f16(
            self.reg,
            self.stream,
            crate::SoftmaxMaskedBuffers { scores, mask, out },
            crate::SoftmaxMaskedKnobs { m, k, scale },
        )
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
        crate::launch::conv::causal_conv1d_f32(
            self.reg,
            self.stream,
            crate::ConvCausal1dBuffers { conv_input, weight, y },
            crate::ConvCausal1dShape { n_new, conv_channels, conv_kernel },
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
        crate::launch::moe::topk_f32(
            self.reg,
            self.stream,
            crate::TopkBuffers { logits, idx, weights },
            crate::TopkShape { n_tokens, n_experts, k },
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
        crate::launch::moe::apply_per_expert_scale_f32(
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
        crate::launch::moe::indexed_moe_mmvq_q4_k_r2(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q6_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q6_k(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q5_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q5_k(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q3_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q3_k(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq4_xs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq4_xs(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq4_nl(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq4_nl(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq3_xxs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq3_xxs(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq3_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq3_s(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq2_xxs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq2_xxs(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq2_xs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq2_xs(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq2_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq2_s(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq1_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq1_s(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_iq1_m(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_iq1_m(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_0(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q4_0(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_1(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q4_1(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_0_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q4_0_gate_up(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmvq_q8_0(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q8_0(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q8_0_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q8_0_gate_up(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmvq_q4_k_r2_sorted(
        &self,
        buffers: crate::MoeMmvqSortedBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q4_k_r2_sorted(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_k_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_k_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q3_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q3_k_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q3_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q3_k_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q5_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q5_k_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q6_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q6_k_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_0_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_0_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_0_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_0_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_1_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_1_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q8_0_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q8_0_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q8_0_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q8_0_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    // ---- IQ tile8 impls. All follow the same shape: dispatch to the
    // matching free function in `crate::launch::moe`.

    fn indexed_moe_mmq_iq4_xs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq4_xs_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq4_xs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq4_xs_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq4_nl_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq4_nl_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq4_nl_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq4_nl_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq3_xxs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq3_xxs_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq3_xxs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq3_xxs_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq3_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq3_s_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq3_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq3_s_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq2_xxs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq2_xxs_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq2_xxs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq2_xxs_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq2_xs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq2_xs_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq2_xs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq2_xs_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq2_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq2_s_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq2_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq2_s_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq1_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq1_s_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq1_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq1_s_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_iq1_m_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq1_m_gate_up_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }
    fn indexed_moe_mmq_iq1_m_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_iq1_m_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k_gate_up_lds_tiled(
        &self,
        bufs: crate::MoeMmqQ4KGateUpLdsTiledBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_k_gate_up_lds_tiled(self.reg, self.stream, bufs, shape)
    }

    fn indexed_moe_mmq_q4_k_down_lds_tiled(
        &self,
        bufs: crate::MoeMmqQ4KDownLdsTiledBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_k_down_lds_tiled(self.reg, self.stream, bufs, shape)
    }

    fn indexed_moe_mmq_q5_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q5_k_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q6_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q6_k_down_tile8(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_k_gate_up_sorted(
        &self,
        buffers: crate::MoeMmvqGateUpSortedBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q4_k_gate_up_sorted(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmvq_q4_k_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmvq_q4_k_gate_up(
            self.reg,
            self.stream,
            buffers,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_k(
        &self,
        bufs: crate::MoeMmqQ4KBuffers,
        shape: crate::MoeMmqQ4KShape,
    ) -> Result<()> {
        crate::launch::moe::indexed_moe_mmq_q4_k(self.reg, self.stream, bufs, shape)
    }

    fn shared_expert_scale_f32(
        &self,
        shared_out: DevicePtr,
        x: DevicePtr,
        gate_w: DevicePtr,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<()> {
        crate::launch::moe::shared_expert_scale_f32(
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
        crate::launch::moe::moe_combine_f16(self.reg, self.stream, buf, shape)
    }

    fn moe_combine_no_residual_f16(
        &self,
        buf: crate::MoeCombineNoResidualBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        crate::launch::moe::moe_combine_no_residual_f16(self.reg, self.stream, buf, shape)
    }

    fn moe_combine_no_residual_f32(
        &self,
        buf: crate::MoeCombineNoResidualBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        crate::launch::moe::moe_combine_no_residual_f32(self.reg, self.stream, buf, shape)
    }

    fn moe_combine_two_residuals_f16(
        &self,
        buf: crate::MoeCombineTwoResidualsBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()> {
        crate::launch::moe::moe_combine_two_residuals_f16(self.reg, self.stream, buf, shape)
    }

    fn moe_sort_by_expert(
        &self,
        buf: crate::MoeSortBuffers,
        shape: crate::MoeSortShape,
    ) -> Result<()> {
        crate::launch::moe::moe_sort_by_expert(self.reg, self.stream, buf, shape)
    }

    fn moe_sort_by_expert_padded_16(
        &self,
        buf: crate::MoeSortPaddedBuffers,
        shape: crate::MoeSortPaddedShape,
    ) -> Result<()> {
        crate::launch::moe::moe_sort_by_expert_padded_16(self.reg, self.stream, buf, shape)
    }

    fn moe_sort_by_expert_padded(
        &self,
        buf: crate::MoeSortPaddedBuffers,
        shape: crate::MoeSortPaddedShape,
    ) -> Result<()> {
        crate::launch::moe::moe_sort_by_expert_padded(self.reg, self.stream, buf, shape)
    }

    fn apply_softcap_f32(&self, x: DevicePtr, y: DevicePtr, n: usize, cap: f32) -> Result<()> {
        crate::launch::softcap::apply_softcap_f32(self.reg, self.stream, x, y, n, cap)
    }

    fn gelu_f32_to_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::gelu_f32_to_f16(self.reg, self.stream, a, b, y, n)
    }

    fn gelu_mul_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        crate::launch::mlp::gelu_mul_f32(self.reg, self.stream, a, b, y, n)
    }
}
