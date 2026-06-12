//! Backend-portable op surface.
//!
//! Implementors carry their own `(registry, stream)` pair internally;
//! method args are pure shape scalars + device pointers. CUDA's
//! `CudaOps` will mirror this surface 1:1.
//!
//! Graph-capture slot methods (`attention_decode_f16_slots`,
//! `attention_prefill_f16_slots`) reference `ScalarSlot` from
//! `flambeau-backend-hip`. The whole trait is gated under
//! `feature = "hip"` at the crate root, so HIP-specific types in
//! method signatures are consistent — when CUDA arrives, the slot
//! types either go behind a `Slot` trait at flambeau-model-ops or split
//! into per-backend trait extensions.

use crate::MoeShape;
use anyhow::Result;
use flambeau_core::device::DevicePtr;
use flambeau_core::op::QDtype;

/// Portable op surface implemented per backend (HIP today, CUDA in V2).
///
/// The trait is intentionally one fat surface: every method a model
/// composition crate needs lands here, grouped by family in the
/// declaration below. We do not split into sub-traits because every
/// real consumer pulls from multiple groups in the same forward call.
pub trait Ops {
    // -- qmatmul (quantised weight × activation) --

    fn qmatmul(
        &self,
        buf: crate::QmatmulBuffers,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> Result<()>;

    fn mmvq_q4_0_t128(
        &self,
        weights: DevicePtr,
        y_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq_q4_0_gate_up_t128(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()>;

    /// Q4_0 gate+up row-tile batched MMVQ for n_slots ∈ [2, 4]. Each
    /// block handles R=4 consecutive output rows and shares one LDS-
    /// resident Q8_1 activation strip across the N decode slots. The
    /// per-call work is `(n_rows_gate + n_rows_up) × n_slots` outputs
    /// vs. `N × per-slot mmvq_q4_0_gate_up_t128` launches in the
    /// per-slot fallback. Activations slot-major `[N, k]` Q8_1;
    /// outputs slot-major `[N, n_rows_*]` F32.
    fn mmvq_q4_0_gate_up_row_tile_batched(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpBatchShape,
    ) -> Result<()>;

    /// Row-tiled Q4_0 MMVQ for non-fused projections (single weight
    /// matrix). Each block owns 4 output rows and shares one LDS-
    /// resident Q8_1 activation strip across the N decode slots.
    /// Activations slot-major `[N, k]` Q8_1; outputs slot-major
    /// `[N, n_rows]` F32.
    fn mmvq_q4_0_row_tile_batched(
        &self,
        buffers: crate::MmvqBuffers,
        shape: crate::MmvqBatchShape,
    ) -> Result<()>;

    /// Row-tiled Q8_0 sibling of [`Ops::mmvq_q4_0_row_tile_batched`].
    /// Same ABI; for GDN α/β (Q8_0 on Qwen3.6 hybrids) and any other
    /// non-gate+up Q8_0 projection at decode N∈{2,3,4}.
    fn mmvq_q8_0_row_tile_batched(
        &self,
        buffers: crate::MmvqBuffers,
        shape: crate::MmvqBatchShape,
    ) -> Result<()>;

    fn mmvq_q4_0_warpcoop64(
        &self,
        weights: DevicePtr,
        y_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq_q4_0_kv_f16dst(
        &self,
        buf: crate::MmvqKvF16Buffers,
        n_rows_kv: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq_q4_0_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()>;

    fn mmvq_q4_1_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()>;

    fn mmvq_q8_0_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()>;

    fn mmvq_q5_k_gate_up(
        &self,
        buffers: crate::MmvqGateUpBuffers,
        shape: crate::MmvqGateUpShape,
    ) -> Result<()>;

    fn mmvq(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MmvqShape,
        dtype_weight: QDtype,
    ) -> Result<()>;

    /// Q4_K r4 MMVQ for the wide-vocab LM head at decode: each block
    /// owns 4 output rows, halving the block count over r2. `n_rows` is
    /// the vocab; `n_superblocks` is `k / 256`. The 4-row amortisation
    /// only pays off at vocab ≥ 131072 with `k % 256 == 0` — the caller
    /// gates on that and falls back to `qmatmul` otherwise.
    fn mmvq_q4_k_r4(
        &self,
        buf: crate::MmvqBuffers,
        n_rows: usize,
        n_superblocks: usize,
    ) -> Result<()>;

    /// Weight × Q8_1 MMVQ writing directly into an F16 destination
    /// (saturating at ±65504). Skips the F32 scratch + `cast_f32_to_f16`
    /// two-step path for consumers whose downstream kernel expects F16
    /// (e.g. K projection feeding `rmsnorm_f16`). Supports Q4_0 / Q4_1
    /// / Q8_0 today; other dtypes bail. #120.
    fn mmvq_f16_direct(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MmvqShape,
        dtype_weight: QDtype,
    ) -> Result<()>;

    fn mmq(
        &self,
        buf: crate::MmvqBuffers,
        shape: crate::MatmulShape,
        dtype_weight: QDtype,
    ) -> Result<()>;

    // -- attention (decode + prefill) --

    fn attention_decode_f16(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    /// Graph-capture variant of `attention_decode_f16`. When `slots`
    /// is `Some`, the recorder tags the `n_tokens_kv` kernel arg so the
    /// caller can update it per replay via `HipGraphExec::set_slot`.
    fn attention_decode_f16_slots(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnDecodeSlots>,
    ) -> Result<()>;

    fn attention_decode_f16_batched(
        &self,
        buffers: crate::AttnBatchedBuffers,
        shape: crate::AttnDecodeBatchedShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    /// Single-launch per-slot K/V append for the batched-decode path.
    /// `slot_{k,v}_dst_ptrs` are `[n_slots] u64` device arrays of
    /// per-slot KV-cache base pointers; `slot_write_pos` is `[n_slots]
    /// i32` with each slot's pre-bump tail index. Writes one row of
    /// `kv_width` F16 from `k_src` / `v_src` (slot-major) at
    /// `dst + write_pos * kv_width` per slot.
    fn kv_append_f16_batched_slots(
        &self,
        buf: crate::KvAppendBatchedSlotsBuffers,
        shape: crate::KvAppendBatchedSlotsShape,
    ) -> Result<()>;

    /// PagedAttention prefill attention. Same flash-attn-v2 body as
    /// `attention_prefill_f16`; per-`t` K/V row resolved via
    /// `block_table[t / page_size] * page_size + (t & (page_size - 1))`.
    /// `page_size` must be a power of two.
    fn attention_prefill_f16_paged(
        &self,
        buffers: crate::AttnPagedPrefillBuffers,
        shape: crate::AttnPrefillPagedShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    /// PagedAttention prefill K + V append. Writes L K + V rows for
    /// a single slot's prefill into the slot's paged KV cache, walking
    /// the slot's row of the block table per token. The host must
    /// pre-populate `block_table` for `[start_pos, start_pos +
    /// n_tokens)`. `page_size` must be a power of two.
    fn kv_append_f16_paged_prefill(
        &self,
        buf: crate::KvAppendPagedPrefillBuffers,
        shape: crate::KvAppendPagedPrefillShape,
    ) -> Result<()>;

    /// PagedAttention sibling of `kv_append_f16_batched_slots`. Writes
    /// the per-slot K+V row into the page that the slot's block table
    /// currently maps to. `block_tables` is `[n_slots,
    /// max_pages_per_slot]` u32 row-major. `page_size` must be a power
    /// of two.
    fn kv_append_f16_paged_slots(
        &self,
        buf: crate::KvAppendPagedSlotsBuffers,
        shape: crate::KvAppendPagedSlotsShape,
    ) -> Result<()>;

    /// PagedAttention sibling of `attention_decode_f16_batched`. Reads
    /// K/V per-token rows through the slot's block-table indirection.
    /// `page_size` must be a power of two.
    fn attention_decode_f16_paged(
        &self,
        buffers: crate::AttnPagedDecodeBuffers,
        shape: crate::AttnDecodePagedShape,
        scale: f32,
    ) -> Result<()>;

    /// Fused V unit-RMSNorm + KV-cache append (K direct copy, V normed).
    /// See `hip::attention::kv_append_v_unit_norm_f16`.
    fn kv_append_v_unit_norm_f16(
        &self,
        buf: crate::KvAppendBuffers,
        shape: crate::KvAppendVUnitShape,
        write_pos: usize,
        eps: f32,
    ) -> Result<()>;

    fn attention_decode_f16_splitk(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    fn attention_decode_f16_splitk_h2(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    fn attention_decode_q8_kv(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnDecodeShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    fn attention_decode_q8_kv_splitk(
        &self,
        buffers: crate::AttnBuffers,
        partials: crate::AttnSplitkPartials,
        shape: crate::AttnSplitkShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    fn attention_prefill_q8_kv(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    fn attention_prefill_f16(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
    ) -> Result<()>;

    /// Graph-capture variant of `attention_prefill_f16`. When `slots`
    /// is `Some`, the recorder tags both `n_k_tokens` and `q_offset`
    /// kernel args for per-replay updates.
    fn attention_prefill_f16_slots(
        &self,
        buffers: crate::AttnBuffers,
        shape: crate::AttnPrefillShape,
        knobs: crate::AttnKnobs,
        slots: Option<crate::AttnPrefillSlots>,
    ) -> Result<()>;

    fn split_q_gate_f16(
        &self,
        fused_qg: DevicePtr,
        q_out: DevicePtr,
        gate_out: DevicePtr,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<()>;

    // -- norm (RMSNorm + L2-norm + Q8_1 quantize) --

    fn rmsnorm_f16(&self, buf: crate::NormBuffers, shape: crate::NormShape, eps: f32) -> Result<()>;

    fn rmsnorm_f16_add_residual(
        &self,
        buf: crate::NormFusedAddBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()>;

    /// Per-head V unit RMSNorm in place on
    /// `[n_tokens, n_kv_heads, head_dim]` F16. Unit weights (no
    /// learnable gamma). Composes with `kv_append_f16` and
    /// `kv_append_f16_batched_slots` for the mixed-batch path on
    /// gemma4-style archs whose `attn_v_unit_norm` flag is set.
    fn v_unit_norm_per_head_f16(
        &self,
        v: DevicePtr,
        n_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()>;

    fn rmsnorm_quant_q8_1(&self, buf: crate::NormBuffers, shape: crate::NormShape, eps: f32) -> Result<()>;

    fn rmsnorm_f32(&self, buf: crate::NormBuffers, shape: crate::NormShape, eps: f32) -> Result<()>;

    /// F32-in / F16-out fused RMSNorm. Replaces `cast_f32_to_f16 →
    /// rmsnorm_f16` for the gemma4 post-attn / post-ffn norm site.
    fn rmsnorm_f32_to_f16(&self, buf: crate::NormBuffers, shape: crate::NormShape, eps: f32) -> Result<()>;

    /// Fused rmsnorm_f32_to_f16 + residual add. Writes
    /// `resid_out = resid_in + rmsnorm(x * weight)`. `resid_out` may
    /// alias `resid_in` for in-place. Same launch shape as
    /// `rmsnorm_f32_to_f16`.
    fn rmsnorm_f32_to_f16_add_residual(
        &self,
        buf: crate::NormResidualBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()>;

    /// F16-input sibling of `rmsnorm_f32_to_f16_add_residual`. Used
    /// after an F16 AR sum (gemma4 post-attn / post-ffn path when the
    /// safety predicate allows skipping the F32 AR widening).
    fn rmsnorm_f16_to_f16_add_residual(
        &self,
        buf: crate::NormResidualBuffers,
        shape: crate::NormShape,
        eps: f32,
    ) -> Result<()>;

    fn l2_norm_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        eps: f32,
    ) -> Result<()>;

    fn quantize_q8_1(&self, x_f32: DevicePtr, y_q8_1: DevicePtr, n_elems: usize) -> Result<()>;

    fn quantize_q8_1_mmq(
        &self,
        x_f32: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()>;

    fn quantize_f16_q8_1_mmq(
        &self,
        x_f16: DevicePtr,
        y_q8_1_mmq: DevicePtr,
        ncols: usize,
        total_b: usize,
    ) -> Result<()>;

    fn quantize_f16_q8_1(&self, x_f16: DevicePtr, y_q8_1: DevicePtr, n_elems: usize) -> Result<()>;

    fn quantize_f16_q8_0(&self, x_f16: DevicePtr, y_q8_0: DevicePtr, n_elems: usize) -> Result<()>;

    // -- mlp (pointwise + gated activations) --

    fn silu_f32(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn swiglu_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn swiglu_f32_to_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn swiglu_f32_to_q8_1(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y_q8_1: DevicePtr,
        n: usize,
    ) -> Result<()>;

    /// Fused `y_f16[i] = (fp16)(gelu(a[i]) * b[i])` — Gemma 4 dense
    /// FFN. GELU = ggml tanh-approximation form.
    fn gelu_f32_to_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    /// Fused `y_f32[i] = gelu(a[i]) * b[i]` — Gemma 4 per-layer
    /// side-channel embedding gate.
    fn gelu_mul_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn scale_f32(&self, x: DevicePtr, y: DevicePtr, n: usize, scale: f32) -> Result<()>;

    /// F16 variant of [`Ops::scale_f32`].
    fn scale_f16(&self, x: DevicePtr, y: DevicePtr, n: usize, scale: f32) -> Result<()>;

    fn add_f16(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn add_f32(&self, a: DevicePtr, b: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn swiglu_f16(&self, gate: DevicePtr, up: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn sigmoid_mul_f16(&self, gate: DevicePtr, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    // -- pe (rope) --

    fn rope_f16(
        &self,
        buf: crate::RopeBuffers,
        shape: crate::RopeShape,
        theta_base: f32,
    ) -> Result<()>;

    fn rope_neox_partial_f16(
        &self,
        buf: crate::RopeBuffers,
        shape: crate::RopePartialShape,
        theta_base: f32,
    ) -> Result<()>;

    /// Fused per-head rmsnorm + partial NeoX RoPE, F16 in-place. See
    /// `hip::pe::rmsnorm_rope_neox_partial_f16` for the math.
    fn rmsnorm_rope_neox_partial_f16(
        &self,
        buf: crate::RopeFusedBuffers,
        shape: crate::RopePartialShape,
        theta_base: f32,
        eps: f32,
    ) -> Result<()>;

    // -- cast (dtype conversion) --

    fn cast_f32_to_f16(&self, x_f32: DevicePtr, y_f16: DevicePtr, n: usize) -> Result<()>;
    fn cast_f16_to_f32(&self, x_f16: DevicePtr, y_f32: DevicePtr, n: usize) -> Result<()>;

    // -- sampling --

    fn apply_penalties_f32(
        &self,
        bufs: crate::PenaltyBuffers,
        n_pairs: usize,
        vocab: usize,
        knobs: crate::PenaltyKnobs,
    ) -> Result<()>;

    fn topk_softmax_f32(
        &self,
        logits: DevicePtr,
        out_ids: DevicePtr,
        out_probs: DevicePtr,
        vocab: usize,
        k: usize,
        inv_temp: f32,
    ) -> Result<()>;

    // -- recurrent (gated delta-net state) --

    fn gdn_state_step_f32_s128(
        &self,
        bufs: crate::GdnStepBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()>;

    fn gdn_alpha_beta_f32(
        &self,
        bufs: crate::GdnAlphaBetaBuffers,
        shape: crate::GdnAlphaBetaShape,
    ) -> Result<()>;

    fn gdn_state_step_alphabeta_f32_s128(
        &self,
        bufs: crate::GdnStepAlphaBetaBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()>;

    fn gdn_assemble_conv_input_f32(
        &self,
        history: DevicePtr,
        current: DevicePtr,
        conv_input: DevicePtr,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Result<()>;

    /// Batched-slots single-token GDN recurrent step. Each slot owns
    /// an independent state buffer; `state_in_ptrs` / `state_out_ptrs`
    /// are `[B] u64` device arrays of those base pointers. Activations
    /// (q/k/v/alpha/beta/attn_out) are slot-major `[B, L, H, S_v]`.
    /// Same compute as `gdn_state_step_alphabeta_f32_s128`.
    fn gdn_state_step_alphabeta_f32_s128_batched_slots(
        &self,
        bufs: crate::GdnStepAlphaBetaBatchedSlotsBuffers,
        shape: crate::GdnStepShape,
    ) -> Result<()>;

    /// Batched-slots single-token conv trio (assemble + causal_conv1d
    /// + history shift) collapsed into one launch across N slots, each
    ///   with its own conv-history buffer. `slot_history_ptrs` is
    ///   `[N] u64`; `qkv_mixed` and `conv_out` are slot-major
    ///   `[N, conv_channels]`.
    fn gdn_conv_trio_decode_f32_batched_slots(
        &self,
        bufs: crate::GdnConvTrioBatchedSlotsBuffers,
        shape: crate::GdnConvTrioShape,
    ) -> Result<()>;

    fn gdn_split_qkv_f32(
        &self,
        bufs: crate::GdnSplitQkvBuffers,
        shape: crate::GdnSplitQkvShape,
    ) -> Result<()>;

    // -- router (dense F16/F32 GEMV for routers + LM head fragments) --

    fn dense_gemv_f32_f16(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()>;

    fn dense_gemv_f16_f16(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
    ) -> Result<()>;

    fn dense_gemv_f16_f16_batched(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<()>;

    fn dense_gemv_f32_f16_batched(
        &self,
        w: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n_rows: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<()>;

    // -- softmax --

    fn softmax_masked_f16(
        &self,
        scores: DevicePtr,
        mask: DevicePtr,
        out: DevicePtr,
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<()>;

    // -- conv (causal 1-D) --

    fn causal_conv1d_f32(
        &self,
        conv_input: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        n_new: usize,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Result<()>;

    // -- moe (router → indexed expert matmul → combine) --

    fn topk_f32(
        &self,
        logits: DevicePtr,
        idx: DevicePtr,
        weights: DevicePtr,
        n_tokens: usize,
        n_experts: usize,
        k: usize,
    ) -> Result<()>;

    /// `expert_weights[k] *= expert_scales[expert_ids[k]]` for k in 0..top_k.
    /// Folds gemma4's per-expert `ffn_down_exps.scale` into the routing
    /// weights so `moe_combine_*` picks up the post-down scaling for free
    /// (equivalent to multiplying each expert's down output by the
    /// scalar before the weighted sum — see candle's `quantized_gemma4`
    /// reference at line 2521 of `quantized_gemma4.rs`).
    fn apply_per_expert_scale_f32(
        &self,
        expert_weights: DevicePtr,
        expert_ids: DevicePtr,
        expert_scales: DevicePtr,
        n_tokens: usize,
        top_k: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_r2(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q6_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q5_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q3_k(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq4_xs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq4_nl(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq3_xxs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq3_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq2_xxs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq2_xs(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq2_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq1_s(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_iq1_m(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_0(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_1(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_0_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q8_0(
        &self,
        buffers: crate::MoeMmvqBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q8_0_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_r2_sorted(
        &self,
        buffers: crate::MoeMmvqSortedBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q3_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q3_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q5_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q6_k_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_0_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_0_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_1_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q8_0_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q8_0_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    // ---- IQ tile8: gate_up + down for the 9 IQ families ----
    // Same signature pattern as the q4_0/q8_0/q4_k tile8 trait methods
    // above. Phase-4 free functions live at
    // `crates/ops/src/hip/moe.rs::indexed_moe_mmq_iq*_{gate_up,down}_tile8`.

    fn indexed_moe_mmq_iq4_xs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq4_xs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq4_nl_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq4_nl_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq3_xxs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq3_xxs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq3_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq3_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq2_xxs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq2_xxs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq2_xs_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq2_xs_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq2_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq2_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq1_s_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq1_s_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq1_m_gate_up_tile8(
        &self,
        buffers: crate::MoeMmqTile8GateUpBuffers,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq1_m_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_gate_up_turbo(
        &self,
        bufs: crate::MoeMmqQ4KGateUpTurboBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_down_turbo(
        &self,
        bufs: crate::MoeMmqQ4KDownTurboBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q5_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q6_k_down_tile8(
        &self,
        buffers: crate::MoeMmqTile8DownBuffers,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_gate_up_sorted(
        &self,
        buffers: crate::MoeMmvqGateUpSortedBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_gate_up(
        &self,
        buffers: crate::MoeMmvqGateUpBuffers,
        shape: crate::MoeMmvqShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k(
        &self,
        bufs: crate::MoeMmqQ4KBuffers,
        shape: crate::MoeMmqQ4KShape,
    ) -> Result<()>;

    fn shared_expert_scale_f32(
        &self,
        shared_out: DevicePtr,
        x: DevicePtr,
        gate_w: DevicePtr,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<()>;

    fn moe_combine_f16(
        &self,
        buf: crate::MoeCombineBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()>;

    fn moe_combine_no_residual_f16(
        &self,
        buf: crate::MoeCombineNoResidualBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()>;

    fn moe_combine_no_residual_f32(
        &self,
        buf: crate::MoeCombineNoResidualBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()>;

    fn moe_combine_two_residuals_f16(
        &self,
        buf: crate::MoeCombineTwoResidualsBuffers,
        shape: crate::MoeCombineShape,
    ) -> Result<()>;

    fn moe_sort_by_expert(
        &self,
        buf: crate::MoeSortBuffers,
        shape: crate::MoeSortShape,
    ) -> Result<()>;

    fn moe_sort_by_expert_padded_16(
        &self,
        buf: crate::MoeSortPaddedBuffers,
        shape: crate::MoeSortPaddedShape,
    ) -> Result<()>;

    fn moe_sort_by_expert_padded(
        &self,
        buf: crate::MoeSortPaddedBuffers,
        shape: crate::MoeSortPaddedShape,
    ) -> Result<()>;

    /// Final-logit softcap: `y[i] = tanh(x[i] / cap) * cap`. In-place
    /// safe (`x` may equal `y`). Used by Gemma4 on the LM-head logits.
    fn apply_softcap_f32(&self, x: DevicePtr, y: DevicePtr, n: usize, cap: f32) -> Result<()>;
}
