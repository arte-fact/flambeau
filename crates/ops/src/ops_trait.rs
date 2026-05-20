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
//! types either go behind a `Slot` trait at flambeau-blocks or split
//! into per-backend trait extensions.

use crate::MoeShape;
use anyhow::Result;
use flambeau_backend_hip::ScalarSlot;
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
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        act_q8_1_mmq: DevicePtr,
        dst: DevicePtr,
        m: usize,
        k: usize,
        n: usize,
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
        gate_w: DevicePtr,
        up_w: DevicePtr,
        y_q8_1: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows_gate: usize,
        n_rows_up: usize,
        k: usize,
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
        k_w: DevicePtr,
        v_w: DevicePtr,
        y_q8_1: DevicePtr,
        k_out_f16: DevicePtr,
        v_out_f16: DevicePtr,
        n_rows_kv: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq_q4_0_gate_up(
        &self,
        gate_w: DevicePtr,
        up_w: DevicePtr,
        y_q8_1: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows_gate: usize,
        n_rows_up: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq_q4_1_gate_up(
        &self,
        gate_w: DevicePtr,
        up_w: DevicePtr,
        y_q8_1: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows_gate: usize,
        n_rows_up: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq_q8_0_gate_up(
        &self,
        gate_w: DevicePtr,
        up_w: DevicePtr,
        y_q8_1: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows_gate: usize,
        n_rows_up: usize,
        k: usize,
    ) -> Result<()>;

    fn mmvq(
        &self,
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
        dtype_weight: QDtype,
    ) -> Result<()>;

    /// Weight × Q8_1 MMVQ writing directly into an F16 destination
    /// (saturating at ±65504). Skips the F32 scratch + `cast_f32_to_f16`
    /// two-step path for consumers whose downstream kernel expects F16
    /// (e.g. K projection feeding `rmsnorm_f16`). Supports Q4_0 / Q4_1
    /// / Q8_0 today; other dtypes bail. #120.
    fn mmvq_f16_direct(
        &self,
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        dst_f16: DevicePtr,
        n_rows: usize,
        k: usize,
        dtype_weight: QDtype,
    ) -> Result<()>;

    fn mmq(
        &self,
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        dst: DevicePtr,
        m: usize,
        k: usize,
        n: usize,
        dtype_weight: QDtype,
    ) -> Result<()>;

    // -- attention (decode + prefill) --

    #[allow(clippy::too_many_arguments)]
    fn attention_decode_f16(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_tokens_kv: usize,
        scale: f32,
        window_size: i32,
    ) -> Result<()>;

    /// Graph-capture variant of `attention_decode_f16`. When
    /// `n_tokens_kv_slot` is `Some`, the recorder tags the
    /// `n_tokens_kv` kernel arg so the caller can update it per
    /// replay via `HipGraphExec::set_slot`.
    #[allow(clippy::too_many_arguments)]
    fn attention_decode_f16_slots(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_tokens_kv: usize,
        scale: f32,
        window_size: i32,
        n_tokens_kv_slot: Option<ScalarSlot>,
    ) -> Result<()>;

    fn attention_decode_f16_batched(
        &self,
        q_batched: DevicePtr,
        k_cache_ptrs: DevicePtr,
        v_cache_ptrs: DevicePtr,
        out_batched: DevicePtr,
        n_tokens_kv: DevicePtr,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_slots: usize,
        scale: f32,
    ) -> Result<()>;

    /// Single-launch per-slot K/V append for the batched-decode path.
    /// `slot_{k,v}_dst_ptrs` are `[n_slots] u64` device arrays of
    /// per-slot KV-cache base pointers; `slot_write_pos` is `[n_slots]
    /// i32` with each slot's pre-bump tail index. Writes one row of
    /// `kv_width` F16 from `k_src` / `v_src` (slot-major) at
    /// `dst + write_pos * kv_width` per slot.
    fn kv_append_f16_batched_slots(
        &self,
        k_src: DevicePtr,
        v_src: DevicePtr,
        slot_k_dst_ptrs: DevicePtr,
        slot_v_dst_ptrs: DevicePtr,
        slot_write_pos: DevicePtr,
        n_slots: usize,
        kv_width: usize,
    ) -> Result<()>;

    /// Fused V unit-RMSNorm + KV-cache append (K direct copy, V normed).
    /// See `hip::attention::kv_append_v_unit_norm_f16`.
    #[allow(clippy::too_many_arguments)]
    fn kv_append_v_unit_norm_f16(
        &self,
        k_src: DevicePtr,
        v_src: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        n_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        write_pos: usize,
        eps: f32,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn attention_decode_f16_splitk(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        partials_m: DevicePtr,
        partials_s: DevicePtr,
        partials_o: DevicePtr,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_tokens_kv: usize,
        chunk_size: usize,
        scale: f32,
        window_size: i32,
    ) -> Result<()>;

    fn attention_decode_q8_kv(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_tokens_kv: usize,
        scale: f32,
    ) -> Result<()>;

    fn attention_decode_q8_kv_splitk(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        partials_m: DevicePtr,
        partials_s: DevicePtr,
        partials_o: DevicePtr,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_tokens_kv: usize,
        chunk_size: usize,
        scale: f32,
    ) -> Result<()>;

    fn attention_prefill_q8_kv(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        n_q_tokens: usize,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_k_tokens: usize,
        q_offset: usize,
        scale: f32,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_f16(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        n_q_tokens: usize,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_k_tokens: usize,
        q_offset: usize,
        scale: f32,
        window_size: i32,
    ) -> Result<()>;

    /// Graph-capture variant of `attention_prefill_f16`. When
    /// `n_k_slot` / `q_off_slot` are `Some`, the recorder tags those
    /// kernel args for per-replay updates.
    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_f16_slots(
        &self,
        q: DevicePtr,
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        out: DevicePtr,
        n_q_tokens: usize,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_k_tokens: usize,
        q_offset: usize,
        scale: f32,
        window_size: i32,
        n_k_slot: Option<ScalarSlot>,
        q_off_slot: Option<ScalarSlot>,
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

    fn rmsnorm_f16(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()>;

    fn rmsnorm_f16_add_residual(
        &self,
        x_in: DevicePtr,
        delta: DevicePtr,
        weight: DevicePtr,
        mid: DevicePtr,
        mid_norm: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()>;

    fn rmsnorm_quant_q8_1(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        y_q8_1: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()>;

    fn rmsnorm_f32(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()>;

    /// F32-in / F16-out fused RMSNorm. Replaces `cast_f32_to_f16 →
    /// rmsnorm_f16` for the gemma4 post-attn / post-ffn norm site.
    fn rmsnorm_f32_to_f16(
        &self,
        x_f32: DevicePtr,
        weight_f16: DevicePtr,
        y_f16: DevicePtr,
        m: usize,
        k: usize,
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

    fn quantize_q8_1(
        &self,
        x_f32: DevicePtr,
        y_q8_1: DevicePtr,
        n_elems: usize,
    ) -> Result<()>;

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

    fn quantize_f16_q8_1(
        &self,
        x_f16: DevicePtr,
        y_q8_1: DevicePtr,
        n_elems: usize,
    ) -> Result<()>;

    fn quantize_f16_q8_0(
        &self,
        x_f16: DevicePtr,
        y_q8_0: DevicePtr,
        n_elems: usize,
    ) -> Result<()>;

    // -- mlp (pointwise + gated activations) --

    fn silu_f32(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()>;

    fn swiglu_f32(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    fn swiglu_f32_to_f16(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    fn swiglu_f32_to_q8_1(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y_q8_1: DevicePtr,
        n: usize,
    ) -> Result<()>;

    /// Fused `y_f16[i] = (fp16)(gelu(a[i]) * b[i])` — Gemma 4 dense
    /// FFN. GELU = ggml tanh-approximation form.
    fn gelu_f32_to_f16(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    /// Fused `y_f32[i] = gelu(a[i]) * b[i]` — Gemma 4 per-layer
    /// side-channel embedding gate.
    fn gelu_mul_f32(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    fn scale_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
        scale: f32,
    ) -> Result<()>;

    /// F16 variant of [`Ops::scale_f32`].
    fn scale_f16(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
        scale: f32,
    ) -> Result<()>;

    fn add_f16(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    fn add_f32(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    fn swiglu_f16(
        &self,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    fn sigmoid_mul_f16(
        &self,
        gate: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()>;

    // -- pe (rope) --

    fn rope_f16(
        &self,
        x: DevicePtr,
        positions: DevicePtr,
        theta_base: f32,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<()>;

    fn rope_neox_partial_f16(
        &self,
        x: DevicePtr,
        positions: DevicePtr,
        theta_base: f32,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        rotated_dims: usize,
    ) -> Result<()>;

    /// Fused per-head rmsnorm + partial NeoX RoPE, F16 in-place. See
    /// `hip::pe::rmsnorm_rope_neox_partial_f16` for the math.
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_rope_neox_partial_f16(
        &self,
        x: DevicePtr,
        norm_w: DevicePtr,
        positions: DevicePtr,
        theta_base: f32,
        eps: f32,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        rotated_dims: usize,
    ) -> Result<()>;

    // -- cast (dtype conversion) --

    fn cast_f32_to_f16(&self, x_f32: DevicePtr, y_f16: DevicePtr, n: usize) -> Result<()>;
    fn cast_f16_to_f32(&self, x_f16: DevicePtr, y_f32: DevicePtr, n: usize) -> Result<()>;

    // -- sampling --

    fn apply_penalties_f32(
        &self,
        logits: DevicePtr,
        token_counts: DevicePtr,
        n_pairs: usize,
        vocab: usize,
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
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
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        state_in: DevicePtr,
        state_out: DevicePtr,
        attn_out: DevicePtr,
        b: usize,
        h_v: usize,
        l: usize,
        n_rep: usize,
        rep_inner_layout: bool,
    ) -> Result<()>;

    fn gdn_alpha_beta_f32(
        &self,
        alpha_in: DevicePtr,
        beta_in: DevicePtr,
        ssm_dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        gate_out: DevicePtr,
        beta_out: DevicePtr,
        num_v_heads: usize,
        n_tokens: usize,
    ) -> Result<()>;

    fn gdn_state_step_alphabeta_f32_s128(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        alpha_in: DevicePtr,
        beta_in: DevicePtr,
        ssm_dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        state_in: DevicePtr,
        state_out: DevicePtr,
        attn_out: DevicePtr,
        b: usize,
        h_v: usize,
        l: usize,
        n_rep: usize,
        rep_inner_layout: bool,
    ) -> Result<()>;

    fn gdn_assemble_conv_input_f32(
        &self,
        history: DevicePtr,
        current: DevicePtr,
        conv_input: DevicePtr,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Result<()>;

    fn gdn_split_qkv_f32(
        &self,
        silu_out: DevicePtr,
        q_out: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        n_tokens: usize,
        qk_size: usize,
        v_size: usize,
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
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q6_k(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q5_k(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_0(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_blocks_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_1(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_blocks_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_0_gate_up(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_blocks_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q8_0(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_blocks_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q8_0_gate_up(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_blocks_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_r2_sorted(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_0_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_0_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_1_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q8_0_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q8_0_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    // ---- IQ tile8: gate_up + down for the 9 IQ families ----
    // Same signature pattern as the q4_0/q8_0/q4_k tile8 trait methods
    // above. Phase-4 free functions live at
    // `crates/ops/src/hip/moe.rs::indexed_moe_mmq_iq*_{gate_up,down}_tile8`.

    fn indexed_moe_mmq_iq4_xs_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq4_xs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq4_nl_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq4_nl_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq3_xxs_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq3_xxs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq3_s_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq3_s_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq2_xxs_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq2_xxs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq2_xs_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq2_xs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq2_s_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq2_s_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq1_s_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq1_s_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_iq1_m_gate_up_tile8(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;
    fn indexed_moe_mmq_iq1_m_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_gate_up_turbo(
        &self,
        gate_w: DevicePtr,
        up_w: DevicePtr,
        y_mmq: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k_down_turbo(
        &self,
        down_w: DevicePtr,
        y_mmq: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q5_k_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmq_q6_k_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_gate_up_sorted(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmvq_q4_k_gate_up(
        &self,
        w_gate: DevicePtr,
        w_up: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()>;

    fn indexed_moe_mmq_q4_k(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        bucket_expert: DevicePtr,
        bucket_slots: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_sb_per_row: usize,
        top_k: usize,
        n_buckets: usize,
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
        expert_outs: DevicePtr,
        weights: DevicePtr,
        residual: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()>;

    fn moe_combine_no_residual_f16(
        &self,
        expert_outs: DevicePtr,
        weights: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()>;

    fn moe_combine_no_residual_f32(
        &self,
        expert_outs: DevicePtr,
        weights: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()>;

    fn moe_combine_two_residuals_f16(
        &self,
        expert_outs: DevicePtr,
        weights: DevicePtr,
        residual1: DevicePtr,
        residual2: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()>;

    fn moe_sort_by_expert(
        &self,
        expert_ids: DevicePtr,
        counts: DevicePtr,
        offsets: DevicePtr,
        cursors: DevicePtr,
        sorted_pair_idx: DevicePtr,
        total: usize,
        n_experts: usize,
    ) -> Result<()>;

    fn moe_sort_by_expert_padded_16(
        &self,
        expert_ids: DevicePtr,
        counts: DevicePtr,
        offsets: DevicePtr,
        cursors: DevicePtr,
        sorted_pair_idx: DevicePtr,
        padded_offsets: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        total: usize,
        n_experts: usize,
        max_tokens: usize,
        top_k: usize,
    ) -> Result<()>;

    fn moe_sort_by_expert_padded(
        &self,
        expert_ids: DevicePtr,
        counts: DevicePtr,
        offsets: DevicePtr,
        cursors: DevicePtr,
        sorted_pair_idx: DevicePtr,
        padded_offsets: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        total: usize,
        n_experts: usize,
        max_tokens: usize,
        top_k: usize,
    ) -> Result<()>;

    /// Final-logit softcap: `y[i] = tanh(x[i] / cap) * cap`. In-place
    /// safe (`x` may equal `y`). Used by Gemma4 on the LM-head logits.
    fn apply_softcap_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
        cap: f32,
    ) -> Result<()>;
}
