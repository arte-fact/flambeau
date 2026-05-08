//! Backend-portable op surface.
//!
//! Implementors carry their own `(registry, stream)` pair internally;
//! method args are pure shape scalars + device pointers. CUDA's
//! `CudaOps` will mirror this surface 1:1.
//!
//! Methods that today reference HIP-specific scaffolding (graph-capture
//! `ScalarSlot` overloads, the `splitk_chunk_size` utility) stay as
//! inherent methods on `HipOps` — not part of this portable surface.
//! When CUDA arrives we will revisit slot abstractions if and only if a
//! caller actually needs them through the trait.

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

    fn mmvq_bf16_bf16(
        &self,
        weights: DevicePtr,
        act_bf16: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
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

    fn attention_decode_bf16(
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

    fn split_q_gate_bf16(
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

    fn rmsnorm_bf16(
        &self,
        x_bf16: DevicePtr,
        weight_f16: DevicePtr,
        y_bf16: DevicePtr,
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

    fn swiglu_f32_to_bf16(
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

    fn scale_f32(
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

    fn sigmoid_mul_bf16(
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

    fn rope_neox_partial_bf16(
        &self,
        x: DevicePtr,
        positions: DevicePtr,
        theta_base: f32,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        rotated_dims: usize,
    ) -> Result<()>;

    // -- cast (dtype conversion) --

    fn cast_f32_to_f16(&self, x_f32: DevicePtr, y_f16: DevicePtr, n: usize) -> Result<()>;
    fn cast_f16_to_f32(&self, x_f16: DevicePtr, y_f32: DevicePtr, n: usize) -> Result<()>;
    fn cast_f32_to_bf16(&self, x_f32: DevicePtr, y_bf16: DevicePtr, n: usize) -> Result<()>;
    fn cast_bf16_to_f32(&self, x_bf16: DevicePtr, y_f32: DevicePtr, n: usize) -> Result<()>;
    fn cast_f16_to_bf16(&self, x_f16: DevicePtr, y_bf16: DevicePtr, n: usize) -> Result<()>;
    fn cast_bf16_to_f16(&self, x_bf16: DevicePtr, y_f16: DevicePtr, n: usize) -> Result<()>;

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
}
