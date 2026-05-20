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
}

impl<'a> Ops for HipOps<'a> {
    // -- qmatmul --

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
    ) -> Result<()> {
        super::qmatmul::qmatmul(
            self.reg, self.stream, weights, act_q8_1, act_q8_1_mmq, dst, m, k, n, dtype_weight,
        )
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
        gate_w: DevicePtr,
        up_w: DevicePtr,
        y_q8_1: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        n_rows_gate: usize,
        n_rows_up: usize,
        k: usize,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_gate_up_t128(
            self.reg, self.stream, gate_w, up_w, y_q8_1, gate_out, up_out, n_rows_gate, n_rows_up,
            k,
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
        k_w: DevicePtr,
        v_w: DevicePtr,
        y_q8_1: DevicePtr,
        k_out_f16: DevicePtr,
        v_out_f16: DevicePtr,
        n_rows_kv: usize,
        k: usize,
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_kv_f16dst(
            self.reg, self.stream, k_w, v_w, y_q8_1, k_out_f16, v_out_f16, n_rows_kv, k,
        )
    }

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
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_0_gate_up(
            self.reg, self.stream, gate_w, up_w, y_q8_1, gate_out, up_out, n_rows_gate, n_rows_up,
            k,
        )
    }

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
    ) -> Result<()> {
        super::qmatmul::mmvq_q4_1_gate_up(
            self.reg, self.stream, gate_w, up_w, y_q8_1, gate_out, up_out, n_rows_gate, n_rows_up,
            k,
        )
    }

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
    ) -> Result<()> {
        super::qmatmul::mmvq_q8_0_gate_up(
            self.reg, self.stream, gate_w, up_w, y_q8_1, gate_out, up_out, n_rows_gate, n_rows_up,
            k,
        )
    }

    fn mmvq(
        &self,
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        k: usize,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::mmvq(
            self.reg, self.stream, weights, act_q8_1, dst, n_rows, k, dtype_weight,
        )
    }

    fn mmvq_f16_direct(
        &self,
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        dst_f16: DevicePtr,
        n_rows: usize,
        k: usize,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::mmvq_f16_direct(
            self.reg, self.stream, weights, act_q8_1, dst_f16, n_rows, k, dtype_weight,
        )
    }

    fn mmq(
        &self,
        weights: DevicePtr,
        act_q8_1: DevicePtr,
        dst: DevicePtr,
        m: usize,
        k: usize,
        n: usize,
        dtype_weight: QDtype,
    ) -> Result<()> {
        super::qmatmul::mmq(
            self.reg, self.stream, weights, act_q8_1, dst, m, k, n, dtype_weight,
        )
    }

    // -- attention --

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
    ) -> Result<()> {
        super::attention::attention_decode_f16(
            self.reg, self.stream, q, k_cache, v_cache, out, n_heads_q, n_heads_kv, head_dim,
            n_tokens_kv, scale, window_size,
        )
    }

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
        n_tokens_kv_slot: Option<flambeau_backend_hip::ScalarSlot>,
    ) -> Result<()> {
        super::attention::attention_decode_f16_slots(
            self.reg, self.stream, q, k_cache, v_cache, out, n_heads_q, n_heads_kv, head_dim,
            n_tokens_kv, scale, window_size, n_tokens_kv_slot,
        )
    }

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
    ) -> Result<()> {
        super::attention::attention_decode_f16_batched(
            self.reg, self.stream, q_batched, k_cache_ptrs, v_cache_ptrs, out_batched, n_tokens_kv,
            n_heads_q, n_heads_kv, head_dim, n_slots, scale,
        )
    }

    fn kv_append_f16_batched_slots(
        &self,
        k_src: DevicePtr,
        v_src: DevicePtr,
        slot_k_dst_ptrs: DevicePtr,
        slot_v_dst_ptrs: DevicePtr,
        slot_write_pos: DevicePtr,
        n_slots: usize,
        kv_width: usize,
    ) -> Result<()> {
        super::attention::kv_append_f16_batched_slots(
            self.reg,
            self.stream,
            k_src,
            v_src,
            slot_k_dst_ptrs,
            slot_v_dst_ptrs,
            slot_write_pos,
            n_slots,
            kv_width,
        )
    }

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
    ) -> Result<()> {
        super::attention::kv_append_v_unit_norm_f16(
            self.reg, self.stream, k_src, v_src, k_cache, v_cache,
            n_tokens, n_kv_heads, head_dim, write_pos, eps,
        )
    }

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
    ) -> Result<()> {
        super::attention::attention_decode_f16_splitk(
            self.reg, self.stream, q, k_cache, v_cache, out, partials_m, partials_s, partials_o,
            n_heads_q, n_heads_kv, head_dim, n_tokens_kv, chunk_size, scale, window_size,
        )
    }

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
    ) -> Result<()> {
        super::attention::attention_decode_q8_kv(
            self.reg, self.stream, q, k_cache, v_cache, out, n_heads_q, n_heads_kv, head_dim,
            n_tokens_kv, scale,
        )
    }

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
    ) -> Result<()> {
        super::attention::attention_decode_q8_kv_splitk(
            self.reg, self.stream, q, k_cache, v_cache, out, partials_m, partials_s, partials_o,
            n_heads_q, n_heads_kv, head_dim, n_tokens_kv, chunk_size, scale,
        )
    }

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
    ) -> Result<()> {
        super::attention::attention_prefill_q8_kv(
            self.reg, self.stream, q, k_cache, v_cache, out, n_q_tokens, n_heads_q, n_heads_kv,
            head_dim, n_k_tokens, q_offset, scale,
        )
    }

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
    ) -> Result<()> {
        super::attention::attention_prefill_f16(
            self.reg, self.stream, q, k_cache, v_cache, out, n_q_tokens, n_heads_q, n_heads_kv,
            head_dim, n_k_tokens, q_offset, scale, window_size,
        )
    }

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
        n_k_slot: Option<flambeau_backend_hip::ScalarSlot>,
        q_off_slot: Option<flambeau_backend_hip::ScalarSlot>,
    ) -> Result<()> {
        super::attention::attention_prefill_f16_slots(
            self.reg, self.stream, q, k_cache, v_cache, out, n_q_tokens, n_heads_q, n_heads_kv,
            head_dim, n_k_tokens, q_offset, scale, window_size, n_k_slot, q_off_slot,
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
            self.reg, self.stream, fused_qg, q_out, gate_out, n_tokens, n_heads, head_dim,
        )
    }

    // -- norm --

    fn rmsnorm_f16(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f16(self.reg, self.stream, x, weight, y, m, k, eps)
    }

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
    ) -> Result<()> {
        super::norm::rmsnorm_f16_add_residual(
            self.reg, self.stream, x_in, delta, weight, mid, mid_norm, m, k, eps,
        )
    }

    fn rmsnorm_quant_q8_1(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        y_q8_1: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_quant_q8_1(self.reg, self.stream, x, weight, y_q8_1, m, k, eps)
    }

    fn rmsnorm_f32(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f32(self.reg, self.stream, x, weight, y, m, k, eps)
    }

    fn rmsnorm_f32_to_f16(
        &self,
        x_f32: DevicePtr,
        weight_f16: DevicePtr,
        y_f16: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f32_to_f16(self.reg, self.stream, x_f32, weight_f16, y_f16, m, k, eps)
    }

    fn rmsnorm_f32_to_f16_add_residual(
        &self,
        x_f32: DevicePtr,
        weight_f16: DevicePtr,
        resid_in_f16: DevicePtr,
        resid_out_f16: DevicePtr,
        m: usize,
        k: usize,
        eps: f32,
    ) -> Result<()> {
        super::norm::rmsnorm_f32_to_f16_add_residual(
            self.reg, self.stream, x_f32, weight_f16, resid_in_f16, resid_out_f16, m, k, eps,
        )
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

    fn quantize_q8_1(
        &self,
        x_f32: DevicePtr,
        y_q8_1: DevicePtr,
        n_elems: usize,
    ) -> Result<()> {
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

    fn quantize_f16_q8_1(
        &self,
        x_f16: DevicePtr,
        y_q8_1: DevicePtr,
        n_elems: usize,
    ) -> Result<()> {
        super::norm::quantize_f16_q8_1(self.reg, self.stream, x_f16, y_q8_1, n_elems)
    }

    fn quantize_f16_q8_0(
        &self,
        x_f16: DevicePtr,
        y_q8_0: DevicePtr,
        n_elems: usize,
    ) -> Result<()> {
        super::norm::quantize_f16_q8_0(self.reg, self.stream, x_f16, y_q8_0, n_elems)
    }

    // -- mlp --

    fn silu_f32(&self, x: DevicePtr, y: DevicePtr, n: usize) -> Result<()> {
        super::mlp::silu_f32(self.reg, self.stream, x, y, n)
    }

    fn swiglu_f32(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::swiglu_f32(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f32_to_f16(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
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

    fn scale_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
        scale: f32,
    ) -> Result<()> {
        super::mlp::scale_f32(self.reg, self.stream, x, y, n, scale)
    }

    fn scale_f16(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
        scale: f32,
    ) -> Result<()> {
        super::mlp::scale_f16(self.reg, self.stream, x, y, n, scale)
    }

    fn add_f16(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::add_f16(self.reg, self.stream, a, b, y, n)
    }

    fn add_f32(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::add_f32(self.reg, self.stream, a, b, y, n)
    }

    fn swiglu_f16(
        &self,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::swiglu_f16(self.reg, self.stream, gate, up, y, n)
    }

    fn sigmoid_mul_f16(
        &self,
        gate: DevicePtr,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::sigmoid_mul_f16(self.reg, self.stream, gate, x, y, n)
    }

    // -- pe --

    fn rope_f16(
        &self,
        x: DevicePtr,
        positions: DevicePtr,
        theta_base: f32,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        super::pe::rope_f16(
            self.reg, self.stream, x, positions, theta_base, n_tokens, n_heads, head_dim,
        )
    }

    fn rope_neox_partial_f16(
        &self,
        x: DevicePtr,
        positions: DevicePtr,
        theta_base: f32,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        rotated_dims: usize,
    ) -> Result<()> {
        super::pe::rope_neox_partial_f16(
            self.reg, self.stream, x, positions, theta_base, n_tokens, n_heads, head_dim,
            rotated_dims,
        )
    }

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
    ) -> Result<()> {
        super::pe::rmsnorm_rope_neox_partial_f16(
            self.reg, self.stream, x, norm_w, positions, theta_base, eps, n_tokens, n_heads,
            head_dim, rotated_dims,
        )
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
        logits: DevicePtr,
        token_counts: DevicePtr,
        n_pairs: usize,
        vocab: usize,
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> Result<()> {
        super::sampling::apply_penalties_f32(
            self.reg, self.stream, logits, token_counts, n_pairs, vocab, repetition_penalty,
            presence_penalty, frequency_penalty,
        )
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
            self.reg, self.stream, logits, out_ids, out_probs, vocab, k, inv_temp,
        )
    }

    // -- recurrent --

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
    ) -> Result<()> {
        super::recurrent::gdn_state_step_f32_s128(
            self.reg, self.stream, q, k, v, gate, beta, state_in, state_out, attn_out, b, h_v, l,
            n_rep, rep_inner_layout,
        )
    }

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
    ) -> Result<()> {
        super::recurrent::gdn_alpha_beta_f32(
            self.reg, self.stream, alpha_in, beta_in, ssm_dt_bias, ssm_a, gate_out, beta_out,
            num_v_heads, n_tokens,
        )
    }

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
    ) -> Result<()> {
        super::recurrent::gdn_state_step_alphabeta_f32_s128(
            self.reg, self.stream, q, k, v, alpha_in, beta_in, ssm_dt_bias, ssm_a, state_in,
            state_out, attn_out, b, h_v, l, n_rep, rep_inner_layout,
        )
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
            self.reg, self.stream, history, current, conv_input, conv_channels, conv_kernel,
        )
    }

    fn gdn_split_qkv_f32(
        &self,
        silu_out: DevicePtr,
        q_out: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        n_tokens: usize,
        qk_size: usize,
        v_size: usize,
    ) -> Result<()> {
        super::recurrent::gdn_split_qkv_f32(
            self.reg, self.stream, silu_out, q_out, k_out, v_out, n_tokens, qk_size, v_size,
        )
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
            self.reg, self.stream, w, x, y, n_rows, k, n_tokens,
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
            self.reg, self.stream, w, x, y, n_rows, k, n_tokens,
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
            self.reg, self.stream, conv_input, weight, y, n_new, conv_channels, conv_kernel,
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
            self.reg, self.stream, logits, idx, weights, n_tokens, n_experts, k,
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
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        dst: DevicePtr,
        n_rows: usize,
        n_tokens: usize,
        top_k: usize,
        n_sb_per_row: usize,
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_r2(
            self.reg, self.stream, w, y, expert_ids, dst, n_rows, n_tokens, top_k, n_sb_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q6_k(
            self.reg, self.stream, w, y, expert_ids, dst, n_rows, n_tokens, top_k, n_sb_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q5_k(
            self.reg, self.stream, w, y, expert_ids, dst, n_rows, n_tokens, top_k, n_sb_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_0(
            self.reg, self.stream, w, y, expert_ids, dst, n_rows, n_tokens, top_k,
            n_blocks_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_1(
            self.reg, self.stream, w, y, expert_ids, dst, n_rows, n_tokens, top_k,
            n_blocks_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_0_gate_up(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, gate_out, up_out, n_rows, n_tokens,
            top_k, n_blocks_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q8_0(
            self.reg, self.stream, w, y, expert_ids, dst, n_rows, n_tokens, top_k,
            n_blocks_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q8_0_gate_up(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, gate_out, up_out, n_rows,
            n_tokens, top_k, n_blocks_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_r2_sorted(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx, dst, n_rows, n_tokens, top_k,
            n_sb_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, sorted_pair_idx_padded,
            padded_offsets, gate_out, up_out, shape,
        )
    }

    fn indexed_moe_mmq_q4_k_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_down_tile8(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst,
            shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_0_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, sorted_pair_idx_padded,
            padded_offsets, gate_out, up_out, shape,
        )
    }

    fn indexed_moe_mmq_q4_0_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_0_down_tile8(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst,
            shape,
        )
    }

    fn indexed_moe_mmq_q4_1_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_1_down_tile8(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst,
            shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q8_0_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, sorted_pair_idx_padded,
            padded_offsets, gate_out, up_out, shape,
        )
    }

    fn indexed_moe_mmq_q8_0_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q8_0_down_tile8(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst,
            shape,
        )
    }

    // ---- IQ tile8 impls. All follow the same shape: dispatch to the
    // matching free function in `super::moe`.

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_xs_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq4_xs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_xs_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_nl_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq4_nl_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq4_nl_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_xxs_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq3_xxs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_xxs_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_s_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq3_s_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq3_s_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xxs_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq2_xxs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xxs_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xs_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq2_xs_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_xs_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_s_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq2_s_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq2_s_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_s_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq1_s_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_s_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_m_gate_up_tile8(
            self.reg, self.stream, w_gate, w_up, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, gate_out, up_out, shape,
        )
    }
    fn indexed_moe_mmq_iq1_m_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_iq1_m_down_tile8(
            self.reg, self.stream, w, y, expert_ids,
            sorted_pair_idx_padded, padded_offsets, dst, shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_gate_up_turbo(
            self.reg, self.stream, gate_w, up_w, y_mmq, expert_ids, sorted_pair_idx_padded,
            padded_offsets, gate_out, up_out, shape,
        )
    }

    fn indexed_moe_mmq_q4_k_down_turbo(
        &self,
        down_w: DevicePtr,
        y_mmq: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k_down_turbo(
            self.reg, self.stream, down_w, y_mmq, expert_ids, sorted_pair_idx_padded,
            padded_offsets, dst, shape,
        )
    }

    fn indexed_moe_mmq_q5_k_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q5_k_down_tile8(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst,
            shape,
        )
    }

    fn indexed_moe_mmq_q6_k_down_tile8(
        &self,
        w: DevicePtr,
        y: DevicePtr,
        expert_ids: DevicePtr,
        sorted_pair_idx_padded: DevicePtr,
        padded_offsets: DevicePtr,
        dst: DevicePtr,
        shape: MoeShape,
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q6_k_down_tile8(
            self.reg, self.stream, w, y, expert_ids, sorted_pair_idx_padded, padded_offsets, dst,
            shape,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_gate_up_sorted(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, sorted_pair_idx, gate_out, up_out,
            n_rows, n_tokens, top_k, n_sb_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmvq_q4_k_gate_up(
            self.reg, self.stream, w_gate, w_up, y, expert_ids, gate_out, up_out, n_rows, n_tokens,
            top_k, n_sb_per_row,
        )
    }

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
    ) -> Result<()> {
        super::moe::indexed_moe_mmq_q4_k(
            self.reg, self.stream, w, y, bucket_expert, bucket_slots, dst, n_rows, n_sb_per_row,
            top_k, n_buckets,
        )
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
            self.reg, self.stream, shared_out, x, gate_w, n_tokens, hidden,
        )
    }

    fn moe_combine_f16(
        &self,
        expert_outs: DevicePtr,
        weights: DevicePtr,
        residual: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        super::moe::moe_combine_f16(
            self.reg, self.stream, expert_outs, weights, residual, out, n_tokens, top_k, hidden,
        )
    }

    fn moe_combine_no_residual_f16(
        &self,
        expert_outs: DevicePtr,
        weights: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        super::moe::moe_combine_no_residual_f16(
            self.reg, self.stream, expert_outs, weights, out, n_tokens, top_k, hidden,
        )
    }

    fn moe_combine_no_residual_f32(
        &self,
        expert_outs: DevicePtr,
        weights: DevicePtr,
        out: DevicePtr,
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        super::moe::moe_combine_no_residual_f32(
            self.reg, self.stream, expert_outs, weights, out, n_tokens, top_k, hidden,
        )
    }

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
    ) -> Result<()> {
        super::moe::moe_combine_two_residuals_f16(
            self.reg, self.stream, expert_outs, weights, residual1, residual2, out, n_tokens,
            top_k, hidden,
        )
    }

    fn moe_sort_by_expert(
        &self,
        expert_ids: DevicePtr,
        counts: DevicePtr,
        offsets: DevicePtr,
        cursors: DevicePtr,
        sorted_pair_idx: DevicePtr,
        total: usize,
        n_experts: usize,
    ) -> Result<()> {
        super::moe::moe_sort_by_expert(
            self.reg, self.stream, expert_ids, counts, offsets, cursors, sorted_pair_idx, total,
            n_experts,
        )
    }

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
    ) -> Result<()> {
        super::moe::moe_sort_by_expert_padded_16(
            self.reg, self.stream, expert_ids, counts, offsets, cursors, sorted_pair_idx,
            padded_offsets, sorted_pair_idx_padded, total, n_experts, max_tokens, top_k,
        )
    }

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
    ) -> Result<()> {
        super::moe::moe_sort_by_expert_padded(
            self.reg, self.stream, expert_ids, counts, offsets, cursors, sorted_pair_idx,
            padded_offsets, sorted_pair_idx_padded, total, n_experts, max_tokens, top_k,
        )
    }

    fn apply_softcap_f32(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        n: usize,
        cap: f32,
    ) -> Result<()> {
        super::softcap::apply_softcap_f32(self.reg, self.stream, x, y, n, cap)
    }

    fn gelu_f32_to_f16(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::gelu_f32_to_f16(self.reg, self.stream, a, b, y, n)
    }

    fn gelu_mul_f32(
        &self,
        a: DevicePtr,
        b: DevicePtr,
        y: DevicePtr,
        n: usize,
    ) -> Result<()> {
        super::mlp::gelu_mul_f32(self.reg, self.stream, a, b, y, n)
    }
}
