//! Per-call scratch views for a Gemma 4 layer decode step. Caller
//! owns the underlying device allocations; the scratch struct hands
//! out a borrowed view per forward call.

#![cfg(feature = "hip")]

use flambeau_core::DevicePtr;

/// Borrowed view of a caller-owned per-layer decode scratch.
///
/// Lifetimes: every `DevicePtr` is a plain `Copy` value pointing at
/// caller-owned device memory; only `positions_host` actually borrows
/// (a 1-element host slice for the position upload).
pub struct LayerDecodeScratch<'a> {
    /// Q8_1 staging for the (pre-attn + pre-FFN) x-quant. Sized for
    /// `hidden / 32` Q8_0 blocks (max(hidden, q_width)).
    pub x_q8_1: DevicePtr,
    /// F32 accumulator for the various MMVQ outputs. Sized for the
    /// largest single matmul output: `max(q_width, kv_width, hidden,
    /// ff_len)`.
    pub mmvq_f32: DevicePtr,
    /// F16 Q post-cast / post-norm / post-RoPE.
    pub q_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    /// F16 attention output (post-attn kernel).
    pub attn_out_f16: DevicePtr,
    /// F16 attention output post-`post_attention_norm`.
    pub post_attn_norm_f16: DevicePtr,
    /// F16 first-residual `attn_residual = x_in + post_attn_norm(attn_out)`.
    pub attn_residual_f16: DevicePtr,
    /// F16 buffer for the FFN pre-norm output.
    pub ffn_norm_f16: DevicePtr,
    /// F32 dense FFN intermediate buffers (gate, up).
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    /// F16 fused `gelu(gate) * up` output.
    pub activated_f16: DevicePtr,
    /// Q8_1 staging of `activated_f16` for `ffn_down` MMVQ.
    pub activated_q8_1: DevicePtr,
    /// F32 FFN-down output, cast to F16 before residual add.
    pub down_f32: DevicePtr,
    /// F16 buffer holding `post_ffw_norm(down_proj_f16)`.
    pub post_ffw_norm_f16: DevicePtr,
    /// Position buffer (1-element i32).
    pub positions: DevicePtr,
    /// Persistent host backing for the position memcpy (lifetime: the
    /// call's bounded synchronize).
    pub positions_host: &'a mut [i32],
    /// Unit-weight `[head_dim]` F16 buffer used for V's unlearned
    /// RMSNorm. Pre-filled with `1.0`. Optional — caller passes `None`
    /// when the layer has its own V proj **and** no V-norm is needed
    /// (gemma4 always normalises V, so callers populate it).
    pub v_ones_f16: DevicePtr,
}

/// Borrowed view of a caller-owned per-layer **prefill** scratch.
/// Buffers are sized for `max_tokens` rows — caller picks `max_tokens`
/// at session-init and chunks longer prompts upstream.
pub struct LayerPrefillScratch<'a> {
    pub max_tokens: usize,
    /// F16 [max_tokens, hidden] — RMSNorm(x_in, attn_norm) output.
    pub x_norm_f16: DevicePtr,
    /// Q8_1 [max_tokens, hidden/32].
    pub x_q8_1: DevicePtr,
    /// Q8_1-MMQ [max_tokens, hidden/32].
    pub x_q8_1_mmq: DevicePtr,
    /// F32 staging for QMatMul outputs. Sized for
    /// `max(q_width, kv_width, hidden, ff_len) * max_tokens`.
    pub mmvq_f32: DevicePtr,
    /// F16 [max_tokens, n_heads * head_dim].
    pub q_f16: DevicePtr,
    /// F16 [max_tokens, n_kv_heads * head_dim].
    pub k_f16: DevicePtr,
    /// F16 [max_tokens, n_kv_heads * head_dim].
    pub v_f16: DevicePtr,
    /// F16 [max_tokens, n_heads * head_dim] (attention output).
    pub attn_out_f16: DevicePtr,
    /// F16 [max_tokens, hidden] — post_attn_norm output.
    pub post_attn_norm_f16: DevicePtr,
    /// F16 [max_tokens, hidden] — attn-side residual stream
    /// (x_in + post_attn_norm(attn_out)).
    pub attn_residual_f16: DevicePtr,
    /// F32 [max_tokens, ff_len] for FFN gate output.
    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    /// F16 [max_tokens, ff_len].
    pub activated_f16: DevicePtr,
    /// Q8_1 [max_tokens, ff_len/32].
    pub activated_q8_1: DevicePtr,
    pub activated_q8_1_mmq: DevicePtr,
    /// F32 [max_tokens, hidden].
    pub down_f32: DevicePtr,
    /// F16 [max_tokens, hidden].
    pub post_ffw_norm_f16: DevicePtr,
    /// i32 [max_tokens] device buffer.
    pub positions: DevicePtr,
    /// Host-side positions vec, used to populate `positions` per call.
    pub positions_host: &'a mut [i32],
    /// Unit-weight `[head_dim_max]` F16 buffer used for V's unlearned
    /// RMSNorm.
    pub v_ones_f16: DevicePtr,
}
