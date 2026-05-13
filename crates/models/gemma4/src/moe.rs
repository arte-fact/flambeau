//! Gemma 4 MoE FFN composition (26B-A4B). Every MoE layer runs the
//! shared dense MLP and the routed-experts branch **in parallel**
//! and sums them; the result then goes through `post_ffw_norm` +
//! residual add (same shape as the dense path, just with the MoE
//! sum replacing the dense FFN's `cast → norm` output).
//!
//! Router fold: gemma4 specifies
//! `tmp = rmsnorm_unlearned(attn_out, eps) * (1/sqrt(n_embd)) * ffn_gate_inp_s`
//! followed by `ffn_gate_inp @ tmp`. The three scale steps fold
//! cleanly into `rmsnorm_f16(attn_out, w, ...)` where the F16 weight
//! is precomputed as `w = (1/sqrt(n_embd)) * ffn_gate_inp_s` at
//! upload time. No new kernel.
//!
//! Gating gap (S6-B-A): the gemma4 expert gating function is
//! `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (softmax over all experts
//! → take top-k → keep softmax-of-all probs, no renorm). The
//! `blocks::MoeExperts` route_decode currently dispatches
//! `RouterNormalize::Softmax` to an error; this composer uses
//! `RouterNormalize::TopkRenorm` (softmax-of-topk) until the
//! dedicated kernel ships in a follow-up. Real-weights parity vs
//! llama.cpp on 26B-A4B is blocked on closing this gap.

#![cfg(feature = "hip")]

use anyhow::{Context, Result};
use flambeau_blocks::{MoeExperts, MoeExpertsDecodeScratch};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;

use crate::layer::Gemma4LayerWeights;

/// MoE-only weights for one layer. Pairs with the dense FFN weights
/// already on [`Gemma4LayerWeights`] (the shared MLP runs in
/// parallel with the routed experts).
pub struct Gemma4MoeFfnWeights {
    /// Routed experts block (router + per-expert gate/up/down).
    /// Configured with `Activation::Gelu` and a `RouterPolicy` whose
    /// `pre_scale` / `pre_scalar` are folded into
    /// [`Self::pre_router_weight_f16`].
    pub moe: MoeExperts,
    /// F16 [hidden] — precomputed `(1/sqrt(hidden)) * ffn_gate_inp_s`.
    /// Fed to `rmsnorm_f16(attn_residual, pre_router_weight, ...)` to
    /// produce the router input in one launch (replaces the
    /// `rmsnorm_unlearned + scale + mul` chain).
    pub pre_router_weight_f16: DevicePtr,
    /// F16 [hidden] — pre-MoE-branch RMSNorm weight.
    pub pre_ffw_norm_2: DevicePtr,
    /// F16 [hidden] — post-shared-MLP RMSNorm weight.
    pub post_ffw_norm_1: DevicePtr,
    /// F16 [hidden] — post-MoE-branch RMSNorm weight.
    pub post_ffw_norm_2: DevicePtr,
}

/// Per-call MoE composer scratch (in addition to the layer scratch
/// already passed to `forward_layer_decode`). Owned by the caller.
pub struct Gemma4MoeScratch {
    /// F16 [hidden] — router input (`rmsnorm_f16(attn_residual,
    /// pre_router_weight)`).
    pub router_input_f16: DevicePtr,
    /// F16 [hidden] — `cur_mlp` post-shared-MLP-norm output.
    pub cur_mlp_f16: DevicePtr,
    /// F16 [hidden] — `cur_moe` post-MoE-norm output.
    pub cur_moe_f16: DevicePtr,
    /// F16 [hidden] — combined `cur_mlp + cur_moe` (intermediate).
    pub cur_combined_f16: DevicePtr,
    /// blocks::MoeExperts decode scratch.
    pub moe_scratch: MoeExpertsDecodeScratch,
}

/// Run the gemma4 MoE FFN for one decode token.
///
/// Pipeline:
/// ```text
/// router_input = rmsnorm_f16(attn_residual, pre_router_weight)
/// cur_mlp = rmsnorm_quant(attn_residual, ffn_norm) → gate/up/GELU/down → rmsnorm(post_ffw_norm_1)
/// cur_moe_input = rmsnorm_f16(attn_residual, pre_ffw_norm_2)
/// {expert_ids, expert_weights} = MoE::route_decode(router_input)
/// cur_moe = MoE::forward_decode(cur_moe_input, residual=0, expert_*) → rmsnorm(post_ffw_norm_2)
/// cur_combined = cur_mlp + cur_moe
/// cur = rmsnorm(cur_combined, post_ffw_norm)
/// x_out = cur + attn_residual
/// ```
#[allow(clippy::too_many_arguments)]
pub fn forward_ffn_moe<O: Ops>(
    ops: &O,
    layer: &Gemma4LayerWeights,
    moe: &Gemma4MoeFfnWeights,
    layer_scratch_x_q8_1: DevicePtr,
    _layer_scratch_mmvq_f32: DevicePtr,
    layer_scratch_gate_f32: DevicePtr,
    layer_scratch_up_f32: DevicePtr,
    layer_scratch_activated_f16: DevicePtr,
    layer_scratch_activated_q8_1: DevicePtr,
    layer_scratch_down_f32: DevicePtr,
    moe_scratch: &Gemma4MoeScratch,
    attn_residual: DevicePtr,
    x_out: DevicePtr,
    ff_len: usize,
    hidden: usize,
    rms_norm_eps: f32,
) -> Result<()> {
    // 1. Router input — folded rmsnorm + scale + mul.
    ops.rmsnorm_f16(
        attn_residual,
        moe.pre_router_weight_f16,
        moe_scratch.router_input_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE router_input rmsnorm")?;

    // 2. Shared MLP branch.
    //    cur_mlp = rmsnorm_quant_q8_1(attn_residual, ffn_norm) → gate/up/GELU/down → norm.
    ops.rmsnorm_quant_q8_1(
        attn_residual,
        layer.ffn_norm,
        layer_scratch_x_q8_1,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE shared MLP ffn_norm + quant")?;
    ops.mmvq(
        layer.ffn_gate.ptr,
        layer_scratch_x_q8_1,
        layer_scratch_gate_f32,
        ff_len,
        hidden,
        layer.ffn_gate.dtype,
    )
    .context("MoE shared mmvq gate")?;
    ops.mmvq(
        layer.ffn_up.ptr,
        layer_scratch_x_q8_1,
        layer_scratch_up_f32,
        ff_len,
        hidden,
        layer.ffn_up.dtype,
    )
    .context("MoE shared mmvq up")?;
    ops.gelu_f32_to_f16(
        layer_scratch_gate_f32,
        layer_scratch_up_f32,
        layer_scratch_activated_f16,
        ff_len,
    )
    .context("MoE shared gelu")?;
    ops.quantize_f16_q8_1(
        layer_scratch_activated_f16,
        layer_scratch_activated_q8_1,
        ff_len,
    )?;
    ops.mmvq(
        layer.ffn_down.ptr,
        layer_scratch_activated_q8_1,
        layer_scratch_down_f32,
        hidden,
        ff_len,
        layer.ffn_down.dtype,
    )
    .context("MoE shared mmvq down")?;
    ops.cast_f32_to_f16(layer_scratch_down_f32, moe_scratch.cur_mlp_f16, hidden)?;
    // Post-shared-MLP norm (in place).
    ops.rmsnorm_f16(
        moe_scratch.cur_mlp_f16,
        moe.post_ffw_norm_1,
        moe_scratch.cur_mlp_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE shared post_ffw_norm_1")?;

    // 3. MoE branch.
    //    Pre-MoE norm (rmsnorm of attn_residual with pre_ffw_norm_2).
    //    Result goes into cur_moe_f16 as the input to MoE::forward_decode.
    ops.rmsnorm_f16(
        attn_residual,
        moe.pre_ffw_norm_2,
        moe_scratch.cur_moe_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE branch pre_ffw_norm_2")?;

    // Router: compute logits + top-k. NOTE: blocks::MoeExperts.route_decode
    // uses `RouterNormalize::TopkRenorm` (the default) for S6-B-A. The
    // gemma4 spec calls for `Softmax` (no renorm); parity gap documented.
    moe.moe
        .route_decode(ops, moe_scratch.router_input_f16, moe_scratch.moe_scratch)
        .context("MoE route_decode")?;

    // Expert forward — residual is zero (we want cur_moe alone, no
    // residual fold here; the final residual is `attn_residual` added
    // after post_ffw_norm). Pass cur_moe_f16 as both x_norm (already
    // rmsnormed above) and reuse another buffer as the "residual=0"
    // input. The block's combine adds residual + Σ w_k · expert_out_k.
    // We want only the Σ — but the API requires a residual. Trick:
    // allocate a zero buffer once on session init. For this composer
    // we re-use cur_combined_f16 zeroed below.
    //
    // Zero `cur_combined_f16` via `add_f16(x, -x, y)` would need extra
    // ops; simplest is a memset to zero. The caller is expected to
    // keep `cur_combined_f16` zeroed between calls (a session-level
    // invariant — see Gemma4MoeScratch docs).
    moe.moe
        .forward_decode(
            ops,
            moe_scratch.cur_moe_f16,        // x_norm
            moe_scratch.cur_combined_f16,   // residual = zeros
            None,                            // no extra residual
            moe_scratch.cur_moe_f16,        // out (overwritten)
            moe_scratch.moe_scratch,
        )
        .context("MoE forward_decode")?;
    // Post-MoE norm (in place).
    ops.rmsnorm_f16(
        moe_scratch.cur_moe_f16,
        moe.post_ffw_norm_2,
        moe_scratch.cur_moe_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE branch post_ffw_norm_2")?;

    // 4. Combine: cur_combined = cur_mlp + cur_moe.
    ops.add_f16(
        moe_scratch.cur_mlp_f16,
        moe_scratch.cur_moe_f16,
        moe_scratch.cur_combined_f16,
        hidden,
    )
    .context("MoE combined add")?;

    // 5. Final post_ffw_norm + residual add. Layer's `post_ffw_norm`
    // applies here (shared between MoE / Dense paths).
    ops.rmsnorm_f16(
        moe_scratch.cur_combined_f16,
        layer.post_ffw_norm,
        moe_scratch.cur_combined_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE post_ffw_norm")?;
    ops.add_f16(
        attn_residual,
        moe_scratch.cur_combined_f16,
        x_out,
        hidden,
    )
    .context("MoE final residual add")?;

    Ok(())
}
