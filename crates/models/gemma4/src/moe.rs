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
//! Gating gap: the gemma4 expert gating function is
//! `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (softmax over all experts
//! → take top-k → keep softmax-of-all probs, no renorm). The
//! `blocks::MoeExperts` route_decode currently dispatches
//! `RouterNormalize::Softmax` to an error; this composer uses
//! `RouterNormalize::TopkRenorm` (softmax-of-topk) until the
//! dedicated kernel ships. Real-weights bit-exact parity vs
//! llama.cpp on 26B-A4B is blocked on closing this gap; output is
//! finite + plausible without it.

#![cfg(feature = "hip")]

use anyhow::{anyhow, Context, Result};
use flambeau_blocks::{MoeExperts, MoeExpertsDecodeScratch, RawAllocTracker};
use flambeau_core::DevicePtr;
use flambeau_ops::hip::HipDevice;
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
    /// F32 [hidden] — post-shared-MLP RMSNorm weight. F32 (not F16)
    /// to keep the post-norm cascade in F32 — mirror of the TP path
    /// (see `feedback_gemma4_moe_f16_overflow`).
    pub post_ffw_norm_1_f32: DevicePtr,
    /// F32 [hidden] — post-MoE-branch RMSNorm weight. Same F32-cascade
    /// rationale as `post_ffw_norm_1_f32`.
    pub post_ffw_norm_2_f32: DevicePtr,
    /// F32 [hidden] — gemma4 `post_ffw_norm` (final). Duplicated
    /// alongside the F16 copy on `Gemma4LayerWeights::post_ffw_norm`
    /// (used by the dense composer); MoE-mode reads this F32 variant.
    pub post_ffw_norm_f32: DevicePtr,
    /// F32 [n_experts] — gemma4 `ffn_down_exps.scale`. Per-expert
    /// scalar applied post down-projection; folded into the routing
    /// weights before combine. Same shape as TP path.
    pub ffn_down_exps_scale_f32: DevicePtr,
}

/// Per-call MoE composer scratch (in addition to the layer scratch
/// already passed to `forward_layer_decode`). Owned by the caller.
/// All cascade buffers are F32 — mirrors the TP path's F32 cascade.
pub struct Gemma4MoeScratch {
    /// F16 [hidden] — router input (`rmsnorm_f16(attn_residual,
    /// pre_router_weight)`).
    pub router_input_f16: DevicePtr,
    /// F16 [hidden] — pre-MoE-branch rmsnorm output, fed as F16 input
    /// to the MoE forward gate/up matmul.
    pub cur_moe_input_f16: DevicePtr,
    /// F32 [hidden] — shared-MLP row-parallel partial (F32 throughout,
    /// no F16 cast at the down output).
    pub partial_shared_mlp_f32: DevicePtr,
    /// F32 [hidden] — routed-MoE partial (F32 combine output).
    pub partial_moe_f32: DevicePtr,
    /// F32 [hidden] — `rmsnorm_f32(partial_shared_mlp_f32, post_ffw_norm_1_f32)`.
    pub cur_mlp_f32: DevicePtr,
    /// F32 [hidden] — `rmsnorm_f32(partial_moe_f32, post_ffw_norm_2_f32)`.
    pub cur_moe_f32: DevicePtr,
    /// F32 [hidden] — `cur_mlp_f32 + cur_moe_f32`.
    pub cur_combined_f32: DevicePtr,
    /// F32 [hidden] — `rmsnorm_f32(cur_combined_f32, post_ffw_norm_f32)`
    /// before the F16 cast + residual add.
    pub tmp_f32: DevicePtr,
    /// blocks::MoeExperts decode scratch.
    pub moe_scratch: MoeExpertsDecodeScratch,
}

impl Gemma4MoeScratch {
    /// Allocate per-rank scratch on `device`. Sizes mirror the TP
    /// composer's `Gemma4TpMoeScratch::alloc` but with full
    /// `n_ff_exp` (no per-rank slicing).
    pub fn alloc(
        device: &HipDevice,
        hidden: usize,
        n_ff_exp: usize,
        n_experts: usize,
        top_k: usize,
        raw_alloc: &mut RawAllocTracker,
    ) -> Result<Self> {
        if hidden % 32 != 0 {
            return Err(anyhow!(
                "Gemma4MoeScratch::alloc: hidden={hidden} not a multiple of 32"
            ));
        }
        let inter_total = top_k * n_ff_exp;
        if inter_total % 32 != 0 {
            return Err(anyhow!(
                "Gemma4MoeScratch::alloc: top_k*n_ff_exp={inter_total} not a multiple of 32"
            ));
        }
        let router_input_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
        let cur_moe_input_f16 = raw_alloc.alloc_f16(device, hidden)?.0;
        let partial_shared_mlp_f32 = raw_alloc.alloc_f32(device, hidden)?.0;
        let partial_moe_f32 = raw_alloc.alloc_f32(device, hidden)?.0;
        let cur_mlp_f32 = raw_alloc.alloc_f32(device, hidden)?.0;
        let cur_moe_f32 = raw_alloc.alloc_f32(device, hidden)?.0;
        let cur_combined_f32 = raw_alloc.alloc_f32(device, hidden)?.0;
        let tmp_f32 = raw_alloc.alloc_f32(device, hidden)?.0;
        let moe_scratch = MoeExpertsDecodeScratch {
            x_q8_1: raw_alloc.alloc_q8_1(device, hidden)?.0,
            router_logits: raw_alloc.alloc_f32(device, n_experts)?.0,
            expert_ids: raw_alloc.alloc_i32(device, top_k)?.0,
            expert_weights: raw_alloc.alloc_f32(device, top_k)?.0,
            gate_out_f32: raw_alloc.alloc_f32(device, inter_total)?.0,
            up_out_f32: raw_alloc.alloc_f32(device, inter_total)?.0,
            activated_f16: raw_alloc.alloc_f16(device, inter_total)?.0,
            activated_q8_1: raw_alloc.alloc_q8_1(device, inter_total)?.0,
            down_f32: raw_alloc.alloc_f32(device, top_k * hidden)?.0,
            down_f16: raw_alloc.alloc_f16(device, top_k * hidden)?.0,
        };
        Ok(Self {
            router_input_f16,
            cur_moe_input_f16,
            partial_shared_mlp_f32,
            partial_moe_f32,
            cur_mlp_f32,
            cur_moe_f32,
            cur_combined_f32,
            tmp_f32,
            moe_scratch,
        })
    }
}

/// Run the gemma4 MoE FFN for one decode token (single-device / PP).
///
/// F32-cascade pipeline (mirror of the TP composer; see
/// `feedback_gemma4_moe_f16_overflow` +
/// `feedback_gemma4_attn_output_proj_f16_saturate`):
/// ```text
/// router_input = rmsnorm_f16(attn_residual, pre_router_weight)
/// {ids, w} = MoE::route_decode(router_input); w *= ffn_down_exps_scale
/// // Shared MLP (F32 partial throughout)
/// rmsnorm_quant_q8_1(attn_residual, ffn_norm) → gate/up/GELU/down →
///   partial_shared_mlp_f32   (no F16 cast at the end)
/// cur_mlp_f32 = rmsnorm_f32(partial_shared_mlp_f32, post_ffw_norm_1_f32)
/// // Routed MoE (F32 partial throughout)
/// cur_moe_input_f16 = rmsnorm_f16(attn_residual, pre_ffw_norm_2)
/// partial_moe_f32   = MoE::forward_decode_tp_f32(cur_moe_input_f16)
/// cur_moe_f32       = rmsnorm_f32(partial_moe_f32, post_ffw_norm_2_f32)
/// // Combine + final post-norm + residual
/// cur_combined_f32  = cur_mlp_f32 + cur_moe_f32
/// tmp_f32           = rmsnorm_f32(cur_combined_f32, post_ffw_norm_f32)
/// x_out = attn_residual + cast_f32_to_f16(tmp_f32)   (F16 residual stream)
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
    _layer_scratch_down_f32: DevicePtr,
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

    // 2. Router top-k + per-expert scale fold.
    moe.moe
        .route_decode(ops, moe_scratch.router_input_f16, moe_scratch.moe_scratch)
        .context("MoE route_decode")?;
    ops.apply_per_expert_scale_f32(
        moe_scratch.moe_scratch.expert_weights,
        moe_scratch.moe_scratch.expert_ids,
        moe.ffn_down_exps_scale_f32,
        1,
        moe.moe.top_k,
    )
    .context("MoE apply ffn_down_exps.scale")?;

    // 3. Shared MLP branch — F32 down output directly into
    //    partial_shared_mlp_f32 (no F16 cast).
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
        moe_scratch.partial_shared_mlp_f32,
        hidden,
        ff_len,
        layer.ffn_down.dtype,
    )
    .context("MoE shared mmvq down → partial_shared_mlp_f32")?;
    ops.rmsnorm_f32(
        moe_scratch.partial_shared_mlp_f32,
        moe.post_ffw_norm_1_f32,
        moe_scratch.cur_mlp_f32,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE shared post_ffw_norm_1 (F32)")?;

    // 4. Routed MoE branch — F32 partial via forward_decode_tp_f32
    //    (reusable on single-device when intermediate is the full
    //    n_ff_exp, no per-rank slicing).
    ops.rmsnorm_f16(
        attn_residual,
        moe.pre_ffw_norm_2,
        moe_scratch.cur_moe_input_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE branch pre_ffw_norm_2")?;
    moe.moe
        .forward_decode_tp_f32(
            ops,
            moe_scratch.cur_moe_input_f16,
            moe_scratch.partial_moe_f32,
            moe_scratch.moe_scratch,
        )
        .context("MoE forward_decode_tp_f32")?;
    ops.rmsnorm_f32(
        moe_scratch.partial_moe_f32,
        moe.post_ffw_norm_2_f32,
        moe_scratch.cur_moe_f32,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE branch post_ffw_norm_2 (F32)")?;

    // 5. Combine: cur_combined_f32 = cur_mlp_f32 + cur_moe_f32.
    ops.add_f32(
        moe_scratch.cur_mlp_f32,
        moe_scratch.cur_moe_f32,
        moe_scratch.cur_combined_f32,
        hidden,
    )
    .context("MoE combine cur_mlp + cur_moe (F32)")?;

    // 6. Final post_ffw_norm (F32) + cast to F16 + residual add.
    ops.rmsnorm_f32(
        moe_scratch.cur_combined_f32,
        moe.post_ffw_norm_f32,
        moe_scratch.tmp_f32,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("MoE post_ffw_norm (F32)")?;
    // Cast into x_out (F16) then add the residual in place. Saves a
    // dedicated F16 cast scratch — x_out is the layer output buffer.
    ops.cast_f32_to_f16(moe_scratch.tmp_f32, x_out, hidden)
        .context("MoE final cast F32→F16")?;
    ops.add_f16(attn_residual, x_out, x_out, hidden)
        .context("MoE final residual add")?;

    Ok(())
}
