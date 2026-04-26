//! TP-4b-i2 — tensor-parallel `forward_moe_ffn_decode`.
//!
//! Per-rank decode of one MoE FFN layer. Mirrors
//! [`super::moe::forward_moe_ffn_decode`] but operates on TP-sliced
//! expert weights:
//!
//! - `ffn_gate_exps` / `ffn_up_exps` are `ColParallel{dim=1}` →
//!   per-rank shape `[n_experts, local_inter, hidden]`.
//! - `ffn_down_exps` is `RowParallel{dim=2}` → per-rank shape
//!   `[n_experts, hidden, local_inter]`.
//!
//! Output writes the per-rank `Σ_k weights[k] * down_local[k, :]` into
//! `partial_ffn_out`. The layer driver schedules `BarP2pAllReduce`
//! immediately after to fold the rank-local partials into `hidden`
//! together with the residual.
//!
//! Bit-exactness: per the analysis in the cert,
//! `Σ_r Σ_k w[k] * down_local_r[k] = Σ_k w[k] * Σ_r down_local_r[k]`
//! = `Σ_k w[k] * down_full[k]` modulo FP32 reduction order.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition; same rationale as super::moe — every \
              unsafe is a kernel launch over session-scoped buffers."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    moe::moe_combine_no_residual_f16,
    norm::quantize_f16_q8_1,
    HipStream, OpsRegistry,
};

use super::common::{run_indexed_moe_down, run_indexed_moe_gate_up, validate_moe_dtypes};
use super::moe::{MoeScratch, SharedExpertScratch};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

/// Per-rank decode for one MoE FFN layer.
///
/// `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` are TP-sliced per
/// the TP-4b layout table. `expert_ids` and `expert_weights` come from
/// the (replicated) router output — same on every rank since the
/// router runs replicated.
///
/// # Errors
/// - `tp_world == 0` or `moe_intermediate_size % tp_world != 0`.
/// - Dtype validation failures (propagated from `validate_moe_dtypes`).
/// - Underlying op-dispatch / kernel-launch failures.
#[expect(
    clippy::too_many_arguments,
    reason = "matches the PP forward_moe_ffn_decode signature shape"
)]
pub fn forward_moe_ffn_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_exps: &DeviceTensor,
    ffn_up_exps: &DeviceTensor,
    ffn_down_exps: &DeviceTensor,
    scratch: &mut MoeScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    if inter % world != 0 {
        bail!("moe_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;
    let top_k = cfg.num_experts_per_tok;

    // 1. Quantise x_norm → Q8_1. x_norm is the AR'd post-attn hidden,
    //    replicated across ranks → every rank produces the same x_q8_1.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("moe (TP) x_norm → Q8_1")?;

    // 2. Validate dtypes (same kernel set as PP).
    validate_moe_dtypes(
        "indexed-MoE (TP)",
        ffn_gate_exps.dtype,
        ffn_up_exps.dtype,
        ffn_down_exps.dtype,
        hidden,
        local_inter,
    )?;

    // 3. Fused gate + up matmul on per-rank `local_inter` slabs.
    run_indexed_moe_gate_up(
        ops,
        stream,
        ffn_gate_exps.dtype,
        ffn_gate_exps.ptr,
        ffn_up_exps.ptr,
        scratch.x_q8_1,
        scratch.expert_ids,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        local_inter,
        1,
        top_k,
        hidden,
    )?;

    // 4+5. Fused SwiGLU + Q8_1 quantise.
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops,
        stream,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        scratch.activated_f16,
        top_k * local_inter,
    )
    .context("moe (TP) swiglu_f32_to_f16")?;
    flambeau_ops::hip::norm::quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        top_k * local_inter,
    )
    .context("moe (TP) quantize activated → Q8_1")?;

    // 6. Per-rank down matmul on RowParallel-sliced ffn_down_exps.
    //    Output is [top_k, hidden] full-H rows where each row is THIS
    //    rank's contribution to the down projection of expert `k`.
    run_indexed_moe_down(
        ops,
        stream,
        ffn_down_exps.dtype,
        ffn_down_exps.ptr,
        scratch.activated_q8_1,
        scratch.expert_ids,
        scratch.down_f32,
        hidden,
        top_k,
        1,
        local_inter,
    )?;

    // 7. Cast down F32→F16 for the combine kernel.
    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        scratch.down_f16,
        top_k * hidden,
    )
    .context("moe (TP) cast down → f16")?;

    // 8. Combine WITHOUT residual: partial_ffn_out = Σ_k w[k] * down[k].
    //    The residual stream is folded by the AllReduce kernel that
    //    follows, not here.
    moe_combine_no_residual_f16(
        ops,
        stream,
        scratch.down_f16,
        scratch.expert_weights,
        partial_ffn_out,
        1,
        top_k,
        hidden,
    )
    .context("moe (TP) combine_no_residual_f16")?;

    Ok(())
}

/// **TP-4c-i2** — per-rank decode for the shared expert.
///
/// Structurally identical to [`super::dense_ffn_tp::forward_dense_ffn_decode_tp`]
/// except `shared_expert_intermediate_size` replaces `moe_intermediate_size`,
/// and the output goes to `shared_delta_out` (the layer driver folds it into
/// `partial_ffn_out` before the AR via `moe_combine_f16` with `residual =
/// shared_delta`).
#[expect(
    clippy::too_many_arguments,
    reason = "matches dense FFN TP signature shape"
)]
pub fn forward_shared_expert_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_shexp: &DeviceTensor,
    ffn_up_shexp: &DeviceTensor,
    ffn_down_shexp: &DeviceTensor,
    scratch: &mut SharedExpertScratch,
    x_norm: DevicePtr,
    shared_delta_out: DevicePtr,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_decode_tp requires cfg.shared_expert_intermediate_size")?;
    if inter % world != 0 {
        bail!("shared_expert_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;

    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
        .context("shexp (TP) x_norm → Q8_1")?;

    let fuse_gate_up = std::env::var("FLAMBEAU_VARIANT").as_deref() != Ok("baseline")
        && ffn_gate_shexp.dtype == flambeau_quant::GgmlDType::Q8_0
        && ffn_up_shexp.dtype == flambeau_quant::GgmlDType::Q8_0;
    if fuse_gate_up {
        flambeau_ops::hip::qmatmul::mmvq_q8_0_gate_up(
            ops,
            stream,
            ffn_gate_shexp.ptr,
            ffn_up_shexp.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            local_inter,
            local_inter,
            hidden,
        )
        .context("shexp (TP) mmvq_q8_0_gate_up")?;
    } else {
        super::common::run_mmvq_from_tensor(
            ops,
            stream,
            ffn_gate_shexp,
            scratch.x_q8_1,
            scratch.gate_f32,
            local_inter,
            hidden,
            "ffn_gate_shexp (TP)",
        )?;
        super::common::run_mmvq_from_tensor(
            ops,
            stream,
            ffn_up_shexp,
            scratch.x_q8_1,
            scratch.up_f32,
            local_inter,
            hidden,
            "ffn_up_shexp (TP)",
        )?;
    }

    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f16,
        local_inter,
    )
    .context("shexp (TP) swiglu_f32_to_f16")?;
    flambeau_ops::hip::norm::quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        local_inter,
    )
    .context("shexp (TP) quantize activated → Q8_1")?;

    super::common::run_mmvq_from_tensor(
        ops,
        stream,
        ffn_down_shexp,
        scratch.activated_q8_1,
        scratch.down_f32,
        hidden,
        local_inter,
        "ffn_down_shexp (TP)",
    )?;

    cast_f32_to_f16(ops, stream, scratch.down_f32, shared_delta_out, hidden)
        .context("shexp (TP) cast down → f16")?;

    Ok(())
}
