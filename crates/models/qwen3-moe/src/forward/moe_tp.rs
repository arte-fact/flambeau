//! tensor-parallel `forward_moe_ffn_decode`.
//! Per-rank decode of one MoE FFN layer. Mirrors
//! [`super::moe::forward_moe_ffn_decode`] but operates on TP-sliced
//! expert weights:
//! - `ffn_gate_exps` / `ffn_up_exps` are `ColParallel{dim=1}` →
//! per-rank shape `[n_experts, local_inter, hidden]`.
//! - `ffn_down_exps` is `RowParallel{dim=2}` → per-rank shape
//! `[n_experts, hidden, local_inter]`.
//! Output writes the per-rank `Σ_k weights[k] * down_local[k, :]` into
//! `partial_ffn_out`. The layer driver schedules `BarP2pAllReduce`
//! immediately after to fold the rank-local partials into `hidden`
//! together with the residual.
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
    cast::{cast_f16_to_f32, cast_f32_to_f16},
    moe::{
        indexed_moe_mmq_q4_0_down_tile8, indexed_moe_mmq_q4_0_gate_up_tile8,
        indexed_moe_mmq_q4_1_down_tile8, indexed_moe_mmq_q4_k_down_tile8,
        indexed_moe_mmq_q4_k_gate_up_tile8, indexed_moe_mmq_q8_0_down_tile8,
        indexed_moe_mmq_q8_0_gate_up_tile8, moe_combine_no_residual_f16,
        moe_sort_by_expert_padded, shared_expert_scale_f32, MoeShape,
    },
    norm::quantize_f16_q8_1,
    HipStream, OpsRegistry,
};
use flambeau_quant::GgmlDType;

use super::common::{run_indexed_moe_down, run_indexed_moe_gate_up, validate_moe_dtypes};
use super::moe::{MoePrefillScratch, MoeScratch, SharedExpertPrefillScratch, SharedExpertScratch};
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

/// Per-rank decode for one MoE FFN layer.
/// `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` are TP-sliced per
/// the layout table. `expert_ids` and `expert_weights` come from
/// the (replicated) router output — same on every rank since the
/// router runs replicated.
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
    // replicated across ranks → every rank produces the same x_q8_1.
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

    // 4+5. fused SwiGLU + Q8_1 quantise.
    let n_total = top_k * local_inter;
    let fuse_swiglu_quant = n_total % 32 == 0;
    if fuse_swiglu_quant {
        flambeau_ops::hip::mlp::swiglu_f32_to_q8_1(
            ops,
            stream,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            scratch.activated_q8_1,
            n_total,
        )
        .context("moe (TP) swiglu_f32_to_q8_1")?;
    } else {
        flambeau_ops::hip::mlp::swiglu_f32_to_f16(
            ops,
            stream,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            scratch.activated_f16,
            n_total,
        )
        .context("moe (TP) swiglu_f32_to_f16")?;
        flambeau_ops::hip::norm::quantize_f16_q8_1(
            ops,
            stream,
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_total,
        )
        .context("moe (TP) quantize activated → Q8_1")?;
    }

    // 6. Per-rank down matmul on RowParallel-sliced ffn_down_exps.
    // Output is [top_k, hidden] full-H rows where each row is THIS
    // rank's contribution to the down projection of expert `k`.
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
    // The residual stream is folded by the AllReduce kernel that
    // follows, not here.
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

/// per-rank decode for the shared expert.
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
    // qwen3next gates the shared expert by sigmoid(x_norm·w);
    // qwen35moe omits this projection (None). Replicated weight, applied
    // pre-AR (linear in the partial down output, so AR commutes with the
    // per-token scalar gate).
    ffn_gate_inp_shexp: Option<&DeviceTensor>,
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

    let fuse_gate_up = ffn_gate_shexp.dtype == flambeau_quant::GgmlDType::Q8_0
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

    // fused swiglu + Q8_1 quantise (TP shared expert).
    let fuse_shexp_swiglu_quant = local_inter % 32 == 0;
    if fuse_shexp_swiglu_quant {
        flambeau_ops::hip::mlp::swiglu_f32_to_q8_1(
            ops,
            stream,
            scratch.gate_f32,
            scratch.up_f32,
            scratch.activated_q8_1,
            local_inter,
        )
        .context("shexp (TP) swiglu_f32_to_q8_1")?;
    } else {
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
    }

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

    // apply qwen3next's per-token sigmoid gate to the partial
    // down before AR. Linear in `down`, so commutes with the cross-rank
    // sum: σ(x·w) · sum_r partial_r = sum_r σ(x·w) · partial_r.
    if let Some(gate_w) = ffn_gate_inp_shexp {
        cast_f16_to_f32(ops, stream, x_norm, scratch.x_norm_f32, hidden)
            .context("shexp (TP) cast x_norm → f32 (gate)")?;
        shared_expert_scale_f32(
            ops, stream, scratch.down_f32, scratch.x_norm_f32, gate_w.ptr, 1, hidden,
        )
        .context("shexp (TP) shared_expert_scale_f32")?;
    }

    cast_f32_to_f16(ops, stream, scratch.down_f32, shared_delta_out, hidden)
        .context("shexp (TP) cast down → f16")?;

    Ok(())
}

/// 2** — L-batched per-rank MoE FFN prefill.
/// Sister of [`forward_moe_ffn_decode_tp`] (M=L instead of M=1). Same
/// per-rank ColParallel (gate/up) + RowParallel (down) sharding; every
/// kernel call is parametrised by `n_tokens` so a single sweep handles
/// the whole prompt. Output is `partial_ffn_out[L, hidden]` —
/// per-rank Σ_k weights[k] * down_local[k, :], no residual. Caller AR's
/// after this layer.
/// Like decode_tp, this is the bandwidth-stable indexed-MoE MMVQ path:
/// no sort+pad MMQ tile8. The PP non-TP `forward_moe_ffn_prefill`
/// auto-routes to tile8 at `n_tokens >= 32`; the TP MMVQ-per-token
/// fallback is correct at any L. tile8 + sort-by-expert TP integration
/// is a V2.x perf lever (would require per-rank sort scratch +
/// expert-id replication invariant).
/// `n_tokens` ≤ `scratch.max_tokens`; caller chunks larger prompts.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_moe_ffn_decode_tp + non-TP forward_moe_ffn_prefill arg shapes"
)]
pub fn forward_moe_ffn_prefill_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_exps: &DeviceTensor,
    ffn_up_exps: &DeviceTensor,
    ffn_down_exps: &DeviceTensor,
    scratch: &mut MoePrefillScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    if n_tokens == 0 {
        bail!("forward_moe_ffn_prefill_tp called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_moe_ffn_prefill_tp: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    if inter % world != 0 {
        bail!("moe_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;
    let top_k = cfg.num_experts_per_tok;

    // 1. Quantise x_norm[L, hidden] → Q8_1.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("moe (TP) prefill x_norm → Q8_1")?;

    // 2. Validate dtypes (same kernel set as decode TP).
    validate_moe_dtypes(
        "indexed-MoE (TP) prefill",
        ffn_gate_exps.dtype,
        ffn_up_exps.dtype,
        ffn_down_exps.dtype,
        hidden,
        local_inter,
    )?;

    // TP tile8 fast path.
    // The non-TP `forward_moe_ffn_prefill` routes Q4_0 / Q8_0 gate/up at
    // `n_tokens >= 32` through the sort+pad MMQ tile8 kernels (2.b /
    // 8.c). The TP path historically fell through to per-token MMVQ
    // because (a) the sort scratch wasn't reachable and (b) Q4_1 down
    // had no tile8 sibling; HipEvent profile attributed
    // 84 % of pp2tp2 prefill wall to this section on Coder-Next.
    // Both gates clear now: `MoePrefillScratch` already carries the
    // sort+pad buffers (used by the non-TP path), and 
    // landed `indexed_moe_mmq_q4_1_down_tile8_dp4a`. The TP layout is
    // intra-expert (every rank holds all `n_experts` with `local_inter`
    // sliced), so the expert-id table + sort layout are identical to
    // the non-TP single-rank case — sort + tile8 just need `local_inter`
    // wired through `MoeShape::n_rows`.
    let n_experts = cfg.num_experts;
    let total_pairs = n_tokens * top_k;
    const TP_TILE8_THRESHOLD: usize = 32;
    let tile8_dt_ok = matches!(
        (ffn_gate_exps.dtype, ffn_down_exps.dtype),
        (GgmlDType::Q4_0, GgmlDType::Q4_0)
            | (GgmlDType::Q4_0, GgmlDType::Q8_0)
            | (GgmlDType::Q4_0, GgmlDType::Q4_1)
            | (GgmlDType::Q8_0, GgmlDType::Q8_0)
            | (GgmlDType::Q4K, GgmlDType::Q4K),
    );
    let gate_block_size = ffn_gate_exps.dtype.block_size();
    let down_block_size = ffn_down_exps.dtype.block_size();
    if tile8_dt_ok && n_tokens >= TP_TILE8_THRESHOLD {
        moe_sort_by_expert_padded(
            ops,
            stream,
            scratch.expert_ids,
            scratch.sort_counts,
            scratch.sort_offsets,
            scratch.sort_cursors,
            scratch.sort_sorted_pair_idx,
            scratch.sort_padded_offsets,
            scratch.sort_sorted_pair_idx_padded,
            total_pairs,
            n_experts,
            scratch.max_tokens,
            top_k,
        )
        .context("moe (TP) prefill moe_sort_by_expert_padded")?;
        let padded_total_ub = total_pairs + n_experts * 8;

        let gate_up_shape = MoeShape {
            n_rows: local_inter,
            n_tokens,
            top_k,
            n_sb_per_row: hidden / gate_block_size,
            n_experts,
            padded_total_upper_bound: padded_total_ub,
        };
        match ffn_gate_exps.dtype {
            GgmlDType::Q4_0 => indexed_moe_mmq_q4_0_gate_up_tile8(
                ops,
                stream,
                ffn_gate_exps.ptr,
                ffn_up_exps.ptr,
                scratch.x_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.gate_out_f32,
                scratch.up_out_f32,
                gate_up_shape,
            )
            .context("moe (TP) prefill gate+up q4_0 tile8")?,
            GgmlDType::Q8_0 => indexed_moe_mmq_q8_0_gate_up_tile8(
                ops,
                stream,
                ffn_gate_exps.ptr,
                ffn_up_exps.ptr,
                scratch.x_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.gate_out_f32,
                scratch.up_out_f32,
                gate_up_shape,
            )
            .context("moe (TP) prefill gate+up q8_0 tile8")?,
            GgmlDType::Q4K => indexed_moe_mmq_q4_k_gate_up_tile8(
                ops,
                stream,
                ffn_gate_exps.ptr,
                ffn_up_exps.ptr,
                scratch.x_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.gate_out_f32,
                scratch.up_out_f32,
                gate_up_shape,
            )
            .context("moe (TP) prefill gate+up q4_k tile8")?,
            _ => unreachable!("tile8_dt_ok already filtered gate dtype"),
        }

        flambeau_ops::hip::mlp::swiglu_f32_to_f16(
            ops,
            stream,
            scratch.gate_out_f32,
            scratch.up_out_f32,
            scratch.activated_f16,
            n_tokens * top_k * local_inter,
        )
        .context("moe (TP) prefill swiglu_f32_to_f16 (tile8)")?;
        quantize_f16_q8_1(
            ops,
            stream,
            scratch.activated_f16,
            scratch.activated_q8_1,
            n_tokens * top_k * local_inter,
        )
        .context("moe (TP) prefill quantize activated → Q8_1 (tile8)")?;

        let down_shape = MoeShape {
            n_rows: hidden,
            n_tokens: total_pairs,
            top_k: 1,
            n_sb_per_row: local_inter / down_block_size,
            n_experts,
            padded_total_upper_bound: padded_total_ub,
        };
        match ffn_down_exps.dtype {
            GgmlDType::Q4_0 => indexed_moe_mmq_q4_0_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                down_shape,
            )
            .context("moe (TP) prefill down q4_0 tile8")?,
            GgmlDType::Q4_1 => indexed_moe_mmq_q4_1_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                down_shape,
            )
            .context("moe (TP) prefill down q4_1 tile8")?,
            GgmlDType::Q8_0 => indexed_moe_mmq_q8_0_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                down_shape,
            )
            .context("moe (TP) prefill down q8_0 tile8")?,
            GgmlDType::Q4K => indexed_moe_mmq_q4_k_down_tile8(
                ops,
                stream,
                ffn_down_exps.ptr,
                scratch.activated_q8_1,
                scratch.expert_ids,
                scratch.sort_sorted_pair_idx_padded,
                scratch.sort_padded_offsets,
                scratch.down_f32,
                down_shape,
            )
            .context("moe (TP) prefill down q4_k tile8")?,
            _ => unreachable!("tile8_dt_ok already filtered down dtype"),
        }

        cast_f32_to_f16(
            ops,
            stream,
            scratch.down_f32,
            scratch.down_f16,
            n_tokens * top_k * hidden,
        )
        .context("moe (TP) prefill cast down → f16 (tile8)")?;
        moe_combine_no_residual_f16(
            ops,
            stream,
            scratch.down_f16,
            scratch.expert_weights,
            partial_ffn_out,
            n_tokens,
            top_k,
            hidden,
        )
        .context("moe (TP) prefill combine_no_residual_f16 (tile8)")?;
        return Ok(());
    }

    // 3. Fused gate + up matmul on per-rank `local_inter` slabs across L tokens.
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
        n_tokens,
        top_k,
        hidden,
    )?;

    // 4+5. Fused SwiGLU → F16 + Q8_1 quantise over [L, top_k, local_inter].
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops,
        stream,
        scratch.gate_out_f32,
        scratch.up_out_f32,
        scratch.activated_f16,
        n_tokens * top_k * local_inter,
    )
    .context("moe (TP) prefill swiglu_f32_to_f16")?;
    flambeau_ops::hip::norm::quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        n_tokens * top_k * local_inter,
    )
    .context("moe (TP) prefill quantize activated → Q8_1")?;

    // 6. Per-rank down matmul on RowParallel-sliced ffn_down_exps. Each
    // (token, slot) pair is its own effective token (top_k_inner=1) —
    // matches the non-TP prefill's down call shape.
    run_indexed_moe_down(
        ops,
        stream,
        ffn_down_exps.dtype,
        ffn_down_exps.ptr,
        scratch.activated_q8_1,
        scratch.expert_ids,
        scratch.down_f32,
        hidden,
        n_tokens * top_k,
        1,
        local_inter,
    )?;

    // 7. Cast down F32→F16 over [L, top_k, hidden].
    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        scratch.down_f16,
        n_tokens * top_k * hidden,
    )
    .context("moe (TP) prefill cast down → f16")?;

    // 8. Combine WITHOUT residual: partial_ffn_out[L, hidden] = Σ_k w[k]*down[k].
    // Residual stream is folded by the AR that follows.
    moe_combine_no_residual_f16(
        ops,
        stream,
        scratch.down_f16,
        scratch.expert_weights,
        partial_ffn_out,
        n_tokens,
        top_k,
        hidden,
    )
    .context("moe (TP) prefill combine_no_residual_f16")?;

    Ok(())
}

/// 2** — L-batched per-rank shared-expert prefill.
/// Sister of [`forward_shared_expert_decode_tp`] (M=L instead of M=1).
/// Per-rank ColParallel (gate/up) + RowParallel (down) shared-expert
/// FFN over L tokens; output is `shared_delta_out[L, hidden]` (no
/// residual, no scale). The layer driver folds it into
/// `partial_ffn_out` before the post-FFN AR.
/// Note: like decode_tp, this omits the `shared_expert_scale_f32` step
/// the non-TP `forward_shared_expert_prefill` applies. The asymmetry
/// pre-dates (it's how forward_shared_expert_decode_tp was
/// landed at ). Out of scope to fix here.
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_shared_expert_decode_tp shape"
)]
pub fn forward_shared_expert_prefill_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate_shexp: &DeviceTensor,
    ffn_up_shexp: &DeviceTensor,
    ffn_down_shexp: &DeviceTensor,
    // qwen3next gates the shared expert by sigmoid(x_norm·w);
    // qwen35moe omits this projection (None). See decode_tp for the AR
    // commutativity argument.
    ffn_gate_inp_shexp: Option<&DeviceTensor>,
    scratch: &mut SharedExpertPrefillScratch,
    x_norm: DevicePtr,
    shared_delta_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
) -> Result<()> {
    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    if n_tokens == 0 {
        bail!("forward_shared_expert_prefill_tp called with n_tokens = 0");
    }
    if n_tokens > scratch.max_tokens {
        bail!(
            "forward_shared_expert_prefill_tp: n_tokens={n_tokens} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg
        .shared_expert_intermediate_size
        .context("forward_shared_expert_prefill_tp requires cfg.shared_expert_intermediate_size")?;
    if inter % world != 0 {
        bail!("shared_expert_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;

    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("shexp (TP) prefill x_norm → Q8_1")?;

    // Fused gate+up MMVQ has no L>1 variant; for prefill, the per-token
    // dispatch is dropped and we always go through the standard
    // run_qmatmul path which handles n_tokens > 1 natively.
    super::common::run_qmatmul_from_tensor(
        ops,
        stream,
        ffn_gate_shexp,
        scratch.x_q8_1,
        DevicePtr(0),
        scratch.gate_f32,
        n_tokens,
        hidden,
        local_inter,
        "ffn_gate_shexp (TP) prefill",
    )?;
    super::common::run_qmatmul_from_tensor(
        ops,
        stream,
        ffn_up_shexp,
        scratch.x_q8_1,
        DevicePtr(0),
        scratch.up_f32,
        n_tokens,
        hidden,
        local_inter,
        "ffn_up_shexp (TP) prefill",
    )?;

    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f16,
        n_tokens * local_inter,
    )
    .context("shexp (TP) prefill swiglu_f32_to_f16")?;
    flambeau_ops::hip::norm::quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        n_tokens * local_inter,
    )
    .context("shexp (TP) prefill quantize activated → Q8_1")?;

    super::common::run_qmatmul_from_tensor(
        ops,
        stream,
        ffn_down_shexp,
        scratch.activated_q8_1,
        DevicePtr(0),
        scratch.down_f32,
        n_tokens,
        local_inter,
        hidden,
        "ffn_down_shexp (TP) prefill",
    )?;

    // apply qwen3next's per-token sigmoid gate to the partial
    // down before AR. See decode_tp comment for the linearity argument.
    if let Some(gate_w) = ffn_gate_inp_shexp {
        cast_f16_to_f32(ops, stream, x_norm, scratch.x_norm_f32, n_tokens * hidden)
            .context("shexp (TP) prefill cast x_norm → f32 (gate)")?;
        shared_expert_scale_f32(
            ops, stream, scratch.down_f32, scratch.x_norm_f32, gate_w.ptr,
            n_tokens, hidden,
        )
        .context("shexp (TP) prefill shared_expert_scale_f32")?;
    }

    cast_f32_to_f16(
        ops,
        stream,
        scratch.down_f32,
        shared_delta_out,
        n_tokens * hidden,
    )
    .context("shexp (TP) prefill cast down → f16")?;

    Ok(())
}
