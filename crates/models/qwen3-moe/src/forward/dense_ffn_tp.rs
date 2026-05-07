//! TP-2c — tensor-parallel `forward_dense_ffn_decode`.
//!
//! Sister of [`super::dense_ffn::forward_dense_ffn_decode`] for the
//! TP-sharded forward path. Same 7-op structure, with two changes:
//!
//! 1. `local_inter = cfg.moe_intermediate_size / tp_world` is used
//!    everywhere the PP version uses `inter`. Sliced weights:
//!    - `ffn_gate.weight` ColParallel{dim=0} → `[local_inter, hidden]`
//!    - `ffn_up.weight`   ColParallel{dim=0} → `[local_inter, hidden]`
//!    - `ffn_down.weight` RowParallel{dim=1} → `[hidden, local_inter]`
//!
//! 2. **No residual add.** The PP path computes
//!    `x_out = residual + down(F16)` in one shot via `add_f16`. The TP
//!    path emits *only the per-rank `down` projection* (cast to F16)
//!    into `partial_ffn_out` — caller schedules
//!    `BarP2pAllReduce::residual_tp{2,4}` immediately after to fold the
//!    rank-local partials into `hidden` together with the residual.
//!
//! ## Same trade-offs as TP-2b
//!
//! - `DenseFfnScratch` reused as-is — its `gate_f32` / `up_f32` /
//!   `activated_*` slabs are sized for full `inter`, so they overflow
//!   the per-rank slice into space that's never touched. 4× wasteful
//!   on TP=4; not a correctness issue.
//! - The fused `mmvq_q8_0_gate_up` branch still works under TP when
//!   both gate and up are Q8_0 — slicing only changes per-rank row
//!   counts. Qwen3.5-27B-Q4_1 takes the unfused branch (Q4_1 weights);
//!   models with Q8_0 dense FFN take the fused branch.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition; same rationale as super::dense_ffn — every \
              unsafe is a kernel launch over session-scoped buffers."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::hip::{
    cast::cast_f32_to_f16,
    norm::quantize_f16_q8_1,
    qmatmul::{
        mmvq_q4_0_gate_up_t128,
        mmvq_q4_1_gate_up, mmvq_q8_0_gate_up, qmatmul,
    },
    HipStream, OpsRegistry,
};

use super::common::qdtype_of;
use super::dense_ffn::DenseFfnScratch;
use crate::config::Qwen3MoEConfig;
use crate::weights::DeviceTensor;

/// Per-rank decode for one dense FFN layer.
///
/// `ffn_gate`, `ffn_up`, `ffn_down` are *already sliced* per the
/// TP-1a layout table.
///
/// Output: writes the per-rank `down(SwiGLU(gate, up))` cast to F16
/// into `partial_ffn_out` (length `hidden`). The caller's AR fold
/// resolves both the cross-rank reduction and the residual add into
/// `hidden`.
///
/// # Errors
/// - `tp_world == 0` or `moe_intermediate_size % tp_world != 0`.
/// - Sliced weight shape mismatch.
/// - Underlying op-dispatch / kernel-launch failures.
#[expect(
    clippy::too_many_arguments,
    reason = "matches the PP forward_dense_ffn_decode signature — flat parameter list \
              avoids struct copies on the decode hot path."
)]
pub fn forward_dense_ffn_decode_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate: &DeviceTensor,
    ffn_up: &DeviceTensor,
    ffn_down: &DeviceTensor,
    scratch: &mut DenseFfnScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    tp_world: u32,
    // **TP-perf-c2** — when true, scratch.x_q8_1 is already populated by
    // the upstream fused AR+RMSNorm+Q8_1 kernel and the leading
    // quantize_f16_q8_1 call is skipped. `x_norm` is then unused.
    pre_quantized: bool,
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

    // Shape sanity (sliced weights only).
    let gate_dims = (ffn_gate.dims[0] as usize, ffn_gate.dims[1] as usize);
    let up_dims = (ffn_up.dims[0] as usize, ffn_up.dims[1] as usize);
    let down_dims = (ffn_down.dims[0] as usize, ffn_down.dims[1] as usize);
    if gate_dims != (local_inter, hidden) {
        bail!("ffn_gate (TP) shape {gate_dims:?} != [{local_inter}, {hidden}]");
    }
    if up_dims != (local_inter, hidden) {
        bail!("ffn_up (TP) shape {up_dims:?} != [{local_inter}, {hidden}]");
    }
    if down_dims != (hidden, local_inter) {
        bail!("ffn_down (TP) shape {down_dims:?} != [{hidden}, {local_inter}]");
    }

    // 1. Quantise x_norm → Q8_1 once. x_norm is the AR'd post-attn
    //    hidden state (replicated across ranks, so every rank
    //    quantises the same input — duplicated but correct).
    //    TP-perf-c2: skip when `pre_quantized` because the upstream
    //    fused AR+RMSNorm+Q8_1 already wrote scratch.x_q8_1.
    if !pre_quantized {
        quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, hidden)
            .context("dense ffn (TP) x_norm → Q8_1")?;
    } else {
        let _ = x_norm;
    }

    // 2+3. gate + up matmuls — fused if both Q8_0 or both Q4_0, else unfused.
    let fuse_q8 = ffn_gate.dtype == flambeau_quant::GgmlDType::Q8_0
        && ffn_up.dtype == flambeau_quant::GgmlDType::Q8_0;
    let fuse_q4 = ffn_gate.dtype == flambeau_quant::GgmlDType::Q4_0
        && ffn_up.dtype == flambeau_quant::GgmlDType::Q4_0;
    // C8-i1 — Q4_1 fused gate+up. Sibling of fuse_q4; closes the dense-FFN
    // launch-pair gap on Qwen3.5-9B-Q4_1 / 27B-Q4_1.
    let fuse_q4_1 = ffn_gate.dtype == flambeau_quant::GgmlDType::Q4_1
        && ffn_up.dtype == flambeau_quant::GgmlDType::Q4_1;
    if fuse_q8 {
        mmvq_q8_0_gate_up(
            ops,
            stream,
            ffn_gate.ptr,
            ffn_up.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            local_inter,
            local_inter,
            hidden,
        )
        .context("dense ffn (TP) gate+up fused mmvq_q8_0")?;
    } else if fuse_q4 {
        // Dense FFN gate+up is always symmetric (n_rows_gate == n_rows_up
        // == local_inter), so the shape-aware default picks t128.
        mmvq_q4_0_gate_up_t128(
            ops,
            stream,
            ffn_gate.ptr,
            ffn_up.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            local_inter,
            local_inter,
            hidden,
        )
        .context("dense ffn (TP) gate+up fused mmvq_q4_0_t128")?;
    } else if fuse_q4_1 {
        // C8-i1 — Q4_1 dense FFN gate+up fusion. Symmetric (both rows =
        // local_inter). One launch instead of two, single Q8_1 activation
        // read per block.
        mmvq_q4_1_gate_up(
            ops,
            stream,
            ffn_gate.ptr,
            ffn_up.ptr,
            scratch.x_q8_1,
            scratch.gate_f32,
            scratch.up_f32,
            local_inter,
            local_inter,
            hidden,
        )
        .context("dense ffn (TP) gate+up fused mmvq_q4_1")?;
    } else {
        qmatmul(
            ops,
            stream,
            ffn_gate.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.gate_f32,
            1,
            hidden,
            local_inter,
            qdtype_of(ffn_gate.dtype)?,
        )
        .context("dense ffn (TP) gate qmatmul")?;
        qmatmul(
            ops,
            stream,
            ffn_up.ptr,
            scratch.x_q8_1,
            DevicePtr(0),
            scratch.up_f32,
            1,
            hidden,
            local_inter,
            qdtype_of(ffn_up.dtype)?,
        )
        .context("dense ffn (TP) up qmatmul")?;
    }

    // 4+5. Fused SwiGLU(gate, up) → F16 + quantise on the per-rank slice.
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops,
        stream,
        scratch.gate_f32,
        scratch.up_f32,
        scratch.activated_f16,
        local_inter,
    )
    .context("dense ffn (TP) swiglu_f32_to_f16")?;
    quantize_f16_q8_1(
        ops,
        stream,
        scratch.activated_f16,
        scratch.activated_q8_1,
        local_inter,
    )
    .context("dense ffn (TP) activated → Q8_1")?;

    // 6. Row-parallel down matmul: weight[hidden, local_inter] × activated[local_inter]
    //    → down_f32[hidden]. Decode m=1.
    qmatmul(
        ops,
        stream,
        ffn_down.ptr,
        scratch.activated_q8_1,
        DevicePtr(0),
        scratch.down_f32,
        1,
        local_inter,
        hidden,
        qdtype_of(ffn_down.dtype)?,
    )
    .context("dense ffn (TP) down qmatmul")?;

    // 7. Cast down F32→F16 directly into partial_ffn_out. NO residual
    //    add here — the AR fold (caller's
    //    BarP2pAllReduce::residual_tp{2,4}) sums the 4 rank-local
    //    partials into hidden together with the residual.
    cast_f32_to_f16(ops, stream, scratch.down_f32, partial_ffn_out, hidden)
        .context("dense ffn (TP) cast down → partial_ffn_out")?;

    Ok(())
}

/// **AUTO-6b3** — per-rank L-batched dense FFN prefill. Counterpart of
/// [`forward_dense_ffn_decode_tp`] for n_tokens > 1. Mirrors the
/// kernel shape of [`super::dense_ffn::forward_dense_ffn_prefill`]
/// (the PP version) but emits a `[L, hidden]` partial that the caller
/// folds via one `BarP2pAllReduce::residual_tp{2,4}` call across `L *
/// hidden` elements (instead of L per-token ARs).
///
/// Reuses [`super::dense_ffn::DenseFfnPrefillScratch`] (sized for full
/// `inter` — same waste profile as the decode TP path; slicing only
/// changes per-rank row counts).
#[expect(
    clippy::too_many_arguments,
    reason = "matches forward_dense_ffn_decode_tp's flat parameter list."
)]
pub fn forward_dense_ffn_prefill_tp(
    ops: &OpsRegistry,
    stream: &HipStream,
    cfg: &Qwen3MoEConfig,
    ffn_gate: &DeviceTensor,
    ffn_up: &DeviceTensor,
    ffn_down: &DeviceTensor,
    scratch: &mut super::dense_ffn::DenseFfnPrefillScratch,
    x_norm: DevicePtr,
    partial_ffn_out: DevicePtr,
    n_tokens: usize,
    tp_world: u32,
) -> Result<()> {
    use flambeau_ops::hip::norm::quantize_f16_q8_1_mmq;

    if tp_world == 0 {
        bail!("tp_world must be >= 1");
    }
    if n_tokens == 0 {
        bail!("forward_dense_ffn_prefill_tp called with n_tokens = 0");
    }
    let world = tp_world as usize;
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    if inter % world != 0 {
        bail!("moe_intermediate_size {inter} not divisible by tp_world {tp_world}");
    }
    let local_inter = inter / world;

    let gate_dims = (ffn_gate.dims[0] as usize, ffn_gate.dims[1] as usize);
    let up_dims = (ffn_up.dims[0] as usize, ffn_up.dims[1] as usize);
    let down_dims = (ffn_down.dims[0] as usize, ffn_down.dims[1] as usize);
    if gate_dims != (local_inter, hidden) {
        bail!("ffn_gate (TP prefill) shape {gate_dims:?} != [{local_inter}, {hidden}]");
    }
    if up_dims != (local_inter, hidden) {
        bail!("ffn_up (TP prefill) shape {up_dims:?} != [{local_inter}, {hidden}]");
    }
    if down_dims != (hidden, local_inter) {
        bail!("ffn_down (TP prefill) shape {down_dims:?} != [{hidden}, {local_inter}]");
    }

    // 1. Quantise x_norm to BOTH Q8_1 layouts.
    quantize_f16_q8_1(ops, stream, x_norm, scratch.x_q8_1, n_tokens * hidden)
        .context("dense ffn prefill (TP) x_norm → Q8_1 std")?;
    quantize_f16_q8_1_mmq(ops, stream, x_norm, scratch.x_q8_1_mmq, hidden, n_tokens)
        .context("dense ffn prefill (TP) x_norm → Q8_1 MMQ")?;

    // 2+3. gate + up matmuls. Per-rank rows = local_inter.
    qmatmul(
        ops, stream, ffn_gate.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.gate_f32, n_tokens, hidden, local_inter,
        qdtype_of(ffn_gate.dtype)?,
    )
    .context("dense ffn prefill (TP) gate qmatmul")?;
    qmatmul(
        ops, stream, ffn_up.ptr,
        scratch.x_q8_1, scratch.x_q8_1_mmq,
        scratch.up_f32, n_tokens, hidden, local_inter,
        qdtype_of(ffn_up.dtype)?,
    )
    .context("dense ffn prefill (TP) up qmatmul")?;

    // 4+5. Fused SwiGLU → F16, then quantise to BOTH Q8_1 layouts.
    flambeau_ops::hip::mlp::swiglu_f32_to_f16(
        ops, stream, scratch.gate_f32, scratch.up_f32, scratch.activated_f16,
        n_tokens * local_inter,
    )
    .context("dense ffn prefill (TP) swiglu_f32_to_f16")?;
    quantize_f16_q8_1(
        ops, stream, scratch.activated_f16, scratch.activated_q8_1,
        n_tokens * local_inter,
    )
    .context("dense ffn prefill (TP) activated → Q8_1 std")?;
    quantize_f16_q8_1_mmq(
        ops, stream, scratch.activated_f16, scratch.activated_q8_1_mmq,
        local_inter, n_tokens,
    )
    .context("dense ffn prefill (TP) activated → Q8_1 MMQ")?;

    // 6. Row-parallel down matmul. Per-rank cols = local_inter, rows = hidden.
    qmatmul(
        ops, stream, ffn_down.ptr,
        scratch.activated_q8_1, scratch.activated_q8_1_mmq,
        scratch.down_f32, n_tokens, local_inter, hidden,
        qdtype_of(ffn_down.dtype)?,
    )
    .context("dense ffn prefill (TP) down qmatmul")?;

    // 7. Cast F32→F16 directly into partial_ffn_out. AR fold (caller's
    //    BarP2pAllReduce::residual_tp{2,4}) sums the partials together
    //    with the residual.
    cast_f32_to_f16(ops, stream, scratch.down_f32, partial_ffn_out, n_tokens * hidden)
        .context("dense ffn prefill (TP) cast down → partial_ffn_out")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Substantive tests need GPU; covered by TP-2d's parity smoke
    // (compares TP at world=1 against PP forward_dense_ffn_decode).
}
