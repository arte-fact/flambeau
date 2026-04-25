//! Small cross-cutting helpers used by every submodule under `forward/`.
//!
//! Nothing here is on the tight inner loop — these are setup /
//! one-shot conversions that appear in both decode and prefill paths.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::hip::{
    qmatmul::{mmvq, qmatmul},
    HipDevice, HipStream, OpsRegistry,
};
use flambeau_quant::GgmlDType;

use flambeau_ops::hip::cast::cast_f32_to_f16;
use flambeau_ops::hip::moe::{
    indexed_moe_mmvq_q4_0, indexed_moe_mmvq_q4_k_gate_up, indexed_moe_mmvq_q4_k_r2,
    indexed_moe_mmvq_q5_k, indexed_moe_mmvq_q6_k, indexed_moe_mmvq_q8_0,
};
use flambeau_ops::hip::norm::quantize_f16_q8_1;

use crate::weights::DeviceTensor;

/// K-quant super-block size. Q4_K / Q5_K / Q6_K weights pack 256 elements
/// per super-block; MoE weight dims `[n_experts, inter, hidden]` must be
/// divisible by `QK_K` on the `hidden` and `inter` axes so our indexed-MoE
/// kernels can walk super-blocks without a row-straddling tail.
pub(crate) const QK_K: usize = 256;

/// Assert MoE expert dtypes fall inside the supported set and that
/// `hidden` / `inter` are `QK_K`-aligned. Shared by decode and prefill.
///
/// Gate and up must agree in dtype and be one of `Q4_K / Q8_0 / Q4_0`.
/// Down may additionally be `Q6_K` (UD-Q4_K_S `ffn_down_exps` promotion).
///
/// `label` disambiguates decode vs prefill in error messages; pass
/// `"indexed-MoE"` for decode and `"indexed-MoE prefill"` for prefill to
/// preserve the existing text that downstream tests grep against.
///
/// # Errors
/// Returns an error if any dtype is outside the supported set or
/// `hidden` / `inter` are not multiples of `QK_K`.
pub(crate) fn validate_moe_dtypes(
    label: &str,
    gate_dt: GgmlDType,
    up_dt: GgmlDType,
    down_dt: GgmlDType,
    hidden: usize,
    inter: usize,
) -> Result<()> {
    if hidden % QK_K != 0 {
        bail!("MoE expects hidden={hidden} divisible by QK_K={QK_K}");
    }
    if inter % QK_K != 0 {
        bail!("MoE expects moe_intermediate_size={inter} divisible by QK_K={QK_K}");
    }
    if !(gate_dt == up_dt
        && (gate_dt == GgmlDType::Q4K
            || gate_dt == GgmlDType::Q8_0
            || gate_dt == GgmlDType::Q4_0))
    {
        bail!(
            "{label} gate/up dtypes must match and be Q4_K, Q8_0 or Q4_0; got gate={gate_dt:?}, up={up_dt:?}"
        );
    }
    if down_dt != GgmlDType::Q4K
        && down_dt != GgmlDType::Q5K
        && down_dt != GgmlDType::Q6K
        && down_dt != GgmlDType::Q8_0
        && down_dt != GgmlDType::Q4_0
    {
        bail!(
            "{label} ffn_down_exps must be Q4_K, Q5_K, Q6_K, Q8_0 or Q4_0; got {down_dt:?}"
        );
    }
    Ok(())
}

/// Run the MoE gate+up matmul for one dispatch shape. Parametrised on
/// `n_tokens` so both decode (n_tokens=1) and prefill share a single
/// implementation. Dispatches:
/// - Q4_K → fused `indexed_moe_mmvq_q4_k_gate_up` (1 kernel, 2 outputs)
/// - Q8_0 / Q4_0 → two separate `indexed_moe_mmvq_q{8_0,4_0}` launches
///
/// Caller must have validated `dtype` via [`validate_moe_dtypes`].
///
/// # Errors
/// Returns an error if the underlying kernel launch fails.
#[expect(
    clippy::too_many_arguments,
    reason = "thin parametric wrapper over a family of indexed-MoE kernels; flattens into \
              the caller's existing scratch-pointer flow so introducing a context struct \
              would just rewrap the same pointers."
)]
pub(crate) fn run_indexed_moe_gate_up(
    ops: &OpsRegistry,
    stream: &HipStream,
    dtype: GgmlDType,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    x_q8_1: DevicePtr,
    expert_ids: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    inter: usize,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<()> {
    match dtype {
        GgmlDType::Q4K => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_q4_k_gate_up(
                ops, stream, w_gate, w_up, x_q8_1, expert_ids, gate_out, up_out, inter,
                n_tokens, top_k, nb,
            )
            .context("indexed_moe gate+up q4_k")
        }
        GgmlDType::Q8_0 => {
            let nb = hidden / 32;
            indexed_moe_mmvq_q8_0(
                ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens,
                top_k, nb,
            )
            .context("indexed_moe gate q8_0")?;
            indexed_moe_mmvq_q8_0(
                ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k,
                nb,
            )
            .context("indexed_moe up q8_0")
        }
        GgmlDType::Q4_0 => {
            // V2.23.b.1 — fused gate+up reads Q8_1 activation once per block
            // and produces both outputs. Halves launch count for Q4_0 MoE
            // decode (where the tile8 MMQ path does not fire at n_tokens<32).
            let nb = hidden / 32;
            flambeau_ops::hip::moe::indexed_moe_mmvq_q4_0_gate_up(
                ops, stream, w_gate, w_up, x_q8_1, expert_ids, gate_out, up_out,
                inter, n_tokens, top_k, nb,
            )
            .context("indexed_moe gate+up q4_0 fused")
        }
        _ => bail!("run_indexed_moe_gate_up: unsupported gate dtype {dtype:?} (expected Q4_K / Q8_0 / Q4_0)"),
    }
}

/// Run the MoE down matmul for one dispatch shape. Parametrised on
/// `n_tokens_eff` and `top_k_inner` so callers that have already flattened
/// the `(n_tokens, top_k)` routing into per-expert effective tokens can
/// pass `(n_tokens * top_k, 1)` (decode + prefill down-step pattern), and
/// the standard prefill path can pass `(n_tokens, top_k)` directly.
/// Dispatches Q4_K (r2 variant), Q6_K, Q8_0, Q4_0.
///
/// # Errors
/// Returns an error if the underlying kernel launch fails or the dtype is
/// outside the supported set.
#[expect(
    clippy::too_many_arguments,
    reason = "thin parametric wrapper over a family of indexed-MoE kernels; flattens into \
              the caller's existing scratch-pointer flow so introducing a context struct \
              would just rewrap the same pointers."
)]
pub(crate) fn run_indexed_moe_down(
    ops: &OpsRegistry,
    stream: &HipStream,
    dtype: GgmlDType,
    w_down: DevicePtr,
    activated_q8_1: DevicePtr,
    expert_ids: DevicePtr,
    down_out: DevicePtr,
    hidden: usize,
    n_tokens_eff: usize,
    top_k_inner: usize,
    inter: usize,
) -> Result<()> {
    match dtype {
        GgmlDType::Q4K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q4_k_r2(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q4_k r2")
        }
        GgmlDType::Q5K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q5_k(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q5_k")
        }
        GgmlDType::Q6K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q6_k(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q6_k")
        }
        GgmlDType::Q8_0 => {
            let nb = inter / 32;
            indexed_moe_mmvq_q8_0(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q8_0")
        }
        GgmlDType::Q4_0 => {
            let nb = inter / 32;
            indexed_moe_mmvq_q4_0(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q4_0")
        }
        _ => bail!("run_indexed_moe_down: unsupported down dtype {dtype:?} (expected Q4_K / Q6_K / Q8_0 / Q4_0)"),
    }
}

/// Cast F32 → F16 then quantize F16 → Q8_1, in that order. The two-kernel
/// pipeline is used on every MoE + shared-expert down-input path; a fused
/// F32→Q8_1 kernel was tried (V2.x memory) and regressed by ~1% because the
/// F32 quantise path lacks the packed fp16 max-reduction.
///
/// `label` is a short prefix that propagates into both step's error
/// contexts so backtraces stay readable.
///
/// # Errors
/// Returns an error if either kernel launch fails.
pub(crate) fn cast_and_quantize_f32_to_q8_1(
    ops: &OpsRegistry,
    stream: &HipStream,
    src_f32: DevicePtr,
    tmp_f16: DevicePtr,
    dst_q8_1: DevicePtr,
    n_elems: usize,
    label: &str,
) -> Result<()> {
    cast_f32_to_f16(ops, stream, src_f32, tmp_f16, n_elems)
        .with_context(|| format!("{label}: cast f32 → f16"))?;
    quantize_f16_q8_1(ops, stream, tmp_f16, dst_q8_1, n_elems)
        .with_context(|| format!("{label}: quantize f16 → Q8_1"))
}

/// Write `[position]` as an i32 into the 4-byte `positions` scratch slot.
/// Used by RoPE to pick the angle per token.
pub(super) fn upload_position(
    device: &HipDevice,
    stream: &HipStream,
    dst: DevicePtr,
    position: i32,
) -> Result<()> {
    let host = [position];
    // SAFETY: `dst` has 4 valid bytes; `host` is 4 valid host bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            dst,
            DevicePtr(host.as_ptr() as usize),
            4,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Map our `GgmlDType` (from GGUF) to the `QDtype` the qmatmul dispatcher
/// uses. Only the dtypes our V1 kernels support are allowed here; everything
/// else is a load-time error.
///
/// # Errors
/// Returns an error if `dtype` is not in the V1 qmatmul dispatch set.
pub(super) fn qdtype_of(dtype: GgmlDType) -> Result<QDtype> {
    Ok(match dtype {
        GgmlDType::Q4K => QDtype::Q4_K,
        GgmlDType::Q5K => QDtype::Q5_K,
        GgmlDType::Q6K => QDtype::Q6_K,
        GgmlDType::Q8_0 => QDtype::Q8_0,
        GgmlDType::Q4_1 => QDtype::Q4_1,
        // V2.21.b — UD-Q8_K_XL reserves F16 for i-matrix-flagged layers
        // (Qwen3.6-27B-UD-Q8_K_XL: all 48 attn_gate + 48 ssm_out + scattered
        // attn_q/k + ffn_gate/up/down). `mmvq()` special-cases F16 to skip
        // the dispatch table and call the direct F16×Q8_1 kernel.
        GgmlDType::F16 => QDtype::F16,
        // V2.23.a — Q4_0 and Q5_0 unblock Qwen3.6-35B-A3B-Q4_0.
        GgmlDType::Q4_0 => QDtype::Q4_0,
        GgmlDType::Q5_0 => QDtype::Q5_0,
        // V2.26.a — Q5_1 (llama.cpp parity; no Qwen3 model currently uses it
        // but unblocks any incoming GGUF mix).
        GgmlDType::Q5_1 => QDtype::Q5_1,
        other => bail!("weight dtype {other:?} not supported by V1 qmatmul dispatch"),
    })
}

/// Pull the `(n_rows, k)` pair out of a weight tensor's GGUF dims.
///
/// `flambeau_quant::GgufFile` reverses the on-wire dim order at parse time,
/// so `dims` is **outermost-first**: for a 2D weight `[n_rows, k]` we have
/// `dims = [n_rows, k]`.
///
/// # Errors
/// Returns an error if the weight is not 2D.
pub(super) fn mat_shape(w: &DeviceTensor) -> Result<(usize, usize)> {
    if w.dims.len() != 2 {
        bail!(
            "expected a 2D weight tensor for `{}`, got dims {:?}",
            w.name,
            w.dims
        );
    }
    let n_rows = w.dims[0] as usize;
    let k = w.dims[1] as usize;
    Ok((n_rows, k))
}

/// Byte count per vocabulary row for a 2D weight `[vocab, hidden]` of the
/// given dtype.
///
/// # Errors
/// Returns an error if `hidden` is not a multiple of the dtype's block size.
pub(super) fn row_bytes_for_dtype(dtype: GgmlDType, hidden: usize) -> Result<usize> {
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    if block_size > 1 && hidden % block_size != 0 {
        bail!(
            "token_embd hidden {hidden} is not a multiple of block_size {block_size} for {dtype:?}"
        );
    }
    let n_blocks = hidden / block_size;
    Ok(n_blocks * type_size)
}

/// Run an MMVQ against a weight `DeviceTensor`, validating dims and
/// dispatching on dtype. Keeps forward bodies readable.
///
/// # Errors
/// Returns an error if the tensor shape doesn't match the caller's
/// `(expected_rows, expected_k)`, or if the mmvq dispatch fails.
pub(super) fn run_mmvq_from_tensor(
    ops: &OpsRegistry,
    stream: &HipStream,
    w: &DeviceTensor,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    expected_rows: usize,
    expected_k: usize,
    label: &str,
) -> Result<()> {
    let dtype = qdtype_of(w.dtype)?;
    let (rows, k) = mat_shape(w)?;
    if rows != expected_rows || k != expected_k {
        bail!(
            "{label} shape [{rows}, {k}] != expected [{expected_rows}, {expected_k}]"
        );
    }
    mmvq(ops, stream, w.ptr, act_q8_1, dst, rows, k, dtype)
        .with_context(|| format!("mmvq {label}"))
}

/// Prefill counterpart to [`run_mmvq_from_tensor`]. Dispatches on dtype and
/// routes through MMQ when the caller's recipe applies; callers that don't
/// exercise the MmqLdsX64 kernel can pass [`DevicePtr(0)`] for
/// `act_q8_1_mmq` (no access).
///
/// # Errors
/// Returns an error if the tensor shape doesn't match the caller's
/// `(expected_rows, expected_k)`, or if the qmatmul dispatch fails.
pub(super) fn run_qmatmul_from_tensor(
    ops: &OpsRegistry,
    stream: &HipStream,
    w: &DeviceTensor,
    act_q8_1: DevicePtr,
    act_q8_1_mmq: DevicePtr,
    dst: DevicePtr,
    m: usize,
    expected_k: usize,
    expected_rows: usize,
    label: &str,
) -> Result<()> {
    let dtype = qdtype_of(w.dtype)?;
    let (rows, k) = mat_shape(w)?;
    if rows != expected_rows || k != expected_k {
        bail!(
            "{label} shape [{rows}, {k}] != expected [{expected_rows}, {expected_k}]"
        );
    }
    qmatmul(ops, stream, w.ptr, act_q8_1, act_q8_1_mmq, dst, m, k, rows, dtype)
        .with_context(|| format!("qmatmul {label}"))
}
