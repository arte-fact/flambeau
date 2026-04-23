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

use crate::weights::DeviceTensor;

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
            "token_embd hidden {hidden} is not a multiple of block_size {block_size} for {:?}",
            dtype
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
