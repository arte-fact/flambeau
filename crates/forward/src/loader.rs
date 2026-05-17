//! Generic GGUF → device-side weight helpers.
//!
//! Arch-agnostic primitives that every model crate's loader composes:
//!
//! - `upload_bytes` / `upload_f16_from_f32` — bytes-as-is HtoD upload.
//! - `wrap_quant` — wraps a `(DevicePtr, n_elems, GgmlDType)` in the
//!   right `QuantWeight` variant.
//! - `upload_quant_weight` — raw GGUF tensor bytes → device →
//!   `QuantWeight`.
//! - `upload_dequant_to_f16` — host-dequant any tensor → cast to F16 →
//!   device. Used for norm weights (F32 GGUF) and quantised embedding
//!   tables.
//! - `upload_col_sharded_quant` / `upload_row_sharded_quant` — TP
//!   sharded variants of `upload_quant_weight`. Col-shard slices the
//!   GGUF dim-0 (output rows, contiguous bytes); row-shard slices
//!   dim-1 (input cols, requires per-row stride copy).
//!
//! Every helper appends `(DevicePtr, bytes)` to a caller-supplied
//! `&mut Vec<(DevicePtr, usize)>` so the model can walk one disposer
//! over all of its allocations.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
use flambeau_quant::{GgmlDType, GgufFile};
use half::f16;

use crate::ctx::QuantWeight;

/// Allocate `bytes.len()` device bytes, HtoD `bytes`, append the alloc
/// to `allocs`. Returns the device pointer.
pub fn upload_bytes(
    device: &HipDevice,
    bytes: &[u8],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let ptr = device.alloc(bytes.len()).context("alloc")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns bytes.len() bytes; bytes has bytes.len() host bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(bytes.as_ptr() as usize),
            bytes.len(),
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after upload")?;
    allocs.push((ptr, bytes.len()));
    Ok(ptr)
}

/// Host: F32 → F16 cast → HtoD as F16. Returns the wrapped tensor.
pub fn upload_f16_from_f32(
    device: &HipDevice,
    f32_vec: &[f32],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let f16_vec: Vec<f16> = f32_vec.iter().map(|&v| f16::from_f32(v)).collect();
    let bytes = f16_vec.len() * 2;
    let ptr = device.alloc(bytes).context("alloc F16")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns bytes; f16_vec has the matching byte count.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(f16_vec.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F16>::from_raw(ptr, f16_vec.len()) })
}

/// Block size (elems) + type size (bytes) for the GGUF dtypes
/// model-ops supports for `qmatmul`. K-quant / IQ / MXFP4 aren't
/// in this set yet.
pub fn quant_block_info(dtype: GgmlDType) -> Result<(usize, usize)> {
    Ok(match dtype {
        GgmlDType::Q4_0 => (32, 18),
        GgmlDType::Q4_1 => (32, 20),
        GgmlDType::Q5_0 => (32, 22),
        GgmlDType::Q5_1 => (32, 24),
        GgmlDType::Q8_0 => (32, 34),
        other => bail!(
            "quant_block_info: dtype {other:?} not in model-ops's qmatmul set; \
             K-quant / IQ / MXFP4 wrappers land in a later phase"
        ),
    })
}

/// Wrap a `(ptr, n_elems, dtype)` triple in the matching `QuantWeight`
/// variant. Errors if the dtype isn't a model-ops `qmatmul` target.
pub fn wrap_quant(ptr: DevicePtr, n_elems: usize, dtype: GgmlDType) -> Result<QuantWeight> {
    Ok(match dtype {
        GgmlDType::Q4_0 => QuantWeight::Q4_0(unsafe { Tensor::<Q4_0>::from_raw(ptr, n_elems) }),
        GgmlDType::Q4_1 => QuantWeight::Q4_1(unsafe { Tensor::<Q4_1>::from_raw(ptr, n_elems) }),
        GgmlDType::Q5_0 => QuantWeight::Q5_0(unsafe { Tensor::<Q5_0>::from_raw(ptr, n_elems) }),
        GgmlDType::Q5_1 => QuantWeight::Q5_1(unsafe { Tensor::<Q5_1>::from_raw(ptr, n_elems) }),
        GgmlDType::Q8_0 => QuantWeight::Q8_0(unsafe { Tensor::<Q8_0>::from_raw(ptr, n_elems) }),
        other => bail!(
            "wrap_quant: dtype {other:?} not in model-ops's qmatmul set"
        ),
    })
}

/// Bytes-as-is HtoD upload of a GGUF tensor's raw payload. Returns
/// the device pointer (no `Tensor` wrapping — caller picks the dtype
/// + element count).
pub fn upload_raw(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    upload_bytes(device, raw, allocs)
}

/// Dequant a tensor to F32 on host, cast to F16, HtoD as F16. Used
/// for norm weights (F32 in GGUF → F16 on device) and quantised
/// embedding tables (Q8_0 → F16 row table).
pub fn upload_dequant_to_f16(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    expected_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != expected_elems {
        bail!(
            "loader: {name} dequant produced {} elems, expected {expected_elems}",
            f32_vec.len()
        );
    }
    upload_f16_from_f32(device, &f32_vec, allocs)
}

/// Upload a quant tensor's raw bytes and wrap in `QuantWeight`.
pub fn upload_quant_weight(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let info = file
        .info(name)
        .with_context(|| format!("tensor info {name}"))?;
    let ptr = upload_raw(file, device, name, allocs)?;
    wrap_quant(ptr, n_elems, info.dtype)
}

/// Column-shard a quant tensor along its GGUF dim-0 (the matmul
/// output dim). Each rank gets `n_rows / n_ranks` rows worth of
/// contiguous bytes.
#[allow(clippy::too_many_arguments)]
pub fn upload_col_sharded_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    if n_rows % n_ranks != 0 {
        bail!(
            "{name}: n_rows {n_rows} not divisible by n_ranks {n_ranks}"
        );
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let (block_size, type_size) = quant_block_info(info.dtype)?;
    if n_cols % block_size != 0 {
        bail!(
            "{name}: n_cols {n_cols} not divisible by block_size {block_size}"
        );
    }
    let row_bytes = n_cols / block_size * type_size;
    let rows_per_rank = n_rows / n_ranks;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let expected_total = n_rows * row_bytes;
    if raw.len() != expected_total {
        bail!(
            "{name}: raw bytes {} != expected {expected_total} (n_rows={n_rows}, row_bytes={row_bytes})",
            raw.len()
        );
    }
    let start = rank * rows_per_rank * row_bytes;
    let end = start + rows_per_rank * row_bytes;
    let shard = &raw[start..end];
    let ptr = upload_bytes(device, shard, allocs)?;
    wrap_quant(ptr, rows_per_rank * n_cols, info.dtype)
}

/// Row-shard a quant tensor along its GGUF dim-1 (the matmul input
/// dim). Each rank gets `n_cols / n_ranks` cols of every output row;
/// `cols_per_rank` must be block-aligned (no intra-block splits).
#[allow(clippy::too_many_arguments)]
pub fn upload_row_sharded_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    if n_cols % n_ranks != 0 {
        bail!("{name}: n_cols {n_cols} not divisible by n_ranks {n_ranks}");
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let (block_size, type_size) = quant_block_info(info.dtype)?;
    let cols_per_rank = n_cols / n_ranks;
    if cols_per_rank % block_size != 0 {
        bail!(
            "{name}: cols_per_rank {cols_per_rank} not divisible by block_size {block_size}; \
             row-shard requires block-aligned splits"
        );
    }
    let row_bytes = n_cols / block_size * type_size;
    let half_row_bytes = cols_per_rank / block_size * type_size;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let mut shard: Vec<u8> = Vec::with_capacity(n_rows * half_row_bytes);
    for r in 0..n_rows {
        let row_start = r * row_bytes;
        let col_offset = rank * half_row_bytes;
        shard.extend_from_slice(
            &raw[row_start + col_offset..row_start + col_offset + half_row_bytes],
        );
    }
    let ptr = upload_bytes(device, &shard, allocs)?;
    wrap_quant(ptr, n_rows * cols_per_rank, info.dtype)
}
