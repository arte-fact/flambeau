//! Device-aware weight-upload helpers.
//!
//! The host-side byte slicing lives in [`flambeau_runtime::tp_slice`];
//! this module pairs slicing with a HIP `alloc` + `memcpy_async` and
//! tracks every allocation in a [`RawAllocTracker`] for the caller's
//! `dispose()` chain. Model crates use these instead of hand-rolling
//! the same `slice → alloc → memcpy → push-to-tracker` sequence per
//! tensor.
//!
//! Three entry points cover every upload pattern in the codebase:
//!
//! - [`upload_sharded_tensor`] — TP-sharded weights (column-parallel,
//!   row-parallel, fused-QKV). Returns an [`UploadedTensor`] holding
//!   the per-rank device buffer.
//! - [`upload_replicated_tensor`] — full tensor uploaded as-is, dtype
//!   preserved (token_embd, LM head, MoE expert weights when
//!   replicated, …).
//! - [`upload_replicated_norm_f32_to_f16`] — F32 norm tensor cast to
//!   F16 at upload (Gemma 4 stores every learned norm as F32 but the
//!   `rmsnorm_f16` kernel reads F16; same pattern in Qwen variants).
//!
//! All helpers synchronise the stream before returning so the buffer
//! is immediately readable by the caller's next kernel launch on the
//! same stream.

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::HipStream;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::{GgmlDType, GgufFile, TensorInfo};
use flambeau_runtime::tp_slice::slice_for_tp;
use flambeau_runtime::WeightLayout;
use half::f16;

use crate::driver_utils::RawAllocTracker;

/// One uploaded device tensor. Owns its allocation through the
/// [`RawAllocTracker`] the caller passed to the upload function; the
/// tracker's `dispose()` is what frees it.
#[derive(Debug, Clone, Copy)]
pub struct UploadedTensor {
    pub ptr: DevicePtr,
    pub dtype: GgmlDType,
    pub bytes: usize,
}

/// Slice + upload `info` per `layout` for `rank`. Allocates a fresh
/// device buffer sized to the rank's local bytes, copies the slice
/// from the GGUF mmap (zero-copy for Replicated / ColParallel; host
/// gather for RowParallel and FusedQkvParallel), and records the
/// allocation in `tracker` for later dispose.
pub fn upload_sharded_tensor(
    file: &GgufFile,
    info: &TensorInfo,
    layout: WeightLayout,
    rank: u32,
    device: &HipDevice,
    stream: &HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<UploadedTensor> {
    let slice = slice_for_tp(file, &info.name, layout, rank)
        .with_context(|| format!("slice_for_tp `{}`", info.name))?;
    let bytes = slice.len();
    htod_alloc_copy(&info.name, &slice, info.dtype, device, stream, tracker, bytes)
}

/// Upload `info` fully replicated (no slicing). Dtype preserved.
pub fn upload_replicated_tensor(
    file: &GgufFile,
    info: &TensorInfo,
    device: &HipDevice,
    stream: &HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<UploadedTensor> {
    let bytes = info.size_in_bytes() as usize;
    let raw = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if raw.len() < bytes {
        bail!(
            "tensor `{}` mmap slice {} < declared {}",
            info.name,
            raw.len(),
            bytes
        );
    }
    htod_alloc_copy(
        &info.name,
        &raw[..bytes],
        info.dtype,
        device,
        stream,
        tracker,
        bytes,
    )
}

/// Upload `info` as F16, casting from F32 host-side. Used for learned
/// norms: gemma4 stores them all as F32 but the `rmsnorm_f16` kernel
/// reads F16 — uploading raw would silently corrupt downstream norms.
/// Asserts `info.dtype == F32` and `info.elements() == expected_len`.
pub fn upload_replicated_norm_f32_to_f16(
    file: &GgufFile,
    info: &TensorInfo,
    expected_len: usize,
    device: &HipDevice,
    stream: &HipStream,
    tracker: &mut RawAllocTracker,
) -> Result<UploadedTensor> {
    if info.dtype != GgmlDType::F32 {
        bail!(
            "norm `{}` expected F32, got {:?}",
            info.name,
            info.dtype
        );
    }
    let elems: usize = info.dims.iter().product::<u64>() as usize;
    if elems != expected_len {
        bail!(
            "norm `{}` elems {} != expected {}",
            info.name,
            elems,
            expected_len
        );
    }
    let raw = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if raw.len() < elems * 4 {
        bail!(
            "norm `{}` mmap slice {} < expected {}",
            info.name,
            raw.len(),
            elems * 4
        );
    }
    // SAFETY: F32 dtype + page-aligned mmap = 4-byte alignment OK.
    let src: &[f32] = bytemuck::cast_slice(&raw[..elems * 4]);
    let host_f16: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();
    let new_bytes = elems * 2;
    let host_bytes: &[u8] = bytemuck::cast_slice(&host_f16);
    htod_alloc_copy(
        &info.name,
        host_bytes,
        GgmlDType::F16,
        device,
        stream,
        tracker,
        new_bytes,
    )
}

/// Shared HtoD path: alloc → memcpy → sync → push to tracker.
fn htod_alloc_copy(
    name: &str,
    host: &[u8],
    out_dtype: GgmlDType,
    device: &HipDevice,
    stream: &HipStream,
    tracker: &mut RawAllocTracker,
    bytes: usize,
) -> Result<UploadedTensor> {
    if host.len() < bytes {
        bail!(
            "htod_alloc_copy `{name}`: host slice {} < requested {bytes}",
            host.len()
        );
    }
    let ptr = device
        .alloc(bytes)
        .map_err(|e| anyhow!("hipMalloc {bytes} B for `{name}`: {e}"))?;
    // SAFETY: ptr was just allocated for `bytes`; host slice has ≥ bytes;
    // the bounded synchronize below ensures host stays alive through the
    // copy.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(host.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow!("memcpy_async `{name}`: {e}"))?;
    }
    stream.synchronize()?;
    tracker.track(ptr, bytes);
    Ok(UploadedTensor {
        ptr,
        dtype: out_dtype,
        bytes,
    })
}
