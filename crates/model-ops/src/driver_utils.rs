//! Shared driver-scaffolding helpers used by every model crate's
//! `PpStage` / `TpStage` / `HybridStage` upload + dispose flow.
//!
//! Move-here policy: this module is for boilerplate that has zero
//! coupling to model layer composition or layout. Anything that
//! reasons about per-layer shape (KV cache sizing, scratch shape,
//! weight slicing) stays in the model crate. The pieces collected
//! here are kernel-launch-free utility (allocate, zero, memcpy) plus
//! the host-side token-embedding lookup that every model uses
//! identically.

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_quant::{GgmlDType, QK8_0};
use half::f16;

/// Allocate `bytes` on `device` and zero-fill via a host→device memcpy
/// on the default stream. Synchronises before returning. Used by every
/// per-stage scratch-builder for buffers that must start at zero
/// (residual streams, partial accumulators).
pub fn alloc_zeroed(device: &HipDevice, bytes: usize) -> Result<DevicePtr> {
    let p = device
        .alloc(bytes)
        .map_err(|e| anyhow!("alloc {bytes}: {e}"))?;
    let zero = vec![0u8; bytes];
    // SAFETY: dst has `bytes` allocation; src is a host vec of the
    // same length; we sync the stream before returning so the host
    // buffer outlives the copy.
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            p,
            DevicePtr(zero.as_ptr() as usize),
            bytes,
        )?;
    }
    device.default_stream().synchronize()?;
    Ok(p)
}

/// Allocate an F16 `[n]` buffer on `device` and fill with `1.0` via a
/// host→device memcpy on the default stream. Used by gemma4 + future
/// models that apply an "unlearned" RMSNorm (norm-with-unit-weight) —
/// the V-norm pass in gemma4's full-attn layer is the canonical
/// caller.
pub fn upload_f16_ones(device: &HipDevice, n: usize) -> Result<DevicePtr> {
    let ones: Vec<f16> = vec![f16::from_f32(1.0); n];
    let bytes = ones.len() * 2;
    let p = device
        .alloc(bytes)
        .map_err(|e| anyhow!("alloc {bytes}: {e}"))?;
    // SAFETY: dst has `bytes` allocation; ones outlives the bounded sync.
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            p,
            DevicePtr(ones.as_ptr() as usize),
            bytes,
        )?;
    }
    device.default_stream().synchronize()?;
    Ok(p)
}

/// Map a GGUF [`GgmlDType`] to the [`QDtype`] the qmatmul dispatcher
/// uses. Covers every quant family the V1 kernels support; returns
/// an error for unsupported dtypes (e.g. MXFP4, which is at-load
/// dequantised to Q8_0 by callers).
pub fn ggml_to_qdtype(d: GgmlDType) -> Result<QDtype> {
    Ok(match d {
        GgmlDType::F32 => QDtype::F32,
        GgmlDType::F16 => QDtype::F16,
        GgmlDType::BF16 => QDtype::BF16,
        GgmlDType::Q8_0 => QDtype::Q8_0,
        GgmlDType::Q8_1 => QDtype::Q8_1,
        GgmlDType::Q4_0 => QDtype::Q4_0,
        GgmlDType::Q4_1 => QDtype::Q4_1,
        GgmlDType::Q5_0 => QDtype::Q5_0,
        GgmlDType::Q5_1 => QDtype::Q5_1,
        GgmlDType::Q2K => QDtype::Q2_K,
        GgmlDType::Q3K => QDtype::Q3_K,
        GgmlDType::Q4K => QDtype::Q4_K,
        GgmlDType::Q5K => QDtype::Q5_K,
        GgmlDType::Q6K => QDtype::Q6_K,
        GgmlDType::Q8K => QDtype::Q8_K,
        GgmlDType::Iq4Nl => QDtype::IQ4_NL,
        GgmlDType::Iq4Xs => QDtype::IQ4_XS,
        GgmlDType::Iq3Xxs => QDtype::IQ3_XXS,
        GgmlDType::Iq3S => QDtype::IQ3_S,
        GgmlDType::Iq2Xxs => QDtype::IQ2_XXS,
        GgmlDType::Iq2Xs => QDtype::IQ2_XS,
        GgmlDType::Iq2S => QDtype::IQ2_S,
        GgmlDType::Iq1S => QDtype::IQ1_S,
        GgmlDType::Iq1M => QDtype::IQ1_M,
        other => bail!("ggml_to_qdtype: unsupported dtype {other:?}"),
    })
}

/// Host-side F32 → Q8_0 quantisation. Used as a `WeightSpec.pre_upload`
/// hook for tensors that ship F32 in the GGUF but are consumed by
/// Q8_0 kernels (e.g. qwen3-moe's `ssm_alpha` / `ssm_beta`). Element
/// count must be a multiple of `QK8_0` (32). Block layout matches
/// `flambeau_quant::BlockQ8_0`: 2-byte `d` (F16 scale) + 32-byte `qs`
/// = 34 bytes per block.
///
/// Pass-through if the source dtype is already `Q8_0` — returns the
/// original bytes verbatim so callers can use this as a single
/// `pre_upload` hook for tensors that *might* already be quantised.
pub fn quant_f32_to_q8_0(raw: &[u8], src_dtype: GgmlDType) -> Result<(Vec<u8>, GgmlDType)> {
    if src_dtype == GgmlDType::Q8_0 {
        return Ok((raw.to_vec(), GgmlDType::Q8_0));
    }
    if src_dtype != GgmlDType::F32 {
        bail!("quant_f32_to_q8_0: unsupported source dtype {src_dtype:?} (expected F32 or Q8_0)");
    }
    // SAFETY-cast: raw is the F32 mmap view; alignment is 4 bytes
    // (mmap is page-aligned, exceeds f32 alignment).
    let src: &[f32] = bytemuck::cast_slice(raw);
    let elems = src.len();
    if elems == 0 || elems % QK8_0 != 0 {
        bail!("quant_f32_to_q8_0: elem count {elems} not a positive multiple of QK8_0={QK8_0}");
    }
    let n_blocks = elems / QK8_0;
    let block_size = 34usize; // 2 (d) + 32 (qs)
    let mut buf: Vec<u8> = Vec::with_capacity(n_blocks * block_size);
    for block in src.chunks_exact(QK8_0) {
        let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = absmax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        for &v in block {
            let q = (v * id).round_ties_even() as i32;
            let q = q.clamp(-127, 127) as i8;
            buf.push(q as u8);
        }
    }
    Ok((buf, GgmlDType::Q8_0))
}

/// Bytes-per-row for a 2-D quantised weight with `hidden` columns.
/// Asserts the contraction dim divides cleanly into the dtype's
/// block size — otherwise the GGUF layout is malformed.
pub fn row_bytes_for_dtype(dtype: GgmlDType, hidden: usize) -> Result<usize> {
    let bs = dtype.block_size() as usize;
    let ts = dtype.type_size() as usize;
    if hidden % bs != 0 {
        bail!("hidden {hidden} % block_size {bs} != 0 for {dtype:?}");
    }
    Ok((hidden / bs) * ts)
}

/// Host-side single-token embedding lookup. Downloads the
/// `token_id`-th row of `token_embd_ptr` (a 2-D `[vocab, hidden]`
/// weight in GGUF-native dtype), dequantises it on the host, casts
/// to F16, and uploads the F16 row into `out_f16_dev`. Returns after
/// a stream sync so the caller can read the result immediately.
///
/// Per-token cost is two memcpys + one host dequantise + two syncs —
/// negligible at any realistic decode throughput. A device-side
/// gather kernel that bypasses the host round-trip is a future
/// optimisation candidate but has not been measured to be on any
/// hot-path bottleneck.
#[allow(clippy::too_many_arguments)]
pub fn embed_token_host(
    device: &HipDevice,
    stream: &HipStream,
    token_embd_ptr: DevicePtr,
    token_embd_dtype: GgmlDType,
    token_embd_bytes: usize,
    vocab: usize,
    hidden: usize,
    token_id: u32,
    out_f16_dev: DevicePtr,
) -> Result<()> {
    if (token_id as usize) >= vocab {
        bail!("token_id {token_id} >= vocab {vocab}");
    }
    let row_bytes = row_bytes_for_dtype(token_embd_dtype, hidden)?;
    let offset = (token_id as usize) * row_bytes;
    if offset + row_bytes > token_embd_bytes {
        bail!(
            "token_embd row out of bounds: token_id={token_id} row_bytes={row_bytes} \
             total_bytes={token_embd_bytes}"
        );
    }
    let src = token_embd_ptr.offset_bytes(offset);

    // 1. Download the row's raw bytes.
    let mut row_raw = vec![0u8; row_bytes];
    // SAFETY: src points at >= row_bytes valid device bytes; row_raw
    // owns row_bytes host bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(row_raw.as_mut_ptr() as usize),
            src,
            row_bytes,
        )?;
    }
    stream.synchronize()?;

    // 2. Dequantise on host. F16 fast-path avoids the F32 round-trip.
    let row_f16: Vec<f16> = if token_embd_dtype == GgmlDType::F16 {
        bytemuck::cast_slice::<u8, f16>(&row_raw).to_vec()
    } else {
        let row_f32 = flambeau_quant::dequantize_to_vec(token_embd_dtype, &row_raw, hidden)
            .with_context(|| format!("dequant token_embd row {token_id}"))?;
        row_f32.into_iter().map(f16::from_f32).collect()
    };
    drop(row_raw);

    // 3. Upload to the F16 scratch slot.
    let upload_bytes = hidden * 2;
    // SAFETY: out_f16_dev has hidden*2 valid bytes; row_f16 outlives
    // the bounded sync.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            out_f16_dev,
            DevicePtr(row_f16.as_ptr() as usize),
            upload_bytes,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Tracks `(DevicePtr, bytes)` pairs for every raw allocation made
/// while building a per-rank stage's scratch / weight buffers, so
/// the stage's `dispose(device)` can walk them in one place. Mirrors
/// the ad-hoc `raw_alloc_bytes: Vec<(DevicePtr, usize)>` field every
/// model crate previously carried.
///
/// `disposed` lets the stage's `Drop` impl warn-on-leak without
/// double-free risk if `dispose` was already called.
#[derive(Default)]
pub struct RawAllocTracker {
    pub allocs: Vec<(DevicePtr, usize)>,
    disposed: bool,
}

impl RawAllocTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate `bytes` on `device`, zero-fill, and record the
    /// `(ptr, bytes)` pair for later dispose. Returns the new ptr.
    pub fn alloc_zeroed_tracked(&mut self, device: &HipDevice, bytes: usize) -> Result<DevicePtr> {
        let p = alloc_zeroed(device, bytes)?;
        self.allocs.push((p, bytes));
        Ok(p)
    }

    /// Record an externally-made allocation (e.g. the result of
    /// `upload_f16_ones` whose host source we want to free before
    /// dispose).
    pub fn track(&mut self, ptr: DevicePtr, bytes: usize) {
        self.allocs.push((ptr, bytes));
    }

    /// Allocate + track an F32 buffer of `n` elements (4 B / elem).
    /// Returns `(ptr, bytes)` — the `(DevicePtr, usize)` pair used by
    /// stage-scratch structs.
    pub fn alloc_f32(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        let bytes = n * 4;
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    /// Allocate + track an F16 buffer of `n` elements (2 B / elem).
    pub fn alloc_f16(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        let bytes = n * 2;
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    /// Allocate + track a Q8_0 buffer of `n` elements
    /// (`(n / 32) * 34` B; asserts `n % 32 == 0`).
    pub fn alloc_q8_0(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        if n % 32 != 0 {
            bail!("alloc_q8_0: n={n} not a multiple of 32");
        }
        let bytes = (n / 32) * 34;
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    /// Allocate + track a Q8_1 buffer of `n` elements
    /// (`(n / 32) * 36` B; asserts `n % 32 == 0`).
    pub fn alloc_q8_1(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        if n % 32 != 0 {
            bail!("alloc_q8_1: n={n} not a multiple of 32");
        }
        let bytes = (n / 32) * 36;
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    /// Allocate + track an `i32` scratch of `n` elements (4 B / elem).
    /// Used for `positions`, `expert_ids`, and other small index
    /// buffers.
    pub fn alloc_i32(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        let bytes = n * 4;
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    /// Allocate + track a Q8_1_MMQ buffer of `n` elements. The 4-warp
    /// LDS-tiled MMQ kernels read Q8_1 in 128-elem super-blocks
    /// (`BlockQ8_1Mmq`, 144 B). Asserts `n % 128 == 0`.
    pub fn alloc_q8_1_mmq(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        if n % 128 != 0 {
            bail!("alloc_q8_1_mmq: n={n} not a multiple of 128");
        }
        let bytes = (n / 128) * std::mem::size_of::<flambeau_quant::BlockQ8_1Mmq>();
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    /// Allocate + track a `u64` scratch of `n` elements (8 B / elem).
    /// Used for batched-decode per-slot pointer tables (`slot_k_ptrs`,
    /// `slot_v_ptrs`).
    pub fn alloc_u64(&mut self, device: &HipDevice, n: usize) -> Result<(DevicePtr, usize)> {
        let bytes = n * 8;
        let ptr = self.alloc_zeroed_tracked(device, bytes)?;
        Ok((ptr, bytes))
    }

    pub fn is_empty(&self) -> bool {
        self.allocs.is_empty()
    }

    pub fn disposed(&self) -> bool {
        self.disposed
    }

    /// Drain the tracked allocs and mark the tracker disposed without
    /// freeing any of them. Use when the caller has taken ownership of
    /// the (`ptr`, `bytes`) pairs via [`UploadedTensor`] (or similar)
    /// and is tracking them on its own — e.g. qwen3-moe's per-shard
    /// byte counter + DeviceTensor list, or gemma4's PP `raw:
    /// &mut Vec<(DevicePtr, usize)>`. Returns the drained alloc list
    /// so the caller can integrate it.
    ///
    /// After this call the tracker behaves as if `dispose` had been
    /// called — `Drop` won't log, subsequent `dispose` is a no-op.
    pub fn forget_allocs(&mut self) -> Vec<(DevicePtr, usize)> {
        self.disposed = true;
        std::mem::take(&mut self.allocs)
    }

    /// Free every tracked allocation on `device` and mark disposed.
    /// Idempotent — a second call is a no-op.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: every ptr came from `device.alloc(bytes)` (or a
            // tracked sibling that uses the same allocator).
            unsafe {
                let _ = device.dealloc(ptr, bytes);
            }
        }
        Ok(())
    }
}

impl Drop for RawAllocTracker {
    fn drop(&mut self) {
        if !self.disposed && !self.allocs.is_empty() {
            tracing::warn!(
                "RawAllocTracker dropped without dispose(); {} allocations leaked",
                self.allocs.len()
            );
        }
    }
}
