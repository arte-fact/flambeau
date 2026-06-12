//! Byte / tensor upload primitives + dtype mapping. Building blocks
//! for the sharded helpers and per-layer composers in sibling
//! modules.

use anyhow::{bail, Context, Result};
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32};
use flambeau_quant::{GgmlDType, GgufFile};
use half::f16;

use crate::ctx::QuantWeight;

pub fn upload_bytes(
    device: &impl Device,
    bytes: &[u8],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let ptr = device.alloc(bytes.len()).context("alloc")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns bytes.len(); bytes is a host slice of the same len.
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

/// Allocate + upload a `[n]` F16 buffer filled with `1.0`. Used for
/// gemma4's `v_ones` per-head V-norm weight. Caller adds the
/// allocation to `allocs` for lifetime tracking via this function.
pub fn upload_f16_ones(
    device: &impl Device,
    n: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let host = vec![f16::ONE; n];
    let bytes = n * 2;
    let ptr = device.alloc(bytes).context("alloc F16 ones")?;
    let stream = device.default_stream();
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after f16 ones upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F16>::from_raw(ptr, n) })
}

pub fn upload_f16_from_f32(
    device: &impl Device,
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

/// Map a GGUF dtype to the runtime `QDtype` the HIP qmatmul kernel
/// dispatcher uses. Covers every dtype `flambeau_ops::Ops::qmatmul`
/// has a kernel for. Float dtypes (F32 / F16 / BF16) are NOT in this
/// set — they take the dequant-and-requantise-to-Q8_0 fallback path
/// in `upload_quant_weight`.
pub fn ggml_to_qdtype(dtype: GgmlDType) -> Result<QDtype> {
    use flambeau_quant::GgmlDType as G;
    Ok(match dtype {
        G::Q4_0 => QDtype::Q4_0,
        G::Q4_1 => QDtype::Q4_1,
        G::Q5_0 => QDtype::Q5_0,
        G::Q5_1 => QDtype::Q5_1,
        G::Q8_0 => QDtype::Q8_0,
        G::Q2K => QDtype::Q2_K,
        G::Q3K => QDtype::Q3_K,
        G::Q4K => QDtype::Q4_K,
        G::Q5K => QDtype::Q5_K,
        G::Q6K => QDtype::Q6_K,
        G::Q8K => QDtype::Q8_K,
        G::Iq1S => QDtype::IQ1_S,
        G::Iq1M => QDtype::IQ1_M,
        G::Iq2Xxs => QDtype::IQ2_XXS,
        G::Iq2Xs => QDtype::IQ2_XS,
        G::Iq2S => QDtype::IQ2_S,
        G::Iq3Xxs => QDtype::IQ3_XXS,
        G::Iq3S => QDtype::IQ3_S,
        G::Iq4Nl => QDtype::IQ4_NL,
        G::Iq4Xs => QDtype::IQ4_XS,
        other => bail!("ggml_to_qdtype: {other:?} has no qmatmul kernel; use dequant fallback"),
    })
}

/// True when `flambeau_ops::Ops::qmatmul` has a kernel for `dtype`,
/// i.e. the loader can upload it bytes-as-is. False for F16 / BF16 /
/// F32 (handled by the dequant-to-Q8_0 fallback).
pub fn dtype_qmatmul_native(dtype: GgmlDType) -> bool {
    ggml_to_qdtype(dtype).is_ok()
}

pub fn wrap_quant(ptr: DevicePtr, n_elems: usize, dtype: GgmlDType) -> Result<QuantWeight> {
    Ok(QuantWeight {
        ptr,
        dtype: ggml_to_qdtype(dtype)?,
        n_elems,
    })
}

pub fn upload_raw(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    upload_bytes(device, raw, allocs)
}

/// Host-dequant any tensor → cast to F16 → HtoD. Used for norm
/// weights (F32 in GGUF) and quantised embedding tables.
pub fn upload_dequant_to_f16(
    file: &GgufFile,
    device: &impl Device,
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

/// Derive gemma4's pre-router rmsnorm weight from the on-disk
/// `ffn_gate_inp.scale` F32 array: `weight = scale * 1 / sqrt(hidden)`,
/// cast F16, then HtoD. Mirrors legacy
/// `crates/models/gemma4/src/weights_hip.rs` upload path.
pub fn upload_gemma4_pre_router_weight_f16(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    hidden: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let scale_info = file.info(name).with_context(|| format!("info {name}"))?;
    if scale_info.dtype != flambeau_quant::GgmlDType::F32 {
        bail!("{name}: expected F32, got {:?}", scale_info.dtype);
    }
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    if raw.len() < hidden * 4 {
        bail!(
            "{name}: raw bytes {} < expected {} (hidden={hidden})",
            raw.len(),
            hidden * 4
        );
    }
    let scale_f32: &[f32] = bytemuck::cast_slice(&raw[..hidden * 4]);
    let inv_sqrt = 1.0f32 / (hidden as f32).sqrt();
    let host: Vec<f16> = scale_f32
        .iter()
        .map(|&v| f16::from_f32(v * inv_sqrt))
        .collect();
    let bytes = hidden * 2;
    let ptr = device.alloc(bytes).context("alloc pre_router_weight")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns `bytes`; host has the matching byte count.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after pre_router_weight upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F16>::from_raw(ptr, hidden) })
}

/// HtoD a host-dequantised tensor as `Tensor<F32>` (no quantise
/// step). Used for F32 scalars some arches store on disk (GDN's
/// `ssm_dt_bias`, `ssm_a`, `ssm_conv1d`).
pub fn upload_f32_tensor(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    expected_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != expected_elems {
        bail!(
            "{name}: dequant produced {} elems, expected {expected_elems}",
            f32_vec.len()
        );
    }
    let bytes = f32_vec.len() * 4;
    let ptr = device.alloc(bytes).context("alloc F32")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns `bytes`; f32_vec has the matching byte count.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(f32_vec.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after F32 upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F32>::from_raw(ptr, expected_elems) })
}

/// Quantise an F32 buffer to Q8_0 (32-elem blocks). The buffer must
/// be a multiple of 32 elements long.
pub(crate) fn f32_to_q8_0_bytes(name: &str, f32_buf: &[f32]) -> Result<Vec<u8>> {
    if f32_buf.len() % 32 != 0 {
        bail!(
            "{name}: Q8_0 fallback needs len %% 32 == 0 (got {})",
            f32_buf.len()
        );
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(f32_buf.len() / 32 * 34);
    let mut cursor = 0usize;
    while cursor < f32_buf.len() {
        flambeau_quant::quantize_k::quantize_row_q8_0(&f32_buf[cursor..cursor + 32], &mut bytes);
        cursor += 32;
    }
    Ok(bytes)
}
