//! Device-resident weights for Gemma 4. Mirrors `qwen3-moe::weights`
//! (one alloc per tensor, dtype stays GGUF-native) but trimmed to
//! the Gemma 4 surface: full-attn + dense FFN today; MoE expert
//! split + per-layer-embed land alongside S6.

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_blocks::WeightHandle;
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::{GgmlDType, GgufFile, TensorInfo};
use half::f16;

use crate::config::Gemma4Config;
use crate::layer::Gemma4LayerWeights;
use crate::layout::{FfnKind, ModelLayout};
use crate::moe::Gemma4MoeFfnWeights;
use crate::names::{AttnNames, DenseFfnNames, GlobalNames, MoeFfnNames, PerLayerEmbedNames};
use crate::per_layer_embd::PerLayerEmbedLayerWeights;
use crate::weights::resolve_weights;
use flambeau_blocks::{Activation, MoeExperts};

/// One device-resident tensor. Owns its allocation; freed by
/// [`Gemma4DeviceWeights::dispose`].
#[derive(Debug, Clone, Copy)]
pub struct DeviceTensor {
    pub ptr: DevicePtr,
    pub dtype: GgmlDType,
    pub bytes: usize,
}

impl DeviceTensor {
    fn is_live(&self) -> bool {
        !self.ptr.is_null() && self.bytes > 0
    }

    /// Convert to a [`WeightHandle`] for the blocks API. `dims` is
    /// the resolver's `[out_rows, hidden]` (or analogous) shape.
    pub fn as_weight_handle(&self, dims: [usize; 2]) -> Result<WeightHandle> {
        Ok(WeightHandle {
            ptr: self.ptr,
            dtype: ggml_to_qdtype(self.dtype)?,
            dims,
        })
    }
}

fn ggml_to_qdtype(d: GgmlDType) -> Result<QDtype> {
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
        other => bail!("unsupported GGML dtype for Gemma 4 weight: {other:?}"),
    })
}

/// Per-layer-embd global tensors (E2B / E4B only). Each is uploaded
/// once at session init; `build_inp_per_layer_table` reads the mmap
/// directly per token, but the device copies stay here for future
/// on-device build paths and for the `raw_tensors` dispose list.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedDeviceGlobals {
    pub per_layer_token_embd: DeviceTensor,
    pub per_layer_model_proj: DeviceTensor,
    pub per_layer_proj_norm: DeviceTensor,
}

/// Device-resident weights. Wraps a `Vec<Gemma4LayerWeights>` plus
/// global tensors (token_embd, output_norm, output, and the optional
/// per-layer-embd globals for E2B/E4B variants).
pub struct Gemma4DeviceWeights {
    pub token_embd: DeviceTensor,
    pub token_embd_dims: [usize; 2],
    pub output_norm: DeviceTensor,
    pub output: Option<DeviceTensor>,
    pub layers: Vec<Gemma4LayerWeights>,
    /// `Some` for E2B / E4B variants (`cfg.per_layer_embed.is_some()`).
    /// `None` for 26B-A4B / 31B (no side-channel embedding).
    pub per_layer_embd_globals: Option<PerLayerEmbedDeviceGlobals>,
    /// Raw device tensors held for `dispose()`. Mirrors
    /// `qwen3-moe::ModelWeights::iter_tensors_mut` but kept as a flat
    /// list to keep the upload-time bookkeeping simple.
    pub raw_tensors: Vec<DeviceTensor>,
    pub total_bytes: usize,
    pub device_id: i32,
    disposed: bool,
}

impl Gemma4DeviceWeights {
    /// Build a `Gemma4DeviceWeights` from pre-allocated device buffers
    /// for tests. The caller is responsible for keeping `raw_tensors`
    /// consistent with the per-layer `Gemma4LayerWeights` (every
    /// device pointer the layer references must appear in
    /// `raw_tensors` so `dispose()` frees it).
    pub fn from_pieces(
        token_embd: DeviceTensor,
        token_embd_dims: [usize; 2],
        output_norm: DeviceTensor,
        output: Option<DeviceTensor>,
        layers: Vec<Gemma4LayerWeights>,
        raw_tensors: Vec<DeviceTensor>,
        device_id: i32,
    ) -> Self {
        let total_bytes = raw_tensors.iter().map(|t| t.bytes).sum();
        Self {
            token_embd,
            token_embd_dims,
            output_norm,
            output,
            layers,
            per_layer_embd_globals: None,
            raw_tensors,
            total_bytes,
            device_id,
            disposed: false,
        }
    }
}

/// Cast an F32 norm tensor on host to F16, upload it, free the
/// original F32 buffer. Mirrors `qwen3-moe::cast_one_norm`. Caller
/// passes a freshly-uploaded F32 `DeviceTensor`; this returns the
/// new F16 tensor (same elem count, half the bytes) and the old F32
/// allocation is deallocated.
#[allow(dead_code)] // kept available for callers that upload + cast in two steps
fn cast_f32_norm_to_f16(
    file: &GgufFile,
    info: &TensorInfo,
    device: &HipDevice,
    f32_tensor: DeviceTensor,
) -> Result<DeviceTensor> {
    if f32_tensor.dtype != GgmlDType::F32 {
        // Already cast or never was F32 — return as-is.
        return Ok(f32_tensor);
    }
    let stream = device.default_stream();
    let elems: usize = info.dims.iter().product::<u64>() as usize;
    let raw = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if raw.len() < elems * 4 {
        bail!(
            "cast_f32_norm_to_f16 `{}`: mmap slice {} < expected {}",
            info.name,
            raw.len(),
            elems * 4
        );
    }
    // SAFETY: F32 dtype + page-aligned mmap = 4-byte alignment OK.
    let src: &[f32] = bytemuck::cast_slice(&raw[..elems * 4]);
    let host: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();
    let new_bytes = elems * 2;
    let new_ptr = device
        .alloc(new_bytes)
        .map_err(|e| anyhow!("alloc F16 norm `{}`: {e}", info.name))?;
    // SAFETY: new_ptr owns new_bytes; host outlives the bounded sync.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                new_ptr,
                DevicePtr(host.as_ptr() as usize),
                new_bytes,
            )
            .map_err(|e| anyhow!("memcpy F32→F16 `{}`: {e}", info.name))?;
    }
    stream.synchronize()?;
    // Free the old F32 buffer.
    // SAFETY: f32_tensor.ptr came from device.alloc above (in upload_one).
    unsafe {
        let _ = device.dealloc(f32_tensor.ptr, f32_tensor.bytes);
    }
    Ok(DeviceTensor {
        ptr: new_ptr,
        dtype: GgmlDType::F16,
        bytes: new_bytes,
    })
}

/// Upload the MoE-specific weights for one layer (26B-A4B). Splits the
/// fused `ffn_gate_up_exps` host-side into separate gate/up device
/// slabs (the indexed-MMVQ kernels expect contiguous per-expert
/// blocks), folds the per-channel `ffn_gate_inp.scale` and the
/// `1/sqrt(hidden)` router pre-scalar into a single F16 `[hidden]`
/// `pre_router_weight` (fed as the rmsnorm weight in the composer),
/// and uploads the three extra MoE norms with the standard F32→F16
/// cast.
///
/// Known gaps tracked under follow-ups:
/// - Router gating uses `RouterNormalize::TopkRenorm` (softmax-of-topk)
///   pending the `softmax_topk_f32` kernel. Output is finite + plausible
///   but not bit-exact vs llama.cpp.
/// - The per-expert `ffn_down_exps.scale` post-multiply is not yet
///   applied — pending the dedicated kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_moe_layer(
    file: &GgufFile,
    layer_index: usize,
    hidden: usize,
    moe_dims: crate::config::MoeDims,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    raw_tensors: &mut Vec<DeviceTensor>,
    total_bytes: &mut usize,
) -> Result<Gemma4MoeFfnWeights> {
    let names = MoeFfnNames::for_layer(layer_index);
    let n_experts = moe_dims.num_experts;
    let n_ff_exp = moe_dims.moe_intermediate_size;
    let top_k = moe_dims.num_experts_per_tok;

    // 1. Router weight (F32 [n_experts, hidden]).
    let router_info = file
        .tensors
        .get(&names.ffn_gate_inp)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_gate_inp))?;
    if router_info.dtype != GgmlDType::F32 {
        bail!(
            "{} expected F32, got {:?}",
            names.ffn_gate_inp,
            router_info.dtype
        );
    }
    if router_info.dims != [n_experts as u64, hidden as u64] {
        bail!(
            "{} dims {:?} != [{}, {}]",
            names.ffn_gate_inp,
            router_info.dims,
            n_experts,
            hidden
        );
    }
    let router = raw_upload(file, router_info, device, stream, raw_tensors, total_bytes)?;

    // 2. Pre-router weight: `(1/sqrt(hidden)) * ffn_gate_inp.scale`, cast F16.
    let scale_info = file
        .tensors
        .get(&names.ffn_gate_inp_scale)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_gate_inp_scale))?;
    if scale_info.dtype != GgmlDType::F32 {
        bail!(
            "{} expected F32, got {:?}",
            names.ffn_gate_inp_scale,
            scale_info.dtype
        );
    }
    if scale_info.dims != [hidden as u64] {
        bail!(
            "{} dims {:?} != [{}]",
            names.ffn_gate_inp_scale,
            scale_info.dims,
            hidden
        );
    }
    let scale_raw = file
        .tensor_raw(&scale_info.name)
        .with_context(|| format!("tensor_raw `{}`", scale_info.name))?;
    let scale_f32: &[f32] = bytemuck::cast_slice(&scale_raw[..hidden * 4]);
    let inv_sqrt = 1.0f32 / (hidden as f32).sqrt();
    let pre_router_host: Vec<f16> = scale_f32
        .iter()
        .map(|&v| f16::from_f32(v * inv_sqrt))
        .collect();
    let pre_router_bytes = hidden * 2;
    let pre_router_ptr = device
        .alloc(pre_router_bytes)
        .map_err(|e| anyhow!("alloc pre_router_weight: {e}"))?;
    // SAFETY: pre_router_ptr owns `pre_router_bytes`; host outlives the bounded sync.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                pre_router_ptr,
                DevicePtr(pre_router_host.as_ptr() as usize),
                pre_router_bytes,
            )
            .map_err(|e| anyhow!("memcpy pre_router_weight: {e}"))?;
    }
    stream.synchronize()?;
    drop(pre_router_host);
    *total_bytes += pre_router_bytes;
    raw_tensors.push(DeviceTensor {
        ptr: pre_router_ptr,
        dtype: GgmlDType::F16,
        bytes: pre_router_bytes,
    });

    // 3. Split fused `ffn_gate_up_exps` (Q8_0 [n_experts, 2*n_ff_exp, hidden])
    //    into separate gate / up device slabs.
    let fused_info = file
        .tensors
        .get(&names.ffn_gate_up_exps)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_gate_up_exps))?;
    if fused_info.dims
        != [n_experts as u64, (2 * n_ff_exp) as u64, hidden as u64]
    {
        bail!(
            "{} dims {:?} != [{}, {}, {}]",
            names.ffn_gate_up_exps,
            fused_info.dims,
            n_experts,
            2 * n_ff_exp,
            hidden
        );
    }
    let (gate_ptr, up_ptr, gate_up_bytes_each) = split_fused_gate_up(
        file,
        fused_info,
        n_experts,
        n_ff_exp,
        hidden,
        device,
        stream,
        raw_tensors,
        total_bytes,
    )?;

    // 4. Down experts (`Q8_0 [n_experts, hidden, n_ff_exp]`).
    let down_info = file
        .tensors
        .get(&names.ffn_down_exps)
        .ok_or_else(|| anyhow!("{} missing", names.ffn_down_exps))?;
    if down_info.dims != [n_experts as u64, hidden as u64, n_ff_exp as u64] {
        bail!(
            "{} dims {:?} != [{}, {}, {}]",
            names.ffn_down_exps,
            down_info.dims,
            n_experts,
            hidden,
            n_ff_exp
        );
    }
    let down = raw_upload(file, down_info, device, stream, raw_tensors, total_bytes)?;

    // 5. Three extra MoE norms (F32→F16 cast).
    let pre_ffw_norm_2_info = file
        .tensors
        .get(&names.pre_ffw_norm_2)
        .ok_or_else(|| anyhow!("{} missing", names.pre_ffw_norm_2))?;
    let post_ffw_norm_1_info = file
        .tensors
        .get(&names.post_ffw_norm_1)
        .ok_or_else(|| anyhow!("{} missing", names.post_ffw_norm_1))?;
    let post_ffw_norm_2_info = file
        .tensors
        .get(&names.post_ffw_norm_2)
        .ok_or_else(|| anyhow!("{} missing", names.post_ffw_norm_2))?;
    let pre_ffw_norm_2 = norm_upload_f32_to_f16(
        file, pre_ffw_norm_2_info, hidden, device, stream, raw_tensors, total_bytes,
    )?;
    let post_ffw_norm_1 = norm_upload_f32_to_f16(
        file, post_ffw_norm_1_info, hidden, device, stream, raw_tensors, total_bytes,
    )?;
    let post_ffw_norm_2 = norm_upload_f32_to_f16(
        file, post_ffw_norm_2_info, hidden, device, stream, raw_tensors, total_bytes,
    )?;

    // 6. Build WeightHandles + MoeExperts.
    let router_handle = router.as_weight_handle([n_experts, hidden])?;
    let gate_handle = WeightHandle {
        ptr: gate_ptr,
        dtype: ggml_to_qdtype(fused_info.dtype)?,
        dims: [n_experts * n_ff_exp, hidden],
    };
    let up_handle = WeightHandle {
        ptr: up_ptr,
        dtype: ggml_to_qdtype(fused_info.dtype)?,
        dims: [n_experts * n_ff_exp, hidden],
    };
    let down_handle = WeightHandle {
        ptr: down.ptr,
        dtype: ggml_to_qdtype(down_info.dtype)?,
        dims: [n_experts * hidden, n_ff_exp],
    };
    let _ = gate_up_bytes_each;

    let moe = MoeExperts::new(
        router_handle,
        gate_handle,
        up_handle,
        down_handle,
        hidden,
        n_ff_exp,
        n_experts,
        top_k,
    )
    .context("MoeExperts::new")?
    .with_activation(Activation::Gelu);
    // Router policy stays at default `TopkRenorm` (softmax-of-topk).
    // Gemma4 spec calls for `Softmax` (softmax-of-all → take top-k,
    // no renorm); that requires the `softmax_topk_f32` kernel which
    // is pending. Output is finite + plausible but not bit-exact vs
    // llama.cpp until the kernel lands.

    Ok(Gemma4MoeFfnWeights {
        moe,
        pre_router_weight_f16: pre_router_ptr,
        pre_ffw_norm_2: pre_ffw_norm_2.ptr,
        post_ffw_norm_1: post_ffw_norm_1.ptr,
        post_ffw_norm_2: post_ffw_norm_2.ptr,
    })
}

/// Raw HtoD upload of one GGUF tensor — kernel-free, no dtype cast.
/// Sister of the `upload_one` closure used inside `upload()`; exposed
/// here so [`upload_moe_layer`] can share the same allocation +
/// dispose-list bookkeeping.
fn raw_upload(
    file: &GgufFile,
    info: &TensorInfo,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    raw_tensors: &mut Vec<DeviceTensor>,
    total_bytes: &mut usize,
) -> Result<DeviceTensor> {
    let bytes = info.size_in_bytes() as usize;
    let data = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if data.len() < bytes {
        bail!(
            "tensor `{}` mmap slice {} < declared {}",
            info.name,
            data.len(),
            bytes
        );
    }
    let ptr = device
        .alloc(bytes)
        .map_err(|e| anyhow!("hipMalloc {} B for `{}`: {e}", bytes, info.name))?;
    // SAFETY: ptr is a fresh HIP alloc of `bytes`; data is mmap view ≥ bytes.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(data.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow!("memcpy_async `{}`: {e}", info.name))?;
    }
    *total_bytes += bytes;
    let t = DeviceTensor {
        ptr,
        dtype: info.dtype,
        bytes,
    };
    raw_tensors.push(t);
    Ok(t)
}

/// F32 norm tensor → F16 upload, picked from the inline `upload_norm`
/// closure for reuse by [`upload_moe_layer`].
#[allow(clippy::too_many_arguments)]
fn norm_upload_f32_to_f16(
    file: &GgufFile,
    info: &TensorInfo,
    expected_len: usize,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    raw_tensors: &mut Vec<DeviceTensor>,
    total_bytes: &mut usize,
) -> Result<DeviceTensor> {
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
    let data = file
        .tensor_raw(&info.name)
        .with_context(|| format!("tensor_raw `{}`", info.name))?;
    if data.len() < elems * 4 {
        bail!(
            "norm `{}` mmap slice {} < expected {}",
            info.name,
            data.len(),
            elems * 4
        );
    }
    // SAFETY: F32 + page-aligned mmap = 4-byte alignment OK.
    let src: &[f32] = bytemuck::cast_slice(&data[..elems * 4]);
    let host: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();
    let new_bytes = elems * 2;
    let ptr = device
        .alloc(new_bytes)
        .map_err(|e| anyhow!("alloc F16 norm `{}`: {e}", info.name))?;
    // SAFETY: ptr owns new_bytes; host outlives the bounded sync.
    unsafe {
        device
            .memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(host.as_ptr() as usize),
                new_bytes,
            )
            .map_err(|e| anyhow!("memcpy F32→F16 `{}`: {e}", info.name))?;
    }
    stream.synchronize()?;
    drop(host);
    *total_bytes += new_bytes;
    let t = DeviceTensor {
        ptr,
        dtype: GgmlDType::F16,
        bytes: new_bytes,
    };
    raw_tensors.push(t);
    Ok(t)
}

/// Host-side split of fused `ffn_gate_up_exps` (Q8_0 / Q8_1 / etc.)
/// into separate per-expert gate and up device buffers. The GGUF
/// layout is `[n_experts, 2 * n_ff_exp, hidden]` with the first
/// `n_ff_exp` rows per expert being gate and the next `n_ff_exp`
/// being up — but stored interleaved across experts. The indexed
/// MMVQ kernels need each expert's gate and up rows in their own
/// contiguous slab.
///
/// Returns `(gate_ptr, up_ptr, bytes_per_half_per_expert)`.
#[allow(clippy::too_many_arguments)]
fn split_fused_gate_up(
    file: &GgufFile,
    fused_info: &TensorInfo,
    n_experts: usize,
    n_ff_exp: usize,
    hidden: usize,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
    raw_tensors: &mut Vec<DeviceTensor>,
    total_bytes: &mut usize,
) -> Result<(DevicePtr, DevicePtr, usize)> {
    let row_bytes = row_bytes_for_dtype(fused_info.dtype, hidden).with_context(|| {
        format!(
            "row_bytes_for_dtype `{}` ({:?})",
            fused_info.name, fused_info.dtype
        )
    })?;
    let gate_bytes_per_expert = n_ff_exp * row_bytes;
    let up_bytes_per_expert = n_ff_exp * row_bytes;
    let fused_bytes_per_expert = gate_bytes_per_expert + up_bytes_per_expert;
    let total_per_half = n_experts * gate_bytes_per_expert;

    let fused_raw = file
        .tensor_raw(&fused_info.name)
        .with_context(|| format!("tensor_raw `{}`", fused_info.name))?;
    let expected_bytes = n_experts * fused_bytes_per_expert;
    if fused_raw.len() < expected_bytes {
        bail!(
            "{} mmap slice {} < expected {}",
            fused_info.name,
            fused_raw.len(),
            expected_bytes
        );
    }

    let gate_ptr = device
        .alloc(total_per_half)
        .map_err(|e| anyhow!("alloc gate_exps split: {e}"))?;
    let up_ptr = device
        .alloc(total_per_half)
        .map_err(|e| anyhow!("alloc up_exps split: {e}"))?;

    for e in 0..n_experts {
        let src_base = e * fused_bytes_per_expert;
        // SAFETY: gate_ptr owns total_per_half bytes; gate/up source slices
        // are bounded by the expected_bytes check above.
        unsafe {
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    DevicePtr(gate_ptr.0 + e * gate_bytes_per_expert),
                    DevicePtr(fused_raw.as_ptr() as usize + src_base),
                    gate_bytes_per_expert,
                )
                .map_err(|e| anyhow!("memcpy gate expert: {e}"))?;
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    DevicePtr(up_ptr.0 + e * up_bytes_per_expert),
                    DevicePtr(fused_raw.as_ptr() as usize + src_base + gate_bytes_per_expert),
                    up_bytes_per_expert,
                )
                .map_err(|e| anyhow!("memcpy up expert: {e}"))?;
        }
    }
    stream.synchronize()?;

    *total_bytes += 2 * total_per_half;
    raw_tensors.push(DeviceTensor {
        ptr: gate_ptr,
        dtype: fused_info.dtype,
        bytes: total_per_half,
    });
    raw_tensors.push(DeviceTensor {
        ptr: up_ptr,
        dtype: fused_info.dtype,
        bytes: total_per_half,
    });
    Ok((gate_ptr, up_ptr, gate_bytes_per_expert))
}

/// Bytes per row at the given dtype for a row of `cols` elements.
/// Q-quant rows are block-packed; `cols` must be a multiple of the
/// block size (32 for Q8_0/Q8_1 family, 256 for K-quants). F16/F32
/// scale linearly with element count.
fn row_bytes_for_dtype(dtype: GgmlDType, cols: usize) -> Result<usize> {
    match dtype {
        GgmlDType::F32 => Ok(cols * 4),
        GgmlDType::F16 => Ok(cols * 2),
        GgmlDType::Q8_0 => {
            if cols % 32 != 0 {
                bail!("Q8_0 cols {} not a multiple of 32", cols);
            }
            // Q8_0 block: 2-byte F16 scale + 32 i8.
            Ok((cols / 32) * 34)
        }
        GgmlDType::Q8_1 => {
            if cols % 32 != 0 {
                bail!("Q8_1 cols {} not a multiple of 32", cols);
            }
            Ok((cols / 32) * 36)
        }
        other => bail!(
            "row_bytes_for_dtype: dtype {:?} not yet handled for the fused-split path",
            other
        ),
    }
}

impl Gemma4DeviceWeights {
    pub fn upload(
        file: &GgufFile,
        cfg: &Gemma4Config,
        layout: &ModelLayout,
        device: &HipDevice,
    ) -> Result<Self> {
        device.bind()?;
        let stream = device.default_stream();
        let mut total_bytes = 0usize;
        let mut raw_tensors: Vec<DeviceTensor> = Vec::new();

        let resolved = resolve_weights(file, cfg, layout).context("resolve_weights")?;
        let has_per_layer_embed = cfg.per_layer_embed.is_some();
        for spec in &layout.layers {
            if spec.ffn_kind == FfnKind::Moe && cfg.moe.is_none() {
                bail!(
                    "Gemma4DeviceWeights::upload: layer {} ffn_kind=Moe but cfg.moe is None",
                    spec.index
                );
            }
            if !spec.has_kv && spec.kv_share_src.is_none() {
                bail!(
                    "Gemma4DeviceWeights::upload: shared-KV tail layer {} has no \
                     kv_share_src; caller must run `ModelLayout::resolve_kv_sharing()` \
                     before upload",
                    spec.index
                );
            }
        }

        let upload_one = |info: &TensorInfo,
                          raw: &mut Vec<DeviceTensor>,
                          total: &mut usize|
         -> Result<DeviceTensor> {
            let bytes = info.size_in_bytes() as usize;
            let data = file
                .tensor_raw(&info.name)
                .with_context(|| format!("tensor_raw `{}`", info.name))?;
            if data.len() < bytes {
                bail!(
                    "tensor `{}` mmap slice {} < declared {}",
                    info.name,
                    data.len(),
                    bytes
                );
            }
            let ptr = device.alloc(bytes).map_err(|e| {
                anyhow!("hipMalloc {} B for `{}`: {e}", bytes, info.name)
            })?;
            // SAFETY: ptr is a fresh HIP alloc of `bytes`; data is an
            // mmap view of ≥ bytes host bytes.
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ptr,
                        DevicePtr(data.as_ptr() as usize),
                        bytes,
                    )
                    .map_err(|e| anyhow!("memcpy_async `{}`: {e}", info.name))?;
            }
            *total += bytes;
            let t = DeviceTensor {
                ptr,
                dtype: info.dtype,
                bytes,
            };
            raw.push(t);
            Ok(t)
        };

        // Helper for norm tensors: upload, then if dtype was F32, cast
        // to F16 in place (gemma4 GGUFs store norms F32 but
        // `rmsnorm_f16` reads them as F16 — silent corruption otherwise).
        // Updates `raw_tensors` to point at the new F16 buffer so
        // `dispose()` frees the right allocation.
        let upload_norm = |info: &TensorInfo,
                           raw: &mut Vec<DeviceTensor>,
                           total: &mut usize|
         -> Result<DeviceTensor> {
            let bytes = info.size_in_bytes() as usize;
            let data = file
                .tensor_raw(&info.name)
                .with_context(|| format!("tensor_raw `{}`", info.name))?;
            if data.len() < bytes {
                bail!(
                    "norm `{}` mmap slice {} < declared {}",
                    info.name,
                    data.len(),
                    bytes
                );
            }
            if info.dtype == GgmlDType::F32 {
                // F32 → F16 cast at upload time. Single allocation,
                // single memcpy.
                let elems: usize = info.dims.iter().product::<u64>() as usize;
                if data.len() < elems * 4 {
                    bail!(
                        "norm `{}` mmap slice {} < expected {}",
                        info.name,
                        data.len(),
                        elems * 4
                    );
                }
                // SAFETY: F32 dtype + page-aligned mmap = 4-byte alignment OK.
                let src: &[f32] = bytemuck::cast_slice(&data[..elems * 4]);
                let host: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();
                let new_bytes = elems * 2;
                let ptr = device
                    .alloc(new_bytes)
                    .map_err(|e| anyhow!("alloc F16 norm `{}`: {e}", info.name))?;
                // SAFETY: ptr owns new_bytes; host outlives the bounded sync.
                unsafe {
                    device
                        .memcpy_async(
                            stream,
                            CopyDirection::HostToDevice,
                            ptr,
                            DevicePtr(host.as_ptr() as usize),
                            new_bytes,
                        )
                        .map_err(|e| anyhow!("memcpy F32→F16 `{}`: {e}", info.name))?;
                }
                *total += new_bytes;
                let t = DeviceTensor {
                    ptr,
                    dtype: GgmlDType::F16,
                    bytes: new_bytes,
                };
                raw.push(t);
                Ok(t)
            } else if info.dtype == GgmlDType::F16 {
                // Already F16; upload raw.
                let ptr = device
                    .alloc(bytes)
                    .map_err(|e| anyhow!("hipMalloc {} B for `{}`: {e}", bytes, info.name))?;
                // SAFETY: ptr is a fresh HIP alloc of `bytes`; data is an
                // mmap view of ≥ bytes host bytes.
                unsafe {
                    device
                        .memcpy_async(
                            stream,
                            CopyDirection::HostToDevice,
                            ptr,
                            DevicePtr(data.as_ptr() as usize),
                            bytes,
                        )
                        .map_err(|e| anyhow!("memcpy_async `{}`: {e}", info.name))?;
                }
                *total += bytes;
                let t = DeviceTensor {
                    ptr,
                    dtype: info.dtype,
                    bytes,
                };
                raw.push(t);
                Ok(t)
            } else {
                bail!(
                    "norm `{}`: unsupported dtype {:?} (expected F32 or F16)",
                    info.name,
                    info.dtype
                );
            }
        };

        let g_names = GlobalNames::default_names();
        let token_embd_info = file
            .tensors
            .get(&g_names.token_embd)
            .ok_or_else(|| anyhow!("token_embd missing"))?;
        let token_embd_dims = [
            token_embd_info.dims[0] as usize,
            token_embd_info.dims[1] as usize,
        ];
        let token_embd = upload_one(token_embd_info, &mut raw_tensors, &mut total_bytes)?;
        let output_norm = upload_norm(
            file.tensors
                .get(&g_names.output_norm)
                .ok_or_else(|| anyhow!("output_norm missing"))?,
            &mut raw_tensors,
            &mut total_bytes,
        )?;
        let output = if let Some(t) = file.tensors.get(&g_names.output) {
            Some(upload_one(t, &mut raw_tensors, &mut total_bytes)?)
        } else {
            None
        };

        // Per-layer-embd globals (E2B/E4B only).
        let per_layer_embd_globals = if has_per_layer_embed {
            let tokembd_info = file.tensors.get(&g_names.per_layer_token_embd).ok_or_else(|| {
                anyhow!("per_layer_token_embd missing despite cfg.per_layer_embed.is_some()")
            })?;
            let modelproj_info = file
                .tensors
                .get(&g_names.per_layer_model_proj)
                .ok_or_else(|| anyhow!("per_layer_model_proj missing"))?;
            let projnorm_info = file
                .tensors
                .get(&g_names.per_layer_proj_norm)
                .ok_or_else(|| anyhow!("per_layer_proj_norm missing"))?;
            let per_layer_token_embd =
                upload_one(tokembd_info, &mut raw_tensors, &mut total_bytes)?;
            let per_layer_model_proj =
                upload_one(modelproj_info, &mut raw_tensors, &mut total_bytes)?;
            let per_layer_proj_norm =
                upload_one(projnorm_info, &mut raw_tensors, &mut total_bytes)?;
            Some(PerLayerEmbedDeviceGlobals {
                per_layer_token_embd,
                per_layer_model_proj,
                per_layer_proj_norm,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for (i, spec) in layout.layers.iter().enumerate() {
            let _ = i;
            let an = AttnNames::for_layer(spec.index);
            let dn = DenseFfnNames::for_layer(spec.index);

            let attn_norm = upload_norm(
                file.tensors.get(&an.attn_norm).ok_or_else(|| anyhow!("{}", an.attn_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;
            let attn_q_info = file.tensors.get(&an.attn_q).ok_or_else(|| anyhow!("{}", an.attn_q))?;
            let attn_q = upload_one(attn_q_info, &mut raw_tensors, &mut total_bytes)?;
            let attn_q_dims = [
                attn_q_info.dims[0] as usize,
                attn_q_info.dims[1] as usize,
            ];

            // `attn_k` / `attn_v` / `attn_k_norm` are TENSOR_NOT_REQUIRED for
            // shared-KV tail layers (mirrors llama.cpp PR #21739); attn_v
            // is additionally always optional (alt-attention).
            let (attn_k, attn_k_dims) = if let Some(info) = file.tensors.get(&an.attn_k) {
                let dt = upload_one(info, &mut raw_tensors, &mut total_bytes)?;
                let dims = [info.dims[0] as usize, info.dims[1] as usize];
                (Some(dt), Some(dims))
            } else {
                if spec.has_kv {
                    bail!("layer {}: attn_k required but missing", spec.index);
                }
                (None, None)
            };

            let (attn_v, attn_v_dims) = if let Some(info) = file.tensors.get(&an.attn_v) {
                let dt = upload_one(info, &mut raw_tensors, &mut total_bytes)?;
                let dims = [info.dims[0] as usize, info.dims[1] as usize];
                (Some(dt), Some(dims))
            } else {
                (None, None)
            };

            let attn_output_info = file.tensors.get(&an.attn_output).ok_or_else(|| anyhow!("{}", an.attn_output))?;
            let attn_output = upload_one(attn_output_info, &mut raw_tensors, &mut total_bytes)?;
            let attn_output_dims = [
                attn_output_info.dims[0] as usize,
                attn_output_info.dims[1] as usize,
            ];

            let attn_q_norm = upload_norm(
                file.tensors.get(&an.attn_q_norm).ok_or_else(|| anyhow!("{}", an.attn_q_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;
            let attn_k_norm = if let Some(info) = file.tensors.get(&an.attn_k_norm) {
                Some(upload_norm(info, &mut raw_tensors, &mut total_bytes)?)
            } else {
                if spec.has_kv {
                    bail!("layer {}: attn_k_norm required but missing", spec.index);
                }
                None
            };
            let post_attention_norm = upload_norm(
                file.tensors
                    .get(&an.post_attention_norm)
                    .ok_or_else(|| anyhow!("{}", an.post_attention_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;
            // `layer_output_scale` is F32 [1]. Read its value host-side
            // (it's a constant during inference) so the layer composer
            // can apply it via `scale_f16` without an extra
            // broadcast-mul op.
            let layer_output_scale_value: Option<f32> =
                if let Some(info) = file.tensors.get(&an.layer_output_scale) {
                    if info.dtype != GgmlDType::F32 {
                        bail!(
                            "layer {}: layer_output_scale must be F32, got {:?}",
                            spec.index,
                            info.dtype
                        );
                    }
                    let raw = file
                        .tensor_raw(&info.name)
                        .with_context(|| format!("tensor_raw `{}`", info.name))?;
                    if raw.len() < 4 {
                        bail!("layer {}: layer_output_scale row < 4 bytes", spec.index);
                    }
                    Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
                } else {
                    None
                };

            // Dense FFN
            let ffn_norm = upload_norm(
                file.tensors.get(&dn.ffn_norm).ok_or_else(|| anyhow!("{}", dn.ffn_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;
            let ffn_gate_info = file.tensors.get(&dn.ffn_gate).ok_or_else(|| anyhow!("{}", dn.ffn_gate))?;
            let ffn_gate = upload_one(ffn_gate_info, &mut raw_tensors, &mut total_bytes)?;
            let ffn_gate_dims = [
                ffn_gate_info.dims[0] as usize,
                ffn_gate_info.dims[1] as usize,
            ];
            let ffn_up_info = file.tensors.get(&dn.ffn_up).ok_or_else(|| anyhow!("{}", dn.ffn_up))?;
            let ffn_up = upload_one(ffn_up_info, &mut raw_tensors, &mut total_bytes)?;
            let ffn_up_dims = [
                ffn_up_info.dims[0] as usize,
                ffn_up_info.dims[1] as usize,
            ];
            let ffn_down_info = file.tensors.get(&dn.ffn_down).ok_or_else(|| anyhow!("{}", dn.ffn_down))?;
            let ffn_down = upload_one(ffn_down_info, &mut raw_tensors, &mut total_bytes)?;
            let ffn_down_dims = [
                ffn_down_info.dims[0] as usize,
                ffn_down_info.dims[1] as usize,
            ];
            let post_ffw_norm = upload_norm(
                file.tensors.get(&dn.post_ffw_norm).ok_or_else(|| anyhow!("{}", dn.post_ffw_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;

            // Per-layer-embd weights (E2B/E4B only).
            let per_layer_embed = if has_per_layer_embed {
                let ple = PerLayerEmbedNames::for_layer(spec.index);
                let inp_gate_info = file
                    .tensors
                    .get(&ple.inp_gate)
                    .ok_or_else(|| anyhow!("{}", ple.inp_gate))?;
                let proj_info = file
                    .tensors
                    .get(&ple.proj)
                    .ok_or_else(|| anyhow!("{}", ple.proj))?;
                let post_norm_info = file
                    .tensors
                    .get(&ple.post_norm)
                    .ok_or_else(|| anyhow!("{}", ple.post_norm))?;
                let inp_gate = upload_one(inp_gate_info, &mut raw_tensors, &mut total_bytes)?;
                let proj = upload_one(proj_info, &mut raw_tensors, &mut total_bytes)?;
                let post_norm = upload_norm(post_norm_info, &mut raw_tensors, &mut total_bytes)?;
                Some(PerLayerEmbedLayerWeights {
                    inp_gate: inp_gate.ptr,
                    proj: proj.ptr,
                    post_norm_f16: post_norm.ptr,
                })
            } else {
                None
            };

            let attn_k_handle = if let (Some(dt), Some(dims)) = (attn_k, attn_k_dims) {
                Some(dt.as_weight_handle(dims)?)
            } else {
                None
            };
            let attn_v_handle = if let (Some(dt), Some(dims)) = (attn_v, attn_v_dims) {
                Some(dt.as_weight_handle(dims)?)
            } else {
                None
            };
            let moe = if spec.ffn_kind == FfnKind::Moe {
                let moe_dims = cfg.moe.expect("MoE layer with cfg.moe=None already rejected");
                Some(upload_moe_layer(
                    file,
                    spec.index,
                    cfg.hidden_size,
                    moe_dims,
                    device,
                    stream,
                    &mut raw_tensors,
                    &mut total_bytes,
                )?)
            } else {
                None
            };

            layers.push(Gemma4LayerWeights {
                attn_norm: attn_norm.ptr,
                attn_q: attn_q.as_weight_handle(attn_q_dims)?,
                attn_k: attn_k_handle,
                attn_v: attn_v_handle,
                attn_output: attn_output.as_weight_handle(attn_output_dims)?,
                attn_q_norm: attn_q_norm.ptr,
                attn_k_norm: attn_k_norm.map(|dt| dt.ptr),
                post_attention_norm: post_attention_norm.ptr,
                layer_output_scale: layer_output_scale_value,
                ffn_norm: ffn_norm.ptr,
                ffn_gate: ffn_gate.as_weight_handle(ffn_gate_dims)?,
                ffn_up: ffn_up.as_weight_handle(ffn_up_dims)?,
                ffn_down: ffn_down.as_weight_handle(ffn_down_dims)?,
                post_ffw_norm: post_ffw_norm.ptr,
                per_layer_embed,
                moe,
            });
        }

        // One end-of-upload sync — cheaper than per-tensor.
        stream.synchronize()?;

        // Drop the resolver shape table; downstream code uses the per-layer
        // `Gemma4LayerWeights` we just built.
        let _ = resolved;

        Ok(Self {
            token_embd,
            token_embd_dims,
            output_norm,
            output,
            layers,
            per_layer_embd_globals,
            raw_tensors,
            total_bytes,
            device_id: device.id(),
            disposed: false,
        })
    }

    /// Free every device allocation. Idempotent.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut out = Ok(());
        for t in &self.raw_tensors {
            if !t.is_live() {
                continue;
            }
            // SAFETY: every pointer came from `device.alloc(bytes)` above;
            // no aliasing; no outstanding stream work (caller contract).
            unsafe {
                if let Err(e) = device.dealloc(t.ptr, t.bytes) {
                    if out.is_ok() {
                        out = Err(anyhow!("hipFree: {e}"));
                    }
                }
            }
        }
        self.raw_tensors.clear();
        out
    }
}

impl Drop for Gemma4DeviceWeights {
    fn drop(&mut self) {
        if !self.disposed && !self.raw_tensors.is_empty() {
            tracing::warn!(
                "Gemma4DeviceWeights dropped without dispose(); {} tensors leaked on device {}",
                self.raw_tensors.len(),
                self.device_id
            );
        }
    }
}
