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

use crate::config::Gemma4Config;
use crate::layer::Gemma4LayerWeights;
use crate::layout::{FfnKind, ModelLayout};
use crate::names::{AttnNames, DenseFfnNames, GlobalNames};
use crate::weights::resolve_weights;

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

/// Device-resident weights. Wraps a `Vec<Gemma4LayerWeights>` plus
/// global tensors (token_embd, output_norm, output). Per-layer-embed
/// (E2B/E4B) lands separately when S5-B-2 wires it.
pub struct Gemma4DeviceWeights {
    pub token_embd: DeviceTensor,
    pub token_embd_dims: [usize; 2],
    pub output_norm: DeviceTensor,
    pub output: Option<DeviceTensor>,
    pub layers: Vec<Gemma4LayerWeights>,
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
            raw_tensors,
            total_bytes,
            device_id,
            disposed: false,
        }
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

        // Reject MoE + per-layer-embed for S5-B-1; those land alongside S6.
        let resolved = resolve_weights(file, cfg, layout).context("resolve_weights")?;
        if cfg.moe.is_some() {
            bail!(
                "Gemma4DeviceWeights::upload: MoE variants not supported in S5-B-1; \
                 see S6 for the indexed-experts upload + router-policy wiring"
            );
        }
        if cfg.per_layer_embed.is_some() {
            bail!(
                "Gemma4DeviceWeights::upload: per-layer side-channel embedding (E2B/E4B) \
                 not supported in S5-B-1; see S5-B-2 follow-up"
            );
        }
        for spec in &layout.layers {
            if spec.ffn_kind != FfnKind::Dense {
                bail!(
                    "Gemma4DeviceWeights::upload: MoE FFN (layer {}) not supported (S6-B)",
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
        let output_norm = upload_one(
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

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for (i, spec) in layout.layers.iter().enumerate() {
            let _ = i;
            let an = AttnNames::for_layer(spec.index);
            let dn = DenseFfnNames::for_layer(spec.index);

            let attn_norm = upload_one(
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

            let attn_q_norm = upload_one(
                file.tensors.get(&an.attn_q_norm).ok_or_else(|| anyhow!("{}", an.attn_q_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;
            let attn_k_norm = if let Some(info) = file.tensors.get(&an.attn_k_norm) {
                Some(upload_one(info, &mut raw_tensors, &mut total_bytes)?)
            } else {
                if spec.has_kv {
                    bail!("layer {}: attn_k_norm required but missing", spec.index);
                }
                None
            };
            let post_attention_norm = upload_one(
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
            let ffn_norm = upload_one(
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
            let post_ffw_norm = upload_one(
                file.tensors.get(&dn.post_ffw_norm).ok_or_else(|| anyhow!("{}", dn.post_ffw_norm))?,
                &mut raw_tensors,
                &mut total_bytes,
            )?;

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
                // Per-layer-embd weights upload is part of the
                // real-GGUF integration follow-up; this `upload()`
                // path already bails earlier when `cfg.per_layer_embed
                // .is_some()` (see top of the function). Synthetic
                // fixtures construct Gemma4LayerWeights directly and
                // set this field as needed.
                per_layer_embed: None,
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
