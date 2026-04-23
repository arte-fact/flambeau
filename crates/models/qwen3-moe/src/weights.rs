#![expect(
    clippy::redundant_closure,
    reason = "the `.map(|t| upload(t))` pattern keeps the FnMut borrow structure explicit \
              across a dozen adjacent upload sites; clippy's `.map(&mut upload)` rewrite \
              adds a reborrow dance with no readability win"
)]
//! Device-resident weight tensors for a Qwen3.x MoE model.
//!
//! Upload path: [`ModelWeights::upload`] walks a [`ModelLayout`] and
//! transfers every required tensor from the GGUF mmap to HIP global memory
//! via `memcpy_async(HostToDevice)`, allocating a fresh `DevicePtr` per
//! tensor. The structs mirror the layout variants 1:1 so model code can
//! pattern-match without an extra indirection.
//!
//! Memory layout decisions:
//! - **One allocation per tensor.** Matches candle's pattern and makes it
//!   trivial to cert per-tensor uploads. A future "one big arena" change
//!   would only need to rewrite this file.
//! - **Dtype stays whatever the GGUF says.** No on-upload requantisation.
//!   Q4_K weights stay Q4_K; F16 norms stay F16; the Q8_0 ssm_alpha /
//!   ssm_beta that Candle requantises on load (candle P20) will be a
//!   future optimisation, not part of V1.7.3-a.
//! - **Dealloc via [`ModelWeights::dispose`].** Drop alone can't get a
//!   device handle, so the scaffold deliberately surfaces the teardown
//!   call instead of silently leaking.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::{GgmlDType, GgufFile};

use crate::layout::{
    DenseAttnTensors, FullAttnTensors, GdnTensors, LayerAttnBlock, LayerDescriptor, ModelLayout,
    MoeFfnTensors, ResolvedTensor, SharedExpertTensors,
};

/// One device-resident tensor. Points into a `hipMalloc` allocation owned
/// by [`ModelWeights`]; dtype + dims are copied from the GGUF index.
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub ptr: DevicePtr,
    pub dtype: GgmlDType,
    pub dims: Vec<u64>,
    pub bytes: usize,
    /// GGUF tensor name. Kept for diagnostics + layout-diff reporting.
    pub name: Arc<str>,
}

impl DeviceTensor {
    fn is_live(&self) -> bool {
        !self.ptr.is_null() && self.bytes > 0
    }
}

#[derive(Debug, Clone)]
pub struct DenseAttnWeights {
    pub attn_q: DeviceTensor,
    pub attn_k: DeviceTensor,
    pub attn_v: DeviceTensor,
    pub attn_output: DeviceTensor,
    pub attn_q_norm: DeviceTensor,
    pub attn_k_norm: DeviceTensor,
    pub attn_q_bias: Option<DeviceTensor>,
    pub attn_k_bias: Option<DeviceTensor>,
    pub attn_v_bias: Option<DeviceTensor>,
}

#[derive(Debug, Clone)]
pub struct FullAttnWeights {
    pub attn_q: DeviceTensor,
    pub attn_k: DeviceTensor,
    pub attn_v: DeviceTensor,
    pub attn_output: DeviceTensor,
    pub attn_q_norm: DeviceTensor,
    pub attn_k_norm: DeviceTensor,
}

#[derive(Debug, Clone)]
pub struct GdnWeights {
    pub attn_qkv: DeviceTensor,
    pub attn_gate: DeviceTensor,
    pub ssm_alpha: Option<DeviceTensor>,
    pub ssm_beta: Option<DeviceTensor>,
    pub ssm_ba: Option<DeviceTensor>,
    pub ssm_a: DeviceTensor,
    pub ssm_dt_bias: DeviceTensor,
    pub ssm_conv1d: DeviceTensor,
    pub ssm_norm: DeviceTensor,
    pub ssm_out: DeviceTensor,
}

/// One-of-three attention-block weight sets, matching [`LayerAttnBlock`].
#[derive(Debug, Clone)]
pub enum AttnWeights {
    Dense(DenseAttnWeights),
    FullAttn(FullAttnWeights),
    Gdn(GdnWeights),
}

#[derive(Debug, Clone)]
pub struct SharedExpertWeights {
    pub ffn_gate_inp_shexp: DeviceTensor,
    pub ffn_gate_shexp: DeviceTensor,
    pub ffn_up_shexp: DeviceTensor,
    pub ffn_down_shexp: DeviceTensor,
}

#[derive(Debug, Clone)]
pub struct DenseFfnWeights {
    pub ffn_gate: DeviceTensor,
    pub ffn_up: DeviceTensor,
    pub ffn_down: DeviceTensor,
}

/// Per-layer FFN weights. Mirrors [`MoeFfnTensors`] — mutually exclusive
/// MoE vs dense branches, picked by `cfg.is_dense_ffn()`. MoE fields are
/// `Some` iff `dense.is_none()` and vice versa.
#[derive(Debug, Clone)]
pub struct FfnWeights {
    pub ffn_gate_inp: Option<DeviceTensor>,
    pub ffn_gate_exps: Option<DeviceTensor>,
    pub ffn_up_exps: Option<DeviceTensor>,
    pub ffn_down_exps: Option<DeviceTensor>,
    pub shared: Option<SharedExpertWeights>,
    pub dense: Option<DenseFfnWeights>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub layer_idx: usize,
    pub attn_norm: DeviceTensor,
    pub post_attention_norm: Option<DeviceTensor>,
    pub ffn_norm: Option<DeviceTensor>,
    pub attn: AttnWeights,
    pub ffn: FfnWeights,
}

/// All device-resident model weights. Ownership of every `DevicePtr` inside
/// this struct belongs to it — call [`ModelWeights::dispose`] once inference
/// is done to free them.
#[derive(Debug)]
pub struct ModelWeights {
    pub token_embd: DeviceTensor,
    pub output_norm: DeviceTensor,
    pub output: Option<DeviceTensor>,
    pub layers: Vec<LayerWeights>,
    device_id: i32,
    total_bytes: usize,
    // Set to false by `dispose` to suppress the leak warning in Drop.
    disposed: bool,
}

impl ModelWeights {
    /// Build a `ModelWeights` from already-allocated device tensors. Skips
    /// the GGUF upload path — intended for synthetic-model smoke tests
    /// (V1.7.3-e5, forward_one_token). Callers are responsible for
    /// uploading consistent weight data; `total_bytes` is computed by
    /// summing each `DeviceTensor.bytes`.
    pub fn from_parts(
        device_id: i32,
        token_embd: DeviceTensor,
        output_norm: DeviceTensor,
        output: Option<DeviceTensor>,
        layers: Vec<LayerWeights>,
    ) -> Self {
        let mut this = Self {
            token_embd,
            output_norm,
            output,
            layers,
            device_id,
            total_bytes: 0,
            disposed: false,
        };
        this.total_bytes = this.iter_tensors().map(|t| t.bytes).sum();
        this
    }

    /// HIP device id this weight set was uploaded to.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Total bytes consumed on device across every tensor in the set.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Upload every tensor referenced by `layout` from the GGUF mmap to
    /// `device`. Synchronises the default stream before returning so the
    /// caller can rely on the bytes being resident.
    pub fn upload(file: &GgufFile, layout: &ModelLayout, device: &HipDevice) -> Result<Self> {
        device.bind()?;
        let stream = device.default_stream();
        let mut total_bytes = 0usize;

        // Helper: upload one `ResolvedTensor` and hand back a `DeviceTensor`.
        // Uses a closure over the shared state so per-tensor failure mode
        // can bail with the offending name in the error message.
        let mut upload_one = |r: &ResolvedTensor| -> Result<DeviceTensor> {
            let bytes = r.size_bytes as usize;
            let raw = file
                .tensor_raw(&r.name)
                .with_context(|| format!("tensor_raw `{}`", r.name))?;
            if raw.len() < bytes {
                bail!(
                    "tensor `{}` mmap slice {} < declared size {}",
                    r.name,
                    raw.len(),
                    bytes
                );
            }
            let ptr = device
                .alloc(bytes)
                .map_err(|e| anyhow!("hipMalloc {} B for `{}`: {e}", bytes, r.name))?;
            // SAFETY: `ptr` is a fresh HIP allocation of `bytes` bytes;
            // `raw` is an mmap view of at least `bytes` host bytes.
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ptr,
                        DevicePtr(raw.as_ptr() as usize),
                        bytes,
                    )
                    .map_err(|e| anyhow!("memcpy_async for `{}`: {e}", r.name))?;
            }
            total_bytes += bytes;
            Ok(DeviceTensor {
                ptr,
                dtype: r.dtype,
                dims: r.dims.clone(),
                bytes,
                name: Arc::from(r.name.as_str()),
            })
        };

        let token_embd = upload_one(&layout.token_embd)?;
        let output_norm = upload_one(&layout.output_norm)?;
        let output = layout
            .output
            .as_ref()
            .map(&mut upload_one)
            .transpose()?;

        let mut layers = Vec::with_capacity(layout.layers.len());
        for desc in &layout.layers {
            layers.push(upload_layer(desc, &mut upload_one)?);
        }

        // One synchronize at the end covers the full upload — cheaper than
        // per-tensor syncs and still guarantees residency before the caller
        // starts issuing forward-pass kernels.
        stream.synchronize()?;

        Ok(Self {
            token_embd,
            output_norm,
            output,
            layers,
            device_id: device.id(),
            total_bytes,
            disposed: false,
        })
    }

    /// Free every device allocation in the weight set. Mandatory — `Drop`
    /// can't get a device handle and will log a warn if the caller skips
    /// this step. Idempotent.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut out = Ok(());
        for t in self.iter_tensors_mut() {
            if !t.is_live() {
                continue;
            }
            // SAFETY: every pointer came from `device.alloc(bytes)` above;
            // no aliasing; no outstanding stream work (caller contract).
            unsafe {
                if let Err(e) = device.dealloc(t.ptr, t.bytes) {
                    if out.is_ok() {
                        out = Err(anyhow!("hipFree for `{}`: {e}", t.name));
                    }
                }
            }
            t.ptr = DevicePtr::NULL;
            t.bytes = 0;
        }
        out
    }

    /// Iterator over every tensor in the set. Useful for byte-accounting
    /// and leak-detection tests.
    pub fn iter_tensors(&self) -> impl Iterator<Item = &DeviceTensor> + '_ {
        let globals = std::iter::once(&self.token_embd)
            .chain(std::iter::once(&self.output_norm))
            .chain(self.output.iter());
        let layers = self.layers.iter().flat_map(layer_iter);
        globals.chain(layers)
    }

    fn iter_tensors_mut(&mut self) -> impl Iterator<Item = &mut DeviceTensor> + '_ {
        let mut v: Vec<&mut DeviceTensor> = Vec::new();
        v.push(&mut self.token_embd);
        v.push(&mut self.output_norm);
        if let Some(o) = &mut self.output {
            v.push(o);
        }
        for l in &mut self.layers {
            v.push(&mut l.attn_norm);
            if let Some(t) = &mut l.post_attention_norm {
                v.push(t);
            }
            if let Some(t) = &mut l.ffn_norm {
                v.push(t);
            }
            match &mut l.attn {
                AttnWeights::Dense(d) => {
                    v.extend(
                        [
                            &mut d.attn_q,
                            &mut d.attn_k,
                            &mut d.attn_v,
                            &mut d.attn_output,
                            &mut d.attn_q_norm,
                            &mut d.attn_k_norm,
                        ]
                        .into_iter(),
                    );
                    if let Some(b) = &mut d.attn_q_bias {
                        v.push(b);
                    }
                    if let Some(b) = &mut d.attn_k_bias {
                        v.push(b);
                    }
                    if let Some(b) = &mut d.attn_v_bias {
                        v.push(b);
                    }
                }
                AttnWeights::FullAttn(f) => {
                    v.extend(
                        [
                            &mut f.attn_q,
                            &mut f.attn_k,
                            &mut f.attn_v,
                            &mut f.attn_output,
                            &mut f.attn_q_norm,
                            &mut f.attn_k_norm,
                        ]
                        .into_iter(),
                    );
                }
                AttnWeights::Gdn(g) => {
                    v.extend(
                        [
                            &mut g.attn_qkv,
                            &mut g.attn_gate,
                            &mut g.ssm_a,
                            &mut g.ssm_dt_bias,
                            &mut g.ssm_conv1d,
                            &mut g.ssm_norm,
                            &mut g.ssm_out,
                        ]
                        .into_iter(),
                    );
                    if let Some(t) = &mut g.ssm_alpha {
                        v.push(t);
                    }
                    if let Some(t) = &mut g.ssm_beta {
                        v.push(t);
                    }
                    if let Some(t) = &mut g.ssm_ba {
                        v.push(t);
                    }
                }
            }
            if let Some(t) = &mut l.ffn.ffn_gate_inp { v.push(t); }
            if let Some(t) = &mut l.ffn.ffn_gate_exps { v.push(t); }
            if let Some(t) = &mut l.ffn.ffn_up_exps { v.push(t); }
            if let Some(t) = &mut l.ffn.ffn_down_exps { v.push(t); }
            if let Some(s) = &mut l.ffn.shared {
                v.push(&mut s.ffn_gate_inp_shexp);
                v.push(&mut s.ffn_gate_shexp);
                v.push(&mut s.ffn_up_shexp);
                v.push(&mut s.ffn_down_shexp);
            }
            if let Some(d) = &mut l.ffn.dense {
                v.push(&mut d.ffn_gate);
                v.push(&mut d.ffn_up);
                v.push(&mut d.ffn_down);
            }
        }
        v.into_iter()
    }
}

fn layer_iter(l: &LayerWeights) -> Vec<&DeviceTensor> {
    let mut v = vec![&l.attn_norm];
    if let Some(t) = &l.post_attention_norm {
        v.push(t);
    }
    if let Some(t) = &l.ffn_norm {
        v.push(t);
    }
    match &l.attn {
        AttnWeights::Dense(d) => {
            v.extend([
                &d.attn_q,
                &d.attn_k,
                &d.attn_v,
                &d.attn_output,
                &d.attn_q_norm,
                &d.attn_k_norm,
            ]);
            v.extend(
                [&d.attn_q_bias, &d.attn_k_bias, &d.attn_v_bias]
                    .into_iter()
                    .flatten(),
            );
        }
        AttnWeights::FullAttn(f) => {
            v.extend([
                &f.attn_q,
                &f.attn_k,
                &f.attn_v,
                &f.attn_output,
                &f.attn_q_norm,
                &f.attn_k_norm,
            ]);
        }
        AttnWeights::Gdn(g) => {
            v.extend([
                &g.attn_qkv,
                &g.attn_gate,
                &g.ssm_a,
                &g.ssm_dt_bias,
                &g.ssm_conv1d,
                &g.ssm_norm,
                &g.ssm_out,
            ]);
            v.extend(
                [&g.ssm_alpha, &g.ssm_beta, &g.ssm_ba]
                    .into_iter()
                    .flatten(),
            );
        }
    }
    if let Some(t) = &l.ffn.ffn_gate_inp { v.push(t); }
    if let Some(t) = &l.ffn.ffn_gate_exps { v.push(t); }
    if let Some(t) = &l.ffn.ffn_up_exps { v.push(t); }
    if let Some(t) = &l.ffn.ffn_down_exps { v.push(t); }
    if let Some(d) = &l.ffn.dense {
        v.push(&d.ffn_gate);
        v.push(&d.ffn_up);
        v.push(&d.ffn_down);
    }
    if let Some(s) = &l.ffn.shared {
        v.extend([
            &s.ffn_gate_inp_shexp,
            &s.ffn_gate_shexp,
            &s.ffn_up_shexp,
            &s.ffn_down_shexp,
        ]);
    }
    v
}

impl Drop for ModelWeights {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::weights",
                device_id = self.device_id,
                bytes = self.total_bytes,
                layers = self.layers.len(),
                "ModelWeights dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

fn upload_layer<F>(desc: &LayerDescriptor, upload: &mut F) -> Result<LayerWeights>
where
    F: FnMut(&ResolvedTensor) -> Result<DeviceTensor>,
{
    let attn_norm = upload(&desc.attn_norm)?;
    let post_attention_norm = desc
        .post_attention_norm
        .as_ref()
        .map(|t| upload(t))
        .transpose()?;
    let ffn_norm = desc.ffn_norm.as_ref().map(|t| upload(t)).transpose()?;

    let attn = match &desc.attn {
        LayerAttnBlock::Dense(d) => AttnWeights::Dense(upload_dense_attn(d, upload)?),
        LayerAttnBlock::FullAttn(f) => AttnWeights::FullAttn(upload_full_attn(f, upload)?),
        LayerAttnBlock::Gdn(g) => AttnWeights::Gdn(upload_gdn(g, upload)?),
    };

    let ffn = upload_ffn(&desc.ffn, upload)?;

    Ok(LayerWeights {
        layer_idx: desc.layer_idx,
        attn_norm,
        post_attention_norm,
        ffn_norm,
        attn,
        ffn,
    })
}

fn upload_dense_attn<F>(d: &DenseAttnTensors, upload: &mut F) -> Result<DenseAttnWeights>
where
    F: FnMut(&ResolvedTensor) -> Result<DeviceTensor>,
{
    Ok(DenseAttnWeights {
        attn_q: upload(&d.attn_q)?,
        attn_k: upload(&d.attn_k)?,
        attn_v: upload(&d.attn_v)?,
        attn_output: upload(&d.attn_output)?,
        attn_q_norm: upload(&d.attn_q_norm)?,
        attn_k_norm: upload(&d.attn_k_norm)?,
        attn_q_bias: d.attn_q_bias.as_ref().map(|t| upload(t)).transpose()?,
        attn_k_bias: d.attn_k_bias.as_ref().map(|t| upload(t)).transpose()?,
        attn_v_bias: d.attn_v_bias.as_ref().map(|t| upload(t)).transpose()?,
    })
}

fn upload_full_attn<F>(f: &FullAttnTensors, upload: &mut F) -> Result<FullAttnWeights>
where
    F: FnMut(&ResolvedTensor) -> Result<DeviceTensor>,
{
    Ok(FullAttnWeights {
        attn_q: upload(&f.attn_q)?,
        attn_k: upload(&f.attn_k)?,
        attn_v: upload(&f.attn_v)?,
        attn_output: upload(&f.attn_output)?,
        attn_q_norm: upload(&f.attn_q_norm)?,
        attn_k_norm: upload(&f.attn_k_norm)?,
    })
}

fn upload_gdn<F>(g: &GdnTensors, upload: &mut F) -> Result<GdnWeights>
where
    F: FnMut(&ResolvedTensor) -> Result<DeviceTensor>,
{
    Ok(GdnWeights {
        attn_qkv: upload(&g.attn_qkv)?,
        attn_gate: upload(&g.attn_gate)?,
        ssm_alpha: g.ssm_alpha.as_ref().map(|t| upload(t)).transpose()?,
        ssm_beta: g.ssm_beta.as_ref().map(|t| upload(t)).transpose()?,
        ssm_ba: g.ssm_ba.as_ref().map(|t| upload(t)).transpose()?,
        ssm_a: upload(&g.ssm_a)?,
        ssm_dt_bias: upload(&g.ssm_dt_bias)?,
        ssm_conv1d: upload(&g.ssm_conv1d)?,
        ssm_norm: upload(&g.ssm_norm)?,
        ssm_out: upload(&g.ssm_out)?,
    })
}

fn upload_ffn<F>(f: &MoeFfnTensors, upload: &mut F) -> Result<FfnWeights>
where
    F: FnMut(&ResolvedTensor) -> Result<DeviceTensor>,
{
    let dense = f
        .dense
        .as_ref()
        .map(|d| -> Result<DenseFfnWeights> {
            Ok(DenseFfnWeights {
                ffn_gate: upload(&d.ffn_gate)?,
                ffn_up: upload(&d.ffn_up)?,
                ffn_down: upload(&d.ffn_down)?,
            })
        })
        .transpose()?;
    Ok(FfnWeights {
        ffn_gate_inp: f.ffn_gate_inp.as_ref().map(|t| upload(t)).transpose()?,
        ffn_gate_exps: f.ffn_gate_exps.as_ref().map(|t| upload(t)).transpose()?,
        ffn_up_exps: f.ffn_up_exps.as_ref().map(|t| upload(t)).transpose()?,
        ffn_down_exps: f.ffn_down_exps.as_ref().map(|t| upload(t)).transpose()?,
        shared: f.shared.as_ref().map(|s| upload_shared(s, upload)).transpose()?,
        dense,
    })
}

fn upload_shared<F>(s: &SharedExpertTensors, upload: &mut F) -> Result<SharedExpertWeights>
where
    F: FnMut(&ResolvedTensor) -> Result<DeviceTensor>,
{
    Ok(SharedExpertWeights {
        ffn_gate_inp_shexp: upload(&s.ffn_gate_inp_shexp)?,
        ffn_gate_shexp: upload(&s.ffn_gate_shexp)?,
        ffn_up_shexp: upload(&s.ffn_up_shexp)?,
        ffn_down_shexp: upload(&s.ffn_down_shexp)?,
    })
}
