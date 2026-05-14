//! Typed weight roles + a generic uploader.
//!
//! A *role* is a zero-sized marker type that, via the [`WeightRole`]
//! trait, declares everything the uploader needs to know about one
//! tensor:
//!
//! - the GGUF tensor name (templated on layer index for per-layer
//!   weights, plain for globals);
//! - the un-sharded shape, as a function of the model config + layer
//!   index;
//! - the sharding policy ([`WeightLayout`]) for a given TP world;
//! - the dtype filter (which GGUF dtypes the kernel that consumes this
//!   tensor can read);
//! - whether the tensor is required or optional.
//!
//! Shared roles (`AttnQ`, `FfnGate`, …) live in
//! [`weight_roles`](self)::shared and are reused across every dense or
//! gated transformer arch. Arch-specific roles (GDN coefficients,
//! per-layer side-channel embeds, fused QKV) live in the arch crate as
//! additional `impl WeightRole` blocks. Each new arch defines its
//! `Config` type + an `impl ModelConfig` + the per-arch roles, then
//! drops 200+ lines of upload boilerplate by calling
//! [`WeightUploader::upload`] per role.
//!
//! The uploader returns [`UploadedTensor`] uniformly — the arch crate
//! wraps that as a [`WeightHandle`] for matmul roles or unwraps the
//! `ptr` for norm roles. Both paths are one line.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipStream;
use flambeau_core::DevicePtr;
use flambeau_ops::hip::HipDevice;
use flambeau_quant::GgufFile;
use flambeau_runtime::WeightLayout;

use crate::driver_utils::{ggml_to_qdtype, RawAllocTracker};
use crate::sharding::{
    upload_replicated_norm_f32_to_f16, upload_replicated_tensor, upload_sharded_tensor,
    UploadedTensor,
};
use crate::WeightHandle;

/// Minimal arch-config surface the shared roles need. Each arch's
/// config impls this; the role's `shape_for` / `layout_for` walk the
/// trait methods rather than the concrete `Cfg` so the same role
/// implementation works across every arch.
pub trait ModelConfig {
    fn hidden(&self) -> usize;
    fn ff_len(&self) -> usize;
    fn n_heads(&self, layer: usize) -> usize;
    fn n_kv_heads(&self, layer: usize) -> usize;
    fn head_dim(&self, layer: usize) -> usize;
    fn rms_norm_eps(&self) -> f32;
    fn vocab_size(&self) -> usize;
}

/// Dtype constraints for the consuming kernel. `Any` accepts whatever
/// GGUF says the tensor is (norms, embeddings, quantised matmul
/// weights). `RequireF32` is set on tensors the upload path must cast
/// (gemma4's learned norms are stored F32 but read F16 by
/// `rmsnorm_f16`; the upload helper casts at copy time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeFilter {
    /// Pass the source dtype through unchanged.
    Any,
    /// Source must be F32; uploader casts to F16 at copy time.
    F32ToF16Norm,
}

/// What a [`WeightRole`] knows about its tensor at compile time. Each
/// field on this struct is a `fn` pointer or a const because roles are
/// zero-sized — there's nothing to store at runtime, every value comes
/// from the role's `impl` block.
#[derive(Debug, Clone, Copy)]
pub struct WeightSpec {
    /// `{layer}` is substituted with the layer index. For globals
    /// (token_embd, output_norm, lm_head) the template has no
    /// placeholder.
    pub name_template: &'static str,
    /// Original (un-sharded) `[out, in]` shape — used for the
    /// `WeightHandle.dims` field and for any divisibility checks the
    /// uploader runs.
    pub shape_for: fn(cfg: &dyn ModelConfig, layer: usize) -> [usize; 2],
    /// Sharding policy for this tensor at the given TP world.
    /// `world = 1` collapses to `Replicated` for every role.
    pub layout_for: fn(cfg: &dyn ModelConfig, layer: usize, world: u32) -> WeightLayout,
    /// Dtype constraint, see [`DtypeFilter`].
    pub dtype: DtypeFilter,
    /// `false` for tensors that may legitimately be absent on some
    /// layers (e.g. gemma4 `attn_v` on alt-attention layers,
    /// `attn_k_norm` on layers without a learned K norm).
    pub required: bool,
}

/// A weight tensor type. Implementors are zero-sized markers.
///
/// `impl WeightRole for AttnQ` declares (at compile time) everything
/// the uploader needs to handle the `blk.{layer}.attn_q.weight`
/// tensor.
pub trait WeightRole: Sized {
    const SPEC: WeightSpec;

    /// Build the role's logical name for a specific layer. Globals
    /// override this to ignore `layer`.
    fn tensor_name(layer: usize) -> String {
        Self::SPEC.name_template.replace("{layer}", &layer.to_string())
    }
}

/// Tracker + per-call resources used by [`WeightUploader::upload`].
/// The arch driver instantiates one of these per rank then walks its
/// roles.
pub struct WeightUploader<'a> {
    pub device: &'a HipDevice,
    pub stream: &'a HipStream,
    pub tracker: &'a mut RawAllocTracker,
    pub file: &'a GgufFile,
    pub cfg: &'a dyn ModelConfig,
    /// TP world size — 1 for PP / single-device, N for TP / hybrid.
    pub world: u32,
    /// TP rank, 0..world.
    pub rank: u32,
}

impl<'a> WeightUploader<'a> {
    /// Upload role `R` for the given layer. Returns `Some` for
    /// optional roles whose tensor is absent in the GGUF; `Err` for
    /// required roles that are missing.
    pub fn upload<R: WeightRole>(&mut self, layer: usize) -> Result<Option<UploadedTensor>> {
        let name = R::tensor_name(layer);
        let info = match self.file.tensors.get(&name) {
            Some(info) => info,
            None => {
                if R::SPEC.required {
                    bail!("required tensor `{name}` missing from GGUF (role)");
                }
                return Ok(None);
            }
        };
        let uploaded = match R::SPEC.dtype {
            DtypeFilter::F32ToF16Norm => {
                let shape = (R::SPEC.shape_for)(self.cfg, layer);
                let expected_len: usize = shape.iter().product();
                upload_replicated_norm_f32_to_f16(
                    self.file,
                    info,
                    expected_len,
                    self.device,
                    self.stream,
                    self.tracker,
                )
                .with_context(|| format!("upload {name} (F32→F16 norm)"))?
            }
            DtypeFilter::Any => {
                let layout = (R::SPEC.layout_for)(self.cfg, layer, self.world);
                match layout {
                    WeightLayout::Replicated => upload_replicated_tensor(
                        self.file,
                        info,
                        self.device,
                        self.stream,
                        self.tracker,
                    )
                    .with_context(|| format!("upload {name} (replicated)"))?,
                    _ => upload_sharded_tensor(
                        self.file,
                        info,
                        layout,
                        self.rank,
                        self.device,
                        self.stream,
                        self.tracker,
                    )
                    .with_context(|| format!("upload {name} ({layout:?})"))?,
                }
            }
        };
        Ok(Some(uploaded))
    }

    /// Upload `R` for a layer and require it to be present — convenience
    /// wrapper that errors on `None`.
    pub fn upload_required<R: WeightRole>(&mut self, layer: usize) -> Result<UploadedTensor> {
        self.upload::<R>(layer)?
            .ok_or_else(|| anyhow::anyhow!("required tensor for role missing"))
    }

    /// Upload a matmul-shaped role and immediately wrap as a
    /// `WeightHandle`. Shape comes from the role's `shape_for`,
    /// per-rank-sliced if the policy is `ColParallel` / `RowParallel`.
    pub fn upload_matmul<R: WeightRole>(&mut self, layer: usize) -> Result<Option<WeightHandle>> {
        let uploaded = match self.upload::<R>(layer)? {
            Some(u) => u,
            None => return Ok(None),
        };
        let [out, inp] = (R::SPEC.shape_for)(self.cfg, layer);
        let layout = (R::SPEC.layout_for)(self.cfg, layer, self.world);
        let (local_out, local_in) = match layout {
            WeightLayout::Replicated => (out, inp),
            WeightLayout::ColParallel { world, .. } => (out / world as usize, inp),
            WeightLayout::RowParallel { world, .. } => (out, inp / world as usize),
            // FusedQkv slicing is arch-specific; callers should use
            // the role's full upload + hand-build the WeightHandle.
            WeightLayout::FusedQkvParallel { .. } => (out, inp),
        };
        let qdtype = ggml_to_qdtype(uploaded.dtype).with_context(|| {
            format!(
                "upload_matmul: GGUF dtype {:?} is not a flambeau QDtype",
                uploaded.dtype
            )
        })?;
        Ok(Some(WeightHandle {
            ptr: uploaded.ptr,
            dtype: qdtype,
            dims: [local_out, local_in],
        }))
    }

    /// Convenience wrapper for `upload_matmul` that errors on `None`.
    pub fn upload_matmul_required<R: WeightRole>(&mut self, layer: usize) -> Result<WeightHandle> {
        self.upload_matmul::<R>(layer)?
            .ok_or_else(|| anyhow::anyhow!("required matmul tensor for role missing"))
    }

    /// Upload a norm-style role and return the raw `DevicePtr`.
    pub fn upload_norm<R: WeightRole>(&mut self, layer: usize) -> Result<Option<DevicePtr>> {
        Ok(self.upload::<R>(layer)?.map(|u| u.ptr))
    }

    pub fn upload_norm_required<R: WeightRole>(&mut self, layer: usize) -> Result<DevicePtr> {
        self.upload_norm::<R>(layer)?
            .ok_or_else(|| anyhow::anyhow!("required norm tensor for role missing"))
    }
}

// ---------------------------------------------------------------------------
// Shared roles — used across every dense/gated transformer arch.
// ---------------------------------------------------------------------------

/// Pre-attention RMSNorm weight, F16 (or F32→F16 at upload).
pub struct AttnNorm;
impl WeightRole for AttnNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_norm.weight",
        shape_for: |cfg, _| [cfg.hidden(), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: true,
    };
}

/// Q projection. Column-parallel for TP. Required.
pub struct AttnQ;
impl WeightRole for AttnQ {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_q.weight",
        shape_for: |cfg, l| [cfg.n_heads(l) * cfg.head_dim(l), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        dtype: DtypeFilter::Any,
        required: true,
    };
}

/// K projection. Column-parallel for TP. Optional (e.g. shared-KV tail
/// layers in gemma4).
pub struct AttnK;
impl WeightRole for AttnK {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_k.weight",
        shape_for: |cfg, l| [cfg.n_kv_heads(l) * cfg.head_dim(l), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        dtype: DtypeFilter::Any,
        required: false,
    };
}

/// V projection. Column-parallel for TP. Optional (gemma4 alt-attention
/// layers fall back to V = K pre-norm).
pub struct AttnV;
impl WeightRole for AttnV {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_v.weight",
        shape_for: |cfg, l| [cfg.n_kv_heads(l) * cfg.head_dim(l), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        dtype: DtypeFilter::Any,
        required: false,
    };
}

/// Output projection. Row-parallel for TP (callers must AR-sum).
pub struct AttnOutput;
impl WeightRole for AttnOutput {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_output.weight",
        shape_for: |cfg, l| [cfg.hidden(), cfg.n_heads(l) * cfg.head_dim(l)],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::RowParallel { world, dim: 1 }
            }
        },
        dtype: DtypeFilter::Any,
        required: true,
    };
}

/// Per-head RMSNorm on Q. Optional (only the gemma4/qwen3 families
/// learn it).
pub struct AttnQNorm;
impl WeightRole for AttnQNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_q_norm.weight",
        shape_for: |cfg, l| [cfg.head_dim(l), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: false,
    };
}

/// Per-head RMSNorm on K. Optional.
pub struct AttnKNorm;
impl WeightRole for AttnKNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_k_norm.weight",
        shape_for: |cfg, l| [cfg.head_dim(l), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: false,
    };
}

/// Post-attention RMSNorm (gemma4: `post_attention_norm`; qwen3 maps
/// onto `ffn_norm` — that mapping is done by the arch, not the role).
pub struct PostAttnNorm;
impl WeightRole for PostAttnNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.post_attention_norm.weight",
        shape_for: |cfg, _| [cfg.hidden(), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: false,
    };
}

/// Pre-FFN RMSNorm.
pub struct FfnNorm;
impl WeightRole for FfnNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_norm.weight",
        shape_for: |cfg, _| [cfg.hidden(), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: false,
    };
}

/// Dense FFN gate. Column-parallel.
pub struct FfnGate;
impl WeightRole for FfnGate {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_gate.weight",
        shape_for: |cfg, _| [cfg.ff_len(), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        dtype: DtypeFilter::Any,
        required: false,
    };
}

/// Dense FFN up. Column-parallel.
pub struct FfnUp;
impl WeightRole for FfnUp {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_up.weight",
        shape_for: |cfg, _| [cfg.ff_len(), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        dtype: DtypeFilter::Any,
        required: false,
    };
}

/// Dense FFN down. Row-parallel.
pub struct FfnDown;
impl WeightRole for FfnDown {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_down.weight",
        shape_for: |cfg, _| [cfg.hidden(), cfg.ff_len()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::RowParallel { world, dim: 1 }
            }
        },
        dtype: DtypeFilter::Any,
        required: false,
    };
}

/// Post-FFW RMSNorm (gemma4-specific name; the role exists in case the
/// arch needs the cast helper).
pub struct PostFfwNorm;
impl WeightRole for PostFfwNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.post_ffw_norm.weight",
        shape_for: |cfg, _| [cfg.hidden(), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: false,
    };
}

// ---------------------------------------------------------------------------
// Shared globals.
// ---------------------------------------------------------------------------

/// Token embedding. Replicated across ranks.
pub struct TokenEmbd;
impl WeightRole for TokenEmbd {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "token_embd.weight",
        shape_for: |cfg, _| [cfg.vocab_size(), cfg.hidden()],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::Any,
        required: true,
    };
    fn tensor_name(_layer: usize) -> String {
        Self::SPEC.name_template.to_string()
    }
}

/// Output norm — the final post-block RMSNorm before the LM head.
pub struct OutputNorm;
impl WeightRole for OutputNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "output_norm.weight",
        shape_for: |cfg, _| [cfg.hidden(), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: true,
    };
    fn tensor_name(_layer: usize) -> String {
        Self::SPEC.name_template.to_string()
    }
}

/// LM head. Optional when the arch ties it to `token_embd`.
pub struct LmHead;
impl WeightRole for LmHead {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "output.weight",
        shape_for: |cfg, _| [cfg.vocab_size(), cfg.hidden()],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::Any,
        required: false,
    };
    fn tensor_name(_layer: usize) -> String {
        Self::SPEC.name_template.to_string()
    }
}
