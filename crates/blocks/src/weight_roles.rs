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
use flambeau_core::{Device, DevicePtr};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::{GgmlDType, GgufFile};
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
///
/// GDN / MoE extensions are default-`None` so non-GDN arches (gemma4,
/// llama-style) don't have to implement them. Arches that ship GDN
/// (qwen3-moe family) override the `gdn_*` and `moe_*` methods.
pub trait ModelConfig {
    fn hidden(&self) -> usize;
    fn ff_len(&self) -> usize;
    fn n_heads(&self, layer: usize) -> usize;
    fn n_kv_heads(&self, layer: usize) -> usize;
    fn head_dim(&self, layer: usize) -> usize;
    fn rms_norm_eps(&self) -> f32;
    fn vocab_size(&self) -> usize;

    // --- Optional GDN extensions ---
    /// Number of V heads in the GDN attention block. `None` for non-GDN
    /// arches. Drives `FusedQkvParallel` sharding for `attn_qkv`.
    fn gdn_num_v_heads(&self) -> Option<usize> {
        None
    }
    /// Number of K heads in the GDN block. `None` for non-GDN.
    fn gdn_num_k_heads(&self) -> Option<usize> {
        None
    }
    fn gdn_head_v_dim(&self) -> Option<usize> {
        None
    }
    fn gdn_head_k_dim(&self) -> Option<usize> {
        None
    }
    /// `true` for arches whose GQA head mapping forces K and Q to be
    /// replicated across ranks (rep_outer: qwen3.5/3.6 MoE). `false`
    /// for arches where contiguous TP K/Q split works (rep_inner:
    /// qwen3next).
    fn gdn_kq_replicated(&self) -> bool {
        false
    }

    // --- Optional MoE extensions ---
    /// Number of routed experts per layer. `None` for non-MoE arches.
    fn moe_num_experts(&self) -> Option<usize> {
        None
    }
    /// Per-expert FFN intermediate dim. `None` for non-MoE arches.
    fn moe_intermediate(&self) -> Option<usize> {
        None
    }
    /// Shared-expert intermediate dim. `Some` iff the arch carries an
    /// always-on shared expert alongside the routed experts.
    fn shared_expert_intermediate(&self) -> Option<usize> {
        None
    }
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

/// Optional pre-upload byte-level transform. Takes the raw mmap bytes
/// + the source GGUF dtype, returns the transformed bytes + the new
/// dtype that should be written to the device. Used by qwen3-moe for
/// at-load MXFP4→Q8_0 dequant and F32→Q8_0 quantisation. `None` for
/// straight-copy tensors (the common case).
pub type PreUploadFn = fn(raw: &[u8], src_dtype: GgmlDType) -> Result<(Vec<u8>, GgmlDType)>;

/// Optional callback fired after a tensor's HtoD copy completes. Used
/// for `file.advise_drop_tensor` mmap page-cache eviction so loading a
/// 15GB GGUF doesn't pin host RAM. `None` skips the callback.
pub type OnUploadDoneFn = fn(file: &GgufFile, name: &str);

/// What a [`WeightRole`] knows about its tensor at compile time. Each
/// field on this struct is a `fn` pointer or a const because roles are
/// zero-sized — there's nothing to store at runtime, every value comes
/// from the role's `impl` block.
///
/// New roles compose against [`DEFAULT_WEIGHT_SPEC`] via struct update
/// syntax so they only have to set the fields they care about:
/// ```ignore
/// const SPEC: WeightSpec = WeightSpec {
///     name_template: "blk.{layer}.my_tensor.weight",
///     shape_for: |cfg, _| [cfg.hidden(), cfg.hidden()],
///     layout_for: |_, _, w| if w == 1 { WeightLayout::Replicated } else { WeightLayout::ColParallel { world: w, dim: 0 } },
///     required: true,
///     ..DEFAULT_WEIGHT_SPEC
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WeightSpec {
    /// `{layer}` is substituted with the layer index. For globals
    /// (token_embd, output_norm, lm_head) the template has no
    /// placeholder.
    pub name_template: &'static str,
    /// Original (un-sharded) `[out, in]` shape — used for the
    /// `WeightHandle.dims` field and for any divisibility checks the
    /// uploader runs. For 3D tensors (indexed MoE expert weights), set
    /// [`Self::shape_3d_for`] to `Some(..)` and the uploader uses that
    /// instead.
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
    /// Optional 3D shape for indexed MoE expert weights `[n_experts,
    /// ff_len, hidden]`. When `Some`, the uploader uses this instead of
    /// `shape_for` and skips the `WeightHandle` matmul wrapping (3D
    /// tensors are consumed by index-aware kernels, not matmul).
    pub shape_3d_for: Option<fn(cfg: &dyn ModelConfig, layer: usize) -> [usize; 3]>,
    /// Optional pre-upload byte transform — see [`PreUploadFn`]. When
    /// `Some`, the uploader reads the raw mmap bytes, runs the
    /// transform, then HtoDs the result. When `None`, the uploader
    /// uses the standard [`upload_replicated_tensor`] /
    /// [`upload_sharded_tensor`] path. Cannot be combined with
    /// `DtypeFilter::F32ToF16Norm` (the norm helper has its own cast).
    pub pre_upload: Option<PreUploadFn>,
    /// Optional post-upload callback — see [`OnUploadDoneFn`].
    pub on_upload_done: Option<OnUploadDoneFn>,
}

/// Default values used to compose new role SPECs via struct update
/// syntax. Don't construct a `WeightSpec` from this directly — set the
/// fields your role actually uses and pull defaults from here.
pub const DEFAULT_WEIGHT_SPEC: WeightSpec = WeightSpec {
    name_template: "",
    shape_for: |_, _| [0, 0],
    layout_for: |_, _, _| WeightLayout::Replicated,
    dtype: DtypeFilter::Any,
    required: true,
    shape_3d_for: None,
    pre_upload: None,
    on_upload_done: None,
};

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
        let uploaded = if let Some(pre_upload) = R::SPEC.pre_upload {
            // pre_upload path: read raw mmap bytes, transform host-side
            // (MXFP4→Q8_0 dequant, F32→Q8_0 quant), alloc + HtoD copy.
            // The role declares the post-transform dtype via
            // `pre_upload`'s return value.
            if R::SPEC.dtype == DtypeFilter::F32ToF16Norm {
                bail!(
                    "role `{name}`: pre_upload + F32ToF16Norm are mutually exclusive"
                );
            }
            let raw = self
                .file
                .tensor_raw(&info.name)
                .with_context(|| format!("tensor_raw `{}`", info.name))?;
            let (transformed, new_dtype) = (pre_upload)(&raw, info.dtype)
                .with_context(|| format!("pre_upload `{name}`"))?;
            let bytes = transformed.len();
            let ptr = self
                .device
                .alloc(bytes)
                .map_err(|e| anyhow::anyhow!("alloc {bytes} B for `{name}`: {e}"))?;
            self.tracker.allocs.push((ptr, bytes));
            // SAFETY: `transformed` outlives the bounded stream sync below;
            // alloc'd `ptr` owns `bytes`.
            unsafe {
                use flambeau_core::{CopyDirection, Stream};
                self.device
                    .memcpy_async(
                        self.stream,
                        CopyDirection::HostToDevice,
                        ptr,
                        DevicePtr(transformed.as_ptr() as usize),
                        bytes,
                    )
                    .map_err(|e| anyhow::anyhow!("memcpy {name}: {e}"))?;
                self.stream
                    .synchronize()
                    .map_err(|e| anyhow::anyhow!("sync {name}: {e}"))?;
            }
            drop(transformed);
            UploadedTensor {
                ptr,
                dtype: new_dtype,
                bytes,
            }
        } else {
            match R::SPEC.dtype {
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
            }
        };
        // Optional post-upload hook (mmap page eviction).
        if let Some(cb) = R::SPEC.on_upload_done {
            cb(self.file, &name);
        }
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
    ///
    /// Errors if the role declares a 3D shape (`shape_3d_for` is
    /// `Some`) — those tensors don't have a 2D matmul interpretation
    /// and should be consumed via the raw [`Self::upload`] path.
    pub fn upload_matmul<R: WeightRole>(&mut self, layer: usize) -> Result<Option<WeightHandle>> {
        if R::SPEC.shape_3d_for.is_some() {
            bail!("upload_matmul: role declares a 3D shape; use upload() instead");
        }
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
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
        ..DEFAULT_WEIGHT_SPEC
    };
    fn tensor_name(_layer: usize) -> String {
        Self::SPEC.name_template.to_string()
    }
}

// ===========================================================================
// qwen3-moe family roles.
//
// Lives in this module (rather than the qwen3-moe crate) so the
// FusedQkvParallel layout helper + the at-load dequant pre_upload
// hooks can be unit-tested in blocks without circular dependencies.
// The roles consume ModelConfig's GDN / MoE optional methods.
// ===========================================================================

// --- attention biases (qwen3.5 / 3.6 standard layers) ---

/// Q-projection bias `[q_width]`. Optional; many qwen3 variants omit it.
pub struct AttnQBias;
impl WeightRole for AttnQBias {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_q.bias",
        shape_for: |cfg, l| [cfg.n_heads(l) * cfg.head_dim(l), 1],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// K-projection bias `[kv_width]`. Optional.
pub struct AttnKBias;
impl WeightRole for AttnKBias {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_k.bias",
        shape_for: |cfg, l| [cfg.n_kv_heads(l) * cfg.head_dim(l), 1],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// V-projection bias `[kv_width]`. Optional.
pub struct AttnVBias;
impl WeightRole for AttnVBias {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_v.bias",
        shape_for: |cfg, l| [cfg.n_kv_heads(l) * cfg.head_dim(l), 1],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

// --- GDN / hybrid attention block ---

/// Fused Q+K+V projection used by GDN layers (`attn_qkv`). Shape on
/// disk is `[Q_part | K_part | V_part, hidden]` along output dim. Per-
/// rank slicing is via [`WeightLayout::FusedQkvParallel`], whose
/// parameters (`kq_replicated`, head counts) are pulled from the
/// `ModelConfig` GDN extensions.
pub struct FusedAttnQkv;
impl WeightRole for FusedAttnQkv {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_qkv.weight",
        shape_for: |cfg, _| {
            let nv = cfg.gdn_num_v_heads().unwrap_or(0);
            let nk = cfg.gdn_num_k_heads().unwrap_or(0);
            let hv = cfg.gdn_head_v_dim().unwrap_or(0);
            let hk = cfg.gdn_head_k_dim().unwrap_or(0);
            let total = nv * hv + nk * hk * 2; // Q + K + V along out dim
            [total, cfg.hidden()]
        },
        layout_for: |cfg, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::FusedQkvParallel {
                    world,
                    num_v_heads: cfg.gdn_num_v_heads().unwrap_or(0) as u32,
                    num_k_heads: cfg.gdn_num_k_heads().unwrap_or(0) as u32,
                    head_v_dim: cfg.gdn_head_v_dim().unwrap_or(0) as u32,
                    head_k_dim: cfg.gdn_head_k_dim().unwrap_or(0) as u32,
                    kq_replicated: cfg.gdn_kq_replicated(),
                }
            }
        },
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// GDN output-gate projection (`attn_gate`). ColParallel for TP.
pub struct AttnGate;
impl WeightRole for AttnGate {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.attn_gate.weight",
        shape_for: |cfg, _| {
            let d_inner = cfg.gdn_num_v_heads().unwrap_or(0) * cfg.gdn_head_v_dim().unwrap_or(0);
            [d_inner, cfg.hidden()]
        },
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// SSM alpha projection (qwen3.5 / 3.6 GDN — distinct from qwen3next's
/// fused ssm_ba). Optional: the upload path treats absence + presence
/// of `ssm_ba` as a 2-way branch.
pub struct SsmAlpha;
impl WeightRole for SsmAlpha {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_alpha.weight",
        shape_for: |cfg, _| [cfg.gdn_num_v_heads().unwrap_or(0), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// SSM beta projection (qwen3.5 / 3.6 GDN). Optional, paired with ssm_alpha.
pub struct SsmBeta;
impl WeightRole for SsmBeta {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_beta.weight",
        shape_for: |cfg, _| [cfg.gdn_num_v_heads().unwrap_or(0), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Fused `ssm_ba` (qwen3next pattern — alpha+beta interleaved per
/// K-head). Optional, alternative to ssm_alpha+ssm_beta.
pub struct SsmBa;
impl WeightRole for SsmBa {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_ba.weight",
        shape_for: |cfg, _| [2 * cfg.gdn_num_v_heads().unwrap_or(0), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// SSM A — per-head decay rate, F32 `[num_v_heads]`. Replicated.
pub struct SsmA;
impl WeightRole for SsmA {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_a",
        shape_for: |cfg, _| [cfg.gdn_num_v_heads().unwrap_or(0), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// SSM dt bias — per-head, F32 `[num_v_heads]`. Replicated.
pub struct SsmDtBias;
impl WeightRole for SsmDtBias {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_dt.bias",
        shape_for: |cfg, _| [cfg.gdn_num_v_heads().unwrap_or(0), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Causal conv1d weights `[conv_channels, conv_kernel]`. Replicated;
/// per-rank slicing is hand-coded in qwen3-moe's GDN upload path
/// because conv channels are a function of `local_d_inner` + `2 ×
/// local_qk_size` which doesn't fit a simple ColParallel.
pub struct SsmConv1d;
impl WeightRole for SsmConv1d {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_conv1d.weight",
        shape_for: |_cfg, _l| [0, 0], // arch handles dim computation
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// SSM per-head V RMSNorm `[head_v_dim]`. Replicated.
pub struct SsmNorm;
impl WeightRole for SsmNorm {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_norm.weight",
        shape_for: |cfg, _| [cfg.gdn_head_v_dim().unwrap_or(0), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// SSM output projection `[hidden, d_inner]`. Row-parallel for TP.
pub struct SsmOut;
impl WeightRole for SsmOut {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ssm_out.weight",
        shape_for: |cfg, _| {
            let d_inner = cfg.gdn_num_v_heads().unwrap_or(0) * cfg.gdn_head_v_dim().unwrap_or(0);
            [cfg.hidden(), d_inner]
        },
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::RowParallel { world, dim: 1 }
            }
        },
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

// --- MoE FFN (routed experts) ---

/// Router logits projection (`ffn_gate_inp`), F32 `[n_experts, hidden]`.
/// Replicated. F32→F16 conversion at upload via the F32ToF16Norm filter
/// (router quality survives the cast — it's a discrete top-k).
pub struct MoeRouter;
impl WeightRole for MoeRouter {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_gate_inp.weight",
        shape_for: |cfg, _| [cfg.moe_num_experts().unwrap_or(0), cfg.hidden()],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: true,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Routed expert gate `[n_experts, moe_intermediate, hidden]`. Indexed
/// 3D tensor — uploads as raw bytes; consumer kernel reads the 3D layout.
pub struct MoeExpertsGate;
impl WeightRole for MoeExpertsGate {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_gate_exps.weight",
        shape_for: |cfg, _| {
            let n_e = cfg.moe_num_experts().unwrap_or(0);
            let ff = cfg.moe_intermediate().unwrap_or(0);
            [n_e * ff, cfg.hidden()]
        },
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        shape_3d_for: Some(|cfg, _| {
            [
                cfg.moe_num_experts().unwrap_or(0),
                cfg.moe_intermediate().unwrap_or(0),
                cfg.hidden(),
            ]
        }),
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Routed expert up `[n_experts, moe_intermediate, hidden]`.
pub struct MoeExpertsUp;
impl WeightRole for MoeExpertsUp {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_up_exps.weight",
        shape_for: |cfg, _| {
            let n_e = cfg.moe_num_experts().unwrap_or(0);
            let ff = cfg.moe_intermediate().unwrap_or(0);
            [n_e * ff, cfg.hidden()]
        },
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        shape_3d_for: Some(|cfg, _| {
            [
                cfg.moe_num_experts().unwrap_or(0),
                cfg.moe_intermediate().unwrap_or(0),
                cfg.hidden(),
            ]
        }),
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Routed expert down `[n_experts, hidden, moe_intermediate]`.
pub struct MoeExpertsDown;
impl WeightRole for MoeExpertsDown {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_down_exps.weight",
        shape_for: |cfg, _| {
            let n_e = cfg.moe_num_experts().unwrap_or(0);
            let ff = cfg.moe_intermediate().unwrap_or(0);
            [n_e * cfg.hidden(), ff]
        },
        layout_for: |_, _, _| WeightLayout::Replicated,
        required: true,
        shape_3d_for: Some(|cfg, _| {
            [
                cfg.moe_num_experts().unwrap_or(0),
                cfg.hidden(),
                cfg.moe_intermediate().unwrap_or(0),
            ]
        }),
        ..DEFAULT_WEIGHT_SPEC
    };
}

// --- Shared expert (qwen3.5 / 3.6 hybrid arches) ---

/// Shared-expert router scalar `[hidden]` F32. Replicated, F32→F16 cast.
pub struct SharedExpertRouter;
impl WeightRole for SharedExpertRouter {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_gate_inp_shexp.weight",
        shape_for: |cfg, _| [cfg.hidden(), 1],
        layout_for: |_, _, _| WeightLayout::Replicated,
        dtype: DtypeFilter::F32ToF16Norm,
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Shared-expert gate `[shared_intermediate, hidden]`. ColParallel.
pub struct SharedExpertGate;
impl WeightRole for SharedExpertGate {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_gate_shexp.weight",
        shape_for: |cfg, _| [cfg.shared_expert_intermediate().unwrap_or(0), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Shared-expert up `[shared_intermediate, hidden]`. ColParallel.
pub struct SharedExpertUp;
impl WeightRole for SharedExpertUp {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_up_shexp.weight",
        shape_for: |cfg, _| [cfg.shared_expert_intermediate().unwrap_or(0), cfg.hidden()],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::ColParallel { world, dim: 0 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}

/// Shared-expert down `[hidden, shared_intermediate]`. RowParallel.
pub struct SharedExpertDown;
impl WeightRole for SharedExpertDown {
    const SPEC: WeightSpec = WeightSpec {
        name_template: "blk.{layer}.ffn_down_shexp.weight",
        shape_for: |cfg, _| [cfg.hidden(), cfg.shared_expert_intermediate().unwrap_or(0)],
        layout_for: |_, _, world| {
            if world == 1 {
                WeightLayout::Replicated
            } else {
                WeightLayout::RowParallel { world, dim: 1 }
            }
        },
        required: false,
        ..DEFAULT_WEIGHT_SPEC
    };
}
