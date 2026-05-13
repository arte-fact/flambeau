//! TP-sharded model + topology selector.
//! `Qwen3MoETpModel` is the TP analogue of [`crate::sharded::Qwen3MoEShardedModel`]:
//! every rank holds *all* layers, but each layer's weight tensors are
//! sliced according to [`crate::Qwen35DenseTpLayout`]. The peer model
//! (PP) stays untouched in `sharded.rs`; the two are selected via the
//! [`Topology`] enum.
//! ## Design split with 
//! (`tp_slice.rs`) provides the host-side byte-slicing primitive
//! (`slice_for_tp` + `slice_bytes_for_tp`). composes it with HIP
//! upload to produce per-rank `LayerWeights` structures.
//! ## What's in scope this session
//! - `Topology { Pp(LayerAssignment), Tp(Qwen35DenseTpLayout) }` — the
//! selector consumed by future loaders.
//! - `Qwen3MoETpRankShard` / `Qwen3MoETpModel` types holding sliced
//! weights per rank.
//! - `load()` that walks the model layout, slices per-tensor, allocates
//! device memory, and uploads. **Dtype conversions** (F32→F16 norms,
//! F32→Q8_0 ssm scalars, BF16→Q8_0) the PP path applies on load are
//! intentionally **deferred to uploads bytes-as-is so
//! the forward path can reach for typed loaders when it needs them
//! (this matches 2.a's posture: "do conversion where it pays off",
//! not at every load site).
//! ## What's deferred
//! - Per-layer upload of `attn_qkv` and `ssm_conv1d` is correct (they
//! stay `Replicated` per the layout table) but will replace
//! them with head-aware sharding.
//! - Tied LM head gets a Replicated `output` ptr at every rank; no
//! special-case sharing.
//! - The TP-aware forward path lands in 
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_quant::{GgmlDType, GgufFile};
use flambeau_runtime::{LayerAssignment, RankId, WeightLayout};

use crate::config::Qwen3MoEConfig;
use crate::layout::ModelLayout;
use crate::tp_layout::Qwen35DenseTpLayout;
use crate::tp_slice::slice_for_tp;
use crate::weights::DeviceTensor;

const QK8_0: usize = 32;

/// How weights are distributed across the mesh.
/// Forward paths and the loader switch on this. Pure-PP runs continue
/// to use [`Topology::Pp`]; new TP runs use [`Topology::Tp`]. Hybrid
/// `PpTp` is reserved for V2-tp-5b.
#[derive(Debug, Clone)]
pub enum Topology {
    /// Pipeline parallelism — `LayerAssignment` distributes whole
    /// layers across ranks. The existing path.
    Pp(LayerAssignment),
    /// Tensor parallelism — every rank owns *all* layers, sharded
    /// per-tensor. `Qwen35DenseTpLayout` knows the per-name layout.
    Tp(Qwen35DenseTpLayout),
}

impl Topology {
    /// Mesh size implied by the topology. PP uses `LayerAssignment`'s
    /// rank count; TP uses the configured world.
    pub fn world(&self) -> u32 {
        match self {
            Topology::Pp(a) => a.num_ranks(),
            Topology::Tp(tp) => tp.world(),
        }
    }
}

/// Per-rank shard for the TP path. Mirrors
/// [`crate::sharded::Qwen3MoERankShard`] but every rank carries every
/// layer (sliced) plus replicated globals.
#[derive(Debug)]
pub struct Qwen3MoETpRankShard {
    pub rank: RankId,
    pub device_id: i32,
    /// Replicated globals — present on every rank.
    pub token_embd: DeviceTensor,
    pub output_norm: DeviceTensor,
    /// Replicated `output.weight` when the model has an explicit LM head.
    /// `None` when `cfg.tied_lm_head` (the LM head reuses `token_embd`).
    pub output: Option<DeviceTensor>,
    /// Sliced per-layer tensors. One entry per layer in model order.
    /// The exact contents depend on each tensor's `WeightLayout`:
    /// ColParallel/RowParallel are sliced; Replicated is full.
    pub layers: Vec<Vec<TpLayerTensor>>,
    pub total_bytes: usize,
    disposed: bool,
}

/// One per-tensor entry inside a layer. Carries the resolved name,
/// layout, and device tensor so the forward path can look up the
/// sharded slice without re-walking the layout table.
#[derive(Debug)]
pub struct TpLayerTensor {
    pub name: Arc<str>,
    pub layout: WeightLayout,
    pub tensor: DeviceTensor,
}

impl Qwen3MoETpRankShard {
    /// Free every device allocation in this shard.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut first_err: Option<anyhow::Error> = None;
        let mut free = |t: &mut DeviceTensor| {
            if !t.ptr.is_null() && t.bytes > 0 {
                // SAFETY: every pointer came from `device.alloc()` in `load`.
                if let Err(e) = unsafe { device.dealloc(t.ptr, t.bytes) } {
                    if first_err.is_none() {
                        first_err = Some(anyhow!("dealloc `{}`: {e}", t.name));
                    }
                }
                t.ptr = DevicePtr::NULL;
                t.bytes = 0;
            }
        };
        free(&mut self.token_embd);
        free(&mut self.output_norm);
        if let Some(t) = &mut self.output {
            free(t);
        }
        for layer in &mut self.layers {
            for tlt in layer.iter_mut() {
                free(&mut tlt.tensor);
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Qwen3MoETpRankShard {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::tp_sharded",
                rank = self.rank.0,
                bytes = self.total_bytes,
                "Qwen3MoETpRankShard dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Tensor-parallel sharded model — one [`Qwen3MoETpRankShard`] per
/// rank, every shard carrying every layer in sliced form.
pub struct Qwen3MoETpModel {
    pub config: Qwen3MoEConfig,
    pub layout: ModelLayout,
    pub tp: Qwen35DenseTpLayout,
    pub shards: Vec<Qwen3MoETpRankShard>,
    /// per-rank `OpsRegistry` cache. `OpsRegistry::new`
    /// loads every kernel HSACO via `hipModuleLoadData`, which is a
    /// real driver call (no global module cache). Constructing one per
    /// per-op-block per-layer per-rank is the dominant cost on the
    /// decode hot path (was ~228 ms/layer on 9B before this cache was
    /// added). Built once at load time and reused across every forward
    /// call. Indexed by rank.
    pub ops: Vec<flambeau_ops::hip::OpsRegistry>,
    /// range of layer indices actually populated on this
    /// model's shards. `0..config.num_layers` for a pure-TP load (the
    /// historical V2.* path). For a hybrid PP+TP stage, this is the
    /// stage's layer range; layer indices outside the range hold empty
    /// `Vec<TpLayerTensor>` placeholders in `shards[r].layers`.
    pub layer_range: std::ops::Range<usize>,
    /// `true` iff `token_embd` was actually uploaded on
    /// every rank (always `true` for pure-TP; only `true` on hybrid
    /// stage 0).
    pub has_token_embd: bool,
    /// `true` iff `output_norm` (and `output` when the
    /// model has an explicit LM head) were uploaded on every rank
    /// (always `true` for pure-TP; only `true` on the last hybrid stage).
    pub has_output_head: bool,
}

impl Qwen3MoETpModel {
    /// `true` iff the loader had to fall back to
    /// Replicated upload for layer `il`'s MoE expert tensors (K-quant
    /// block-size misalignment). The TP forward path detects this and
    /// runs that layer's MoE with `world=1` semantics + skips the
    /// post-MoE AllReduce. Layers outside the loaded `layer_range` are
    /// reported as `false` (they don't run at all on this shard).
    pub fn moe_replicated_at(&self, il: usize) -> bool {
        // Inspect rank 0's `ffn_gate_exps` layout for layer `il`.
        let Some(shard) = self.shards.first() else {
            return false;
        };
        let Some(layer_tensors) = shard.layers.get(il) else {
            return false;
        };
        for tlt in layer_tensors {
            if tlt.name.ends_with("ffn_gate_exps.weight")
                || tlt.name.ends_with("ffn_up_exps.weight")
                || tlt.name.ends_with("ffn_down_exps.weight")
            {
                return matches!(tlt.layout, WeightLayout::Replicated);
            }
        }
        false
    }
}

/// knobs the hybrid loader uses to construct a
/// "partial" `Qwen3MoETpModel` covering only one PP stage. Pure-TP
/// callers ignore this and use [`Qwen3MoETpModel::load`] (which is a
/// thin wrapper for `Default::default()`).
/// Forward paths that have not been taught about partial stages will
/// panic if asked to access an out-of-range layer; that wiring is
#[derive(Debug, Clone)]
pub struct TpLoadOpts {
    /// Range of layer indices to actually upload. `None` ≡ all layers.
    pub layer_range: Option<std::ops::Range<usize>>,
    /// Upload `token_embd`. Set on stage 0 of a hybrid mesh; the embed
    /// lookup is local to that stage.
    pub load_token_embd: bool,
    /// Upload `output_norm` + `output`. Set on the last stage of a
    /// hybrid mesh; the LM head runs there.
    pub load_output_head: bool,
}

impl Default for TpLoadOpts {
    fn default() -> Self {
        Self {
            layer_range: None,
            load_token_embd: true,
            load_output_head: true,
        }
    }
}

impl Qwen3MoETpModel {
    /// Open `file`, slice each tensor by the layout in `tp`, and upload
    /// to every rank in `cluster`. The cluster's rank count must match
    /// `tp.world()`.
    /// # Errors
    /// - Cluster size ≠ `tp.world()`.
    /// - Any per-tensor slice / upload failure (propagated).
    pub fn load(file: &GgufFile, cluster: &HipCluster, tp: Qwen35DenseTpLayout) -> Result<Self> {
        Self::load_with_opts(file, cluster, tp, &TpLoadOpts::default())
    }

    /// like [`Self::load`], but honours `opts` to upload
    /// only a layer subrange and/or skip the embedding / LM-head globals.
    /// Used by [`crate::hybrid::Qwen3MoEHybridModel`] to construct one
    /// "stage shard" per PP stage. Layer indices outside
    /// `opts.layer_range` are still present in `shards[r].layers` but
    /// hold an empty `Vec<TpLayerTensor>`; the forward path must check
    /// `model.layer_range` before indexing.
    pub fn load_with_opts(
        file: &GgufFile,
        cluster: &HipCluster,
        tp: Qwen35DenseTpLayout,
        opts: &TpLoadOpts,
    ) -> Result<Self> {
        if cluster.ranks() as u32 != tp.world() {
            bail!(
                "cluster has {} ranks, tp expects world={}",
                cluster.ranks(),
                tp.world()
            );
        }
        let config = Qwen3MoEConfig::from_gguf(file)?;
        let layout = ModelLayout::from_gguf(file, &config)?;
        let layer_range = opts
            .layer_range
            .clone()
            .unwrap_or(0..config.num_layers);
        if layer_range.start > layer_range.end || layer_range.end > config.num_layers {
            bail!(
                "TpLoadOpts.layer_range {:?} out of range for num_layers={}",
                layer_range,
                config.num_layers
            );
        }

        let mut shards = Vec::with_capacity(cluster.ranks());
        let mut ops_registries: Vec<flambeau_ops::hip::OpsRegistry> =
            Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() as u32 {
            let rank = RankId(rank_idx);
            let device = cluster.device(rank_idx as usize);
            device.bind()?;
            // load every kernel module once per rank at
            // model-load. Reused across every forward call.
            ops_registries.push(
                flambeau_ops::hip::OpsRegistry::new(device)
                    .map_err(|e| anyhow!("OpsRegistry::new (rank {rank_idx}): {e}"))?,
            );

            // Globals — gated by opts (always-on for pure-TP, per-stage
            // for hybrid). Skipped tensors get a NULL placeholder so the
            // shard still compiles; dispose treats NULL ptrs as no-ops.
            let (token_embd, b1) = if opts.load_token_embd {
                upload_tp(file, &layout.token_embd.name, &tp, rank_idx, device)?
            } else {
                (null_device_tensor(&layout.token_embd.name), 0)
            };
            let (output_norm, b2) = if opts.load_output_head {
                upload_tp(file, &layout.output_norm.name, &tp, rank_idx, device)?
            } else {
                (null_device_tensor(&layout.output_norm.name), 0)
            };
            let (output, b3) = if let Some(o) = &layout.output {
                if opts.load_output_head {
                    let (t, b) = upload_tp(file, &o.name, &tp, rank_idx, device)?;
                    (Some(t), b)
                } else {
                    (None, 0)
                }
            } else {
                (None, 0)
            };
            let mut total_bytes = b1 + b2 + b3;

            // Per-layer tensors — only those in `layer_range` get
            // uploaded; the rest get an empty tensor list at the same
            // index so layer-indexed forward code still resolves
            // (provided it pre-checks `layer_range`).
            let mut layers: Vec<Vec<TpLayerTensor>> = Vec::with_capacity(layout.layers.len());
            for (il, desc) in layout.layers.iter().enumerate() {
                if !layer_range.contains(&il) {
                    layers.push(Vec::new());
                    continue;
                }
                let mut layer_tensors: Vec<TpLayerTensor> = Vec::new();
                let world = tp.world();
                for name in collect_layer_tensor_names(desc) {
                    // qwen3next packs ssm_alpha+beta as
                    // one fused `ssm_ba.weight`. Split at load-time so the
                    // forward path (which expects the split form) sees them
                    // exactly as if the GGUF had shipped them split. See
                    // upload_tp_ssm_ba_split for the rank-aware slicing.
                    if name.ends_with(".ssm_ba.weight") {
                        let ((alpha_t, alpha_b), (beta_t, beta_b)) =
                            upload_tp_ssm_ba_split(file, &name, rank_idx, world, device, &config)?;
                        total_bytes += alpha_b + beta_b;
                        let alpha_name: Arc<str> = alpha_t.name.clone();
                        let beta_name: Arc<str> = beta_t.name.clone();
                        // Both halves are ColParallel{dim=0} on num_v_heads.
                        let half_layout = WeightLayout::ColParallel { world, dim: 0 };
                        layer_tensors.push(TpLayerTensor {
                            name: alpha_name,
                            layout: half_layout,
                            tensor: alpha_t,
                        });
                        layer_tensors.push(TpLayerTensor {
                            name: beta_name,
                            layout: half_layout,
                            tensor: beta_t,
                        });
                        continue;
                    }
                    let (tensor, b, layout_for) =
                        upload_tp_with_layout(file, &name, &tp, rank_idx, device)?;
                    total_bytes += b;
                    layer_tensors.push(TpLayerTensor {
                        name: Arc::from(name.as_str()),
                        layout: layout_for,
                        tensor,
                    });
                }
                layers.push(layer_tensors);
            }

            // One stream-sync at the end covers every memcpy_async on
            // this rank's default stream.
            device.default_stream().synchronize()?;

            shards.push(Qwen3MoETpRankShard {
                rank,
                device_id: device.id(),
                token_embd,
                output_norm,
                output,
                layers,
                total_bytes,
                disposed: false,
            });
        }

        let model = Self {
            config,
            layout,
            tp,
            shards,
            ops: ops_registries,
            layer_range,
            has_token_embd: opts.load_token_embd,
            has_output_head: opts.load_output_head,
        };
        // the Replicated-MoE forward branch skips the
        // post-FFN AllReduce because each rank's MoE output is already
        // a full-hidden update. That's only correct when there's no
        // shared expert in the same layer (a sharded shared-expert
        // partial would still need AR-folding). Bail at load time
        // rather than producing silent corruption.
        if model.config.shared_expert_intermediate_size.is_some() {
            for il in model.layer_range.clone() {
                if model.moe_replicated_at(il) {
                    bail!(
                        "fallback engaged on layer {il} (K-quant MoE expert \
                         misalignment) but model has a shared expert; that combination \
                         requires per-layer mixed-mode AR which is not yet implemented. \
                         Run with --mesh-mode pp instead, or use a Q4_0/Q4_1 quant of \
                         the model whose MoE expert dims align under the requested TP world."
                    );
                }
            }
        }
        Ok(model)
    }

    /// Total bytes uploaded across all ranks. Useful for the smoke
    /// invariant in `total_bytes == ranks × per_rank_target`.
    pub fn total_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.total_bytes).sum()
    }

    /// Bytes uploaded to a specific rank. Diagnostic for the cert.
    pub fn rank_bytes(&self, rank: usize) -> Option<usize> {
        self.shards.get(rank).map(|s| s.total_bytes)
    }

    /// Free every rank's shard. Mirrors the PP path.
    pub fn dispose(mut self, cluster: &HipCluster) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for shard in self.shards.drain(..) {
            let rank_idx = shard.rank.0 as usize;
            let device = cluster.device(rank_idx);
            if let Err(e) = shard.dispose(device) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// placeholder `DeviceTensor` for slots a hybrid stage
/// shard does not own (e.g. `token_embd` on stages > 0). Free of any
/// device allocation; dispose's `is_live`-style check sees `ptr ==
/// NULL` + `bytes == 0` and skips it. The `name` is kept for
/// diagnostics so a forward path that mistakenly dereferences this
/// slot produces a useful error.
fn null_device_tensor(name: &str) -> DeviceTensor {
    DeviceTensor {
        ptr: DevicePtr::NULL,
        dtype: GgmlDType::F16,
        dims: Vec::new(),
        bytes: 0,
        name: Arc::from(name),
    }
}

/// Per-tensor dtype-conversion target (mirroring `sharded.rs`'s
/// `up_f16` / `up_q8_0` policy). Returns `None` for tensors uploaded
/// as-is. Returns `Some(dtype)` when the tensor's source bytes must
/// be host-converted to `dtype` before upload — required because the
/// consuming kernels expect a specific dtype the GGUF doesn't always
/// store (Qwen3.6 ships norms as F32 and `ssm_alpha`/`ssm_beta` as F32,
/// but the rmsnorm + mmvq_q8_0 kernels expect F16 / Q8_0 respectively).
fn tp_target_dtype(name: &str, source_dtype: GgmlDType) -> Option<GgmlDType> {
    if source_dtype != GgmlDType::F32 {
        return None;
    }
    // Norms consumed by F16 rmsnorm kernels — same set as `upload_layer`
    // in sharded.rs (attn_norm, post_attention_norm, ffn_norm,
    // attn_q_norm, attn_k_norm) plus the global output_norm.
    // `ssm_norm` is intentionally excluded — `gdn_tp.rs` calls
    // `rmsnorm_f32` for it, which wants the F32 weight as-is.
    const F16_NORM_SUFFIXES: &[&str] = &[
        "attn_norm.weight",
        "post_attention_norm.weight",
        "ffn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
    ];
    if name == "output_norm.weight"
        || F16_NORM_SUFFIXES
            .iter()
            .any(|s| name.ends_with(s) && !name.ends_with("ssm_norm.weight"))
    {
        return Some(GgmlDType::F16);
    }
    // ssm_alpha / ssm_beta (per-v-head [num_v_heads, hidden]) — consumed
    // by mmvq_q8_0_gate_up / mmvq_q8_0; PP host-quantises F32 → Q8_0.
    if name.ends_with("ssm_alpha.weight") || name.ends_with("ssm_beta.weight") {
        return Some(GgmlDType::Q8_0);
    }
    // (mirror of iter-3, sharded.rs::upload_ffn). The
    // MoE router weight `ffn_gate_inp` is F32 in every Qwen3.x GGUF.
    // Convert to F16 at upload so the dense_gemv_f16_f16 router kernel
    // () is exercised on the TP path too — pp2tp2 didn't
    // auto-pick up iter-3's prefill lift because the conversion was
    // PP-loader-only. Quality preserved (router is a coarse top-k
    // discriminator; F16 noise can't flip top-1 except on near-tie).
    if name.ends_with("ffn_gate_inp.weight") {
        return Some(GgmlDType::F16);
    }
    None
}

/// Convert a F32 byte slice to F16 host-side. Returns the converted
/// bytes (length = `elems * 2`).
fn convert_f32_to_f16(src: &[u8], elems: usize) -> Result<Vec<u8>> {
    if src.len() < elems * 4 {
        bail!(
            "convert_f32_to_f16: src {} < elems*4 ({})",
            src.len(),
            elems * 4
        );
    }
    let f32s: &[f32] = bytemuck::cast_slice(&src[..elems * 4]);
    let mut out = Vec::with_capacity(elems * 2);
    for &v in f32s {
        let h = half::f16::from_f32(v);
        out.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    Ok(out)
}

/// quantise a BF16 byte slice to Q8_0 host-side.
/// BF16 is the upper 16 bits of an F32, so widen byte-by-byte then run
/// the standard per-32-element absmax/127 quantise. Mirrors
/// `sharded.rs::upload_bf16_as_q8_0` for the TP slicing path; needed
/// for UD-Q8_K_XL-class GGUFs (Qwen3.6-35B-A3B-UD-Q8_K_XL ships 10
/// BF16 tensors that no V1 MMVQ/MMQ path consumes).
fn quantize_bf16_to_q8_0(src: &[u8], elems: usize) -> Result<Vec<u8>> {
    if src.len() < elems * 2 {
        bail!(
            "quantize_bf16_to_q8_0: src {} < elems*2 ({})",
            src.len(),
            elems * 2
        );
    }
    if elems % QK8_0 != 0 {
        bail!(
            "quantize_bf16_to_q8_0: elems {elems} not multiple of QK8_0={QK8_0}"
        );
    }
    let src_u16: &[u16] = bytemuck::cast_slice(&src[..elems * 2]);
    let f32s: Vec<f32> = src_u16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let n_blocks = elems / QK8_0;
    let block_bytes = 34usize;
    let mut out = Vec::with_capacity(n_blocks * block_bytes);
    for block in f32s.chunks_exact(QK8_0) {
        let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = absmax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        for &v in block {
            let q = (v * id).round_ties_even() as i32;
            let q = q.clamp(-127, 127) as i8;
            out.push(q as u8);
        }
    }
    Ok(out)
}

/// Quantise a F32 byte slice to Q8_0 host-side (32-element blocks,
/// 34 B/block: half scale + 32 i8 quants). Delegates to the rayon-parallel
/// `flambeau_quant::quantize_k::quantize_row_q8_0`.
fn quantize_f32_to_q8_0(src: &[u8], elems: usize) -> Result<Vec<u8>> {
    if src.len() < elems * 4 {
        bail!(
            "quantize_f32_to_q8_0: src {} < elems*4 ({})",
            src.len(),
            elems * 4
        );
    }
    if elems % QK8_0 != 0 {
        bail!(
            "quantize_f32_to_q8_0: elems {elems} not multiple of QK8_0={QK8_0}"
        );
    }
    let f32s: &[f32] = bytemuck::cast_slice(&src[..elems * 4]);
    let mut out = Vec::with_capacity(elems / QK8_0 * 34);
    flambeau_quant::quantize_k::quantize_row_q8_0(f32s, &mut out);
    Ok(out)
}

/// quantise an F32 buffer to Q8_0 (in-memory variant of
/// `quantize_f32_to_q8_0` that takes `&[f32]` directly, used by the MXFP4
/// + ssm_ba paths below where we already hold an F32 Vec). Delegates to
/// the rayon-parallel `flambeau_quant::quantize_k::quantize_row_q8_0`.
fn quantize_f32_slice_to_q8_0(f32s: &[f32]) -> Result<Vec<u8>> {
    if f32s.len() % QK8_0 != 0 {
        bail!(
            "quantize_f32_slice_to_q8_0: elems {} not multiple of QK8_0={QK8_0}",
            f32s.len()
        );
    }
    let mut out = Vec::with_capacity(f32s.len() / QK8_0 * 34);
    flambeau_quant::quantize_k::quantize_row_q8_0(f32s, &mut out);
    Ok(out)
}

/// F32 slicer for the per-rank TP shard.
/// Applied **after** dequant in the MXFP4 + ssm_ba paths because slicing
/// pre-dequant would require MXFP4-block-grain (17 B / 32 elems) byte
/// arithmetic. Operating on F32 is straightforward; the ~4× memory blow-up
/// for the full F32 buffer is bounded by the largest single tensor we
/// dequant (Coder-Next shared-expert at hidden×shared_inter ≤ ~64 MiB
/// before quantise; freed immediately after).
/// Supported layouts: `Replicated`, `ColParallel{dim=0}`,
/// `RowParallel{dim=1}`. Returns the per-rank dims alongside the bytes
/// so the caller can record them on the resulting `DeviceTensor`.
fn slice_f32_for_tp(
    f32s: &[f32],
    dims: &[u64],
    layout: WeightLayout,
    rank: u32,
) -> Result<(Vec<f32>, Vec<u64>)> {
    let total: usize = dims.iter().product::<u64>() as usize;
    if f32s.len() != total {
        bail!(
            "slice_f32_for_tp: f32s.len={} != prod(dims)={total}",
            f32s.len()
        );
    }
    // Replicated and Col/RowParallel{world=1} are full copies.
    let (world, axis) = match layout {
        WeightLayout::Replicated => return Ok((f32s.to_vec(), dims.to_vec())),
        WeightLayout::ColParallel { world, dim } | WeightLayout::RowParallel { world, dim } => {
            (world as usize, dim)
        }
        other => bail!(
            "slice_f32_for_tp: layout {:?} not supported (Replicated / Col / RowParallel)",
            other
        ),
    };
    if world <= 1 {
        return Ok((f32s.to_vec(), dims.to_vec()));
    }
    if axis >= dims.len() {
        bail!(
            "slice_f32_for_tp: axis {axis} out of bounds for dims {:?}",
            dims
        );
    }
    let axis_len = dims[axis] as usize;
    if axis_len % world != 0 {
        bail!(
            "slice_f32_for_tp: dims[{axis}]={axis_len} not divisible by world {world}"
        );
    }
    let outer: usize = dims[..axis].iter().product::<u64>() as usize;
    let inner: usize = dims[axis + 1..].iter().product::<u64>() as usize;
    let per_rank_axis = axis_len / world;
    let r0 = rank as usize * per_rank_axis;
    let group_stride = axis_len * inner;
    let mut out = Vec::with_capacity(outer * per_rank_axis * inner);
    for o in 0..outer {
        let base = o * group_stride + r0 * inner;
        out.extend_from_slice(&f32s[base..base + per_rank_axis * inner]);
    }
    let mut new_dims = dims.to_vec();
    new_dims[axis] = per_rank_axis as u64;
    Ok((out, new_dims))
}

/// Generic TP at-load conversion: dequant `src_dtype` → F32, apply the
/// per-rank slice, quantise to Q8_0, upload. Used for any weight dtype
/// that has no native V1 kernel and reaches the TP loader — MXFP4
/// (Coder-Next-Q4_0 shared-expert FFN), IQ4_XS (UD-XL builds), and any
/// future source dtype that has a `dequantize_into` impl. Without this
/// path the TP loader bails on dtype-mismatch when handing src bytes to
/// a Q8_0-shaped dispatch.
fn upload_tp_via_dequant_to_q8_0(
    file: &GgufFile,
    name: &str,
    dims: &[u64],
    layout: WeightLayout,
    rank: u32,
    device: &HipDevice,
    src_dtype: GgmlDType,
) -> Result<(DeviceTensor, usize)> {
    let total_elems: usize = dims.iter().product::<u64>() as usize;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;
    let mut f32_full = vec![0.0f32; total_elems];
    flambeau_quant::dequantize_into(src_dtype, raw, &mut f32_full)
        .with_context(|| format!("{src_dtype:?} dequant `{name}` rank={rank}"))?;

    let (f32_slice, per_rank_dims) = slice_f32_for_tp(&f32_full, dims, layout, rank)
        .with_context(|| format!("slice F32 (post-{src_dtype:?}) `{name}` rank={rank}"))?;
    drop(f32_full);

    let target = pick_convert_target(src_dtype);
    let elems_slice = f32_slice.len();
    let bytes: Vec<u8> = match target {
        GgmlDType::Q8_0 => quantize_f32_slice_to_q8_0(&f32_slice)
            .with_context(|| format!("quantize F32→Q8_0 ({src_dtype:?}) `{name}` rank={rank}"))?,
        GgmlDType::Q4K => {
            if elems_slice % flambeau_quant::QK_K != 0 {
                bail!("`{name}` rank={rank} elem {elems_slice} not multiple of QK_K");
            }
            let mut buf = Vec::with_capacity(elems_slice / flambeau_quant::QK_K * 144);
            flambeau_quant::quantize_k::quantize_row_q4_k(&f32_slice, &mut buf);
            buf
        }
        GgmlDType::Q3K => {
            if elems_slice % flambeau_quant::QK_K != 0 {
                bail!("`{name}` rank={rank} elem {elems_slice} not multiple of QK_K");
            }
            let mut buf = Vec::with_capacity(elems_slice / flambeau_quant::QK_K * 110);
            flambeau_quant::quantize_k::quantize_row_q3_k(&f32_slice, &mut buf);
            buf
        }
        GgmlDType::Q2K => {
            if elems_slice % flambeau_quant::QK_K != 0 {
                bail!("`{name}` rank={rank} elem {elems_slice} not multiple of QK_K");
            }
            let mut buf = Vec::with_capacity(elems_slice / flambeau_quant::QK_K * 84);
            flambeau_quant::quantize_k::quantize_row_q2_k(&f32_slice, &mut buf);
            buf
        }
        other => bail!("upload_tp_via_dequant_to: unsupported target {other:?}"),
    };
    let n = bytes.len();

    let ptr = device
        .alloc(n)
        .map_err(|e| anyhow!("hipMalloc {n} B `{name}`: {e}"))?;
    // SAFETY: ptr is a fresh device alloc of n bytes; bytes is a host
    // Vec we own that lives through the synchronize at the call-site of
    // `upload_tp_with_layout`'s caller.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(bytes.as_ptr() as usize),
                n,
            )
            .map_err(|e| anyhow!("memcpy {src_dtype:?}→{target:?} `{name}` rank={rank}: {e}"))?;
    }
    device.default_stream().synchronize()?;

    let tensor = DeviceTensor {
        ptr,
        dtype: target,
        dims: per_rank_dims,
        bytes: n,
        name: Arc::from(name),
    };
    Ok((tensor, n))
}

/// Source-dtype → at-load conversion-target policy. Mirror of the same
/// function in sharded.rs — kept duplicate-but-local so the TP path
/// doesn't reach across modules for one match arm.
fn pick_convert_target(src: GgmlDType) -> GgmlDType {
    match src {
        GgmlDType::Iq4Xs | GgmlDType::Iq4Nl => GgmlDType::Q4K,
        GgmlDType::Iq3Xxs | GgmlDType::Iq3S => GgmlDType::Q3K,
        GgmlDType::Iq2Xxs | GgmlDType::Iq2Xs | GgmlDType::Iq2S
        | GgmlDType::Iq1S | GgmlDType::Iq1M => GgmlDType::Q2K,
        _ => GgmlDType::Q8_0,
    }
}

/// / #142 — TP-aware ssm_ba split.
/// qwen3next packs `ssm_alpha + ssm_beta` as one fused tensor
/// `ssm_ba.weight [2*num_v_heads, hidden]`. The on-disk layout is
/// **interleaved per K-head** as
/// `[β..., α...] × num_k_heads`, where `n_rep = num_v_heads / num_k_heads`
/// — see `sharded.rs::split_ssm_ba_to_q8_0` for the full derivation
/// against llama.cpp's `ssm_beta_alpha` view.
/// Pre-fix this routine took `alpha = rows[..num_v_heads]; beta =
/// rows[num_v_heads..]`, which (a) named them backwards and (b) mixed
/// β/α across k-heads. Latent on perf benches (synthetic prompts hide
/// the wrong scalars) but produced a degenerate logit attractor live —
/// see cert.
/// TP slicing comes after the de-interleave: split first into
/// `[num_v_heads, hidden]` α and β, then apply `ColParallel{dim=0}`
/// per rank so each rank gets `num_v_heads/world` rows of BOTH α and β
/// (vs the broken `ColParallel{dim=0}` on the fused tensor which would
/// give rank 0 all-α-mixed and rank 1 all-β-mixed at world=2).
/// Returns two TP layer-tensors: `*.ssm_alpha.weight` and
/// `*.ssm_beta.weight`, dtype Q8_0, ready for `mmvq_q8_0` consumption.
fn upload_tp_ssm_ba_split(
    file: &GgufFile,
    name: &str,
    rank: u32,
    world: u32,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
) -> Result<((DeviceTensor, usize), (DeviceTensor, usize))> {
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    if info.dims.len() != 2 {
        bail!(
            "upload_tp_ssm_ba_split: `{name}` expected 2D [2*num_v_heads, hidden], got {:?}",
            info.dims
        );
    }
    let total_rows = info.dims[0] as usize;
    let cols = info.dims[1] as usize;
    let gdn = cfg.gdn.as_ref().ok_or_else(|| {
        anyhow!(
            "upload_tp_ssm_ba_split: cfg.gdn is None for arch {} — fused ssm_ba.weight present without GDN dims",
            cfg.arch,
        )
    })?;
    let num_v_heads = gdn.num_v_heads;
    let num_k_heads = gdn.num_k_heads;
    if total_rows != 2 * num_v_heads {
        bail!(
            "upload_tp_ssm_ba_split: `{name}` rows {total_rows} != 2*num_v_heads={}",
            2 * num_v_heads,
        );
    }
    if num_v_heads % world as usize != 0 {
        bail!(
            "upload_tp_ssm_ba_split: num_v_heads {num_v_heads} not divisible by world {world}"
        );
    }
    if num_v_heads % num_k_heads != 0 {
        bail!(
            "upload_tp_ssm_ba_split: num_v_heads {num_v_heads} not divisible by num_k_heads {num_k_heads}",
        );
    }
    let n_rep = num_v_heads / num_k_heads;
    if cols % QK8_0 != 0 {
        bail!(
            "upload_tp_ssm_ba_split: cols {cols} not multiple of QK8_0={QK8_0}"
        );
    }

    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;
    let elems_total = total_rows * cols;
    let mut f32_full = vec![0.0f32; elems_total];
    flambeau_quant::dequantize_into(info.dtype, raw, &mut f32_full)
        .with_context(|| format!("dequant `{name}` (dtype={:?})", info.dtype))?;

    // De-interleave per K-head: dst row `kh*n_rep + r` ← src row
    // `kh*2*n_rep + r` (β) or `kh*2*n_rep + n_rep + r` (α).
    let half_elems = num_v_heads * cols;
    let mut alpha_dei = vec![0.0f32; half_elems];
    let mut beta_dei = vec![0.0f32; half_elems];
    for kh in 0..num_k_heads {
        for r_idx in 0..n_rep {
            let dst_row = kh * n_rep + r_idx;
            let src_beta_row = kh * (2 * n_rep) + r_idx;
            let src_alpha_row = kh * (2 * n_rep) + n_rep + r_idx;
            let dst_off = dst_row * cols;
            beta_dei[dst_off..dst_off + cols]
                .copy_from_slice(&f32_full[src_beta_row * cols..src_beta_row * cols + cols]);
            alpha_dei[dst_off..dst_off + cols]
                .copy_from_slice(&f32_full[src_alpha_row * cols..src_alpha_row * cols + cols]);
        }
    }
    drop(f32_full);
    let alpha_full = &alpha_dei[..];
    let beta_full = &beta_dei[..];
    let half_dims = vec![num_v_heads as u64, cols as u64];

    let half_layout = WeightLayout::ColParallel { world, dim: 0 };
    let (alpha_slice, alpha_dims) =
        slice_f32_for_tp(alpha_full, &half_dims, half_layout, rank)
            .context("alpha slice")?;
    let (beta_slice, beta_dims) =
        slice_f32_for_tp(beta_full, &half_dims, half_layout, rank)
            .context("beta slice")?;
    drop(alpha_dei);
    drop(beta_dei);

    let alpha_q8 = quantize_f32_slice_to_q8_0(&alpha_slice).context("alpha → Q8_0")?;
    let beta_q8 = quantize_f32_slice_to_q8_0(&beta_slice).context("beta → Q8_0")?;
    let alpha_n = alpha_q8.len();
    let beta_n = beta_q8.len();

    let stem = name.strip_suffix(".ssm_ba.weight").ok_or_else(|| {
        anyhow!("upload_tp_ssm_ba_split: name `{name}` doesn't end with .ssm_ba.weight")
    })?;
    let alpha_name = format!("{stem}.ssm_alpha.weight");
    let beta_name = format!("{stem}.ssm_beta.weight");

    let alpha_ptr = device.alloc(alpha_n)?;
    // SAFETY: fresh device alloc of `alpha_n` bytes; host buffer alive through
    // the synchronize at the load() call site.
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            alpha_ptr,
            DevicePtr(alpha_q8.as_ptr() as usize),
            alpha_n,
        )?;
    }
    let beta_ptr = device.alloc(beta_n)?;
    // SAFETY: same as above.
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            beta_ptr,
            DevicePtr(beta_q8.as_ptr() as usize),
            beta_n,
        )?;
    }
    device.default_stream().synchronize()?;

    let alpha_tensor = DeviceTensor {
        ptr: alpha_ptr,
        dtype: GgmlDType::Q8_0,
        dims: alpha_dims,
        bytes: alpha_n,
        name: Arc::from(alpha_name.as_str()),
    };
    let beta_tensor = DeviceTensor {
        ptr: beta_ptr,
        dtype: GgmlDType::Q8_0,
        dims: beta_dims,
        bytes: beta_n,
        name: Arc::from(beta_name.as_str()),
    };
    Ok(((alpha_tensor, alpha_n), (beta_tensor, beta_n)))
}

/// Slice + upload a single tensor for the given rank. Returns the
/// device tensor (with per-rank dims) and the byte count uploaded.
/// **Bug 2/3 fix**: applies the same F32→F16 norm conversion and
/// F32→Q8_0 ssm_alpha/ssm_beta quantisation that `sharded.rs::upload_layer`
/// does. The original design deferred conversion to forward
/// (see comment in `tp_target_dtype`) but the forward never actually
/// did it — `rmsnorm_quant_q8_1` reinterpreted F32 norm bytes as F16,
/// producing all-NaN logits on the first live execution.
fn upload_tp(
    file: &GgufFile,
    name: &str,
    tp: &Qwen35DenseTpLayout,
    rank: u32,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    let (t, b, _layout) = upload_tp_with_layout(file, name, tp, rank, device)?;
    Ok((t, b))
}

/// like [`upload_tp`] but returns the layout that was
/// **actually** used. Identical to the configured layout in the common
/// case. For MoE expert tensors (`ffn_*_exps.weight`) whose K-quant
/// block size doesn't divide the per-rank slice (Coder-30B Q4_K with
/// `moe_intermediate=768`, world ∈ {2, 4}, block_size=256), this
/// falls back to [`WeightLayout::Replicated`] — uploads the full
/// tensor on every rank — and records the override so the forward
/// path can skip the post-MoE AllReduce on that layer.
fn upload_tp_with_layout(
    file: &GgufFile,
    name: &str,
    tp: &Qwen35DenseTpLayout,
    rank: u32,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize, WeightLayout)> {
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    let configured = tp
        .for_tensor(name)
        .ok_or_else(|| anyhow!("no TP layout entry for tensor `{name}`"))?;

    // every IQ family now has native kernels and aligns on the
    // TP dispatch axis. MXFP4 is the only remaining source dtype that
    // needs the F32 dequant → re-slice → re-quant detour because its
    // 17-byte / 32-elem block + E8M0 scale can't be cleanly byte-sliced
    // on the TP axis.
    if matches!(info.dtype, GgmlDType::Mxfp4) {
        let layout = configured;
        let (tensor, n) = upload_tp_via_dequant_to_q8_0(
            file, name, &info.dims, layout, rank, device, info.dtype,
        )?;
        return Ok((tensor, n, layout));
    }
    let (bytes_cow, layout) = match slice_for_tp(file, name, configured, rank) {
        Ok(b) => (b, configured),
        Err(e) if is_moe_block_misalignment(&e, name) => {
            tracing::warn!(
                target: "flambeau_qwen3_moe::tp_sharded",
                tensor = name,
                rank,
                "K-quant MoE expert misalignment — falling back to Replicated upload "
            );
            let full_layout = WeightLayout::Replicated;
            let full = slice_for_tp(file, name, full_layout, rank)
                .with_context(|| format!("Replicated fallback slice `{name}` rank={rank}"))?;
            (full, full_layout)
        }
        Err(e) => {
            return Err(e).with_context(|| format!("slice_for_tp `{name}` rank={rank}"));
        }
    };
    if bytes_cow.is_empty() {
        bail!("empty slice for tensor `{name}` rank={rank}");
    }
    let per_rank_dims = compute_per_rank_dims(&info.dims, layout);
    let elems: usize = per_rank_dims.iter().product::<u64>() as usize;

    // BF16 → Q8_0 transparent quantise at load. Mirror
    // of the PP path's `upload_bf16_as_q8_0` (2.a). UD-Q8_K_XL
    // GGUFs (e.g. Qwen3.6-35B-A3B-UD-Q8_K_XL) ship a handful of BF16
    // tensors that no V1 MMVQ/MMQ path consumes; without this branch
    // the TP loader would hand BF16 bytes to a Q8_0-shaped dispatch
    // and produce garbage. ~0.4 % precision delta vs BF16, dominated
    // by Q8_0 noise elsewhere in the model.
    let (final_bytes_cow, final_dtype): (std::borrow::Cow<'_, [u8]>, GgmlDType) =
        if info.dtype == GgmlDType::BF16 {
            let converted = quantize_bf16_to_q8_0(&bytes_cow, elems)
                .with_context(|| format!("quantize BF16→Q8_0 `{name}` rank={rank}"))?;
            (std::borrow::Cow::Owned(converted), GgmlDType::Q8_0)
        } else {
            // Apply per-tensor F32→{F16,Q8_0} conversion mirroring `sharded.rs`.
            let target = tp_target_dtype(name, info.dtype);
            match target {
                None => (bytes_cow, info.dtype),
                Some(GgmlDType::F16) => {
                    let converted = convert_f32_to_f16(&bytes_cow, elems)
                        .with_context(|| format!("convert F32→F16 `{name}` rank={rank}"))?;
                    (std::borrow::Cow::Owned(converted), GgmlDType::F16)
                }
                Some(GgmlDType::Q8_0) => {
                    let converted = quantize_f32_to_q8_0(&bytes_cow, elems)
                        .with_context(|| format!("quantize F32→Q8_0 `{name}` rank={rank}"))?;
                    (std::borrow::Cow::Owned(converted), GgmlDType::Q8_0)
                }
                Some(other) => bail!("tp_target_dtype returned unsupported {other:?} for `{name}`"),
            }
        };

    let n = final_bytes_cow.len();
    let ptr = device
        .alloc(n)
        .map_err(|e| anyhow!("hipMalloc {n} B `{name}`: {e}"))?;
    // SAFETY: ptr is a fresh device alloc of n bytes; final_bytes_cow is
    // a host buffer (mmap, owned converted Vec, or owned quantised Vec)
    // of n bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(final_bytes_cow.as_ptr() as usize),
                n,
            )
            .map_err(|e| anyhow!("memcpy `{name}`: {e}"))?;
    }
    Ok((
        DeviceTensor {
            ptr,
            dtype: final_dtype,
            dims: per_rank_dims,
            bytes: n,
            name: Arc::from(name),
        },
        n,
        layout,
    ))
}

/// recognise the specific slice failure mode that
/// warrants the Replicated fallback: K-quant MoE expert tensors
/// (`ffn_*_exps.weight`) whose per-rank inner dim doesn't align with
/// the dtype's block size. Other shape mismatches still bail.
fn is_moe_block_misalignment(err: &anyhow::Error, tensor_name: &str) -> bool {
    let suffix = tensor_name.rsplit('.').next().unwrap_or(tensor_name);
    let is_moe_expert = matches!(
        suffix,
        "weight"
    ) && (tensor_name.ends_with("ffn_gate_exps.weight")
        || tensor_name.ends_with("ffn_up_exps.weight")
        || tensor_name.ends_with("ffn_down_exps.weight"));
    if !is_moe_expert {
        return false;
    }
    err.downcast_ref::<crate::tp_slice::SliceError>()
        .is_some_and(|e| matches!(e, crate::tp_slice::SliceError::InnerBlockMisaligned { .. }))
}

/// Per-rank dims after applying `layout`. Replicated returns the full
/// dims; ColParallel/RowParallel/FusedQkvParallel divide the outer
/// (or specified) dim by `world`.
fn compute_per_rank_dims(full_dims: &[u64], layout: WeightLayout) -> Vec<u64> {
    match layout {
        WeightLayout::Replicated => full_dims.to_vec(),
        WeightLayout::ColParallel { world, dim } | WeightLayout::RowParallel { world, dim } => {
            let mut d = full_dims.to_vec();
            if let Some(slot) = d.get_mut(dim) {
                *slot /= world as u64;
            }
            d
        }
        WeightLayout::FusedQkvParallel {
            world,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            kq_replicated,
        } => {
            // FusedQkv slices outer dim (dim 0). When `kq_replicated`
            // is false, all of V/K/Q divide by world (per-rank outer =
            // full_outer / world). When true, only V splits and K/Q
            // are full per rank, so per-rank outer is asymmetric.
            let mut d = full_dims.to_vec();
            if let Some(slot) = d.get_mut(0) {
                if kq_replicated {
                    let v_part = (num_v_heads as u64) * (head_v_dim as u64);
                    let k_part = (num_k_heads as u64) * (head_k_dim as u64);
                    *slot = v_part / (world as u64) + 2 * k_part;
                } else {
                    *slot /= world as u64;
                }
            }
            d
        }
    }
}

/// per-rank session holding per-layer caches sized for the
/// local head subset. Sister of [`crate::session::Qwen3MoESession`].
pub struct Qwen3MoETpSession {
    /// `caches[rank]` is the layer-cache vector for that rank, sized
    /// for `local_num_v_heads = num_v_heads / world` (GDN) and
    /// `local_num_kv_heads = num_kv_heads / world` (full-attn).
    pub caches: Vec<Vec<crate::session::LayerCache>>,
    disposed: bool,
}

impl Qwen3MoETpSession {
    /// Allocate per-rank layer caches for `model` over `cluster`.
    pub fn new(
        model: &Qwen3MoETpModel,
        cluster: &HipCluster,
        kv_layout: crate::session::KvLayout,
    ) -> Result<Self> {
        let world = model.tp.world();
        if cluster.ranks() as u32 != world {
            anyhow::bail!(
                "cluster has {} ranks, tp expects world={world}",
                cluster.ranks()
            );
        }
        let cfg = &model.config;
        let gdn_kq_replicated = model.tp.gdn_kq_replicated();
        let mut caches = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let mut layer_caches = Vec::with_capacity(cfg.num_layers);
            for il in 0..cfg.num_layers {
                layer_caches.push(crate::session::alloc_layer_cache_tp(
                    cfg,
                    device,
                    il,
                    world,
                    gdn_kq_replicated,
                    kv_layout,
                )?);
            }
            device.default_stream().synchronize()?;
            caches.push(layer_caches);
        }
        Ok(Self { caches, disposed: false })
    }

    /// true if any rank has a Q8_0 KV cache. Mirrors
    /// `Qwen3MoESession::is_q8_kv` / `Qwen3MoEShardedSession::is_q8_kv`.
    /// Used to gate the batched TP prefill path.
    pub fn is_q8_kv(&self) -> bool {
        self.caches.iter().any(|caches| {
            caches.iter().any(|c| {
                matches!(c, crate::session::LayerCache::FullAttnQ8(_))
            })
        })
    }

    /// Total bytes across all ranks. Useful for the perf-cert "session
    /// memory" line.
    pub fn total_bytes(&self) -> usize {
        let mut total = 0usize;
        for rank_caches in &self.caches {
            for c in rank_caches {
                match c {
                    crate::session::LayerCache::FullAttn(kv) => {
                        total += kv.bytes_per_tensor() * 2;
                    }
                    crate::session::LayerCache::FullAttnQ8(kv) => {
                        total += kv.bytes_per_tensor() * 2;
                    }
                    crate::session::LayerCache::Gdn(g) => {
                        total += g.state_bytes + g.conv_history_bytes;
                    }
                }
            }
        }
        total
    }

    /// save GDN state across all ranks. TP analog of
    /// [`crate::sharded::Qwen3MoEShardedSession::save_gdn_snapshot`].
    pub fn save_gdn_snapshot(&mut self, cluster: &HipCluster) -> Result<()> {
        for (rank_idx, rank_caches) in self.caches.iter_mut().enumerate() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let stream = device.default_stream();
            for (il, cache) in rank_caches.iter_mut().enumerate() {
                if let crate::session::LayerCache::Gdn(g) = cache {
                    if g.snapshot_state.is_none() {
                        let p = device.alloc(g.state_bytes).map_err(|e| {
                            anyhow::anyhow!("alloc GDN snap rank={rank_idx} layer={il}: {e}")
                        })?;
                        g.snapshot_state = Some(p);
                    }
                    if g.snapshot_conv_history.is_none() {
                        let p = device.alloc(g.conv_history_bytes).map_err(|e| {
                            anyhow::anyhow!(
                                "alloc GDN snap conv rank={rank_idx} layer={il}: {e}"
                            )
                        })?;
                        g.snapshot_conv_history = Some(p);
                    }
                    let snap_state = g.snapshot_state.unwrap();
                    let snap_conv = g.snapshot_conv_history.unwrap();
                    // SAFETY: shadow buffers same size as live; D2D memcpy.
                    unsafe {
                        device.memcpy_async(
                            stream, CopyDirection::DeviceToDevice,
                            snap_state, g.state, g.state_bytes,
                        )?;
                        device.memcpy_async(
                            stream, CopyDirection::DeviceToDevice,
                            snap_conv, g.conv_history, g.conv_history_bytes,
                        )?;
                    }
                }
            }
            <flambeau_backend_hip::HipStream as flambeau_core::Stream>::synchronize(stream)?;
        }
        Ok(())
    }

    /// restore GDN state from snapshot across all ranks.
    pub fn restore_gdn_snapshot(&mut self, cluster: &HipCluster) -> Result<()> {
        for (rank_idx, rank_caches) in self.caches.iter_mut().enumerate() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let stream = device.default_stream();
            for (il, cache) in rank_caches.iter_mut().enumerate() {
                if let crate::session::LayerCache::Gdn(g) = cache {
                    let snap_state = g.snapshot_state.ok_or_else(|| {
                        anyhow::anyhow!("restore GDN: rank={rank_idx} layer={il} no snap")
                    })?;
                    let snap_conv = g.snapshot_conv_history.ok_or_else(|| {
                        anyhow::anyhow!(
                            "restore GDN conv: rank={rank_idx} layer={il} no snap"
                        )
                    })?;
                    // SAFETY: shadow buffers same size as live; D2D memcpy.
                    unsafe {
                        device.memcpy_async(
                            stream, CopyDirection::DeviceToDevice,
                            g.state, snap_state, g.state_bytes,
                        )?;
                        device.memcpy_async(
                            stream, CopyDirection::DeviceToDevice,
                            g.conv_history, snap_conv, g.conv_history_bytes,
                        )?;
                    }
                }
            }
            <flambeau_backend_hip::HipStream as flambeau_core::Stream>::synchronize(stream)?;
        }
        Ok(())
    }

    /// roll back full-attn K/V tail by `n_remove` slots
    /// across every rank's full-attn layers.
    pub fn rollback_full_attn(&mut self, n_remove: usize) -> Result<()> {
        for (rank_idx, rank_caches) in self.caches.iter_mut().enumerate() {
            for (il, cache) in rank_caches.iter_mut().enumerate() {
                match cache {
                    crate::session::LayerCache::FullAttn(kv) => kv
                        .rollback(n_remove)
                        .map_err(|e| anyhow::anyhow!(
                            "rollback rank={rank_idx} layer={il}: {e}"
                        ))?,
                    crate::session::LayerCache::FullAttnQ8(kv) => kv
                        .rollback(n_remove)
                        .map_err(|e| anyhow::anyhow!(
                            "rollback rank={rank_idx} layer={il}: {e}"
                        ))?,
                    crate::session::LayerCache::Gdn(_) => {}
                }
            }
        }
        Ok(())
    }

    /// **P2.9a (slot pool)** — reset every rank's KV state for reuse
    /// on the next request. Same pattern as
    /// [`crate::sharded::Qwen3MoEShardedSession::reset_for_next_request`]
    /// but the per-rank vector is over `Vec<Vec<LayerCache>>` (TP
    /// fans every layer across every rank).
    pub fn reset_for_next_request(&mut self, cluster: &HipCluster) -> Result<()> {
        for (rank, layer_caches) in self.caches.iter_mut().enumerate() {
            let device = cluster.device(rank);
            device.bind()?;
            for cache in layer_caches.iter_mut() {
                match cache {
                    crate::session::LayerCache::FullAttn(kv) => kv.clear(),
                    crate::session::LayerCache::FullAttnQ8(kv) => kv.clear(),
                    crate::session::LayerCache::Gdn(g) => {
                        crate::session::zero_gdn_layer_state(device, g)?;
                    }
                }
            }
            device.default_stream().synchronize()?;
        }
        Ok(())
    }

    /// Free every rank's caches.
    pub fn dispose(mut self, cluster: &HipCluster) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut first_err: Option<anyhow::Error> = None;
        for (rank, layer_caches) in self.caches.drain(..).enumerate() {
            let device = cluster.device(rank);
            // bind once per rank's worth of disposes.
            device.bind()?;
            for cache in layer_caches {
                if let Err(e) = crate::session::dispose_layer_cache(cache, device) {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Qwen3MoETpSession {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::tp_sharded",
                ranks = self.caches.len(),
                "Qwen3MoETpSession dropped without dispose(cluster); device buffers leaked"
            );
        }
    }
}

/// Walk a [`crate::layout::LayerDescriptor`] and emit every tensor name
/// the forward path needs. Used by the TP loader to drive per-layer
/// uploads through `upload_tp`. The PP path doesn't need this because
/// it composes uploads through `upload_layer` which destructures the
/// layer descriptor directly.
fn collect_layer_tensor_names(desc: &crate::layout::LayerDescriptor) -> Vec<String> {
    use crate::layout::LayerAttnBlock;
    let mut out: Vec<String> = Vec::new();
    out.push(desc.attn_norm.name.clone());
    if let Some(t) = &desc.post_attention_norm {
        out.push(t.name.clone());
    }
    if let Some(t) = &desc.ffn_norm {
        out.push(t.name.clone());
    }
    match &desc.attn {
        LayerAttnBlock::Dense(d) => {
            out.extend([
                d.attn_q.name.clone(),
                d.attn_k.name.clone(),
                d.attn_v.name.clone(),
                d.attn_output.name.clone(),
                d.attn_q_norm.name.clone(),
                d.attn_k_norm.name.clone(),
            ]);
            for opt in [&d.attn_q_bias, &d.attn_k_bias, &d.attn_v_bias] {
                if let Some(t) = opt {
                    out.push(t.name.clone());
                }
            }
        }
        LayerAttnBlock::FullAttn(f) => {
            out.extend([
                f.attn_q.name.clone(),
                f.attn_k.name.clone(),
                f.attn_v.name.clone(),
                f.attn_output.name.clone(),
                f.attn_q_norm.name.clone(),
                f.attn_k_norm.name.clone(),
            ]);
        }
        LayerAttnBlock::Gdn(g) => {
            out.extend([
                g.attn_qkv.name.clone(),
                g.attn_gate.name.clone(),
                g.ssm_a.name.clone(),
                g.ssm_dt_bias.name.clone(),
                g.ssm_conv1d.name.clone(),
                g.ssm_norm.name.clone(),
                g.ssm_out.name.clone(),
            ]);
            if let Some(t) = &g.ssm_alpha {
                out.push(t.name.clone());
            }
            if let Some(t) = &g.ssm_beta {
                out.push(t.name.clone());
            }
            if let Some(t) = &g.ssm_ba {
                out.push(t.name.clone());
            }
        }
    }
    // FFN block — `MoeFfnTensors` carries either `dense` (qwen35 path)
    // or the MoE expert slabs.
    if let Some(d) = &desc.ffn.dense {
        out.extend([
            d.ffn_gate.name.clone(),
            d.ffn_up.name.clone(),
            d.ffn_down.name.clone(),
        ]);
    } else {
        // MoE expert tensors. Router (ffn_gate_inp) +
        // 3D expert slabs (gate/up/down).
        if let Some(t) = &desc.ffn.ffn_gate_inp {
            out.push(t.name.clone());
        }
        if let Some(t) = &desc.ffn.ffn_gate_exps {
            out.push(t.name.clone());
        }
        if let Some(t) = &desc.ffn.ffn_up_exps {
            out.push(t.name.clone());
        }
        if let Some(t) = &desc.ffn.ffn_down_exps {
            out.push(t.name.clone());
        }
        // shared expert (Replicated for now). Walked here so
        // hybrid arches (qwen35moe) load cleanly.
        if let Some(s) = &desc.ffn.shared {
            out.extend([
                s.ffn_gate_inp_shexp.name.clone(),
                s.ffn_gate_shexp.name.clone(),
                s.ffn_up_shexp.name.clone(),
                s.ffn_down_shexp.name.clone(),
            ]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AttentionFamily;

    #[test]
    fn topology_world_pp() {
        let a = LayerAssignment::contiguous(40, 4);
        let topo = Topology::Pp(a);
        assert_eq!(topo.world(), 4);
    }

    #[test]
    fn topology_world_tp() {
        let cfg = Qwen3MoEConfig {
            arch: "qwen35".into(),
            family: AttentionFamily::Hybrid,
            hidden_size: 5120,
            vocab_size: 248320,
            num_layers: 64,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            context_length: 32768,
            rms_norm_eps: 1e-6,
            rope: crate::config::RopeSpec {
                freq_base: 1_000_000.0,
                rotated_dims: 256,
                sections: None,
            },
            num_experts: 0,
            num_experts_per_tok: 1,
            moe_intermediate_size: 17408,
            shared_expert_intermediate_size: None,
            full_attention_interval: Some(4),
            gdn: None,
            tied_lm_head: false,
            pooling_type: None,
        };
        let tp = Qwen35DenseTpLayout::new(&cfg, 4).unwrap();
        let topo = Topology::Tp(tp);
        assert_eq!(topo.world(), 4);
    }

    #[test]
    fn per_rank_dims_replicated_unchanged() {
        let d = compute_per_rank_dims(&[5120, 27648], WeightLayout::Replicated);
        assert_eq!(d, vec![5120, 27648]);
    }

    #[test]
    fn per_rank_dims_col_parallel_dim0() {
        let d = compute_per_rank_dims(&[8192, 5120], WeightLayout::col_parallel(4, 0));
        assert_eq!(d, vec![2048, 5120]);
    }

    #[test]
    fn per_rank_dims_row_parallel_dim1() {
        let d = compute_per_rank_dims(&[5120, 27648], WeightLayout::row_parallel(4, 1));
        assert_eq!(d, vec![5120, 6912]);
    }
}
