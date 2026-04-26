//! TP-1c — TP-sharded model + topology selector.
//!
//! `Qwen3MoETpModel` is the TP analogue of [`crate::sharded::Qwen3MoEShardedModel`]:
//! every rank holds *all* layers, but each layer's weight tensors are
//! sliced according to [`crate::Qwen35DenseTpLayout`]. The peer model
//! (PP) stays untouched in `sharded.rs`; the two are selected via the
//! [`Topology`] enum.
//!
//! ## Design split with TP-1b
//!
//! TP-1b (`tp_slice.rs`) provides the host-side byte-slicing primitive
//! (`slice_for_tp` + `slice_bytes_for_tp`). TP-1c composes it with HIP
//! upload to produce per-rank `LayerWeights` structures.
//!
//! ## What's in scope this session
//!
//! - `Topology { Pp(LayerAssignment), Tp(Qwen35DenseTpLayout) }` — the
//!   selector consumed by future loaders.
//! - `Qwen3MoETpRankShard` / `Qwen3MoETpModel` types holding sliced
//!   weights per rank.
//! - `load()` that walks the model layout, slices per-tensor, allocates
//!   device memory, and uploads. **Dtype conversions** (F32→F16 norms,
//!   F32→Q8_0 ssm scalars, BF16→Q8_0) the PP path applies on load are
//!   intentionally **deferred to TP-2**: TP-1c uploads bytes-as-is so
//!   the forward path can reach for typed loaders when it needs them
//!   (this matches V2.32.a's posture: "do conversion where it pays off",
//!   not at every load site).
//!
//! ## What's deferred
//!
//! - Per-layer upload of `attn_qkv` and `ssm_conv1d` is correct (they
//!   stay `Replicated` per the layout table) but TP-4a will replace
//!   them with head-aware sharding.
//! - Tied LM head gets a Replicated `output` ptr at every rank; no
//!   special-case sharing.
//! - The TP-aware forward path lands in TP-2.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_quant::GgufFile;
use flambeau_runtime::{LayerAssignment, RankId, WeightLayout};

use crate::config::Qwen3MoEConfig;
use crate::layout::ModelLayout;
use crate::tp_layout::Qwen35DenseTpLayout;
use crate::tp_slice::slice_for_tp;
use crate::weights::DeviceTensor;

/// How weights are distributed across the mesh.
///
/// Forward paths and the loader switch on this. Pure-PP runs continue
/// to use [`Topology::Pp`]; new TP runs use [`Topology::Tp`]. Hybrid
/// `PpTp` is reserved for V2-tp-5b.
#[derive(Debug, Clone)]
pub enum Topology {
    /// Pipeline parallelism — `LayerAssignment` distributes whole
    /// layers across ranks. The existing V1.7.5 path.
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
#[derive(Debug)]
pub struct Qwen3MoETpModel {
    pub config: Qwen3MoEConfig,
    pub layout: ModelLayout,
    pub tp: Qwen35DenseTpLayout,
    pub shards: Vec<Qwen3MoETpRankShard>,
}

impl Qwen3MoETpModel {
    /// Open `file`, slice each tensor by the layout in `tp`, and upload
    /// to every rank in `cluster`. The cluster's rank count must match
    /// `tp.world()`.
    ///
    /// # Errors
    /// - Cluster size ≠ `tp.world()`.
    /// - Any per-tensor slice / upload failure (propagated).
    pub fn load(file: &GgufFile, cluster: &HipCluster, tp: Qwen35DenseTpLayout) -> Result<Self> {
        if cluster.ranks() as u32 != tp.world() {
            bail!(
                "cluster has {} ranks, tp expects world={}",
                cluster.ranks(),
                tp.world()
            );
        }
        let config = Qwen3MoEConfig::from_gguf(file)?;
        let layout = ModelLayout::from_gguf(file, &config)?;

        let mut shards = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() as u32 {
            let rank = RankId(rank_idx);
            let device = cluster.device(rank_idx as usize);
            device.bind()?;

            // Globals — every rank gets a full copy (V1 layout).
            let (token_embd, b1) = upload_tp(file, &layout.token_embd.name, &tp, rank_idx, device)?;
            let (output_norm, b2) =
                upload_tp(file, &layout.output_norm.name, &tp, rank_idx, device)?;
            let (output, b3) = if let Some(o) = &layout.output {
                let (t, b) = upload_tp(file, &o.name, &tp, rank_idx, device)?;
                (Some(t), b)
            } else {
                (None, 0)
            };
            let mut total_bytes = b1 + b2 + b3;

            // Per-layer tensors — every layer goes onto every rank,
            // each tensor sliced per its layout.
            let mut layers: Vec<Vec<TpLayerTensor>> = Vec::with_capacity(layout.layers.len());
            for desc in &layout.layers {
                let mut layer_tensors: Vec<TpLayerTensor> = Vec::new();
                for name in collect_layer_tensor_names(desc) {
                    let (tensor, b) = upload_tp(file, &name, &tp, rank_idx, device)?;
                    total_bytes += b;
                    let layout_for = tp
                        .for_tensor(&name)
                        .expect("collect_layer_tensor_names returns only known names");
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

        Ok(Self {
            config,
            layout,
            tp,
            shards,
        })
    }

    /// Total bytes uploaded across all ranks. Useful for the smoke
    /// invariant in TP-1c: `total_bytes == ranks × per_rank_target`.
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

/// Slice + upload a single tensor for the given rank. Returns the
/// device tensor (with per-rank dims) and the byte count uploaded.
///
/// Dtype conversions (F32→F16 norms etc.) the PP path applies on load
/// are intentionally **not** applied here; the forward path (TP-2)
/// reads bytes-as-is and converts where it consumes them.
fn upload_tp(
    file: &GgufFile,
    name: &str,
    tp: &Qwen35DenseTpLayout,
    rank: u32,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    let layout = tp
        .for_tensor(name)
        .ok_or_else(|| anyhow!("no TP layout entry for tensor `{name}`"))?;
    let bytes_cow = slice_for_tp(file, name, layout, rank)
        .with_context(|| format!("slice_for_tp `{name}` rank={rank}"))?;
    let n = bytes_cow.len();
    if n == 0 {
        bail!("empty slice for tensor `{name}` rank={rank}");
    }
    let ptr = device
        .alloc(n)
        .map_err(|e| anyhow!("hipMalloc {n} B `{name}`: {e}"))?;
    // SAFETY: ptr is a fresh device alloc of n bytes; bytes_cow is a
    // host buffer (mmap or owned Vec) of n bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(bytes_cow.as_ptr() as usize),
                n,
            )
            .map_err(|e| anyhow!("memcpy `{name}`: {e}"))?;
    }
    let per_rank_dims = compute_per_rank_dims(&info.dims, layout);
    Ok((
        DeviceTensor {
            ptr,
            dtype: info.dtype,
            dims: per_rank_dims,
            bytes: n,
            name: Arc::from(name),
        },
        n,
    ))
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
        WeightLayout::FusedQkvParallel { world, .. } => {
            // FusedQkv slices outer dim (dim 0) into V/K/Q sub-slabs;
            // each sub-slab divides by world, so the resulting outer
            // dim is full_outer / world.
            let mut d = full_dims.to_vec();
            if let Some(slot) = d.get_mut(0) {
                *slot /= world as u64;
            }
            d
        }
    }
}

/// **TP-2e** — per-rank session holding per-layer caches sized for the
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
    pub fn new(model: &Qwen3MoETpModel, cluster: &HipCluster) -> Result<Self> {
        let world = model.tp.world();
        if cluster.ranks() as u32 != world {
            anyhow::bail!(
                "cluster has {} ranks, tp expects world={world}",
                cluster.ranks()
            );
        }
        let cfg = &model.config;
        let mut caches = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let mut layer_caches = Vec::with_capacity(cfg.num_layers);
            for il in 0..cfg.num_layers {
                layer_caches
                    .push(crate::session::alloc_layer_cache_tp(cfg, device, il, world)?);
            }
            device.default_stream().synchronize()?;
            caches.push(layer_caches);
        }
        Ok(Self { caches, disposed: false })
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
                    crate::session::LayerCache::Gdn(g) => {
                        total += g.state_bytes + g.conv_history_bytes;
                    }
                }
            }
        }
        total
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
        // **TP-4b** — MoE expert tensors. Router (ffn_gate_inp) +
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
        // TP-4c: shared expert (Replicated for now). Walked here so
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
