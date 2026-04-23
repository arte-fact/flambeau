//! V1.7.5.A — pipeline-parallel sharded model.
//!
//! `Qwen3MoEShardedModel` holds one `Qwen3MoERankShard` per rank in a
//! `HipCluster`. Each shard owns:
//! - The subset of transformer layers assigned to its rank (by
//!   [`flambeau_runtime::LayerAssignment`]).
//! - The `OpsRegistry` for its rank's HIP device.
//! - Global tensors (`token_embd`, `output_norm`, `output`) only on the
//!   ranks that actually use them in the forward path (rank 0 for
//!   embedding, last rank for the output head).
//!
//! This is the PP-primary topology per CLAUDE.md: each layer's MoE
//! experts live on the same rank as the layer; routing/topk/indexed-MMVQ
//! are all intra-stage; only the hidden-state hand-off crosses PCIe.
//! Wide-EP (router all-to-all across ranks) is deferred to V2+.
//!
//! Load-time memory budget (Qwen3.6-31B-A3B, 4× MI50 16 GB):
//!   - 40 layers ÷ 4 ranks = 10 layers/rank
//!   - Per layer ≈ 500 MB Q4_K experts + 20 MB full-attn/GDN = ~520 MB
//!   - Per rank ≈ 5.2 GB weights + KV/GDN/scratch ≤ 10 GB → fits 16 GB.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::OpsRegistry;
use flambeau_quant::{GgmlDType, GgufFile, QK8_0};
use flambeau_runtime::{LayerAssignment, RankId};

use crate::config::Qwen3MoEConfig;
use crate::layout::{
    DenseAttnTensors, FullAttnTensors, GdnTensors, LayerAttnBlock, LayerDescriptor, ModelLayout,
    MoeFfnTensors, ResolvedTensor, SharedExpertTensors,
};
use crate::session::{alloc_layer_cache, dispose_layer_cache, LayerCache};
use crate::weights::{
    AttnWeights, DenseAttnWeights, DeviceTensor, FfnWeights, FullAttnWeights, GdnWeights,
    LayerWeights, SharedExpertWeights,
};

/// One rank's slice of the sharded model. Owns the device memory for the
/// layers assigned to this rank + the globals this rank uses.
pub struct Qwen3MoERankShard {
    pub rank: RankId,
    /// HIP device id this shard's memory lives on.
    pub device_id: i32,
    /// Kernel registry bound to this rank's device.
    pub ops: OpsRegistry,
    /// `Some` only on rank 0 — the first stage embeds the input token.
    pub token_embd: Option<DeviceTensor>,
    /// `Some` only on the last rank — the final stage applies the output
    /// norm before the LM head.
    pub output_norm: Option<DeviceTensor>,
    /// `Some` only on the last rank and only when `cfg.tied_lm_head == false`.
    /// When tied, the last rank additionally holds its own copy of the
    /// embedding (small compared to the layer weights) so the LM head
    /// matmul can read local memory.
    pub output: Option<DeviceTensor>,
    /// The subset of layers assigned to this rank, in ascending
    /// `layer_idx` order. `LayerWeights.layer_idx` preserves the global
    /// position so downstream code can look up the right residual
    /// connection / RoPE position without an extra map.
    pub layers: Vec<LayerWeights>,
    /// Bytes uploaded to this rank. Useful for pre-OOM budgeting and the
    /// smoke test's per-rank invariant check.
    total_bytes: usize,
    disposed: bool,
}

impl Qwen3MoERankShard {
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Free every device allocation owned by this shard.
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut out: Result<()> = Ok(());
        let mut free = |t: &mut DeviceTensor| {
            if !t.ptr.is_null() && t.bytes > 0 {
                // SAFETY: every pointer came from the load path's
                // `device.alloc()`; no aliasing.
                unsafe {
                    if let Err(e) = device.dealloc(t.ptr, t.bytes) {
                        if out.is_ok() {
                            out = Err(anyhow::anyhow!("hipFree `{}`: {e}", t.name));
                        }
                    }
                }
                t.ptr = DevicePtr::NULL;
                t.bytes = 0;
            }
        };
        if let Some(t) = &mut self.token_embd {
            free(t);
        }
        if let Some(t) = &mut self.output_norm {
            free(t);
        }
        if let Some(t) = &mut self.output {
            free(t);
        }
        for l in &mut self.layers {
            free(&mut l.attn_norm);
            if let Some(t) = &mut l.post_attention_norm {
                free(t);
            }
            if let Some(t) = &mut l.ffn_norm {
                free(t);
            }
            match &mut l.attn {
                AttnWeights::Dense(d) => {
                    free(&mut d.attn_q);
                    free(&mut d.attn_k);
                    free(&mut d.attn_v);
                    free(&mut d.attn_output);
                    free(&mut d.attn_q_norm);
                    free(&mut d.attn_k_norm);
                    if let Some(t) = &mut d.attn_q_bias {
                        free(t);
                    }
                    if let Some(t) = &mut d.attn_k_bias {
                        free(t);
                    }
                    if let Some(t) = &mut d.attn_v_bias {
                        free(t);
                    }
                }
                AttnWeights::FullAttn(f) => {
                    free(&mut f.attn_q);
                    free(&mut f.attn_k);
                    free(&mut f.attn_v);
                    free(&mut f.attn_output);
                    free(&mut f.attn_q_norm);
                    free(&mut f.attn_k_norm);
                }
                AttnWeights::Gdn(g) => {
                    free(&mut g.attn_qkv);
                    free(&mut g.attn_gate);
                    if let Some(t) = &mut g.ssm_alpha {
                        free(t);
                    }
                    if let Some(t) = &mut g.ssm_beta {
                        free(t);
                    }
                    if let Some(t) = &mut g.ssm_ba {
                        free(t);
                    }
                    free(&mut g.ssm_a);
                    free(&mut g.ssm_dt_bias);
                    free(&mut g.ssm_conv1d);
                    free(&mut g.ssm_norm);
                    free(&mut g.ssm_out);
                }
            }
            if let Some(t) = &mut l.ffn.ffn_gate_inp { free(t); }
            if let Some(t) = &mut l.ffn.ffn_gate_exps { free(t); }
            if let Some(t) = &mut l.ffn.ffn_up_exps { free(t); }
            if let Some(t) = &mut l.ffn.ffn_down_exps { free(t); }
            if let Some(s) = &mut l.ffn.shared {
                free(&mut s.ffn_gate_inp_shexp);
                free(&mut s.ffn_gate_shexp);
                free(&mut s.ffn_up_shexp);
                free(&mut s.ffn_down_shexp);
            }
            if let Some(d) = &mut l.ffn.dense {
                free(&mut d.ffn_gate);
                free(&mut d.ffn_up);
                free(&mut d.ffn_down);
            }
        }
        out
    }
}

impl Drop for Qwen3MoERankShard {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::sharded",
                rank = self.rank.0,
                bytes = self.total_bytes,
                "Qwen3MoERankShard dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Pipeline-parallel sharded model. One `Qwen3MoERankShard` per rank.
pub struct Qwen3MoEShardedModel {
    pub config: Qwen3MoEConfig,
    pub layout: ModelLayout,
    pub assignment: LayerAssignment,
    pub shards: Vec<Qwen3MoERankShard>,
}

impl Qwen3MoEShardedModel {
    /// Parse config + layout from `file` and upload each layer's weights
    /// to the rank assigned by `assignment`. Globals (`token_embd`,
    /// `output_norm`, `output`) go only to the ranks that consume them:
    /// rank 0 for the embedding gather, the last rank for the output
    /// head.
    ///
    /// Tied LM head: when `cfg.tied_lm_head == true` the last rank gets
    /// its own copy of `token_embd` too (the first rank needs it for the
    /// embedding gather, the last rank for the LM head matmul — can't
    /// share across PCIe without making every token incur an extra
    /// peer-copy).
    pub fn load(
        file: &GgufFile,
        cluster: &HipCluster,
        assignment: &LayerAssignment,
    ) -> Result<Self> {
        let config = Qwen3MoEConfig::from_gguf(file)?;
        let layout = ModelLayout::from_gguf(file, &config)?;

        if assignment.num_layers() != config.num_layers {
            bail!(
                "assignment.num_layers={} != cfg.num_layers={}",
                assignment.num_layers(),
                config.num_layers
            );
        }
        if assignment.num_ranks() != cluster.ranks() as u32 {
            bail!(
                "assignment.num_ranks={} != cluster.ranks={}",
                assignment.num_ranks(),
                cluster.ranks()
            );
        }

        let mut shards: Vec<Qwen3MoERankShard> = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() as u32 {
            let rank = RankId(rank_idx);
            let device = cluster.device(rank_idx as usize);
            device.bind()?;

            // 1. Upload the layers assigned to this rank.
            let mut layers = Vec::new();
            let mut bytes = 0usize;
            for il in assignment.layers_on(rank) {
                let desc = &layout.layers[il];
                let (lw, lw_bytes) = upload_layer(file, desc, device, &config)?;
                bytes += lw_bytes;
                layers.push(lw);
            }

            // 2. Upload the globals this rank needs.
            let is_first = assignment.is_first(rank);
            let is_last = assignment.is_last(rank);
            let token_embd = if is_first || (is_last && config.tied_lm_head) {
                let (t, b) = upload_one(file, &layout.token_embd, device)?;
                bytes += b;
                Some(t)
            } else {
                None
            };
            let output_norm = if is_last {
                // Global norm: F32 in Qwen3.6; cast to F16 so rmsnorm_quant_q8_1 reads it correctly.
                let (t, b) = upload_as_f16(file, &layout.output_norm, device)?;
                bytes += b;
                Some(t)
            } else {
                None
            };
            let output = if is_last {
                if let Some(r) = &layout.output {
                    let (t, b) = upload_one(file, r, device)?;
                    bytes += b;
                    Some(t)
                } else {
                    None
                }
            } else {
                None
            };

            device.default_stream().synchronize()?;

            // 3. Build the ops registry once per rank.
            let ops = OpsRegistry::new(device)
                .map_err(|e| anyhow::anyhow!("rank {}: OpsRegistry: {e}", rank_idx))?;

            shards.push(Qwen3MoERankShard {
                rank,
                device_id: device.id(),
                ops,
                token_embd,
                output_norm,
                output,
                layers,
                total_bytes: bytes,
                disposed: false,
            });
        }

        Ok(Self {
            config,
            layout,
            assignment: assignment.clone(),
            shards,
        })
    }

    /// Free every rank's shard. Caller must provide the original
    /// `HipCluster` so each shard can use its assigned `HipDevice`.
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

    /// Total bytes uploaded across all ranks. Should match
    /// `layout.total_bytes()` plus the replicated `token_embd` (for tied
    /// LM head) minus per-rank duplication from unowned globals.
    pub fn total_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.total_bytes).sum()
    }

    /// Build a sharded model from already-allocated per-rank shards.
    /// Intended for synthetic-fixture tests that bypass the GGUF upload
    /// path (mirrors `ModelWeights::from_parts`). `shards[r].rank` must
    /// equal `RankId(r as u32)` and match `assignment`'s layer-to-rank
    /// map.
    pub fn from_parts(
        config: Qwen3MoEConfig,
        layout: ModelLayout,
        assignment: LayerAssignment,
        shards: Vec<Qwen3MoERankShard>,
    ) -> Self {
        Self {
            config,
            layout,
            assignment,
            shards,
        }
    }

    /// Build a new `Qwen3MoERankShard` from already-allocated device
    /// tensors — caller guarantees layer_idx / dtype / dims are valid.
    pub fn new_shard(
        rank: RankId,
        device_id: i32,
        ops: OpsRegistry,
        token_embd: Option<DeviceTensor>,
        output_norm: Option<DeviceTensor>,
        output: Option<DeviceTensor>,
        layers: Vec<LayerWeights>,
    ) -> Qwen3MoERankShard {
        let total_bytes: usize = token_embd
            .iter()
            .chain(output_norm.iter())
            .chain(output.iter())
            .map(|t| t.bytes)
            .sum::<usize>()
            + layers
                .iter()
                .flat_map(iter_layer_tensor_bytes)
                .sum::<usize>();
        Qwen3MoERankShard {
            rank,
            device_id,
            ops,
            token_embd,
            output_norm,
            output,
            layers,
            total_bytes,
            disposed: false,
        }
    }
}

fn iter_layer_tensor_bytes(l: &LayerWeights) -> Vec<usize> {
    let mut v = vec![l.attn_norm.bytes];
    if let Some(t) = &l.post_attention_norm {
        v.push(t.bytes);
    }
    if let Some(t) = &l.ffn_norm {
        v.push(t.bytes);
    }
    match &l.attn {
        AttnWeights::Dense(d) => {
            v.extend([
                d.attn_q.bytes,
                d.attn_k.bytes,
                d.attn_v.bytes,
                d.attn_output.bytes,
                d.attn_q_norm.bytes,
                d.attn_k_norm.bytes,
            ]);
            for b in [&d.attn_q_bias, &d.attn_k_bias, &d.attn_v_bias]
                .into_iter()
                .flatten()
            {
                v.push(b.bytes);
            }
        }
        AttnWeights::FullAttn(f) => {
            v.extend([
                f.attn_q.bytes,
                f.attn_k.bytes,
                f.attn_v.bytes,
                f.attn_output.bytes,
                f.attn_q_norm.bytes,
                f.attn_k_norm.bytes,
            ]);
        }
        AttnWeights::Gdn(g) => {
            v.extend([
                g.attn_qkv.bytes,
                g.attn_gate.bytes,
                g.ssm_a.bytes,
                g.ssm_dt_bias.bytes,
                g.ssm_conv1d.bytes,
                g.ssm_norm.bytes,
                g.ssm_out.bytes,
            ]);
            for t in [&g.ssm_alpha, &g.ssm_beta, &g.ssm_ba]
                .into_iter()
                .flatten()
            {
                v.push(t.bytes);
            }
        }
    }
    for t in [
        &l.ffn.ffn_gate_inp,
        &l.ffn.ffn_gate_exps,
        &l.ffn.ffn_up_exps,
        &l.ffn.ffn_down_exps,
    ]
    .into_iter()
    .flatten()
    {
        v.push(t.bytes);
    }
    if let Some(s) = &l.ffn.shared {
        v.extend([
            s.ffn_gate_inp_shexp.bytes,
            s.ffn_gate_shexp.bytes,
            s.ffn_up_shexp.bytes,
            s.ffn_down_shexp.bytes,
        ]);
    }
    if let Some(d) = &l.ffn.dense {
        v.extend([d.ffn_gate.bytes, d.ffn_up.bytes, d.ffn_down.bytes]);
    }
    v
}

// ---------- internal helpers ----------

/// Upload one `ResolvedTensor` to `device`. Returns the `DeviceTensor` +
/// byte count. Fails if the mmap slice is shorter than declared.
fn upload_one(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    if std::env::var("FLAMBEAU_LOAD_TRACE").is_ok() {
        let t0 = std::time::Instant::now();
        let out = upload_one_inner(file, r, device);
        eprintln!("  [load] upload_one {:<50} {:?} {:>7.1} MB {:>6.1} ms",
            r.name, r.dtype, r.size_bytes as f64 / 1e6,
            t0.elapsed().as_secs_f64() * 1000.0);
        return out;
    }
    upload_one_inner(file, r, device)
}

fn upload_one_inner(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    // V2.22.a — BF16 tensors in UD-Q8_K_XL (10 total per 35B-UD-Q8_K_XL:
    // 1 attn_qkv, 1 attn_gate, 1–4 of each ffn_*_exps + shexp) aren't
    // supported by any V1 MMVQ/MMQ path. Transparently quantise to Q8_0 at
    // load so downstream dispatch flows through the Q8_0 kernels V2.22.a
    // just added. Precision loss is ~0.4 % (Q8 step vs BF16 step) — tiny
    // compared to the Q8_0 noise elsewhere in the model.
    if r.dtype == GgmlDType::BF16 {
        return upload_bf16_as_q8_0(file, r, device);
    }
    // V2.23.a — 5 Qwen3.6-35B-A3B-Q4_0 layers store `ffn_down_exps` as
    // Q4_1 (scattered among 35 Q4_0 + 5 Q4_1). Rather than author a Q4_1
    // indexed-MoE kernel for 5 tensors, convert them to Q8_0 on host at
    // load so they flow through `indexed_moe_mmvq_q8_0`. Applied only to
    // tensors whose name contains "_exps" (the MoE expert slabs) — non-MoE
    // Q4_1 (Qwen3.5-9B-Q4_1) still flows through the Q4_1 kernel.
    if r.dtype == GgmlDType::Q4_1 && r.name.contains("_exps") {
        return upload_q4_1_as_q8_0(file, r, device);
    }
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
        .map_err(|e| anyhow::anyhow!("hipMalloc {} B `{}`: {e}", bytes, r.name))?;
    // SAFETY: `ptr` is a fresh HIP allocation of `bytes` bytes;
    // `raw` is an mmap view of at least `bytes` host bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(raw.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy `{}`: {e}", r.name))?;
    }
    // V2.17: sync after every tensor to prevent HIP queue backlog on
    // large loads. `upload_one` stays non-munmapping by default so it's
    // safe for tensors shared across ranks (token_embd, output_norm,
    // output — the last rank reads them AFTER rank 0 uploads them).
    // Per-layer uploads go through `upload_one_and_drop` which also calls
    // `file.advise_drop_tensor` (munmap) to keep page cache under RAM.
    device.default_stream().synchronize()
        .map_err(|e| anyhow::anyhow!("stream sync after `{}`: {e}", r.name))?;
    Ok((
        DeviceTensor {
            ptr,
            dtype: r.dtype,
            dims: r.dims.clone(),
            bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        },
        bytes,
    ))
}

/// Upload a tensor as F16 — if source is already F16, as-is; if source is
/// F32, cast on host then upload. All V1 norm slots (attn_norm,
/// post_attention_norm, ffn_norm, attn_q_norm, attn_k_norm, output_norm)
/// go through here because Qwen3.6's GGUF stores them F32 but our
/// rmsnorm kernels expect F16 weight.
fn upload_as_f16(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    if r.dtype == GgmlDType::F16 {
        return upload_one(file, r, device);
    }
    if r.dtype != GgmlDType::F32 {
        bail!(
            "upload_as_f16: tensor `{}` has unsupported source dtype {:?}",
            r.name,
            r.dtype
        );
    }
    let raw = file
        .tensor_raw(&r.name)
        .with_context(|| format!("tensor_raw `{}`", r.name))?;
    let src: &[f32] = bytemuck::cast_slice(raw);
    let elems: usize = r.dims.iter().product::<u64>() as usize;
    if src.len() < elems {
        bail!(
            "upload_as_f16: `{}` mmap slice {} < expected {}",
            r.name,
            src.len(),
            elems
        );
    }
    let host: Vec<half::f16> = src[..elems]
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let bytes = host.len() * 2;
    let ptr = device.alloc(bytes)?;
    // SAFETY: `ptr` is a fresh HIP allocation; `host` has `bytes` valid host bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(host.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy (f32→f16) `{}`: {e}", r.name))?;
    }
    device.default_stream().synchronize()?;
    drop(host);
    // Same drop-only-for-layer-private rule as upload_one; safe to drop here
    // because upload_as_f16 is called for per-layer norms + the global
    // output_norm, but output_norm is only uploaded by is_last rank, no
    // sharing conflict. Explicit drop of the staged `host` Vec already frees
    // the converted buffer; the mmap source pages are released by madvise
    // in the caller via `up_f16_drop` when appropriate.
    Ok((
        DeviceTensor {
            ptr,
            dtype: GgmlDType::F16,
            dims: r.dims.clone(),
            bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        },
        bytes,
    ))
}

/// Upload a tensor as Q8_0 — if source is already Q8_0, as-is; if source
/// is F32, quantize on host then upload. V1 uses this for Qwen3.6's
/// `ssm_alpha.weight` / `ssm_beta.weight` which are stored F32 but fed
/// to `mmvq` (which requires a Q-quant). Quant scheme matches
/// `flambeau_block_q8_0`: per-32-elem block absmax / 127 → d (F16),
/// qs[i] = round(x[i] / d) clamped to [-127, 127].
/// V2.22.a — BF16 tensor → Q8_0 on device. BF16's 16 bits are the upper half
/// of a F32, so widen to F32 byte-by-byte, then use the standard Q8_0 per-32-
/// block absmax/127 quantise. Preserves dims + name; sets dtype=Q8_0 so
/// the downstream dispatch treats it like any other Q8_0 weight.
fn upload_bf16_as_q8_0(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    let raw = file
        .tensor_raw(&r.name)
        .with_context(|| format!("tensor_raw `{}`", r.name))?;
    let elems: usize = r.dims.iter().product::<u64>() as usize;
    let src_bytes = elems * 2;
    if raw.len() < src_bytes {
        bail!(
            "upload_bf16_as_q8_0: `{}` mmap slice {} < expected {} ({} elems × 2)",
            r.name, raw.len(), src_bytes, elems
        );
    }
    if elems % QK8_0 != 0 {
        bail!(
            "upload_bf16_as_q8_0: `{}` elem count {elems} not multiple of QK8_0={QK8_0}",
            r.name
        );
    }
    let src_u16: &[u16] = bytemuck::cast_slice(&raw[..src_bytes]);
    // Widen to F32 once so the quant inner loop stays hot.
    let src_f32: Vec<f32> = src_u16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let n_blocks = elems / QK8_0;
    let block_size = 34usize;
    let out_bytes = n_blocks * block_size;
    let mut buf: Vec<u8> = Vec::with_capacity(out_bytes);
    for block in src_f32.chunks_exact(QK8_0) {
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
    debug_assert_eq!(buf.len(), out_bytes);
    let ptr = device.alloc(out_bytes)?;
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(buf.as_ptr() as usize),
                out_bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy (bf16→Q8_0) `{}`: {e}", r.name))?;
    }
    device.default_stream().synchronize()?;
    drop(buf);
    drop(src_f32);
    Ok((
        DeviceTensor {
            ptr,
            dtype: GgmlDType::Q8_0,
            dims: r.dims.clone(),
            bytes: out_bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        },
        out_bytes,
    ))
}

/// V2.23.a — Q4_1 MoE tensor → Q8_0 on device. Q4_1 stores 20 bytes/block:
/// `d (f16) + m (f16) + 16 × 4-bit nibbles`, reconstruction `y = d·q + m`
/// where `q ∈ [0, 15]`. Dequantise to F32 on host, then apply the standard
/// absmax/127 Q8_0 encoder. ~0.4 % additional noise vs Q4_1's own noise.
fn upload_q4_1_as_q8_0(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    const QK4_1: usize = 32;
    const Q4_1_BLOCK: usize = 2 + 2 + QK4_1 / 2; // d + m + 16 nibbles = 20 B
    let elems: usize = r.dims.iter().product::<u64>() as usize;
    if elems % QK4_1 != 0 {
        bail!(
            "upload_q4_1_as_q8_0: `{}` elem count {elems} not multiple of QK4_1={QK4_1}",
            r.name
        );
    }
    let raw = file
        .tensor_raw(&r.name)
        .with_context(|| format!("tensor_raw `{}`", r.name))?;
    let n_blocks = elems / QK4_1;
    let expected_bytes = n_blocks * Q4_1_BLOCK;
    if raw.len() < expected_bytes {
        bail!(
            "upload_q4_1_as_q8_0: `{}` mmap slice {} < expected {}",
            r.name, raw.len(), expected_bytes
        );
    }
    // Dequantise to F32 on host. Same ggml layout as mmvq_q4_1: byte i of
    // qs holds low-nibble at element i, high-nibble at element i+16.
    let mut dq = vec![0.0f32; elems];
    for bi in 0..n_blocks {
        let off = bi * Q4_1_BLOCK;
        let d = half::f16::from_bits(u16::from_le_bytes([raw[off], raw[off + 1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([raw[off + 2], raw[off + 3]])).to_f32();
        let qs = &raw[off + 4..off + 20];
        for i in 0..16 {
            let lo = (qs[i] & 0x0F) as f32;
            let hi = ((qs[i] >> 4) & 0x0F) as f32;
            dq[bi * QK4_1 + i] = d * lo + m;
            dq[bi * QK4_1 + i + 16] = d * hi + m;
        }
    }
    // Now encode to Q8_0 via the same absmax/127 path used in upload_as_q8_0.
    let out_n_blocks = elems / QK8_0;
    let out_bytes = out_n_blocks * 34;
    let mut buf: Vec<u8> = Vec::with_capacity(out_bytes);
    for block in dq.chunks_exact(QK8_0) {
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
    debug_assert_eq!(buf.len(), out_bytes);
    let ptr = device.alloc(out_bytes)?;
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(buf.as_ptr() as usize),
                out_bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy (Q4_1→Q8_0) `{}`: {e}", r.name))?;
    }
    device.default_stream().synchronize()?;
    drop(buf);
    drop(dq);
    Ok((
        DeviceTensor {
            ptr,
            dtype: GgmlDType::Q8_0,
            dims: r.dims.clone(),
            bytes: out_bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        },
        out_bytes,
    ))
}

fn upload_as_q8_0(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    if std::env::var("FLAMBEAU_LOAD_TRACE").is_ok() {
        let t0 = std::time::Instant::now();
        let out = upload_as_q8_0_inner(file, r, device);
        eprintln!("  [load] as_q8_0    {:<50} {:?} {:>7.1} MB {:>6.1} ms",
            r.name, r.dtype, r.size_bytes as f64 / 1e6,
            t0.elapsed().as_secs_f64() * 1000.0);
        return out;
    }
    upload_as_q8_0_inner(file, r, device)
}

fn upload_as_q8_0_inner(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    if r.dtype == GgmlDType::Q8_0 {
        return upload_one_inner(file, r, device);
    }
    if r.dtype != GgmlDType::F32 {
        bail!(
            "upload_as_q8_0: tensor `{}` has unsupported source dtype {:?}",
            r.name,
            r.dtype
        );
    }
    let raw = file
        .tensor_raw(&r.name)
        .with_context(|| format!("tensor_raw `{}`", r.name))?;
    let src: &[f32] = bytemuck::cast_slice(raw);
    let elems: usize = r.dims.iter().product::<u64>() as usize;
    if src.len() < elems {
        bail!(
            "upload_as_q8_0: `{}` mmap slice {} < expected {}",
            r.name,
            src.len(),
            elems
        );
    }
    if elems % QK8_0 != 0 {
        bail!(
            "upload_as_q8_0: `{}` elem count {elems} not multiple of QK8_0={QK8_0}",
            r.name
        );
    }
    let n_blocks = elems / QK8_0;
    // `BlockQ8_0` layout (see `flambeau_quant::BlockQ8_0`): 2-byte d +
    // 32-byte qs = 34 bytes per block.
    let block_size = 34usize;
    let bytes = n_blocks * block_size;
    let mut buf: Vec<u8> = Vec::with_capacity(bytes);
    for block in src[..elems].chunks_exact(QK8_0) {
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
    debug_assert_eq!(buf.len(), bytes);
    let ptr = device.alloc(bytes)?;
    // SAFETY: `ptr` has `bytes` valid HIP bytes; `buf` has `bytes` valid host bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(buf.as_ptr() as usize),
                bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy (f32→Q8_0) `{}`: {e}", r.name))?;
    }
    device.default_stream().synchronize()?;
    drop(buf);
    Ok((
        DeviceTensor {
            ptr,
            dtype: GgmlDType::Q8_0,
            dims: r.dims.clone(),
            bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        },
        bytes,
    ))
}

/// Upload a full `LayerDescriptor`'s tensors to `device`, returning a
/// populated `LayerWeights` + total byte count for this layer.
fn upload_layer(
    file: &GgufFile,
    desc: &LayerDescriptor,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
) -> Result<(LayerWeights, usize)> {
    let mut bytes = 0usize;

    // Norm slots: rmsnorm kernels all expect F16 weight. Qwen3.6 stores
    // them F32 in the GGUF; cast on load.
    let attn_norm = up_f16(file, &desc.attn_norm, device, &mut bytes)?;
    let post_attention_norm = desc
        .post_attention_norm
        .as_ref()
        .map(|t| up_f16(file, t, device, &mut bytes))
        .transpose()?;
    let ffn_norm = desc
        .ffn_norm
        .as_ref()
        .map(|t| up_f16(file, t, device, &mut bytes))
        .transpose()?;

    let attn = match &desc.attn {
        LayerAttnBlock::Dense(d) => {
            AttnWeights::Dense(upload_dense(d, file, device, &mut bytes)?)
        }
        LayerAttnBlock::FullAttn(f) => {
            AttnWeights::FullAttn(upload_full(f, file, device, &mut bytes)?)
        }
        LayerAttnBlock::Gdn(g) => {
            AttnWeights::Gdn(upload_gdn(g, file, device, &mut bytes)?)
        }
    };
    let ffn = upload_ffn(&desc.ffn, file, device, &mut bytes)?;
    let _ = cfg; // cfg is future-proofing — may influence optional uploads.

    Ok((
        LayerWeights {
            layer_idx: desc.layer_idx,
            attn_norm,
            post_attention_norm,
            ffn_norm,
            attn,
            ffn,
        },
        bytes,
    ))
}

/// Upload the tensor as-is (dtype preserved) and add byte count to `total`.
fn up_raw(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
    total: &mut usize,
) -> Result<DeviceTensor> {
    let (t, b) = upload_one(file, r, device)?;
    *total += b;
    // V2.17: this helper is called ONLY from per-layer upload paths
    // (upload_dense / upload_full_attn / upload_gdn / upload_ffn). Each
    // layer's tensors are uploaded exactly once (by the owning rank) so
    // it's safe to munmap their mmap ranges here — keeps page cache
    // under host RAM for GGUFs larger than RAM.
    file.advise_drop_tensor(&r.name);
    Ok(t)
}

/// Upload the tensor as F16 (casting F32 source on host if needed).
fn up_f16(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
    total: &mut usize,
) -> Result<DeviceTensor> {
    let (t, b) = upload_as_f16(file, r, device)?;
    *total += b;
    file.advise_drop_tensor(&r.name);
    Ok(t)
}

/// Upload the tensor as Q8_0 (quantising F32 source on host if needed).
fn up_q8_0(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
    total: &mut usize,
) -> Result<DeviceTensor> {
    let (t, b) = upload_as_q8_0(file, r, device)?;
    *total += b;
    file.advise_drop_tensor(&r.name);
    Ok(t)
}

fn upload_dense(
    d: &DenseAttnTensors,
    file: &GgufFile,
    device: &HipDevice,
    total: &mut usize,
) -> Result<DenseAttnWeights> {
    Ok(DenseAttnWeights {
        attn_q: up_raw(file, &d.attn_q, device, total)?,
        attn_k: up_raw(file, &d.attn_k, device, total)?,
        attn_v: up_raw(file, &d.attn_v, device, total)?,
        attn_output: up_raw(file, &d.attn_output, device, total)?,
        // Per-head Q/K rmsnorm weights — F32 in Qwen3.6, F16 elsewhere.
        attn_q_norm: up_f16(file, &d.attn_q_norm, device, total)?,
        attn_k_norm: up_f16(file, &d.attn_k_norm, device, total)?,
        attn_q_bias: d
            .attn_q_bias
            .as_ref()
            .map(|t| up_raw(file, t, device, total))
            .transpose()?,
        attn_k_bias: d
            .attn_k_bias
            .as_ref()
            .map(|t| up_raw(file, t, device, total))
            .transpose()?,
        attn_v_bias: d
            .attn_v_bias
            .as_ref()
            .map(|t| up_raw(file, t, device, total))
            .transpose()?,
    })
}

fn upload_full(
    f: &FullAttnTensors,
    file: &GgufFile,
    device: &HipDevice,
    total: &mut usize,
) -> Result<FullAttnWeights> {
    Ok(FullAttnWeights {
        attn_q: up_raw(file, &f.attn_q, device, total)?,
        attn_k: up_raw(file, &f.attn_k, device, total)?,
        attn_v: up_raw(file, &f.attn_v, device, total)?,
        attn_output: up_raw(file, &f.attn_output, device, total)?,
        attn_q_norm: up_f16(file, &f.attn_q_norm, device, total)?,
        attn_k_norm: up_f16(file, &f.attn_k_norm, device, total)?,
    })
}

fn upload_gdn(
    g: &GdnTensors,
    file: &GgufFile,
    device: &HipDevice,
    total: &mut usize,
) -> Result<GdnWeights> {
    Ok(GdnWeights {
        attn_qkv: up_raw(file, &g.attn_qkv, device, total)?,
        attn_gate: up_raw(file, &g.attn_gate, device, total)?,
        // ssm_alpha / ssm_beta are F32 in Qwen3.6; quantise to Q8_0 on
        // host at load so the existing mmvq_q8_0 path handles them.
        ssm_alpha: g
            .ssm_alpha
            .as_ref()
            .map(|t| up_q8_0(file, t, device, total))
            .transpose()?,
        ssm_beta: g
            .ssm_beta
            .as_ref()
            .map(|t| up_q8_0(file, t, device, total))
            .transpose()?,
        ssm_ba: g
            .ssm_ba
            .as_ref()
            .map(|t| up_raw(file, t, device, total))
            .transpose()?,
        // ssm_a / ssm_dt / ssm_conv1d / ssm_norm stay F32 — they're
        // consumed by F32-native ops.
        ssm_a: up_raw(file, &g.ssm_a, device, total)?,
        ssm_dt_bias: up_raw(file, &g.ssm_dt_bias, device, total)?,
        ssm_conv1d: up_raw(file, &g.ssm_conv1d, device, total)?,
        ssm_norm: up_raw(file, &g.ssm_norm, device, total)?,
        ssm_out: up_raw(file, &g.ssm_out, device, total)?,
    })
}

fn upload_ffn(
    f: &MoeFfnTensors,
    file: &GgufFile,
    device: &HipDevice,
    total: &mut usize,
) -> Result<FfnWeights> {
    let opt_up_raw = |t: &Option<ResolvedTensor>, total: &mut usize| -> Result<Option<DeviceTensor>> {
        t.as_ref().map(|r| up_raw(file, r, device, total)).transpose()
    };
    let dense = f
        .dense
        .as_ref()
        .map(|d| -> Result<crate::weights::DenseFfnWeights> {
            Ok(crate::weights::DenseFfnWeights {
                ffn_gate: up_raw(file, &d.ffn_gate, device, total)?,
                ffn_up: up_raw(file, &d.ffn_up, device, total)?,
                ffn_down: up_raw(file, &d.ffn_down, device, total)?,
            })
        })
        .transpose()?;
    Ok(FfnWeights {
        ffn_gate_inp: opt_up_raw(&f.ffn_gate_inp, total)?,
        ffn_gate_exps: opt_up_raw(&f.ffn_gate_exps, total)?,
        ffn_up_exps: opt_up_raw(&f.ffn_up_exps, total)?,
        ffn_down_exps: opt_up_raw(&f.ffn_down_exps, total)?,
        shared: f
            .shared
            .as_ref()
            .map(|s| upload_shared(s, file, device, total))
            .transpose()?,
        dense,
    })
}

fn upload_shared(
    s: &SharedExpertTensors,
    file: &GgufFile,
    device: &HipDevice,
    total: &mut usize,
) -> Result<SharedExpertWeights> {
    Ok(SharedExpertWeights {
        ffn_gate_inp_shexp: up_raw(file, &s.ffn_gate_inp_shexp, device, total)?,
        ffn_gate_shexp: up_raw(file, &s.ffn_gate_shexp, device, total)?,
        ffn_up_shexp: up_raw(file, &s.ffn_up_shexp, device, total)?,
        ffn_down_shexp: up_raw(file, &s.ffn_down_shexp, device, total)?,
    })
}

// ---------------------------------------------------------------------------
// Per-rank session (KV + GDN state + conv history) for a sharded model.
// ---------------------------------------------------------------------------

/// One rank's slice of the decode-time mutable state. Holds exactly one
/// `LayerCache` per layer this rank owns, in the same order as
/// `Qwen3MoERankShard::layers`.
pub struct Qwen3MoERankSession {
    pub rank: RankId,
    pub device_id: i32,
    /// Parallel to the shard's `layers` vec. Index `i` here is layer
    /// `shard.layers[i].layer_idx` in global space.
    pub caches: Vec<LayerCache>,
    disposed: bool,
}

impl Qwen3MoERankSession {
    pub fn caches_mut(&mut self) -> &mut [LayerCache] {
        &mut self.caches
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut first_err: Option<anyhow::Error> = None;
        for cache in self.caches.drain(..) {
            if let Err(e) = dispose_layer_cache(cache, device) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

impl Drop for Qwen3MoERankSession {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::sharded",
                rank = self.rank.0,
                "Qwen3MoERankSession dropped without dispose(device); caches leaked"
            );
        }
    }
}

/// Pipeline-parallel decode-time session. One `Qwen3MoERankSession` per
/// rank. KV cache for a full-attn layer, GDN state + conv history for a
/// GDN layer — all allocated on the rank that owns the layer.
pub struct Qwen3MoEShardedSession {
    pub per_rank: Vec<Qwen3MoERankSession>,
}

impl Qwen3MoEShardedSession {
    /// Allocate caches for every layer, placing each on the rank that
    /// `model.assignment` assigns it to. `cluster.ranks()` must match
    /// `model.shards.len()`.
    pub fn new(model: &Qwen3MoEShardedModel, cluster: &HipCluster) -> Result<Self> {
        if cluster.ranks() != model.shards.len() {
            bail!(
                "Qwen3MoEShardedSession::new: cluster.ranks={} != model.shards.len()={}",
                cluster.ranks(),
                model.shards.len()
            );
        }
        let mut per_rank = Vec::with_capacity(model.shards.len());
        for (rank_idx, shard) in model.shards.iter().enumerate() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let mut caches = Vec::with_capacity(shard.layers.len());
            for lw in &shard.layers {
                caches.push(alloc_layer_cache(&model.config, device, lw.layer_idx)?);
            }
            device.default_stream().synchronize()?;
            per_rank.push(Qwen3MoERankSession {
                rank: shard.rank,
                device_id: device.id(),
                caches,
                disposed: false,
            });
        }
        Ok(Self { per_rank })
    }

    pub fn dispose(mut self, cluster: &HipCluster) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for s in self.per_rank.drain(..) {
            let rank_idx = s.rank.0 as usize;
            if let Err(e) = s.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}
