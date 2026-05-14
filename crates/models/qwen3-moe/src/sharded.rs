//! pipeline-parallel sharded model.
//! `Qwen3MoEShardedModel` holds one `Qwen3MoERankShard` per rank in a
//! `HipCluster`. Each shard owns:
//! - The subset of transformer layers assigned to its rank (by
//! [`flambeau_runtime::LayerAssignment`]).
//! - The `OpsRegistry` for its rank's HIP device.
//! - Global tensors (`token_embd`, `output_norm`, `output`) only on the
//! ranks that actually use them in the forward path (rank 0 for
//! embedding, last rank for the output head).
//! This is the PP-primary topology per CLAUDE.md: each layer's MoE
//! experts live on the same rank as the layer; routing/topk/indexed-MMVQ
//! are all intra-stage; only the hidden-state hand-off crosses PCIe.
//! Wide-EP (router all-to-all across ranks) is deferred to V2+.
//! Load-time memory budget (Qwen3.6-31B-A3B, 4× MI50 16 GB):
//! - 40 layers ÷ 4 ranks = 10 layers/rank
//! - Per layer ≈ 500 MB Q4_K experts + 20 MB full-attn/GDN = ~520 MB
//! - Per rank ≈ 5.2 GB weights + KV/GDN/scratch ≤ 10 GB → fits 16 GB.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{HipCluster, HipDevice};
use flambeau_blocks::{LmHead, OutputNorm, RawAllocTracker, TokenEmbd, WeightRole, WeightUploader};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::OpsRegistry;
use flambeau_quant::{GgmlDType, GgufFile, QK8_0};
use flambeau_runtime::{LayerAssignment, RankId};

use crate::config::Qwen3MoEConfig;
use crate::layout::{
    DenseAttnTensors, FullAttnTensors, GdnTensors, LayerAttnBlock, LayerDescriptor, ModelLayout,
    MoeFfnTensors, ResolvedTensor, SharedExpertTensors,
};
use crate::session::{alloc_layer_cache, dispose_layer_cache, zero_gdn_layer_state, LayerCache};
use crate::weights::{
    AttnWeights, DenseAttnWeights, DeviceTensor, FfnWeights, FullAttnWeights, GdnWeights,
    LayerWeights, SharedExpertWeights,
};

#[cfg(feature = "dev_trace")]
fn dev_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}
#[cfg(not(feature = "dev_trace"))]
#[inline(always)]
fn dev_flag(_name: &str) -> bool {
    false
}

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

            // 2. Upload the globals this rank needs — typed `WeightRole`
            // dispatch picks the right primitive (raw memcpy for token_embd
            // + output, F32→F16 cast for output_norm).
            let is_first = assignment.is_first(rank);
            let is_last = assignment.is_last(rank);
            let token_embd = if is_first || (is_last && config.tied_lm_head) {
                let (t, b) =
                    upload_global_role::<TokenEmbd>(file, &config, device, &layout.token_embd)?;
                bytes += b;
                Some(t)
            } else {
                None
            };
            let output_norm = if is_last {
                let (t, b) =
                    upload_global_role::<OutputNorm>(file, &config, device, &layout.output_norm)?;
                bytes += b;
                Some(t)
            } else {
                None
            };
            let output = if is_last {
                if let Some(r) = &layout.output {
                    let (t, b) = upload_global_role::<LmHead>(file, &config, device, r)?;
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
/// Upload a global tensor via the typed `WeightRole` uploader. The
/// per-rank shard still tracks the resulting alloc through its own
/// `total_bytes` counter + the `DeviceTensor` returned here; the
/// tracker created inside is drained via `forget_allocs` so its `Drop`
/// doesn't double-free.
fn upload_global_role<R: WeightRole>(
    file: &GgufFile,
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    layout_tensor: &ResolvedTensor,
) -> Result<(DeviceTensor, usize)> {
    let stream = device.default_stream();
    let mut tracker = RawAllocTracker::new();
    let mut up = WeightUploader {
        device,
        stream,
        tracker: &mut tracker,
        file,
        cfg,
        world: 1,
        rank: 0,
    };
    let uploaded = up.upload_required::<R>(0)?;
    drop(up);
    // Tracker still holds the (ptr, bytes) record from the upload.
    // The shard takes ownership via DeviceTensor below, so drain
    // without freeing.
    let _ = tracker.forget_allocs();
    Ok((
        DeviceTensor {
            ptr: uploaded.ptr,
            dtype: uploaded.dtype,
            dims: layout_tensor.dims.clone(),
            bytes: uploaded.bytes,
            name: std::sync::Arc::from(layout_tensor.name.as_str()),
        },
        uploaded.bytes,
    ))
}

fn upload_one(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    if dev_flag("FLAMBEAU_LOAD_TRACE") {
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
    // 2.a — BF16 tensors in UD-Q8_K_XL (10 total per 35B-UD-Q8_K_XL:
    // 1 attn_qkv, 1 attn_gate, 1–4 of each ffn_*_exps + shexp) aren't
    // supported by any V1 MMVQ/MMQ path. Transparently quantise to Q8_0 at
    // load so downstream dispatch flows through the Q8_0 kernels 2.a
    // just added. Precision loss is ~0.4 % (Q8 step vs BF16 step) — tiny
    // compared to the Q8_0 noise elsewhere in the model.
    // Source dtypes with no native V1 kernel get dequantised on host then
    // re-encoded as Q8_0 at upload time. Same shape for BF16 (1 element per
    // ggml block), MXFP4 (32-elem block + E8M0 microscale), IQ4_XS (256-elem
    // K-quant + IQ4_NL LUT), and the scattered Q4_1 MoE expert tensors on
    // Qwen3.6-35B-A3B-Q4_0 (5 layers ship `ffn_down_exps` as Q4_1 amid the
    // Q4_0 bulk; converting on load avoids authoring a Q4_1 indexed-MoE
    // kernel for 5 tensors). Non-MoE Q4_1 still flows through the Q4_1 path.
    // every IQ family now has a native MMVQ;
    // they flow through `upload_one` (mmap → memcpy, llama.cpp-style).
    // Only BF16, MXFP4, and the scattered Q4_1 MoE expert tensors on
    // Qwen3.6-35B-A3B-Q4_0 still take the host F32 → Q8_0 convert path.
    let needs_q8_0_convert = matches!(r.dtype, GgmlDType::BF16 | GgmlDType::Mxfp4)
        || (r.dtype == GgmlDType::Q4_1 && r.name.contains("_exps"));
    if needs_q8_0_convert {
        return upload_via_dequant_to(file, r, device);
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
    // 7: sync after every tensor to prevent HIP queue backlog on
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


/// Generic at-load conversion path: dequantise `r` from its source dtype
/// to F32 on host, then encode the F32 buffer as Q8_0 and upload. Used
/// for every weight dtype that has no native V1 kernel but reaches the
/// loader: BF16, MXFP4, IQ-family (IQ2_XXS..IQ4_XS, IQ1_S, IQ1_M), and
/// the scattered Q4_1 MoE expert tensors on Qwen3.6-35B-A3B-Q4_0.
///
/// Target dtype is chosen by source dtype to keep the convert close to
/// source bpw (Q8_0 target on a 3-bpw source blows VRAM up ~2.7× —
/// fatal on 122B-class models). Per-target re-quant noise:
/// Q8_0 ~0.1 %, Q4_K ~0.5 %, Q3_K ~1.5 %, Q2_K ~1.8 % — all small vs
/// the source IQ noise itself.
fn upload_via_dequant_to(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
) -> Result<(DeviceTensor, usize)> {
    let elems: usize = r.dims.iter().product::<u64>() as usize;
    let raw = file
        .tensor_raw(&r.name)
        .with_context(|| format!("tensor_raw `{}`", r.name))?;
    let mut f32_full = vec![0.0f32; elems];
    flambeau_quant::dequantize_into(r.dtype, raw, &mut f32_full)
        .with_context(|| format!("dequant {:?} `{}`", r.dtype, r.name))?;

    let target = pick_convert_target(r.dtype);
    let buf: Vec<u8> = match target {
        GgmlDType::Q8_0 => {
            if elems % QK8_0 != 0 {
                bail!(
                    "upload_via_dequant_to: `{}` elem count {elems} not multiple of QK8_0={QK8_0}",
                    r.name
                );
            }
            let mut buf = Vec::with_capacity(elems / QK8_0 * 34);
            flambeau_quant::quantize_k::quantize_row_q8_0(&f32_full, &mut buf);
            buf
        }
        GgmlDType::Q4K => {
            if elems % flambeau_quant::QK_K != 0 {
                bail!("`{}` elem count {elems} not multiple of QK_K", r.name);
            }
            let mut buf = Vec::with_capacity(elems / flambeau_quant::QK_K * 144);
            flambeau_quant::quantize_k::quantize_row_q4_k(&f32_full, &mut buf);
            buf
        }
        GgmlDType::Q3K => {
            if elems % flambeau_quant::QK_K != 0 {
                bail!("`{}` elem count {elems} not multiple of QK_K", r.name);
            }
            let mut buf = Vec::with_capacity(elems / flambeau_quant::QK_K * 110);
            flambeau_quant::quantize_k::quantize_row_q3_k(&f32_full, &mut buf);
            buf
        }
        GgmlDType::Q2K => {
            if elems % flambeau_quant::QK_K != 0 {
                bail!("`{}` elem count {elems} not multiple of QK_K", r.name);
            }
            let mut buf = Vec::with_capacity(elems / flambeau_quant::QK_K * 84);
            flambeau_quant::quantize_k::quantize_row_q2_k(&f32_full, &mut buf);
            buf
        }
        other => bail!("upload_via_dequant_to: unsupported target dtype {other:?}"),
    };
    let out_bytes = buf.len();
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
            .map_err(|e| anyhow::anyhow!("memcpy ({:?}→{:?}) `{}`: {e}", r.dtype, target, r.name))?;
    }
    device.default_stream().synchronize()?;
    drop(buf);
    drop(f32_full);
    Ok((
        DeviceTensor {
            ptr,
            dtype: target,
            dims: r.dims.clone(),
            bytes: out_bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        },
        out_bytes,
    ))
}

/// Source-dtype → at-load conversion-target policy. IQ-family with
/// matching K-quant capacity goes to that K-quant (~1.0–1.4× source
/// bpw); everything else falls back to Q8_0 (the BF16 / MXFP4 /
/// Q4_1-exps cohort, where memory blow-up is bounded and the simpler
/// quant scheme is cheaper).
fn pick_convert_target(src: GgmlDType) -> GgmlDType {
    match src {
        GgmlDType::Iq4Xs | GgmlDType::Iq4Nl => GgmlDType::Q4K,
        GgmlDType::Iq3Xxs | GgmlDType::Iq3S => GgmlDType::Q3K,
        GgmlDType::Iq2Xxs | GgmlDType::Iq2Xs | GgmlDType::Iq2S
        | GgmlDType::Iq1S | GgmlDType::Iq1M => GgmlDType::Q2K,
        _ => GgmlDType::Q8_0,
    }
}

/// V1.x #120 — Split qwen3next's fused `ssm_ba.weight`
/// `[2*num_v_heads, hidden]` tensor into two Q8_0 device tensors
/// `ssm_alpha` + `ssm_beta`, each `[num_v_heads, hidden]`.
/// **Layout** (matches `llama.cpp/src/models/qwen3next.cpp`'s
/// `ssm_beta_alpha` view): the rows are NOT a contiguous `[α | β]`
/// block — they are interleaved per K-head as
/// `[β₀..β_{n_rep-1}, α₀..α_{n_rep-1}] × num_k_heads`, where
/// `n_rep = num_v_heads / num_k_heads`. After matmul against the
/// hidden activation, llama.cpp does
/// `reshape_4d(out, ba_dim=2*n_rep, num_k_heads, …)` → `view(b @ off=0,
/// size=n_rep)` and `view(a @ off=n_rep, size=n_rep)` per K-head, then
/// reshapes the α slice back to `[num_v_heads]` by merging k-head and
/// inner dims. The β slice is the FIRST half of every k-head block,
/// the α slice is the SECOND half — and the names matter (see
/// `pattern_ssm_beta_alpha = "blk\\.\\d*\\.ssm_ba.weight"` in
/// `llama-model.cpp:59`).
/// Pre-fix this routine took
/// `alpha = rows[..num_v_heads]; beta = rows[num_v_heads..]`, which
/// (a) names them backwards and (b) takes contiguous halves that mix
/// β/α across k-heads. The math runs without crashing on any prompt
/// (shape is preserved) but the GDN recurrence evolves with garbage
/// scalars, producing a degenerate logit attractor (e.g. always
/// argmax `**`). 35B-A3B doesn't hit this path because its GGUF
/// stores `ssm_alpha`/`ssm_beta` separately. CLOSES the
/// Coder-Next-80B coherence regression observed live on flambeau
/// serve at curl temp=0 → "** ** ** ** …".
/// Source dtype is whatever the GGUF stored (Q4_0 for Coder-Next-Q4_0,
/// Q4_K for Coder-Next-UD-Q4_K_*): dequant→F32 host-side, gather
/// interleaved rows, re-quantise each half to Q8_0 with the standard
/// absmax/127 encoder. Output is two `DeviceTensor` with dtype=Q8_0
/// ready for `mmvq_q8_0` and the 7.c gate+up fusion path.
fn split_ssm_ba_to_q8_0(
    file: &GgufFile,
    r: &ResolvedTensor,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
) -> Result<(DeviceTensor, DeviceTensor)> {
    if r.dims.len() != 2 {
        bail!(
            "split_ssm_ba_to_q8_0: `{}` expected 2D `[2*num_v_heads, hidden]`, got {:?}",
            r.name, r.dims
        );
    }
    let total_rows = r.dims[0] as usize;
    let cols = r.dims[1] as usize;
    let gdn = cfg.gdn.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "split_ssm_ba_to_q8_0: cfg.gdn is None for arch {:?} — fused ssm_ba.weight present without GDN dims",
            cfg.arch,
        )
    })?;
    let num_v_heads = gdn.num_v_heads;
    let num_k_heads = gdn.num_k_heads;
    if total_rows != 2 * num_v_heads {
        bail!(
            "split_ssm_ba_to_q8_0: `{}` rows {total_rows} != 2*num_v_heads={}",
            r.name, 2 * num_v_heads,
        );
    }
    if num_v_heads % num_k_heads != 0 {
        bail!(
            "split_ssm_ba_to_q8_0: num_v_heads {num_v_heads} not divisible by num_k_heads {num_k_heads}",
        );
    }
    let n_rep = num_v_heads / num_k_heads;
    let elems_total = total_rows * cols;
    if cols % QK8_0 != 0 {
        bail!(
            "split_ssm_ba_to_q8_0: `{}` cols {cols} not multiple of QK8_0={QK8_0}",
            r.name
        );
    }

    // 1. Dequant whatever source dtype to F32, host-side.
    let raw = file
        .tensor_raw(&r.name)
        .with_context(|| format!("tensor_raw `{}`", r.name))?;
    let mut f32_buf = vec![0.0f32; elems_total];
    flambeau_quant::dequantize_into(r.dtype, raw, &mut f32_buf)
        .with_context(|| format!("dequant `{}` (dtype={:?})", r.name, r.dtype))?;

    // 2. Gather interleaved rows. Source layout (row-major):
    // rows = [β_{kh=0,r=0}, β_{kh=0,r=1}, …, β_{kh=0,r=n_rep-1},
    // α_{kh=0,r=0}, …, α_{kh=0,r=n_rep-1},
    // β_{kh=1,r=0}, …]
    // Total rows: 2 * n_rep * num_k_heads = 2 * num_v_heads.
    // Beta rows are at row index `kh * 2*n_rep + r` for r ∈ [0, n_rep).
    // Alpha rows are at row index `kh * 2*n_rep + n_rep + r` for r ∈ [0, n_rep).
    // Outputs:
    // alpha_buf[(kh * n_rep + r) * cols + c] = src[(kh*2*n_rep + n_rep + r)*cols + c]
    // beta_buf [(kh * n_rep + r) * cols + c] = src[(kh*2*n_rep + r)*cols + c]
    // The destination layout `[num_v_heads, hidden]` matches the
    // qwen35moe split-tensor convention so the existing GDN forward
    // path consumes them unchanged.
    let half_elems = num_v_heads * cols;
    let mut alpha_f32 = vec![0.0f32; half_elems];
    let mut beta_f32 = vec![0.0f32; half_elems];
    for kh in 0..num_k_heads {
        for r_idx in 0..n_rep {
            let dst_row = kh * n_rep + r_idx;
            let src_beta_row = kh * (2 * n_rep) + r_idx;
            let src_alpha_row = kh * (2 * n_rep) + n_rep + r_idx;
            let dst_off = dst_row * cols;
            beta_f32[dst_off..dst_off + cols]
                .copy_from_slice(&f32_buf[src_beta_row * cols..src_beta_row * cols + cols]);
            alpha_f32[dst_off..dst_off + cols]
                .copy_from_slice(&f32_buf[src_alpha_row * cols..src_alpha_row * cols + cols]);
        }
    }
    drop(f32_buf);

    // 3. Encode each half as Q8_0 (standard absmax/127 per 32-element block).
    let q8_block_bytes = 2 + QK8_0; // d_f16 + 32 i8
    let half_blocks = half_elems / QK8_0;
    let half_bytes = half_blocks * q8_block_bytes;

    let encode_half = |slice: &[f32]| -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(half_bytes);
        for block in slice.chunks_exact(QK8_0) {
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
        debug_assert_eq!(buf.len(), half_bytes);
        buf
    };

    let alpha_buf = encode_half(&alpha_f32);
    let beta_buf = encode_half(&beta_f32);
    drop(alpha_f32);
    drop(beta_f32);

    // 3. Upload each half. Strip the `.ssm_ba` suffix and append .ssm_{alpha,beta}.
    let stem = r.name.strip_suffix(".ssm_ba.weight").ok_or_else(|| {
        anyhow::anyhow!("split_ssm_ba: name `{}` doesn't end with .ssm_ba.weight", r.name)
    })?;
    let alpha_name: std::sync::Arc<str> = format!("{stem}.ssm_alpha.weight").into();
    let beta_name: std::sync::Arc<str> = format!("{stem}.ssm_beta.weight").into();
    let half_dims: Vec<u64> = vec![num_v_heads as u64, cols as u64];

    let alpha_ptr = device.alloc(half_bytes)?;
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                alpha_ptr,
                DevicePtr(alpha_buf.as_ptr() as usize),
                half_bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy split-alpha `{}`: {e}", r.name))?;
    }
    let beta_ptr = device.alloc(half_bytes)?;
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                beta_ptr,
                DevicePtr(beta_buf.as_ptr() as usize),
                half_bytes,
            )
            .map_err(|e| anyhow::anyhow!("memcpy split-beta `{}`: {e}", r.name))?;
    }
    device.default_stream().synchronize()?;
    drop(alpha_buf);
    drop(beta_buf);

    Ok((
        DeviceTensor {
            ptr: alpha_ptr,
            dtype: GgmlDType::Q8_0,
            dims: half_dims.clone(),
            bytes: half_bytes,
            name: alpha_name,
        },
        DeviceTensor {
            ptr: beta_ptr,
            dtype: GgmlDType::Q8_0,
            dims: half_dims,
            bytes: half_bytes,
            name: beta_name,
        },
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
    // them F32 in the GGUF; cast on load — the shared `AttnNorm` /
    // `PostAttnNorm` / `FfnNorm` roles declare `DtypeFilter::F32ToF16Norm`
    // which does the cast.
    let attn_norm = up_role::<flambeau_blocks::AttnNorm>(
        file,
        cfg,
        &desc.attn_norm,
        desc.layer_idx,
        device,
        &mut bytes,
    )?;
    let post_attention_norm = desc
        .post_attention_norm
        .as_ref()
        .map(|t| {
            up_role::<flambeau_blocks::PostAttnNorm>(
                file,
                cfg,
                t,
                desc.layer_idx,
                device,
                &mut bytes,
            )
        })
        .transpose()?;
    let ffn_norm = desc
        .ffn_norm
        .as_ref()
        .map(|t| {
            up_role::<flambeau_blocks::FfnNorm>(file, cfg, t, desc.layer_idx, device, &mut bytes)
        })
        .transpose()?;

    let attn = match &desc.attn {
        LayerAttnBlock::Dense(d) => {
            AttnWeights::Dense(upload_dense(d, file, desc.layer_idx, cfg, device, &mut bytes)?)
        }
        LayerAttnBlock::FullAttn(f) => AttnWeights::FullAttn(upload_full(
            f,
            file,
            desc.layer_idx,
            cfg,
            device,
            &mut bytes,
        )?),
        LayerAttnBlock::Gdn(g) => {
            AttnWeights::Gdn(upload_gdn(g, file, desc.layer_idx, cfg, device, &mut bytes)?)
        }
    };
    let ffn = upload_ffn(&desc.ffn, file, desc.layer_idx, cfg, device, &mut bytes)?;

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
    // 7: this helper is called ONLY from per-layer upload paths
    // (upload_dense / upload_full_attn / upload_gdn / upload_ffn). Each
    // layer's tensors are uploaded exactly once (by the owning rank) so
    // it's safe to munmap their mmap ranges here — keeps page cache
    // under host RAM for GGUFs larger than RAM.
    file.advise_drop_tensor(&r.name);
    Ok(t)
}

/// Upload one tensor via a typed `WeightRole`, fold the byte count into
/// `total`, and fire `advise_drop_tensor` to free the mmap page cache
/// (matches the legacy `up_raw` / `up_f16` behaviour). Used for the
/// per-layer dense-attn migration.
fn up_role<R: WeightRole>(
    file: &GgufFile,
    cfg: &Qwen3MoEConfig,
    r: &ResolvedTensor,
    layer_idx: usize,
    device: &HipDevice,
    total: &mut usize,
) -> Result<DeviceTensor> {
    let stream = device.default_stream();
    let mut tracker = RawAllocTracker::new();
    let mut up = WeightUploader {
        device,
        stream,
        tracker: &mut tracker,
        file,
        cfg,
        world: 1,
        rank: 0,
    };
    let uploaded = up.upload_required::<R>(layer_idx)?;
    drop(up);
    let _ = tracker.forget_allocs();
    *total += uploaded.bytes;
    file.advise_drop_tensor(&r.name);
    Ok(DeviceTensor {
        ptr: uploaded.ptr,
        dtype: uploaded.dtype,
        dims: r.dims.clone(),
        bytes: uploaded.bytes,
        name: std::sync::Arc::from(r.name.as_str()),
    })
}

/// Optional variant — returns `None` when the role's tensor is absent
/// from the GGUF.
fn up_role_opt<R: WeightRole>(
    file: &GgufFile,
    cfg: &Qwen3MoEConfig,
    r: &ResolvedTensor,
    layer_idx: usize,
    device: &HipDevice,
    total: &mut usize,
) -> Result<Option<DeviceTensor>> {
    let stream = device.default_stream();
    let mut tracker = RawAllocTracker::new();
    let mut up = WeightUploader {
        device,
        stream,
        tracker: &mut tracker,
        file,
        cfg,
        world: 1,
        rank: 0,
    };
    let result = up.upload::<R>(layer_idx)?;
    drop(up);
    let _ = tracker.forget_allocs();
    Ok(result.map(|uploaded| {
        *total += uploaded.bytes;
        file.advise_drop_tensor(&r.name);
        DeviceTensor {
            ptr: uploaded.ptr,
            dtype: uploaded.dtype,
            dims: r.dims.clone(),
            bytes: uploaded.bytes,
            name: std::sync::Arc::from(r.name.as_str()),
        }
    }))
}

fn upload_dense(
    d: &DenseAttnTensors,
    file: &GgufFile,
    layer_idx: usize,
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    total: &mut usize,
) -> Result<DenseAttnWeights> {
    use flambeau_blocks::{AttnK, AttnKBias, AttnKNorm, AttnOutput, AttnQ, AttnQBias, AttnQNorm, AttnV, AttnVBias};
    Ok(DenseAttnWeights {
        attn_q: up_role::<AttnQ>(file, cfg, &d.attn_q, layer_idx, device, total)?,
        attn_k: up_role::<AttnK>(file, cfg, &d.attn_k, layer_idx, device, total)?,
        attn_v: up_role::<AttnV>(file, cfg, &d.attn_v, layer_idx, device, total)?,
        attn_output: up_role::<AttnOutput>(file, cfg, &d.attn_output, layer_idx, device, total)?,
        // Per-head Q/K rmsnorm weights — F32 in Qwen3.6, F16 elsewhere
        // (the role's F32ToF16Norm filter handles both).
        attn_q_norm: up_role::<AttnQNorm>(file, cfg, &d.attn_q_norm, layer_idx, device, total)?,
        attn_k_norm: up_role::<AttnKNorm>(file, cfg, &d.attn_k_norm, layer_idx, device, total)?,
        attn_q_bias: d
            .attn_q_bias
            .as_ref()
            .map(|t| up_role_opt::<AttnQBias>(file, cfg, t, layer_idx, device, total))
            .transpose()?
            .flatten(),
        attn_k_bias: d
            .attn_k_bias
            .as_ref()
            .map(|t| up_role_opt::<AttnKBias>(file, cfg, t, layer_idx, device, total))
            .transpose()?
            .flatten(),
        attn_v_bias: d
            .attn_v_bias
            .as_ref()
            .map(|t| up_role_opt::<AttnVBias>(file, cfg, t, layer_idx, device, total))
            .transpose()?
            .flatten(),
    })
}

fn upload_full(
    f: &FullAttnTensors,
    file: &GgufFile,
    layer_idx: usize,
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    total: &mut usize,
) -> Result<FullAttnWeights> {
    use flambeau_blocks::{AttnK, AttnKNorm, AttnOutput, AttnQ, AttnQNorm, AttnV};
    // Note: qwen3.5's `attn_q` is stored on disk as `[2 * q_width, hidden]`
    // (Q concatenated with the output gate along the outer dim). At PP
    // (world=1) the role's `shape_for = [q_width, hidden]` is unused;
    // upload just memcpys the full tensor at its declared size. TP (S4)
    // slicing of `attn_q` with `ColParallel { dim: 0 }` cleanly splits
    // both halves because the descriptor's `dims` carry the full 2×
    // width.
    Ok(FullAttnWeights {
        attn_q: up_role::<AttnQ>(file, cfg, &f.attn_q, layer_idx, device, total)?,
        attn_k: up_role::<AttnK>(file, cfg, &f.attn_k, layer_idx, device, total)?,
        attn_v: up_role::<AttnV>(file, cfg, &f.attn_v, layer_idx, device, total)?,
        attn_output: up_role::<AttnOutput>(file, cfg, &f.attn_output, layer_idx, device, total)?,
        attn_q_norm: up_role::<AttnQNorm>(file, cfg, &f.attn_q_norm, layer_idx, device, total)?,
        attn_k_norm: up_role::<AttnKNorm>(file, cfg, &f.attn_k_norm, layer_idx, device, total)?,
    })
}

fn upload_gdn(
    g: &GdnTensors,
    file: &GgufFile,
    layer_idx: usize,
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    total: &mut usize,
) -> Result<GdnWeights> {
    use flambeau_blocks::{
        AttnGate, FusedAttnQkv, SsmA, SsmAlpha, SsmBa, SsmBeta, SsmConv1d, SsmDtBias, SsmNorm,
        SsmOut,
    };
    // V1.x #120 / #142 — qwen3next packs ssm_alpha + ssm_beta as one
    // fused `ssm_ba.weight` tensor `[2*num_v_heads, hidden]`. The on-disk
    // layout is INTERLEAVED per K-head as `[β..., α...] × num_k_heads`,
    // matching llama.cpp's `ssm_beta_alpha` view; see
    // `split_ssm_ba_to_q8_0` for details. The split helper produces two
    // Q8_0 DeviceTensors directly — stays inline because it's a 1-in /
    // 2-out transform that doesn't fit the WeightRole shape.
    let split_ba = g.ssm_alpha.is_none() && g.ssm_beta.is_none() && g.ssm_ba.is_some();
    let (alpha_dt, beta_dt, ba_dt) = if split_ba {
        let ba = g.ssm_ba.as_ref().unwrap();
        let (a, b) = split_ssm_ba_to_q8_0(file, ba, device, cfg)?;
        *total += a.bytes + b.bytes;
        (Some(a), Some(b), None)
    } else {
        // SsmAlpha / SsmBeta roles have `pre_upload = quant_f32_to_q8_0`
        // which transcodes F32 → Q8_0 at load (or passes through Q8_0).
        let alpha = g
            .ssm_alpha
            .as_ref()
            .map(|t| up_role::<SsmAlpha>(file, cfg, t, layer_idx, device, total))
            .transpose()?;
        let beta = g
            .ssm_beta
            .as_ref()
            .map(|t| up_role::<SsmBeta>(file, cfg, t, layer_idx, device, total))
            .transpose()?;
        let ba = g
            .ssm_ba
            .as_ref()
            .map(|t| up_role::<SsmBa>(file, cfg, t, layer_idx, device, total))
            .transpose()?;
        (alpha, beta, ba)
    };
    Ok(GdnWeights {
        attn_qkv: up_role::<FusedAttnQkv>(file, cfg, &g.attn_qkv, layer_idx, device, total)?,
        attn_gate: up_role::<AttnGate>(file, cfg, &g.attn_gate, layer_idx, device, total)?,
        ssm_alpha: alpha_dt,
        ssm_beta: beta_dt,
        ssm_ba: ba_dt,
        // ssm_a / ssm_dt / ssm_conv1d / ssm_norm stay F32 — they're
        // consumed by F32-native ops; roles preserve dtype.
        ssm_a: up_role::<SsmA>(file, cfg, &g.ssm_a, layer_idx, device, total)?,
        ssm_dt_bias: up_role::<SsmDtBias>(file, cfg, &g.ssm_dt_bias, layer_idx, device, total)?,
        ssm_conv1d: up_role::<SsmConv1d>(file, cfg, &g.ssm_conv1d, layer_idx, device, total)?,
        ssm_norm: up_role::<SsmNorm>(file, cfg, &g.ssm_norm, layer_idx, device, total)?,
        ssm_out: up_role::<SsmOut>(file, cfg, &g.ssm_out, layer_idx, device, total)?,
    })
}

fn upload_ffn(
    f: &MoeFfnTensors,
    file: &GgufFile,
    layer_idx: usize,
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    total: &mut usize,
) -> Result<FfnWeights> {
    use flambeau_blocks::{
        FfnDown, FfnGate, FfnUp, MoeExpertsDown, MoeExpertsGate, MoeExpertsUp, MoeRouter,
    };
    // `MoeRouter` role uses `DtypeFilter::F32ToF16Norm` so F32 routers
    // are cast to F16 at upload (halves HBM bandwidth for the router
    // GEMV); other dtypes pass through unchanged via the role path.
    // Note: the legacy path only F16-cast when source was F32; the
    // typed role's F32ToF16Norm filter handles both F32 (cast) and F16
    // (raw upload). Sources already quantised (Q8_0 etc) would fail
    // the cast — none of the qwen3-moe variants have non-F32/F16
    // routers, but if that changes the role needs an `Any` variant.
    let dense = f
        .dense
        .as_ref()
        .map(|d| -> Result<crate::weights::DenseFfnWeights> {
            Ok(crate::weights::DenseFfnWeights {
                ffn_gate: up_role::<FfnGate>(file, cfg, &d.ffn_gate, layer_idx, device, total)?,
                ffn_up: up_role::<FfnUp>(file, cfg, &d.ffn_up, layer_idx, device, total)?,
                ffn_down: up_role::<FfnDown>(file, cfg, &d.ffn_down, layer_idx, device, total)?,
            })
        })
        .transpose()?;
    Ok(FfnWeights {
        ffn_gate_inp: f
            .ffn_gate_inp
            .as_ref()
            .map(|r| {
                // The role declares `DtypeFilter::F32ToF16Norm`. Some
                // GGUFs ship the router pre-quantised; in that case
                // fall back to `up_raw` rather than rejecting.
                if r.dtype == GgmlDType::F32 || r.dtype == GgmlDType::F16 {
                    up_role::<MoeRouter>(file, cfg, r, layer_idx, device, total).map(Some)
                } else {
                    up_raw(file, r, device, total).map(Some)
                }
            })
            .transpose()?
            .flatten(),
        ffn_gate_exps: f
            .ffn_gate_exps
            .as_ref()
            .map(|r| up_role::<MoeExpertsGate>(file, cfg, r, layer_idx, device, total))
            .transpose()?,
        ffn_up_exps: f
            .ffn_up_exps
            .as_ref()
            .map(|r| up_role::<MoeExpertsUp>(file, cfg, r, layer_idx, device, total))
            .transpose()?,
        ffn_down_exps: f
            .ffn_down_exps
            .as_ref()
            .map(|r| up_role::<MoeExpertsDown>(file, cfg, r, layer_idx, device, total))
            .transpose()?,
        shared: f
            .shared
            .as_ref()
            .map(|s| upload_shared(s, file, layer_idx, cfg, device, total))
            .transpose()?,
        dense,
    })
}

fn upload_shared(
    s: &SharedExpertTensors,
    file: &GgufFile,
    layer_idx: usize,
    cfg: &Qwen3MoEConfig,
    device: &HipDevice,
    total: &mut usize,
) -> Result<SharedExpertWeights> {
    use flambeau_blocks::{SharedExpertDown, SharedExpertGate, SharedExpertRouter, SharedExpertUp};
    Ok(SharedExpertWeights {
        ffn_gate_inp_shexp: up_role::<SharedExpertRouter>(
            file,
            cfg,
            &s.ffn_gate_inp_shexp,
            layer_idx,
            device,
            total,
        )?,
        ffn_gate_shexp: up_role::<SharedExpertGate>(
            file,
            cfg,
            &s.ffn_gate_shexp,
            layer_idx,
            device,
            total,
        )?,
        ffn_up_shexp: up_role::<SharedExpertUp>(
            file,
            cfg,
            &s.ffn_up_shexp,
            layer_idx,
            device,
            total,
        )?,
        ffn_down_shexp: up_role::<SharedExpertDown>(
            file,
            cfg,
            &s.ffn_down_shexp,
            layer_idx,
            device,
            total,
        )?,
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
    /// true if any rank has a Q8_0 KV cache. Mirrors
    /// `Qwen3MoESession::is_q8_kv`. Used to gate batched-prefill paths
    /// since Q8 KV has no batched-prefill kernel; prefill loops the
    /// per-token decode path instead.
    pub fn is_q8_kv(&self) -> bool {
        self.per_rank.iter().any(|s| {
            s.caches
                .iter()
                .any(|c| matches!(c, crate::session::LayerCache::FullAttnQ8(_)))
        })
    }

    /// Allocate caches for every layer, placing each on the rank that
    /// `model.assignment` assigns it to. `cluster.ranks()` must match
    /// `model.shards.len()`.
    pub fn new(
        model: &Qwen3MoEShardedModel,
        cluster: &HipCluster,
        kv_layout: crate::session::KvLayout,
    ) -> Result<Self> {
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
                caches.push(alloc_layer_cache(&model.config, device, lw.layer_idx, kv_layout)?);
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

    /// **P2.9a (slot pool)** — reset KV state across every rank for
    /// reuse on the next request. Mirrors
    /// [`Qwen3MoESession::reset_for_next_request`] but walks the
    /// per-rank vector. Cheap O(num_layers) — just zeros GDN state +
    /// conv_history (recurrent) and clears full-attn `current_tokens`.
    pub fn reset_for_next_request(&mut self, cluster: &HipCluster) -> Result<()> {
        for rank_session in self.per_rank.iter_mut() {
            let rank_idx = rank_session.rank.0 as usize;
            let device = cluster.device(rank_idx);
            device.bind()?;
            for cache in rank_session.caches.iter_mut() {
                match cache {
                    crate::session::LayerCache::FullAttn(kv) => kv.clear(),
                    crate::session::LayerCache::FullAttnQ8(kv) => kv.clear(),
                    crate::session::LayerCache::Gdn(g) => {
                        zero_gdn_layer_state(device, g)?;
                    }
                }
            }
            device.default_stream().synchronize()?;
        }
        Ok(())
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

    /// speculative-decode snapshot across all ranks. Loops
    /// over each rank's local layers; only GDN layers store data
    /// (full-attn relies on `rollback_full_attn` for cheaper recovery).
    pub fn save_gdn_snapshot(&mut self, cluster: &HipCluster) -> Result<()> {
        for rank_session in &mut self.per_rank {
            let rank_idx = rank_session.rank.0 as usize;
            let device = cluster.device(rank_idx);
            device.bind()?;
            let stream = device.default_stream();
            for (il, cache) in rank_session.caches.iter_mut().enumerate() {
                if let crate::session::LayerCache::Gdn(g) = cache {
                    if g.snapshot_state.is_none() {
                        let p = device.alloc(g.state_bytes).map_err(|e| {
                            anyhow!("alloc GDN snapshot rank={rank_idx} layer={il}: {e}")
                        })?;
                        g.snapshot_state = Some(p);
                    }
                    if g.snapshot_conv_history.is_none() {
                        let p = device.alloc(g.conv_history_bytes).map_err(|e| {
                            anyhow!("alloc GDN snapshot conv rank={rank_idx} layer={il}: {e}")
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
            stream.synchronize()?;
        }
        Ok(())
    }

    /// restore GDN state from snapshot across all ranks.
    pub fn restore_gdn_snapshot(&mut self, cluster: &HipCluster) -> Result<()> {
        for rank_session in &mut self.per_rank {
            let rank_idx = rank_session.rank.0 as usize;
            let device = cluster.device(rank_idx);
            device.bind()?;
            let stream = device.default_stream();
            for (il, cache) in rank_session.caches.iter_mut().enumerate() {
                if let crate::session::LayerCache::Gdn(g) = cache {
                    let snap_state = g.snapshot_state.ok_or_else(|| {
                        anyhow!("restore GDN: rank={rank_idx} layer={il} no snapshot")
                    })?;
                    let snap_conv = g.snapshot_conv_history.ok_or_else(|| {
                        anyhow!("restore GDN conv: rank={rank_idx} layer={il} no snapshot")
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
            stream.synchronize()?;
        }
        Ok(())
    }

    /// roll back full-attn K/V tail by `n_remove` slots
    /// across every rank's full-attn layers.
    pub fn rollback_full_attn(&mut self, n_remove: usize) -> Result<()> {
        for rank_session in &mut self.per_rank {
            let rank_idx = rank_session.rank.0 as usize;
            for (il, cache) in rank_session.caches.iter_mut().enumerate() {
                match cache {
                    crate::session::LayerCache::FullAttn(kv) => kv
                        .rollback(n_remove)
                        .map_err(|e| anyhow!("rollback rank={rank_idx} layer={il}: {e}"))?,
                    crate::session::LayerCache::FullAttnQ8(kv) => kv
                        .rollback(n_remove)
                        .map_err(|e| anyhow!("rollback rank={rank_idx} layer={il}: {e}"))?,
                    crate::session::LayerCache::Gdn(_) => {}
                }
            }
        }
        Ok(())
    }

    /// re-advance GDN state by 1 step per recurrent layer
    /// using the per-layer x_in snapshots saved during a prior
    /// `forward_prefill_pp_logits_paired_l2` call. Used by the
    /// spec-decode reject path to replace the full L=1 redo with a
    /// per-rank-parallel GDN-only re-step (~7 ms vs ~50 ms).
    /// Caller is expected to have already run
    /// [`Self::restore_gdn_snapshot`] (puts every recurrent layer's
    /// state back to "after position-1") and
    /// [`Self::rollback_full_attn`] (drops the L=2 batch's tail K/V
    /// slots). After this call, the layer caches read as "after
    /// committing one token at position with last_token's input" —
    /// the same state a full L=1 redo would have produced, but
    /// without re-running full-attn / MoE / FFN / output-head.
    /// `decode_scratch` provides per-rank `hidden_b` as a discardable
    /// delta_out target plus the per-rank `LayerForwardScratch.gdn`
    /// workspace required by `forward_gdn_layer_decode`.
    pub fn redo_gdn_only_pp(
        &mut self,
        model: &Qwen3MoEShardedModel,
        cluster: &HipCluster,
        decode_scratch: &mut crate::forward::pp::ShardedForwardOneTokenScratch,
        prefill_scratch: &crate::forward::pp::ShardedForwardPrefillScratch,
    ) -> Result<()> {
        let cfg = &model.config;
        let n_ranks = self.per_rank.len();
        if n_ranks != cluster.ranks() {
            bail!(
                "redo_gdn_only_pp: session ranks={} != cluster ranks={}",
                n_ranks,
                cluster.ranks()
            );
        }
        // Per-rank: replay each recurrent layer's GDN forward at L=1
        // against its saved x_in snapshot. Each rank runs on its own
        // default stream → ranks execute concurrently across devices.
        for rank_idx in 0..n_ranks {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let stream = device.default_stream();
            let shard = &model.shards[rank_idx];
            let rank_scratch = decode_scratch
                .per_rank
                .get_mut(rank_idx)
                .context("decode scratch missing rank")?;
            let layer_scratch = rank_scratch
                .layer
                .as_mut()
                .context("decode scratch's LayerForwardScratch missing")?;
            let gdn_scratch = layer_scratch
                .gdn
                .as_mut()
                .context("LayerForwardScratch.gdn missing")?;
            let delta_out = rank_scratch.hidden_b;
            let pre_rank = prefill_scratch
                .per_rank
                .get(rank_idx)
                .context("prefill scratch missing rank")?;
            let rank_session = &mut self.per_rank[rank_idx];
            for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
                if !cfg.is_recurrent(layer_weights.layer_idx) {
                    continue;
                }
                let snap_ptr = pre_rank
                    .gdn_input_snapshots
                    .get(local_idx)
                    .and_then(|s| *s)
                    .ok_or_else(|| anyhow!(
                        "redo_gdn_only_pp: missing snapshot for rank={rank_idx} \
                         layer={} (run forward_prefill_pp_logits_paired_l2 first)",
                        layer_weights.layer_idx
                    ))?;
                let layer_cache = &mut rank_session.caches[local_idx];
                crate::forward::gdn::forward_gdn_layer_decode(
                    &shard.ops,
                    stream,
                    device,
                    cfg,
                    layer_weights,
                    layer_cache,
                    gdn_scratch,
                    snap_ptr,
                    delta_out,
                )
                .with_context(|| {
                    format!(
                        "redo_gdn_only_pp rank={rank_idx} layer={}",
                        layer_weights.layer_idx
                    )
                })?;
            }
            // Stream stays async — caller is expected to either issue
            // its own sync (e.g. before reading `h_for_next` host-side)
            // or to run subsequent forward calls on the same stream
            // which will serialise behind these.
        }
        Ok(())
    }
}
