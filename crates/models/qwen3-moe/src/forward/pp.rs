//! Pipeline-parallel (Mesh&lt;N&gt; for N > 1) forward entry points.
//! Each rank owns ~`num_layers / N` contiguous layers; the hidden state
//! is passed rank-to-rank via `HipCluster::peer_copy_via_host` (pinned-
//! host bounce on PCIe-only rigs per the V1 CLAUDE.md rationale).
//! Mesh&lt;1&gt; is a degenerate instance — the single-device entry points in
//! `forward::single_device` sidestep the peer-copy path entirely.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;

use super::{
    argmax_token_host, download_logits_host, forward_embed_decode_host, forward_layer_decode,
    forward_layer_prefill, forward_output_head_decode, GdnScratch, LayerForwardScratch,
    LayerPrefillScratch, OutputHeadScratch,
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

// ---------------------------------------------------------------------------
// pipeline-parallel forward_one_token.
// ---------------------------------------------------------------------------

/// Per-rank scratch for a pipeline-parallel single-token decode. Only the
/// last rank in the cluster owns an `OutputHeadScratch` (middle ranks
/// never run the LM head).
pub struct RankForwardScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerForwardScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl RankForwardScratch {
    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for RankForwardScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate scratch: one `RankForwardScratch` per rank in the cluster.
pub struct ShardedForwardOneTokenScratch {
    pub per_rank: Vec<RankForwardScratch>,
    /// 7.a-i3 — per-rank graph-capture cache for decode. One
    /// `HipGraphExec` per rank populated lazily on the first token
    /// when `FLAMBEAU_DECODE_GRAPH=1`. Stores only the per-rank
    /// layer chain — embed (rank 0), peer-copy, and argmax
    /// (rank N-1) stay uncaptured.
    pub graph_cache_decode: Vec<Option<GraphCacheDecodeEntry>>,
}

/// 7.a-i3 — one cached decode exec per rank with per-layer slot
/// bundles for full-attn layers.
pub struct GraphCacheDecodeEntry {
    pub exec: flambeau_backend_hip::HipGraphExec,
    /// Per-layer slot bundle. `None` entries correspond to GDN
    /// layers, which advance state in-place and need no slot
    /// updates at replay.
    pub layer_slots: Vec<Option<super::layer::LayerDecodeSlots>>,
}

impl ShardedForwardOneTokenScratch {
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<Self> {
        let hidden_bytes = model.config.hidden_size * 2;
        // C3: pre-size the cluster's pinned bounce buffers to the
        // stage-boundary payload. Decode hand-offs are one hidden-state
        // F16 vector per hop (`hidden_size * 2`). Doing this once here
        // means every subsequent `peer_copy_via_host` hits the lock-free
        // atomic fast path in `ensure_bounce`.
        cluster.reserve_bounce_capacity(hidden_bytes)?;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerForwardScratch::new(&model.config, device)?);
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            per_rank.push(RankForwardScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                hidden_a,
                hidden_b,
                layer,
                output_head,
                hidden_bytes,
                disposed: false,
            });
        }
        let graph_cache_decode = (0..cluster.ranks()).map(|_| None).collect();
        Ok(Self { per_rank, graph_cache_decode })
    }

    pub fn dispose(
        mut self,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Pipeline-parallel single-token decode across an N-rank cluster.
/// Flow:
/// 1. Rank 0 gathers the input embedding into its `hidden_a`.
/// 2. For r in 0..N:
/// - If r > 0: `peer_copy_via_host` pulls the previous rank's
/// final hidden (in `hidden_a` by convention — see step 3)
/// into this rank's `hidden_a`.
/// - Run `forward_layer_decode` over the layers the shard owns,
/// ping-ponging `hidden_a ↔ hidden_b`.
/// - Normalise the final hidden back into `hidden_a` so the
/// next peer-copy has a known source.
/// 3. Last rank runs `forward_output_head_decode` + `argmax_token_host`.
/// Single-token in flight — no micro-batching (the bubble is the sum of
/// each rank's compute; V2 can add 2-micro-batch pipelining). Per-hop
/// cost ≈ 30 µs (measurement), 3 hops for N=4 ≈ 0.5% of the
/// 16.7 ms/token budget at 60 tok/s.
/// `flambeau_blocks::PpDecodeDriver` impl wrapping qwen3-moe's
/// `(model, session, cluster, scratch)` quadruple. Built per-call;
/// holds borrows that the topology orchestrator dereferences through
/// the trait.
struct Qwen3MoEPpDriver<'a> {
    model: &'a crate::sharded::Qwen3MoEShardedModel,
    session: &'a mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &'a flambeau_backend_hip::HipCluster,
    scratch: &'a mut ShardedForwardOneTokenScratch,
}

impl<'a> flambeau_blocks::PpDecodeDriver for Qwen3MoEPpDriver<'a> {
    fn n_ranks(&self) -> usize {
        self.model.shards.len()
    }

    fn layers_per_rank(&self, rank: usize) -> usize {
        self.model.shards[rank].layers.len()
    }

    fn cluster(&self) -> &flambeau_backend_hip::HipCluster {
        self.cluster
    }

    fn hidden_a(&self, rank: usize) -> DevicePtr {
        self.scratch.per_rank[rank].hidden_a
    }

    fn hidden_b(&self, rank: usize) -> DevicePtr {
        self.scratch.per_rank[rank].hidden_b
    }

    fn hidden_bytes(&self) -> usize {
        self.model.config.hidden_size * 2
    }

    fn embed_token(&mut self, token_id: u32) -> Result<()> {
        let rank0 = self.cluster.device(0);
        let shard0 = &self.model.shards[0];
        let scratch0 = &mut self.scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        forward_embed_decode_host(
            rank0,
            rank0.default_stream(),
            token_embd,
            token_id,
            scratch0.hidden_a,
            self.model.config.hidden_size,
        )
    }

    fn forward_layer_decode(
        &mut self,
        rank: usize,
        local_idx: usize,
        x_in: DevicePtr,
        x_out: DevicePtr,
        position: usize,
    ) -> Result<()> {
        let device = self.cluster.device(rank);
        let shard = &self.model.shards[rank];
        let layer_weights = &shard.layers[local_idx];
        let layer_cache = &mut self.session.per_rank[rank].caches[local_idx];
        let rank_scratch = &mut self.scratch.per_rank[rank];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;
        let cfg = &self.model.config;
        forward_layer_decode(
            &shard.ops,
            device.default_stream(),
            device,
            cfg,
            layer_weights,
            layer_cache,
            layer_scratch,
            x_in,
            x_out,
            position,
            None,
        )
        .with_context(|| {
            format!(
                "rank {} layer {} ({})",
                rank,
                layer_weights.layer_idx,
                if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
            )
        })?;
        if dev_flag("FLAMBEAU_PARITY_LAYER_DUMP") && position == 0 {
            let hidden = cfg.hidden_size;
            let mut buf = vec![half::f16::from_f32(0.0); hidden];
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(buf.as_mut_ptr() as usize),
                    x_out,
                    hidden * 2,
                )?;
            }
            device.default_stream().synchronize()?;
            let vals: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
            let l2 = vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            let (mn, mx) = vals
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            eprintln!(
                "[layer-dump] l_out-{} ({}): L2={:.6} min={} max={} head={:?}",
                layer_weights.layer_idx,
                if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
                l2, mn, mx, &vals[..4]
            );
        }
        Ok(())
    }

    fn output_head(&mut self) -> Result<()> {
        let last = self.model.shards.len() - 1;
        let last_device = self.cluster.device(last);
        let last_shard = &self.model.shards[last];
        let last_scratch = &mut self.scratch.per_rank[last];
        let output_norm = last_shard
            .output_norm
            .as_ref()
            .context("last rank missing output_norm")?;
        let lm_head = last_shard
            .output
            .as_ref()
            .or(last_shard.token_embd.as_ref())
            .context("last rank missing both output.weight and tied token_embd")?;
        let head_scratch = last_scratch
            .output_head
            .as_mut()
            .context("last rank missing output_head scratch")?;
        forward_output_head_decode(
            &last_shard.ops,
            last_device.default_stream(),
            &self.model.config,
            output_norm,
            lm_head,
            head_scratch,
            last_scratch.hidden_a,
        )
    }

    fn argmax(&self) -> Result<u32> {
        let last = self.model.shards.len() - 1;
        let last_device = self.cluster.device(last);
        let last_scratch = &self.scratch.per_rank[last];
        let head_scratch = last_scratch
            .output_head
            .as_ref()
            .context("last rank missing output_head scratch")?;
        argmax_token_host(
            last_device,
            last_device.default_stream(),
            head_scratch.logits_f32,
            self.model.config.vocab_size,
        )
    }
}

pub fn forward_one_token_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let mut driver = Qwen3MoEPpDriver { model, session, cluster, scratch };
    flambeau_blocks::forward_one_token_pp(&mut driver, token_id, position)
}

/// Variant of [`forward_one_token_pp`] that downloads the F32 logit row
/// into a caller-owned `Vec<f32>` instead of argmax-ing on host. Used by
/// the HTTP server when `temperature > 0` / `top_p < 1`; greedy callers
/// should keep using [`forward_one_token_pp`] to skip the download +
/// server-side softmax.
pub fn forward_one_token_pp_logits(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    forward_one_token_pp_inner(
        model, session, cluster, scratch, token_id, position,
        PpLogitsSink::Host(logits_out),
    )
    .map(|_| ())
}

/// Like [`forward_one_token_pp_logits`] but skips the logits DtoH; the
/// F32 row stays in `decode.per_rank[last_rank].output_head.logits_f32`
/// for the GPU sampler. Caller must consume it before the next forward.
pub fn forward_one_token_pp_keep_logits_on_device(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<()> {
    forward_one_token_pp_inner(
        model, session, cluster, scratch, token_id, position,
        PpLogitsSink::KeepOnDevice,
    )
    .map(|_| ())
}

/// Logits sink for [`forward_one_token_pp_inner`]:
/// `Host` DtoH-s the row, `Argmax` returns the argmax token id,
/// `KeepOnDevice` leaves the row on the head rank for a GPU sampler.
pub enum PpLogitsSink<'a> {
    Host(&'a mut Vec<f32>),
    Argmax,
    KeepOnDevice,
}

/// Shared body for the three single-token PP entry points. Returns the
/// sampled token id on `Argmax`, `0` on the other two sinks.
fn forward_one_token_pp_inner(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
    sink: PpLogitsSink<'_>,
) -> Result<u32> {
    // Re-run the same composition as `forward_one_token_pp`, but branch on
    // the final reducer. Copy-paste is deliberate — the body is ~150 lines
    // of tightly-ordered HIP calls and threading a branch through would
    // hurt readability more than a second copy that tracks the original
    // line-for-line.
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_one_token_pp_inner: zero-rank cluster");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let hidden_bytes = hidden * 2;

    // section markers. No-op when the thread-local
    // timer in flambeau_backend_hip::profile is disabled. Each mark is
    // recorded on the relevant rank's default stream so the elapsed_ms
    // delta to the next mark on the SAME rank captures the device-side
    // time spent on that section.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        flambeau_backend_hip::profile::mark("step_start", rank0, rank0.default_stream())?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        forward_embed_decode_host(
            rank0,
            rank0.default_stream(),
            token_embd,
            token_id,
            scratch0.hidden_a,
            hidden,
        )?;
        flambeau_backend_hip::profile::mark("embed_done", rank0, rank0.default_stream())?;
    }

    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: shared bounce is reordered by the per-token logits
            // DtoH below; the dst stream picks up the bytes via FIFO.
            unsafe {
                cluster.peer_copy_via_host_event(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    hidden_bytes,
                )?;
            }
        }
        device.bind()?;
        flambeau_backend_hip::profile::mark(
            "stage_start",
            device,
            device.default_stream(),
        )?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        let pp_probe = dev_flag("FLAMBEAU_PP_PROBE");
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_decode(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                position,
                None,
            )
            .with_context(|| {
                format!(
                    "rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            std::mem::swap(&mut x_in, &mut x_out);
            // FLAMBEAU_PP_PROBE — dump x_in (this layer's output, post-swap)
            // for parity comparison vs the TP path (FLAMBEAU_TP_PROBE). Same
            // format: per-layer min/max/L2/head[0..4]. Disabled by default.
            if pp_probe {
                let n = hidden_bytes / 2;
                let mut host = vec![0u16; n];
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToHost,
                        DevicePtr(host.as_mut_ptr() as usize),
                        x_in,
                        hidden_bytes,
                    )?;
                }
                device.default_stream().synchronize()?;
                let mut nan = 0usize;
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                let mut sum = 0.0f64;
                let mut sumsq = 0.0f64;
                for &b in &host {
                    let v = half::f16::from_bits(b).to_f32();
                    if v.is_nan() {
                        nan += 1;
                    } else {
                        if v < min { min = v; }
                        if v > max { max = v; }
                        sum += v as f64;
                        sumsq += (v as f64) * (v as f64);
                    }
                }
                let mean = sum / (n - nan).max(1) as f64;
                let l2 = sumsq.sqrt();
                let head: Vec<f32> = host[..host.len().min(4)]
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect();
                let il = layer_weights.layer_idx;
                eprintln!(
                    "  PP_PROBE after-layer hidden_a rank={rank_idx} il={il:>3} \
                     n={n} nan={nan} L2={l2:.6} min={min:.6} max={max:.6} \
                     mean={mean:.6} head={head:?}"
                );
            }
        }
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    hidden_bytes,
                )?;
            }
        }
        flambeau_backend_hip::profile::mark(
            "stage_end",
            device,
            device.default_stream(),
        )?;
    }

    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    flambeau_backend_hip::profile::mark(
        "output_head_start",
        last_device,
        last_device.default_stream(),
    )?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_scratch.hidden_a,
    )?;

    match sink {
        PpLogitsSink::Host(buf) => {
            download_logits_host(
                last_device,
                last_device.default_stream(),
                output_head_scratch.logits_f32,
                cfg.vocab_size,
                buf,
            )?;
            Ok(0)
        }
        PpLogitsSink::Argmax => argmax_token_host(
            last_device,
            last_device.default_stream(),
            output_head_scratch.logits_f32,
            cfg.vocab_size,
        ),
        PpLogitsSink::KeepOnDevice => {
            // Logits remain in `output_head_scratch.logits_f32`. The
            // caller's downstream kernel (GPU top-K on the head rank's
            // default stream) serialises against the output_head writes
            // via stream ordering — no explicit sync required.
            Ok(0)
        }
    }
}

// ---------------------------------------------------------------------------
// pipeline-parallel forward_prefill.
// ---------------------------------------------------------------------------

/// 5.c — one ubatch lane's ping-pong + per-layer scratch. Each rank
/// holds `u_lanes` of these so 5.d can pipeline ubatches across ranks
/// without aliasing intermediate buffers.
pub struct UbatchLane {
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: LayerPrefillScratch,
    hidden_bytes: usize,
}

impl UbatchLane {
    fn new(
        cfg: &crate::Qwen3MoEConfig,
        device: &HipDevice,
        ubatch_size: usize,
    ) -> Result<Self> {
        let hidden_bytes = ubatch_size * cfg.hidden_size * 2;
        let hidden_a = device.alloc(hidden_bytes)?;
        let hidden_b = device.alloc(hidden_bytes)?;
        let layer = LayerPrefillScratch::new(cfg, device, ubatch_size)?;
        Ok(Self { hidden_a, hidden_b, layer, hidden_bytes })
    }

    fn dispose(self, device: &HipDevice) -> Result<()> {
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        self.layer.dispose(device)?;
        Ok(())
    }
}

/// Per-rank scratch for a pipeline-parallel prefill chunk of up to
/// `max_tokens` tokens. Layout mirrors `RankForwardScratch` with the
/// hidden ping-pong buffers and `LayerPrefillScratch` both sized for `L`
/// tokens. Only the last rank owns an `OutputHeadScratch`.
/// 5.c — `extra_lanes` holds ADDITIONAL `UbatchLane`s beyond the
/// implicit lane 0 (which is the hidden_a/hidden_b/layer fields below).
/// Empty by default; `new_with_lanes(u_lanes > 1)` pre-allocates them so
/// 5.d can pipeline ubatches across ranks without aliasing.
pub struct RankForwardPrefillScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    /// 5.c — additional ubatch lanes beyond lane 0 (= the above
    /// hidden_a/hidden_b/layer fields). Used by 5.d.
    pub extra_lanes: Vec<UbatchLane>,
    /// one device buffer per local layer, holding a single
    /// hidden row (hidden * 2 bytes). Populated lazily during paired-L=2
    /// verify with the per-layer GDN input at L=2 batch position 0, for
    /// use by `Qwen3MoEShardedSession::redo_gdn_only_pp` on spec-decode
    /// reject. `Some(ptr)` for GDN layers, `None` for full-attn layers.
    pub gdn_input_snapshots: Vec<Option<DevicePtr>>,
    /// **P2.9b-i2-A1-wire** — one shared `GdnScratch` (single-token decode
    /// workspace) for the batched-decode driver's per-slot GDN loop. GDN
    /// is recurrent so it can't be batched across slots; the batched
    /// driver loops slots calling `forward_gdn_layer_decode` and reusing
    /// this single scratch sequentially. `None` on archs without GDN.
    /// Only populated when `cfg.gdn.is_some()`; carried alongside the
    /// `LayerPrefillScratch.gdn` (which is the prefill workspace for one
    /// state evolving through L tokens — different shape).
    pub gdn_decode: Option<GdnScratch>,
    hidden_bytes: usize,
    /// Size of one snapshot row = `hidden_size * 2` (F16). Per-rank
    /// constant; cached for the dispose path.
    snapshot_row_bytes: usize,
    disposed: bool,
}

impl RankForwardPrefillScratch {
    /// 5.c — total ubatch lanes (includes lane 0 = the direct fields).
    pub fn u_lanes(&self) -> usize { 1 + self.extra_lanes.len() }

    /// 5.c — hidden_a for lane `idx`. Lane 0 = `self.hidden_a`;
    /// lane i>0 = `self.extra_lanes[i-1].hidden_a`.
    pub fn lane_hidden_a(&self, idx: usize) -> DevicePtr {
        if idx == 0 { self.hidden_a } else { self.extra_lanes[idx - 1].hidden_a }
    }

    /// 5.c — hidden_b for lane `idx`.
    pub fn lane_hidden_b(&self, idx: usize) -> DevicePtr {
        if idx == 0 { self.hidden_b } else { self.extra_lanes[idx - 1].hidden_b }
    }

    /// 5.c — mutable LayerPrefillScratch for lane `idx`.
    pub fn lane_layer_mut(&mut self, idx: usize) -> Option<&mut LayerPrefillScratch> {
        if idx == 0 {
            self.layer.as_mut()
        } else {
            Some(&mut self.extra_lanes[idx - 1].layer)
        }
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        // release per-GDN-layer snapshot buffers.
        for snap in self.gdn_input_snapshots.drain(..) {
            if let Some(ptr) = snap {
                unsafe {
                    device.dealloc(ptr, self.snapshot_row_bytes)?;
                }
            }
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        for lane in self.extra_lanes.drain(..) {
            lane.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.gdn_decode.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for RankForwardPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                rank = self.rank.0,
                "RankForwardPrefillScratch dropped without dispose(device); buffers leaked"
            );
        }
    }
}

/// Aggregate scratch for PP prefill: one `RankForwardPrefillScratch` per rank.
pub struct ShardedForwardPrefillScratch {
    pub per_rank: Vec<RankForwardPrefillScratch>,
    /// 6.a-i5c — per (rank, lane) graph-capture cache used by
    /// `forward_prefill_pp_async` when `FLAMBEAU_ASYNC_GRAPH=1`. The
    /// first ubatch on (rank, lane) captures the whole layer chain;
    /// subsequent ubatches update pos-bearing slots + replay. Empty
    /// Vec (no sub-Vec) means disabled / lazy.
    pub graph_cache: Vec<Vec<Option<GraphCacheEntry>>>,
    /// 6.a-i7b — persistent host-side scratch for rank 0's
    /// batched embed (`forward_embed_prefill_batch`). Replaces per-token
    /// DtoH-sync-HtoD-sync pattern that issued 2·u host barriers per
    /// ubatch and starved async-PP's Rust dispatcher.
    pub embed_host: super::io::EmbedPrefillHostScratch,
    /// 0.a — per (rank, local_layer_idx) HipEvent that serialises
    /// `gdn_state_step` kernel access across lanes on the same rank.
    /// Ubatch N+1 on the opposite lane waits on this event before
    /// running its state_step; ubatch N records it after its state_step.
    /// Same-lane ubatches are already stream-ordered. Eliminates the
    /// 7.d / 8.c.1 / 8.a-i1 GDN race guards.
    /// Shape: outer vec is n_ranks; inner vec is local layer count on
    /// that rank. Entry is None for non-GDN (full-attn) layers.
    pub gdn_state_events: Vec<Vec<Option<flambeau_backend_hip::HipEvent>>>,
}

/// 6.a-i5c — one cached exec per (rank, lane) with the per-layer
/// slot bundles that drive pos updates. Captures only the layer-chain
/// portion (no peer-copy, no embed) — the wrap/unwrap happens outside
/// the capture closure.
pub struct GraphCacheEntry {
    pub exec: flambeau_backend_hip::HipGraphExec,
    pub layer_slots: Vec<super::layer::LayerPrefillSlots>,
}

impl ShardedForwardPrefillScratch {
    /// 5.c back-compat constructor — single lane, lane size = max_tokens.
    /// Equivalent to `new_with_lanes(model, cluster, max_tokens, 1)`.
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::new_with_lanes(model, cluster, max_tokens, 1)
    }

    /// 5.c — construct per-rank scratch with `u_lanes` lanes, each
    /// sized for `ubatch_size` tokens. `u_lanes = 1` is byte-identical to
    /// the pre-5 layout. `u_lanes >= 2` enables 5.d async PP
    /// pipelining (rank k can work on ubatch i+1 while rank k+1 waits
    /// driver-side for ubatch i's peer-copy).
    /// Memory footprint per rank: `u_lanes × (2 × ubatch_size × hidden × 2
    /// bytes hidden ping-pong + LayerPrefillScratch)`.
    pub fn new_with_lanes(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        ubatch_size: usize,
        u_lanes: usize,
    ) -> Result<Self> {
        assert!(ubatch_size >= 1, "ubatch_size must be >= 1");
        assert!(u_lanes >= 1, "u_lanes must be >= 1");
        let hidden_bytes = ubatch_size * model.config.hidden_size * 2;
        // C3: size the cluster's pinned bounces to the max prefill payload
        // (decode hand-offs only need `hidden_bytes / max_tokens`; prefill
        // dominates). This is a max; `reserve_bounce_capacity` never
        // shrinks.
        cluster.reserve_bounce_capacity(hidden_bytes)?;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            // Lane 0 — the legacy fields.
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerPrefillScratch::new(&model.config, device, ubatch_size)?);
            // Extra lanes — one fresh UbatchLane per additional u_lane.
            let mut extra_lanes = Vec::with_capacity(u_lanes.saturating_sub(1));
            for _ in 1..u_lanes {
                extra_lanes.push(UbatchLane::new(&model.config, device, ubatch_size)?);
            }
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            // one snapshot row per local layer, allocated only
            // for GDN-bearing layers. Each snapshot holds a single hidden
            // row (position 0 of the L=2 batch) for the spec-decode reject
            // path's GDN-only re-step.
            let snapshot_row_bytes = model.config.hidden_size * 2;
            let shard = &model.shards[rank_idx];
            let mut gdn_input_snapshots: Vec<Option<DevicePtr>> =
                Vec::with_capacity(shard.layers.len());
            for layer_weights in shard.layers.iter() {
                if model.config.is_recurrent(layer_weights.layer_idx) {
                    let ptr = device.alloc(snapshot_row_bytes)?;
                    gdn_input_snapshots.push(Some(ptr));
                } else {
                    gdn_input_snapshots.push(None);
                }
            }
            // **P2.9b-i2-A1-wire** — shared single-token GDN decode scratch
            // for the batched-decode driver's per-slot GDN loop. Allocated
            // only on archs with GDN (qwen35moe / qwen36moe hybrids); pure
            // full-attn arches (qwen3moe) leave it `None`.
            let gdn_decode = if model.config.gdn.is_some() {
                Some(GdnScratch::new(&model.config, device)?)
            } else {
                None
            };
            per_rank.push(RankForwardPrefillScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                max_tokens: ubatch_size,
                hidden_a,
                hidden_b,
                layer,
                output_head,
                extra_lanes,
                gdn_input_snapshots,
                gdn_decode,
                hidden_bytes,
                snapshot_row_bytes,
                disposed: false,
            });
        }
        // 6.a-i5c — pre-size the graph cache to [ranks][u_lanes]
        // of None. Populated lazily on the first ubatch that hits
        // (rank, lane) under FLAMBEAU_ASYNC_GRAPH.
        let graph_cache: Vec<Vec<Option<GraphCacheEntry>>> = (0..cluster.ranks())
            .map(|_| (0..u_lanes).map(|_| None).collect())
            .collect();
        // 6.a-i7b — rank-0 embed host scratch sized for max ubatch.
        // row_bytes depends on the token_embd dtype which isn't known
        // here; start empty and grow on first use.
        let embed_host = super::io::EmbedPrefillHostScratch { raw: Vec::new(), f16: Vec::new() };
        // 0.a — one HipEvent per (rank, GDN layer) for cross-lane
        // state_step serialisation. Allocated eagerly so no hot-path
        // None-check + device-bind branch; unused on sync / u_lanes=1.
        let mut gdn_state_events: Vec<Vec<Option<flambeau_backend_hip::HipEvent>>> =
            Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let shard = &model.shards[rank_idx];
            let mut rank_events = Vec::with_capacity(shard.layers.len());
            for layer_weights in shard.layers.iter() {
                if model.config.is_recurrent(layer_weights.layer_idx) {
                    rank_events.push(Some(flambeau_backend_hip::HipEvent::new(device.id())?));
                } else {
                    rank_events.push(None);
                }
            }
            gdn_state_events.push(rank_events);
        }
        Ok(Self {
            per_rank,
            graph_cache,
            embed_host,
            gdn_state_events,
        })
    }

    pub fn dispose(
        mut self,
        cluster: &flambeau_backend_hip::HipCluster,
    ) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for rs in self.per_rank.drain(..) {
            let rank_idx = rs.rank.0 as usize;
            if let Err(e) = rs.dispose(cluster.device(rank_idx)) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

/// Pipeline-parallel prefill over `L` tokens across an N-rank cluster.
/// Same per-rank flow as [`forward_one_token_pp`] but each stage processes
/// `L` tokens at once:
/// 1. Rank 0 gathers `L` embeddings (one per token) into `hidden_a[0..L*H]`.
/// 2. For r in 0..N:
/// - If r > 0: `peer_copy_via_host` moves `L * hidden * 2` bytes of
/// F16 hidden state from r-1's `hidden_a` to r's `hidden_a`.
/// - Bind this rank's device.
/// - Ping-pong through the rank's local layers via `forward_layer_prefill`.
/// KV cache / GDN state / conv-history get `L` tokens of history
/// appended per layer.
/// - If the final swap left the output in `hidden_b`, copy back
/// into `hidden_a` so the next rank's peer-copy has a known source.
/// 3. Last rank: output head on the LAST token's hidden row + argmax.
/// No microbatching / 1F1B schedule — the pipeline bubble is the sum of
/// each rank's prefill compute. For Qwen3.6-31B at L=512 across 4 ranks,
/// measured prefill is compute-bound enough that microbatching is a V2+
/// perf lever, not a V1 correctness blocker.
/// `flambeau_blocks::PpPrefillDriver` impl wrapping qwen3-moe's
/// `(model, session, cluster, scratch)` quadruple for prefill.
struct Qwen3MoEPpPrefillDriver<'a> {
    model: &'a crate::sharded::Qwen3MoEShardedModel,
    session: &'a mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &'a flambeau_backend_hip::HipCluster,
    scratch: &'a mut ShardedForwardPrefillScratch,
}

impl<'a> Qwen3MoEPpPrefillDriver<'a> {
    fn last_rank(&self) -> usize {
        self.model.shards.len() - 1
    }

    fn finalize_argmax(&self) -> Result<u32> {
        let last = self.last_rank();
        let last_device = self.cluster.device(last);
        let last_scratch = &self.scratch.per_rank[last];
        let head_scratch = last_scratch
            .output_head
            .as_ref()
            .context("last rank missing output_head scratch")?;
        argmax_token_host(
            last_device,
            last_device.default_stream(),
            head_scratch.logits_f32,
            self.model.config.vocab_size,
        )
    }

    fn finalize_logits(&self, logits_out: &mut Vec<f32>) -> Result<()> {
        let last = self.last_rank();
        let last_device = self.cluster.device(last);
        let last_scratch = &self.scratch.per_rank[last];
        let head_scratch = last_scratch
            .output_head
            .as_ref()
            .context("last rank missing output_head scratch")?;
        download_logits_host(
            last_device,
            last_device.default_stream(),
            head_scratch.logits_f32,
            self.model.config.vocab_size,
            logits_out,
        )
    }
}

impl<'a> flambeau_blocks::PpPrefillDriver for Qwen3MoEPpPrefillDriver<'a> {
    fn n_ranks(&self) -> usize {
        self.model.shards.len()
    }

    fn layers_per_rank(&self, rank: usize) -> usize {
        self.model.shards[rank].layers.len()
    }

    fn cluster(&self) -> &flambeau_backend_hip::HipCluster {
        self.cluster
    }

    fn hidden_a(&self, rank: usize) -> DevicePtr {
        self.scratch.per_rank[rank].hidden_a
    }

    fn hidden_b(&self, rank: usize) -> DevicePtr {
        self.scratch.per_rank[rank].hidden_b
    }

    fn hidden_row_bytes(&self) -> usize {
        self.model.config.hidden_size * 2
    }

    fn max_tokens(&self) -> usize {
        self.scratch.per_rank[0].max_tokens
    }

    fn embed_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        let rank0 = self.cluster.device(0);
        let shard0 = &self.model.shards[0];
        let scratch0 = &mut self.scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        let row_bytes = self.model.config.hidden_size * 2;
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                token_id,
                scratch0.hidden_a.offset_bytes(t * row_bytes),
                self.model.config.hidden_size,
            )?;
        }
        Ok(())
    }

    fn forward_layer_prefill(
        &mut self,
        rank: usize,
        local_idx: usize,
        x_in: DevicePtr,
        x_out: DevicePtr,
        n_tokens: usize,
        start_position: usize,
    ) -> Result<()> {
        let device = self.cluster.device(rank);
        let shard = &self.model.shards[rank];
        let layer_weights = &shard.layers[local_idx];
        let layer_cache = &mut self.session.per_rank[rank].caches[local_idx];
        let rank_scratch = &mut self.scratch.per_rank[rank];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;
        let cfg = &self.model.config;
        forward_layer_prefill(
            &shard.ops,
            device.default_stream(),
            device,
            cfg,
            layer_weights,
            layer_cache,
            layer_scratch,
            x_in,
            x_out,
            n_tokens,
            start_position,
            None,
            None,
        )
        .with_context(|| {
            format!(
                "prefill rank {} layer {} ({})",
                rank,
                layer_weights.layer_idx,
                if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
            )
        })?;
        if dev_flag("FLAMBEAU_PARITY_LAYER_DUMP") {
            let hidden = cfg.hidden_size;
            let row_bytes = hidden * 2;
            for t in 0..n_tokens {
                let mut buf = vec![half::f16::from_f32(0.0); hidden];
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToHost,
                        DevicePtr(buf.as_mut_ptr() as usize),
                        x_out.offset_bytes(t * row_bytes),
                        row_bytes,
                    )?;
                }
                device.default_stream().synchronize()?;
                let vals: Vec<f32> = buf.iter().map(|v| v.to_f32()).collect();
                let l2 = vals.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                eprintln!(
                    "[prefill-dump] l_out-{} t={} ({}): L2={:.6} head={:?}",
                    layer_weights.layer_idx,
                    t,
                    if cfg.is_recurrent(layer_weights.layer_idx) { "gdn" } else { "full_attn" },
                    l2, &vals[..4]
                );
            }
        }
        Ok(())
    }

    fn output_head_last_token(&mut self, l: usize) -> Result<()> {
        let last = self.last_rank();
        let last_device = self.cluster.device(last);
        let last_shard = &self.model.shards[last];
        let last_scratch = &mut self.scratch.per_rank[last];
        let row_bytes = self.model.config.hidden_size * 2;
        let last_token_hidden = last_scratch.hidden_a.offset_bytes((l - 1) * row_bytes);
        let output_norm = last_shard
            .output_norm
            .as_ref()
            .context("last rank missing output_norm")?;
        let lm_head = last_shard
            .output
            .as_ref()
            .or(last_shard.token_embd.as_ref())
            .context("last rank missing both output.weight and tied token_embd")?;
        let head_scratch = last_scratch
            .output_head
            .as_mut()
            .context("last rank missing output_head scratch")?;
        forward_output_head_decode(
            &last_shard.ops,
            last_device.default_stream(),
            &self.model.config,
            output_norm,
            lm_head,
            head_scratch,
            last_token_hidden,
        )
    }
}

pub fn forward_prefill_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
) -> Result<u32> {
    let mut driver = Qwen3MoEPpPrefillDriver { model, session, cluster, scratch };
    flambeau_blocks::forward_prefill_pp(&mut driver, tokens, start_position)?;
    driver.finalize_argmax()
}

/// 5.d — async ubatch-pipelined prefill.
/// Splits `tokens[0..L]` into ubatches of size `ubatch_size` and drives
/// them across the N ranks using per-rank aux streams (5.a) + async
/// peer-copies (5.b) + per-lane scratch (5.c). Ubatch i lives on
/// lane `i % u_lanes`; same-lane kernels serialize via stream ordering
/// (KV/GDN state coherence), different-lane kernels overlap on the
/// driver's DAG.
/// The caller is responsible for:
/// - constructing `scratch` via `new_with_lanes(..., ubatch_size, u_lanes)`
/// so each rank has `u_lanes` hidden+layer scratches sized for ubatch
/// - ensuring `u_lanes >= 2` — the async path has no benefit at u_lanes=1
/// - setting `FLAMBEAU_ASYNC_UBATCH` env opt-in (top-level
/// `forward_prefill_pp` routes here when set)
/// Returns argmax of the LAST token of the LAST ubatch.
pub fn forward_prefill_pp_async(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
    ubatch_size: usize,
) -> Result<u32> {
    use flambeau_backend_hip::HipEvent;

    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_async: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp_async called with empty tokens");
    }
    if ubatch_size == 0 {
        bail!("forward_prefill_pp_async: ubatch_size must be >= 1");
    }
    let max_ubatch = scratch.per_rank[0].max_tokens;
    if ubatch_size > max_ubatch {
        bail!(
            "forward_prefill_pp_async: ubatch_size={ubatch_size} > scratch.max_tokens={max_ubatch}"
        );
    }
    let u_lanes = scratch.per_rank[0].u_lanes();
    if u_lanes < 2 {
        bail!(
            "forward_prefill_pp_async: u_lanes={u_lanes} < 2 — no async benefit; construct scratch via new_with_lanes(..., u_lanes >= 2)"
        );
    }
    // 0.a — the three GDN cross-lane state-race guards (7.d
    // ubatch<128/tail<128, 8.a-i1 gdn_per_rank>10, 8.c.1 K>64)
    // are all the same bug: concurrent ubatches on different aux streams
    // read/write the shared per-layer GDN state tensor with no
    // ordering. 0.a serialises the GDN `state_step` call itself via
    // a per-(rank, layer) HipEvent recorded in `scratch.gdn_state_events`.
    // Each lane's `stream_wait(event)` before `state_step` and `record`
    // after ensures cross-lane state_step executions are ordered, which
    // is all that is needed for parity — the rest of the GDN chain
    // reads/writes only lane-local ubatch-sized buffers.
    cluster.reserve_aux_streams(u_lanes)?;
    // 5.g — per-lane pinned bounces break the single-slab
    // serialisation. Size to a full ubatch worth of F16 hidden.
    let bounce_bytes = ubatch_size * model.config.hidden_size * 2;
    cluster.reserve_lane_bounces(u_lanes, bounce_bytes)?;
    // 6.a-i5c async graph-capture branch was removed in S6:
    // FLAMBEAU_ASYNC_GRAPH=1 measured -2.4 % on qwen36-35b-a3b-q4_0/pp4
    // and null elsewhere; the uncaptured legacy-async path is the only
    // production code now.
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let n_ubatches = l.div_ceil(ubatch_size);

    // Pre-allocate bridge events: one per (src_rank, lane) pair. Used to
    // bridge the DtoH on src_rank to the HtoD on dst_rank inside
    // peer_copy_via_host_async. Reused each ubatch using the same lane —
    // hipEventRecord overwrites the prior record.
    let mut bridge_events: Vec<Vec<HipEvent>> = Vec::with_capacity(n_ranks.saturating_sub(1));
    for r in 0..n_ranks.saturating_sub(1) {
        let device = cluster.device(r);
        device.bind()?;
        let mut row = Vec::with_capacity(u_lanes);
        for _ in 0..u_lanes {
            row.push(HipEvent::new(device.id())?);
        }
        bridge_events.push(row);
    }

    let shards = &model.shards;

    // 5.h — interleaved 1F1B dispatch. At time step t, rank r
    // processes ubatch (t - r) if in [0, n_ubatches). At steady state
    // (t in [n_ranks-1, n_ubatches-1]), every rank is dispatching a
    // different ubatch concurrently, producing real pipeline fill.
    // Pre-5.h dispatched `for ub in 0..n_ubatches { for rank in
    // 0..n_ranks }` which is serial-across-ubatches. That wasted the
    // aux-stream / per-lane-bounce infrastructure because rank r's
    // lane-k work finished before rank r ever started lane-k+1 work.
    let n_timesteps = n_ranks + n_ubatches - 1;
    for t in 0..n_timesteps {
        for rank_idx in 0..n_ranks {
            let ub_idx_signed = t as isize - rank_idx as isize;
            if ub_idx_signed < 0 || (ub_idx_signed as usize) >= n_ubatches {
                continue;
            }
            let ub_idx = ub_idx_signed as usize;
            let lane = ub_idx % u_lanes;
            let start = ub_idx * ubatch_size;
            let end = (start + ubatch_size).min(l);
            let u = end - start;
            let chunk = &tokens[start..end];
            let pos = start_position + start;
            let chunk_bytes = u * row_bytes;

        // ----- Rank 0: embed + layers -----
        if rank_idx == 0 {
            let rank0 = cluster.device(0);
            rank0.bind()?;
            let shard0 = &shards[0];
            let token_embd = shard0
                .token_embd
                .as_ref()
                .context("rank 0 shard missing token_embd")?;
            // Split-borrow: embed_host and per_rank are disjoint
            // fields of scratch; we need mutable access to both.
            let embed_host: *mut super::io::EmbedPrefillHostScratch = &mut scratch.embed_host;
            let scratch0 = &mut scratch.per_rank[0];
            let lane_hidden_a = scratch0.lane_hidden_a(lane);

            // 6.a-i7b — batched embed (one sync per ubatch
            // instead of 2·u). Under 1F1B the Rust driver thread
            // returned here sooner → more time for other-rank
            // dispatches.
            cluster.with_aux_stream(0, lane, |s| -> flambeau_core::DeviceResult<()> {
                // SAFETY: embed_host is a *mut into scratch.embed_host
                // — disjoint from scratch.per_rank we borrowed above,
                // so no aliasing.
                let embed_host_ref = unsafe { &mut *embed_host };
                super::io::forward_embed_prefill_batch(
                    rank0,
                    s,
                    token_embd,
                    chunk,
                    lane_hidden_a,
                    hidden,
                    embed_host_ref,
                )
                .map_err(|e| flambeau_core::DeviceError::Backend {
                    backend: "hip",
                    code: -1,
                    message: format!("embed_batch: {e}"),
                })
            })?;

            // Run rank 0's layers on the same aux stream.
            let rank_session = &mut session.per_rank[0];
            let lane_hidden_b = scratch0.lane_hidden_b(lane);
            let layer_scratch = scratch0
                .lane_layer_mut(lane)
                .context("rank 0 lane_layer_mut")?;
            // 0.a — event vector for this rank's GDN layers. Disjoint
            // from scratch.per_rank borrowed above.
            let rank_events: *mut Vec<Option<flambeau_backend_hip::HipEvent>> =
                &mut scratch.gdn_state_events[0];

            cluster.with_aux_stream(0, lane, |s| -> flambeau_core::DeviceResult<()> {
                // SAFETY: rank_events is *mut into scratch.gdn_state_events[0],
                // disjoint from scratch.per_rank[0] borrowed above.
                let rank_events_ref = unsafe { &mut *rank_events };
                let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                for (local_idx, layer_weights) in shard0.layers.iter().enumerate() {
                    let layer_cache = &mut rank_session.caches[local_idx];
                    let gdn_event = rank_events_ref[local_idx].as_ref();
                    forward_layer_prefill(
                        &shard0.ops,
                        s,
                        rank0,
                        cfg,
                        layer_weights,
                        layer_cache,
                        layer_scratch,
                        x_in,
                        x_out,
                        u,
                        pos,
                        None,
                        gdn_event,
                    )
                    .map_err(|e| flambeau_core::DeviceError::Backend {
                        backend: "hip",
                        code: -1,
                        message: format!("prefill rank 0 layer {}: {e}", layer_weights.layer_idx),
                    })?;
                    std::mem::swap(&mut x_in, &mut x_out);
                }
                // Normalise final hidden into lane's hidden_a for peer-copy.
                if x_in != lane_hidden_a {
                    unsafe {
                        rank0.memcpy_async(
                            s,
                            CopyDirection::DeviceToDevice,
                            lane_hidden_a,
                            x_in,
                            chunk_bytes,
                        )?;
                    }
                }
                Ok(())
            })?;

        } else {
            // ----- Rank r > 0: peer-copy + layers -----
            let device = cluster.device(rank_idx);

            // Gather the raw stream handles for the cross-rank peer-copy.
            // We hold both mutexes simultaneously (different rank entries →
            // different Mutex instances; no deadlock).
            let src_dst_copy_result: Result<()> =
                cluster.with_aux_stream(rank_idx - 1, lane, |src_stream| {
                    cluster.with_aux_stream(rank_idx, lane, |dst_stream| {
                        unsafe {
                            // 5.g — use per-lane bounce to allow
                            // concurrent DtoH across lanes on the same rank.
                            cluster.peer_copy_via_host_async_laned(
                                scratch.per_rank[rank_idx].lane_hidden_a(lane),
                                rank_idx,
                                scratch.per_rank[rank_idx - 1].lane_hidden_a(lane),
                                rank_idx - 1,
                                chunk_bytes,
                                src_stream,
                                dst_stream,
                                &bridge_events[rank_idx - 1][lane],
                                None,
                                Some(lane),
                            )?;
                        }
                        Ok(())
                    })
                })
                .map_err(|e| anyhow::anyhow!("peer_copy_async r{}->{}: {e}", rank_idx - 1, rank_idx));
            src_dst_copy_result?;

            device.bind()?;
            let shard = &shards[rank_idx];
            let rank_session = &mut session.per_rank[rank_idx];
            // 0.a — pull GDN state events out before borrowing per_rank.
            // Disjoint field of scratch.
            let rank_events_ptr: *mut Vec<Option<flambeau_backend_hip::HipEvent>> =
                &mut scratch.gdn_state_events[rank_idx];
            let rank_scratch = &mut scratch.per_rank[rank_idx];
            let lane_hidden_a = rank_scratch.lane_hidden_a(lane);
            let lane_hidden_b = rank_scratch.lane_hidden_b(lane);
            let layer_scratch = rank_scratch
                .lane_layer_mut(lane)
                .context("lane_layer_mut")?;

            cluster.with_aux_stream(rank_idx, lane, |s| -> flambeau_core::DeviceResult<()> {
                // SAFETY: rank_events_ptr is a *mut into
                // scratch.gdn_state_events[rank_idx], disjoint from
                // scratch.per_rank[rank_idx] borrowed above.
                let rank_events_ref = unsafe { &mut *rank_events_ptr };
                let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
                    let layer_cache = &mut rank_session.caches[local_idx];
                    let gdn_event = rank_events_ref[local_idx].as_ref();
                    forward_layer_prefill(
                        &shard.ops,
                        s,
                        device,
                        cfg,
                        layer_weights,
                        layer_cache,
                        layer_scratch,
                        x_in,
                        x_out,
                        u,
                        pos,
                        None,
                        gdn_event,
                    )
                    .map_err(|e| flambeau_core::DeviceError::Backend {
                        backend: "hip",
                        code: -1,
                        message: format!(
                            "prefill rank {} layer {}: {e}",
                            rank_idx, layer_weights.layer_idx
                        ),
                    })?;
                    std::mem::swap(&mut x_in, &mut x_out);
                }
                if x_in != lane_hidden_a {
                    unsafe {
                        device.memcpy_async(
                            s,
                            CopyDirection::DeviceToDevice,
                            lane_hidden_a,
                            x_in,
                            chunk_bytes,
                        )?;
                    }
                }
                Ok(())
            })?;
        }
        } // close for rank_idx
    } // close for t (outer 1F1B timestep loop)

    // ----- Output head on the last rank after the FINAL ubatch -----
    let last_idx = n_ranks - 1;
    let last_ub = n_ubatches - 1;
    let last_lane = last_ub % u_lanes;
    let last_start = last_ub * ubatch_size;
    let last_end = (last_start + ubatch_size).min(l);
    let last_u = last_end - last_start;

    let last_device = cluster.device(last_idx);
    last_device.bind()?;

    // Sync the last rank's lane so the output head sees completed hidden.
    cluster.with_aux_stream(last_idx, last_lane, |s| {
        flambeau_core::Stream::synchronize(s)
    })?;

    let last_shard = &shards[last_idx];
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    // Compute the lane hidden ptr before the mutable borrow of output_head.
    let last_lane_hidden_a = last_scratch.lane_hidden_a(last_lane);
    let last_token_hidden = last_lane_hidden_a.offset_bytes((last_u - 1) * row_bytes);
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
}

/// Variant of [`forward_prefill_pp`] that downloads the F32 logit row for
/// the last token into `logits_out` instead of argmax-ing on host. See
/// [`forward_one_token_pp_logits`] for the streaming rationale.
pub fn forward_prefill_pp_logits(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    let mut driver = Qwen3MoEPpPrefillDriver { model, session, cluster, scratch };
    flambeau_blocks::forward_prefill_pp_chunk(&mut driver, tokens, start_position)?;
    driver.finalize_logits(logits_out)
}

/// paired-logits L=2 primitive for K=1 spec-decode verify.
/// Identical body to [`forward_prefill_pp_logits`] except `tokens.len()`
/// is required to be 2 and the LM-head pass runs **twice** on the last
/// rank: once on `hidden_a[0]` (predicts token at `start_position+1`),
/// once on `hidden_a[1]` (predicts token at `start_position+2` given the
/// draft). On return, `logits_out_pos0` and `logits_out_pos1` each hold
/// `[vocab]` F32 logits.
/// This replaces the 2× sequential L=1 fallback in `forward/spec.rs`
/// — the body of the model only runs once, amortising the per-layer
/// launch overhead and (more importantly on hybrid arch) only paying
/// one GDN state-step pair instead of two.
pub fn forward_prefill_pp_logits_paired_l2(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
    logits_out_pos0: &mut Vec<f32>,
    // when None, skip the pos1 head + download. Caller
    // can fetch pos1 logits later via [`forward_output_head_at_pp`] if
    // accept-path needs them. Eliminates ~3 ms/macro of always-paid head
    // work on reject paths (saves ~0.4 ms/macro avg at 12.5 % rejects).
    logits_out_pos1: Option<&mut Vec<f32>>,
) -> Result<()> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_logits_paired_l2: zero-rank cluster");
    }
    if tokens.len() != 2 {
        bail!(
            "forward_prefill_pp_logits_paired_l2: requires L=2, got L={}",
            tokens.len()
        );
    }
    let l = 2;
    let max_tokens = scratch.per_rank[0].max_tokens;
    if l > max_tokens {
        bail!(
            "forward_prefill_pp_logits_paired_l2: L=2 > scratch.max_tokens={max_tokens}"
        );
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    // Embed both tokens on rank 0.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        flambeau_backend_hip::profile::mark("l2_step_start", rank0, rank0.default_stream())?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        for (t, &token_id) in tokens.iter().enumerate() {
            forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                token_id,
                scratch0.hidden_a.offset_bytes(t * row_bytes),
                hidden,
            )?;
        }
        flambeau_backend_hip::profile::mark("l2_embed_done", rank0, rank0.default_stream())?;
    }

    // Per-rank: peer-copy hidden in, run all owned layers at L=2.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }
        device.bind()?;
        flambeau_backend_hip::profile::mark(
            "l2_stage_start",
            device,
            device.default_stream(),
        )?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        // copy snapshot ptrs out before borrowing the layer
        // scratch mutably. DevicePtr is Copy, so this is a cheap clone.
        let gdn_snapshots: Vec<Option<DevicePtr>> = rank_scratch.gdn_input_snapshots.clone();
        let snapshot_row_bytes = rank_scratch.snapshot_row_bytes;
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            // for GDN-bearing layers, save x_in (position 0
            // only) to the per-layer snapshot buffer for the spec-decode
            // reject path. DtoD memcpy on this rank's default stream;
            // ordered before forward_layer_prefill's own kernels.
            if let Some(snap_ptr) = gdn_snapshots[local_idx] {
                unsafe {
                    device.memcpy_async(
                        device.default_stream(),
                        CopyDirection::DeviceToDevice,
                        snap_ptr,
                        x_in,
                        snapshot_row_bytes,
                    )?;
                }
            }
            let layer_cache = &mut rank_session.caches[local_idx];
            forward_layer_prefill(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                layer_cache,
                layer_scratch,
                x_in,
                x_out,
                l,
                start_position,
                None,
                None,
            )
            .with_context(|| {
                format!(
                    "paired-L2 prefill rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        if x_in != rank_scratch.hidden_a {
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    chunk_bytes,
                )?;
            }
            device.default_stream().synchronize()?;
        }
        flambeau_backend_hip::profile::mark(
            "l2_stage_end",
            device,
            device.default_stream(),
        )?;
    }

    // Last rank: run output head twice and download both logit rows.
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    flambeau_backend_hip::profile::mark(
        "l2_output_head_start",
        last_device,
        last_device.default_stream(),
    )?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;

    // Position 0 of the L=2 batch.
    let h_pos0 = last_scratch.hidden_a;
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        h_pos0,
    )?;
    download_logits_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
        logits_out_pos0,
    )?;
    flambeau_backend_hip::profile::mark(
        "l2_output_pos0_done",
        last_device,
        last_device.default_stream(),
    )?;

    // Position 1. Output-head scratch is reused; logits buffer is
    // overwritten by the next mmvq, so we had to download pos0 first.
    // only run pos1 head when caller asks for it; else
    // skip and let caller defer this work to the accept branch.
    if let Some(logits_out_pos1) = logits_out_pos1 {
        let h_pos1 = last_scratch.hidden_a.offset_bytes(row_bytes);
        forward_output_head_decode(
            &last_shard.ops,
            last_device.default_stream(),
            cfg,
            output_norm,
            lm_head,
            output_head_scratch,
            h_pos1,
        )?;
        download_logits_host(
            last_device,
            last_device.default_stream(),
            output_head_scratch.logits_f32,
            cfg.vocab_size,
            logits_out_pos1,
        )?;
        flambeau_backend_hip::profile::mark(
            "l2_output_pos1_done",
            last_device,
            last_device.default_stream(),
        )?;
    }

    Ok(())
}

/// run the LM head on a single hidden row located at
/// `last_scratch.hidden_a + position * hidden_bytes`. Used by the spec
/// driver to lazily compute pos1 logits only on accept.
pub fn forward_output_head_at_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    position_in_l2: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_output_head_at_pp: zero-rank cluster");
    }
    let cfg = &model.config;
    let row_bytes = cfg.hidden_size * 2;
    let last_idx = n_ranks - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;
    let h_at = last_scratch.hidden_a.offset_bytes(position_in_l2 * row_bytes);
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        h_at,
    )?;
    download_logits_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
        logits_out,
    )?;
    Ok(())
}

