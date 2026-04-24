//! Pipeline-parallel (Mesh&lt;N&gt; for N > 1) forward entry points.
//!
//! Each rank owns ~`num_layers / N` contiguous layers; the hidden state
//! is passed rank-to-rank via `HipCluster::peer_copy_via_host` (pinned-
//! host bounce on PCIe-only rigs per the V1 CLAUDE.md rationale).
//!
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
    forward_layer_prefill, forward_output_head_decode, LayerForwardScratch, LayerPrefillScratch,
    OutputHeadScratch,
};

// ---------------------------------------------------------------------------
// V1.7.5.C — pipeline-parallel forward_one_token.
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
        Ok(Self { per_rank })
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
///
/// Flow:
///   1. Rank 0 gathers the input embedding into its `hidden_a`.
///   2. For r in 0..N:
///        - If r > 0: `peer_copy_via_host` pulls the previous rank's
///          final hidden (in `hidden_a` by convention — see step 3)
///          into this rank's `hidden_a`.
///        - Run `forward_layer_decode` over the layers the shard owns,
///          ping-ponging `hidden_a ↔ hidden_b`.
///        - Normalise the final hidden back into `hidden_a` so the
///          next peer-copy has a known source.
///   3. Last rank runs `forward_output_head_decode` + `argmax_token_host`.
///
/// Single-token in flight — no micro-batching (the bubble is the sum of
/// each rank's compute; V2 can add 2-micro-batch pipelining). Per-hop
/// cost ≈ 30 µs (V1.7.5.B measurement), 3 hops for N=4 ≈ 0.5% of the
/// 16.7 ms/token budget at 60 tok/s.
pub fn forward_one_token_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_one_token_pp: zero-rank cluster");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let hidden_bytes = hidden * 2;

    // 1. Embed on rank 0. Bind first so the embed memcpys land on the
    // right device.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
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
    }

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are hidden_bytes long on their
            // respective devices; no other stream touches them here.
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    hidden_bytes,
                )?;
            }
        }
        // Bind this rank's device before issuing any kernels through its
        // OpsRegistry — HIP's module-launched kernels use the thread's
        // current device context, not whichever device the module was
        // loaded on.
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
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
            // V1.7.4.b per-layer activation dump for A/B vs llama.cpp.
            // Env-gated so the hot path pays zero cost when unset. Pair
            // with `llama-eval-callback` + grep `l_out-<il>` to bisect a
            // future forward divergence.
            if std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok() && position == 0 {
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
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Normalise final hidden into hidden_a for the next hand-off.
        // NO sync: subsequent peer_copy_via_host + downstream kernels all run on
        // the same default_stream, so stream ordering guarantees correctness.
        // Removing this sync saves one ~200 µs CPU-wait per rank per forward.
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
    }

    // 3. Output head on the last rank.
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
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_scratch.hidden_a,
    )?;

    // 4. Host argmax.
    argmax_token_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
    )
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
        model, session, cluster, scratch, token_id, position, /*download_logits=*/ Some(logits_out),
    )
    .map(|_| ())
}

/// Shared body for `forward_one_token_pp` and `forward_one_token_pp_logits`.
/// When `logits_out` is `Some`, downloads the F32 logits into it and returns 0;
/// when `None`, runs host argmax and returns the sampled token id.
fn forward_one_token_pp_inner(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardOneTokenScratch,
    token_id: u32,
    position: usize,
    logits_out: Option<&mut Vec<f32>>,
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

    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
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
    }

    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    hidden_bytes,
                )?;
            }
        }
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerForwardScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
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
    }

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
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_scratch.hidden_a,
    )?;

    match logits_out {
        Some(buf) => {
            download_logits_host(
                last_device,
                last_device.default_stream(),
                output_head_scratch.logits_f32,
                cfg.vocab_size,
                buf,
            )?;
            Ok(0)
        }
        None => argmax_token_host(
            last_device,
            last_device.default_stream(),
            output_head_scratch.logits_f32,
            cfg.vocab_size,
        ),
    }
}

// ---------------------------------------------------------------------------
// V1.7.5.D — pipeline-parallel forward_prefill.
// ---------------------------------------------------------------------------

/// V2.25.c — one ubatch lane's ping-pong + per-layer scratch. Each rank
/// holds `u_lanes` of these so V2.25.d can pipeline ubatches across ranks
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
///
/// V2.25.c — `extra_lanes` holds ADDITIONAL `UbatchLane`s beyond the
/// implicit lane 0 (which is the hidden_a/hidden_b/layer fields below).
/// Empty by default; `new_with_lanes(u_lanes > 1)` pre-allocates them so
/// V2.25.d can pipeline ubatches across ranks without aliasing.
pub struct RankForwardPrefillScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    /// V2.25.c — additional ubatch lanes beyond lane 0 (= the above
    /// hidden_a/hidden_b/layer fields). Used by V2.25.d.
    pub extra_lanes: Vec<UbatchLane>,
    hidden_bytes: usize,
    disposed: bool,
}

impl RankForwardPrefillScratch {
    /// V2.25.c — total ubatch lanes (includes lane 0 = the direct fields).
    pub fn u_lanes(&self) -> usize { 1 + self.extra_lanes.len() }

    /// V2.25.c — hidden_a for lane `idx`. Lane 0 = `self.hidden_a`;
    /// lane i>0 = `self.extra_lanes[i-1].hidden_a`.
    pub fn lane_hidden_a(&self, idx: usize) -> DevicePtr {
        if idx == 0 { self.hidden_a } else { self.extra_lanes[idx - 1].hidden_a }
    }

    /// V2.25.c — hidden_b for lane `idx`.
    pub fn lane_hidden_b(&self, idx: usize) -> DevicePtr {
        if idx == 0 { self.hidden_b } else { self.extra_lanes[idx - 1].hidden_b }
    }

    /// V2.25.c — mutable LayerPrefillScratch for lane `idx`.
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
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        for lane in self.extra_lanes.drain(..) {
            lane.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
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
}

impl ShardedForwardPrefillScratch {
    /// V2.25.c back-compat constructor — single lane, lane size = max_tokens.
    /// Equivalent to `new_with_lanes(model, cluster, max_tokens, 1)`.
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        Self::new_with_lanes(model, cluster, max_tokens, 1)
    }

    /// V2.25.c — construct per-rank scratch with `u_lanes` lanes, each
    /// sized for `ubatch_size` tokens. `u_lanes = 1` is byte-identical to
    /// the pre-V2.25 layout. `u_lanes >= 2` enables V2.25.d async PP
    /// pipelining (rank k can work on ubatch i+1 while rank k+1 waits
    /// driver-side for ubatch i's peer-copy).
    ///
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
            per_rank.push(RankForwardPrefillScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                max_tokens: ubatch_size,
                hidden_a,
                hidden_b,
                layer,
                output_head,
                extra_lanes,
                hidden_bytes,
                disposed: false,
            });
        }
        Ok(Self { per_rank })
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
///
/// Same per-rank flow as [`forward_one_token_pp`] but each stage processes
/// `L` tokens at once:
///   1. Rank 0 gathers `L` embeddings (one per token) into `hidden_a[0..L*H]`.
///   2. For r in 0..N:
///        - If r > 0: `peer_copy_via_host` moves `L * hidden * 2` bytes of
///          F16 hidden state from r-1's `hidden_a` to r's `hidden_a`.
///        - Bind this rank's device.
///        - Ping-pong through the rank's local layers via `forward_layer_prefill`.
///          KV cache / GDN state / conv-history get `L` tokens of history
///          appended per layer.
///        - If the final swap left the output in `hidden_b`, copy back
///          into `hidden_a` so the next rank's peer-copy has a known source.
///   3. Last rank: output head on the LAST token's hidden row + argmax.
///
/// No microbatching / 1F1B schedule — the pipeline bubble is the sum of
/// each rank's prefill compute. For Qwen3.6-31B at L=512 across 4 ranks,
/// measured prefill is compute-bound enough that microbatching is a V2+
/// perf lever, not a V1 correctness blocker.
pub fn forward_prefill_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    session: &mut crate::sharded::Qwen3MoEShardedSession,
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut ShardedForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
) -> Result<u32> {
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp called with empty tokens");
    }
    let max_tokens = scratch.per_rank[0].max_tokens;

    // V2.25.d — opt-in async ubatch path. Requires scratch with
    // `u_lanes >= 2` (from `new_with_lanes`) and `FLAMBEAU_ASYNC_UBATCH`
    // set. `FLAMBEAU_UBATCH` controls the ubatch size (defaults to
    // scratch.max_tokens which keeps behaviour equivalent to sync path).
    let async_enabled = std::env::var("FLAMBEAU_ASYNC_UBATCH").is_ok();
    let u_lanes = scratch.per_rank[0].u_lanes();
    if async_enabled && u_lanes >= 2 {
        let ubatch_size: usize = std::env::var("FLAMBEAU_UBATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(max_tokens);
        if l > ubatch_size {
            return forward_prefill_pp_async(
                model, session, cluster, scratch, tokens, start_position, ubatch_size,
            );
        }
    }

    if l > max_tokens {
        bail!(
            "forward_prefill_pp: L={l} > scratch.max_tokens={max_tokens}; caller must chunk"
        );
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    // 1. Embed all L tokens on rank 0. Row-by-row host dequant + upload —
    //    matches single-device `forward_prefill`'s embedding path.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
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
    }

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are at least `chunk_bytes` long
            // on their respective devices (sized against scratch.max_tokens
            // ≥ L); no other stream touches them here.
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

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
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
            )
            .with_context(|| {
                format!(
                    "prefill rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            if std::env::var("FLAMBEAU_PARITY_LAYER_DUMP").is_ok() {
                // Dump each token's post-layer hidden for A/B vs llama.cpp's
                // per-layer prefill output (`l_out-<il>` covers the full
                // [L, hidden] tensor in the callback trace).
                for t in 0..l {
                    let mut buf = vec![half::f16::from_f32(0.0); hidden];
                    unsafe {
                        device.memcpy_async(
                            device.default_stream(),
                            CopyDirection::DeviceToHost,
                            DevicePtr(buf.as_mut_ptr() as usize),
                            x_out.offset_bytes(t * row_bytes),
                            hidden * 2,
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
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Normalise final hidden into hidden_a for the next peer-copy hop
        // (L rows of F16 hidden).
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
    }

    // 3. Output head on the LAST token row on the last rank.
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
    let last_token_hidden = last_scratch.hidden_a.offset_bytes((l - 1) * row_bytes);
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

/// V2.25.d — async ubatch-pipelined prefill.
///
/// Splits `tokens[0..L]` into ubatches of size `ubatch_size` and drives
/// them across the N ranks using per-rank aux streams (V2.25.a) + async
/// peer-copies (V2.25.b) + per-lane scratch (V2.25.c). Ubatch i lives on
/// lane `i % u_lanes`; same-lane kernels serialize via stream ordering
/// (KV/GDN state coherence), different-lane kernels overlap on the
/// driver's DAG.
///
/// The caller is responsible for:
///   - constructing `scratch` via `new_with_lanes(..., ubatch_size, u_lanes)`
///     so each rank has `u_lanes` hidden+layer scratches sized for ubatch
///   - ensuring `u_lanes >= 2` — the async path has no benefit at u_lanes=1
///   - setting `FLAMBEAU_ASYNC_UBATCH` env opt-in (top-level
///     `forward_prefill_pp` routes here when set)
///
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
    cluster.reserve_aux_streams(u_lanes)?;
    // V2.25.g — per-lane pinned bounces break the single-slab
    // serialisation. Size to a full ubatch worth of F16 hidden.
    let bounce_bytes = ubatch_size * model.config.hidden_size * 2;
    cluster.reserve_lane_bounces(u_lanes, bounce_bytes)?;

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

    // Issue every (ubatch, rank) work item in order. Async peer-copies +
    // per-lane aux streams let the driver overlap across ubatches.
    for ub_idx in 0..n_ubatches {
        let lane = ub_idx % u_lanes;
        let start = ub_idx * ubatch_size;
        let end = (start + ubatch_size).min(l);
        let u = end - start;
        let chunk = &tokens[start..end];
        let pos = start_position + start;
        let chunk_bytes = u * row_bytes;

        // ----- Rank 0: embed + layers -----
        {
            let rank0 = cluster.device(0);
            rank0.bind()?;
            let shard0 = &shards[0];
            let token_embd = shard0
                .token_embd
                .as_ref()
                .context("rank 0 shard missing token_embd")?;
            let scratch0 = &mut scratch.per_rank[0];
            let lane_hidden_a = scratch0.lane_hidden_a(lane);

            // Embed on the lane's aux stream.
            cluster.with_aux_stream(0, lane, |s| -> flambeau_core::DeviceResult<()> {
                for (t, &token_id) in chunk.iter().enumerate() {
                    forward_embed_decode_host(
                        rank0,
                        s,
                        token_embd,
                        token_id,
                        lane_hidden_a.offset_bytes(t * row_bytes),
                        hidden,
                    )
                    .map_err(|e| flambeau_core::DeviceError::Backend {
                        backend: "hip",
                        code: -1,
                        message: format!("embed: {e}"),
                    })?;
                }
                Ok(())
            })?;

            // Run rank 0's layers on the same aux stream.
            let rank_session = &mut session.per_rank[0];
            let lane_hidden_b = scratch0.lane_hidden_b(lane);
            let layer_scratch = scratch0
                .lane_layer_mut(lane)
                .context("rank 0 lane_layer_mut")?;

            cluster.with_aux_stream(0, lane, |s| -> flambeau_core::DeviceResult<()> {
                let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                for (local_idx, layer_weights) in shard0.layers.iter().enumerate() {
                    let layer_cache = &mut rank_session.caches[local_idx];
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
        }

        // ----- Rank r > 0: peer-copy + layers -----
        for rank_idx in 1..n_ranks {
            let device = cluster.device(rank_idx);

            // Gather the raw stream handles for the cross-rank peer-copy.
            // We hold both mutexes simultaneously (different rank entries →
            // different Mutex instances; no deadlock).
            let src_dst_copy_result: Result<()> =
                cluster.with_aux_stream(rank_idx - 1, lane, |src_stream| {
                    cluster.with_aux_stream(rank_idx, lane, |dst_stream| {
                        unsafe {
                            // V2.25.g — use per-lane bounce to allow
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
            let rank_scratch = &mut scratch.per_rank[rank_idx];
            let lane_hidden_a = rank_scratch.lane_hidden_a(lane);
            let lane_hidden_b = rank_scratch.lane_hidden_b(lane);
            let layer_scratch = rank_scratch
                .lane_layer_mut(lane)
                .context("lane_layer_mut")?;

            cluster.with_aux_stream(rank_idx, lane, |s| -> flambeau_core::DeviceResult<()> {
                let (mut x_in, mut x_out) = (lane_hidden_a, lane_hidden_b);
                for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
                    let layer_cache = &mut rank_session.caches[local_idx];
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
    }

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
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_prefill_pp_logits: zero-rank cluster");
    }
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill_pp_logits called with empty tokens");
    }
    let max_tokens = scratch.per_rank[0].max_tokens;
    if l > max_tokens {
        bail!(
            "forward_prefill_pp_logits: L={l} > scratch.max_tokens={max_tokens}; caller must chunk"
        );
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = l * row_bytes;

    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
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
    }

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

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let rank_session = &mut session.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("per-rank LayerPrefillScratch missing")?;

        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
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
            )
            .with_context(|| {
                format!(
                    "prefill rank {} layer {} ({})",
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
    }

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
    let last_token_hidden = last_scratch.hidden_a.offset_bytes((l - 1) * row_bytes);
    forward_output_head_decode(
        &last_shard.ops,
        last_device.default_stream(),
        cfg,
        output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    download_logits_host(
        last_device,
        last_device.default_stream(),
        output_head_scratch.logits_f32,
        cfg.vocab_size,
        logits_out,
    )
}

