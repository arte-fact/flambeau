//! Pipeline-parallel (Mesh&lt;N&gt; for N > 1) forward entry points.
//!
//! Each rank owns ~`num_layers / N` contiguous layers; the hidden state
//! is passed rank-to-rank via `HipCluster::peer_copy_via_host` (pinned-
//! host bounce on PCIe-only rigs per the V1 CLAUDE.md rationale).
//!
//! Mesh&lt;1&gt; is a degenerate instance — the single-device entry points in
//! `forward::single_device` sidestep the peer-copy path entirely.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipCluster;
use flambeau_core::{Device, DevicePtr};
use flambeau_ops::hip::OpsRegistry;
use flambeau_runtime::RankId;

use super::{
    argmax_token_host, forward_embed_decode_host, forward_layer_decode, forward_layer_prefill,
    forward_output_head_decode, LayerForwardScratch, LayerPrefillScratch, OutputHeadScratch,
};
use crate::config::Qwen3MoEConfig;

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

// ---------------------------------------------------------------------------
// V1.7.5.D — pipeline-parallel forward_prefill.
// ---------------------------------------------------------------------------

/// Per-rank scratch for a pipeline-parallel prefill chunk of up to
/// `max_tokens` tokens. Layout mirrors `RankForwardScratch` with the
/// hidden ping-pong buffers and `LayerPrefillScratch` both sized for `L`
/// tokens. Only the last rank owns an `OutputHeadScratch`.
pub struct RankForwardPrefillScratch {
    pub rank: flambeau_runtime::RankId,
    pub device_id: i32,
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl RankForwardPrefillScratch {
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
    pub fn new(
        model: &crate::sharded::Qwen3MoEShardedModel,
        cluster: &flambeau_backend_hip::HipCluster,
        max_tokens: usize,
    ) -> Result<Self> {
        assert!(max_tokens >= 1, "max_tokens must be >= 1");
        let hidden_bytes = max_tokens * model.config.hidden_size * 2;
        let mut per_rank = Vec::with_capacity(cluster.ranks());
        for rank_idx in 0..cluster.ranks() {
            let device = cluster.device(rank_idx);
            device.bind()?;
            let hidden_a = device.alloc(hidden_bytes)?;
            let hidden_b = device.alloc(hidden_bytes)?;
            let layer = Some(LayerPrefillScratch::new(&model.config, device, max_tokens)?);
            let output_head = if rank_idx == cluster.ranks() - 1 {
                Some(OutputHeadScratch::new(&model.config, device)?)
            } else {
                None
            };
            per_rank.push(RankForwardPrefillScratch {
                rank: flambeau_runtime::RankId(rank_idx as u32),
                device_id: device.id(),
                max_tokens,
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

