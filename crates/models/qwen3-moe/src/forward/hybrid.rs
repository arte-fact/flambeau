//! AUTO-4d — hybrid PP-of-TP forward driver.
//!
//! Composes the existing per-layer TP entry points
//! ([`super::tp::forward_full_attn_layer_tp`],
//! [`super::tp::forward_gdn_layer_tp`]) per stage, with one
//! [`HipCluster::peer_copy_via_host`] hop between stages on the
//! **global** cluster (the one that spans all `pp_size * tp_size`
//! ranks). The intra-stage AllReduce uses each stage's per-stage
//! `BarP2pAllReduce` against its sub-cluster — these are constructed
//! by the server at startup (AUTO-4f).
//!
//! ### Embedding & LM head
//!
//! The AUTO-4b loader puts `token_embd` only on **stage 0** and
//! `output_norm` + `output` only on the **last stage**. The driver
//! mirrors that placement: embed runs on stage 0's TP ranks; the LM
//! head runs on the last stage's `head_rank` (rank 0 by default).
//!
//! ### Inter-stage hand-off
//!
//! After stage `s`'s last layer, the residual `hidden_a` on rank 0 of
//! stage `s` is copied to **every rank** of stage `s+1`'s sub-cluster
//! via `peer_copy_via_host`. The cost is `tp_size_next × hidden_size ×
//! 2 B` per token per hop; on Qwen3.5-9B (`hidden=4096`) at
//! `tp_size=2`, that's 16 KB / hop — well under the 4 KB minimum chunk
//! size that pinned-bounce was tuned for, but the path is correct.

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster};
use flambeau_core::{Device, Stream};

use crate::hybrid::{
    Qwen3MoEHybridModel, Qwen3MoEHybridSession, ShardedForwardOneTokenScratchHybrid,
    ShardedForwardPrefillScratchHybrid,
};

use super::io::{argmax_token_host, forward_embed_decode_host, forward_output_head_decode};
use super::tp::{
    forward_full_attn_layer_tp, forward_gdn_layer_tp, forward_prefill_tp_batched_layers,
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

/// AUTO-4e — ingest a `prompt_ids` prompt token-by-token and write
/// the **last** position's F32 logits row into `logits_out`. Mirrors
/// the TP server-side prefill (`model.rs::prefill_logits` for the
/// `LoadedModel::Tp` arm), which is itself a per-token loop because
/// the V1 TP path has no batched prefill kernel — TP-5b "prefill-PP +
/// decode-TP coexistence" deferred batched-TP-prefill to V2 and
/// hybrid inherits the same posture. The hand-off cost between
/// stages stays the same per-token primitive (one F16 hidden vec
/// per hop); a true batched prefill would change the hop payload to
/// `[L, hidden] * F16`, deferred until profile data justifies it.
///
/// `start_position` is the position the *first* prompt token lands
/// at — non-zero when this prefill is appending to a session that
/// already saw earlier tokens.
pub fn forward_prefill_hybrid_logits(
    model: &Qwen3MoEHybridModel,
    scratch: &mut ShardedForwardOneTokenScratchHybrid,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    session: &mut Qwen3MoEHybridSession,
    prompt_ids: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    if prompt_ids.is_empty() {
        bail!("forward_prefill_hybrid_logits: empty prompt");
    }
    // **AUTO-6e** — batched hybrid prefill for prompts ≥ 8 tokens.
    // Per-token fallback (FLAMBEAU_TP_BATCHED=0) was 13× slower; deleted in S3.
    if prompt_ids.len() >= 8 {
        return forward_prefill_hybrid_batched_logits(
            model,
            global_cluster,
            stage_ars,
            session,
            prompt_ids,
            start_position,
            logits_out,
        )
        .context("hybrid batched prefill (AUTO-6e)");
    }
    let last = prompt_ids.len() - 1;
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let pos = start_position + i;
        if i < last {
            // Discard logits for non-final positions; only the cache
            // updates matter, and the per-token forward writes them.
            forward_one_token_hybrid(
                model,
                scratch,
                global_cluster,
                stage_ars,
                session,
                tok,
                pos,
            )
            .with_context(|| format!("hybrid prefill pos={pos}"))?;
        } else {
            forward_one_token_hybrid_logits(
                model,
                scratch,
                global_cluster,
                stage_ars,
                session,
                tok,
                pos,
                logits_out,
            )
            .with_context(|| format!("hybrid prefill final pos={pos}"))?;
        }
    }
    Ok(())
}

/// **AUTO-6e3** — L-batched per-stage hybrid prefill.
///
/// Mirrors [`forward_one_token_hybrid_inner`] structurally (embed →
/// per-stage layers → inter-stage hand-off → output head) but every
/// step is L-aware:
///
///  * Stage 0 embeds **all L tokens** into rank-0..N's `hidden_a`
///    (sized `[L, hidden]` by [`ShardedForwardPrefillScratchHybrid`]).
///  * Each stage runs [`forward_prefill_tp_batched_layers`] over its
///    layer range with `il_cache_offset = stage.layer_range.start`
///    (each stage's session caches were allocated for its slice
///    only — same convention as the per-token hybrid driver).
///  * Inter-stage hand-off transfers `[L, hidden]` F16
///    (`L * hidden * 2` bytes) instead of one hidden vector.
///  * The last stage runs the LM head on the **last position** of
///    `hidden_a` and downloads `cfg.vocab_size` F32 logits.
///
/// Allocates a fresh [`ShardedForwardPrefillScratchHybrid`] per call
/// and disposes on every exit path. V2.x: bind it on
/// `Qwen3MoEHybridSession` to avoid the per-request alloc.
pub fn forward_prefill_hybrid_batched_logits(
    model: &Qwen3MoEHybridModel,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    session: &mut Qwen3MoEHybridSession,
    prompt_ids: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    if prompt_ids.is_empty() {
        bail!("forward_prefill_hybrid_batched_logits: empty prompt");
    }
    let cfg = &model.config;
    let n_stages = model.stages.len();
    let tp_size = model.spec.tp_size as usize;

    if session.stages.len() != n_stages {
        bail!(
            "hybrid session.stages.len()={} != model.stages.len()={n_stages}",
            session.stages.len()
        );
    }
    if stage_ars.len() != n_stages {
        bail!(
            "stage_ars.len()={} != model.stages.len()={n_stages}",
            stage_ars.len()
        );
    }
    if global_cluster.ranks() != n_stages * tp_size {
        bail!(
            "global_cluster.ranks()={} != pp_size*tp_size={}",
            global_cluster.ranks(),
            n_stages * tp_size
        );
    }
    let world = tp_size as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("hybrid batched prefill: per-stage tp_size ∈ {{1, 2, 4}} (got {world})");
    }

    let n_tokens = prompt_ids.len();
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    // 1. Allocate prefill scratch sized to this prompt. RAII guard so
    //    the scratch is disposed on every exit path.
    let prefill = ShardedForwardPrefillScratchHybrid::new(model, n_tokens)
        .context("alloc hybrid prefill scratch")?;
    struct PrefillGuard<'m> {
        scratch: Option<ShardedForwardPrefillScratchHybrid>,
        model: &'m Qwen3MoEHybridModel,
    }
    impl Drop for PrefillGuard<'_> {
        fn drop(&mut self) {
            if let Some(s) = self.scratch.take() {
                let _ = s.dispose(self.model);
            }
        }
    }
    let mut guard = PrefillGuard {
        scratch: Some(prefill),
        model,
    };
    let scratch_ref = guard.scratch.as_mut().expect("scratch present until drop");

    // 2. Stage 0: embed L tokens on every rank's hidden_a.
    {
        let stage = &model.stages[0];
        if !stage.tp_model.has_token_embd {
            bail!("hybrid stage 0 is missing token_embd; loader bug");
        }
        let stage_scratch = &mut scratch_ref.per_stage[0];
        for r in 0..stage.sub_cluster.ranks() {
            let device = stage.sub_cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let dst_base = stage_scratch.per_rank[r].hidden_a;
            for (i, &tok) in prompt_ids.iter().enumerate() {
                let dst_row =
                    flambeau_core::DevicePtr(dst_base.as_usize() + i * row_bytes);
                forward_embed_decode_host(
                    device,
                    stream,
                    &stage.tp_model.shards[r].token_embd,
                    tok,
                    dst_row,
                    hidden,
                )
                .with_context(|| format!("hybrid stage 0 rank {r} embed pos {i}"))?;
            }
        }
    }

    // 3. Per-stage layer loop + L-batched hand-off.
    for s in 0..n_stages {
        let stage = &model.stages[s];
        let stage_scratch = &mut scratch_ref.per_stage[s];
        let stage_session = &mut session.stages[s];
        let stage_ar = &stage_ars[s];

        if stage_session.layer_range != stage.layer_range {
            bail!(
                "hybrid: session stage {s} range {:?} != model range {:?}",
                stage_session.layer_range,
                stage.layer_range
            );
        }

        forward_prefill_tp_batched_layers(
            &stage.tp_model,
            stage_scratch,
            &stage.sub_cluster,
            stage_ar,
            &mut stage_session.caches,
            stage.layer_range.clone(),
            stage.layer_range.start,
            n_tokens,
            start_position,
        )
        .with_context(|| format!("hybrid stage {s} batched layers {:?}", stage.layer_range))?;

        // Hand-off: copy L hidden vectors from stage s rank 0 to every
        // rank of stage s+1. Same pinned-bounce pattern as the per-token
        // path; the only thing that grows is the byte count.
        if s + 1 < n_stages {
            let prod_dev = model.stages[s].sub_cluster.device(0);
            prod_dev.bind()?;
            prod_dev.default_stream().synchronize()?;

            let src_global_rank = s * tp_size;
            let src_ptr = scratch_ref.per_stage[s].per_rank[0].hidden_a;
            let bytes = n_tokens * hidden * 2; // F16 [L, hidden]
            for dst_local in 0..tp_size {
                let dst_global_rank = (s + 1) * tp_size + dst_local;
                let dst_ptr = scratch_ref.per_stage[s + 1].per_rank[dst_local].hidden_a;
                // SAFETY: src/dst are live `[L, hidden] * F16` allocations on
                // their respective devices (sized by ShardedForwardPrefillScratchTp::new
                // for both stages); both ranks belong to global_cluster.
                // Producer sync above covers source freshness; destinations
                // are about to be (re-)written by the next stage's first
                // batched-layer call.
                unsafe {
                    global_cluster
                        .peer_copy_via_host(
                            dst_ptr,
                            dst_global_rank,
                            src_ptr,
                            src_global_rank,
                            bytes,
                        )
                        .with_context(|| {
                            format!(
                                "hybrid batched stage {s} → {} hand-off (rank {src_global_rank} \
                                 → {dst_global_rank}, {bytes} B)",
                                s + 1
                            )
                        })?;
                }
            }
        }
    }

    // 4. Last stage: output head on the LAST POSITION + logits download.
    let head_stage_idx = scratch_ref.head_stage as usize;
    if head_stage_idx + 1 != n_stages {
        bail!(
            "hybrid: head_stage={head_stage_idx} but expected last stage = {}",
            n_stages - 1
        );
    }
    let last_stage = &model.stages[head_stage_idx];
    if !last_stage.tp_model.has_output_head {
        bail!("hybrid last stage is missing output head; loader bug");
    }
    let last_scratch = &mut scratch_ref.per_stage[head_stage_idx];
    let head_rank = last_scratch.head_rank.0 as usize;
    let device = last_stage.sub_cluster.device(head_rank);
    device.bind()?;
    let stream = device.default_stream();
    let head_shard = &last_stage.tp_model.shards[head_rank];
    let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
    let last_row_off = (n_tokens - 1) * row_bytes;
    let hidden_a_last = flambeau_core::DevicePtr(
        last_scratch.per_rank[head_rank].hidden_a.as_usize() + last_row_off,
    );
    let head_scratch = last_scratch.per_rank[head_rank]
        .output_head
        .as_mut()
        .ok_or_else(|| {
            anyhow!("hybrid: head_rank={head_rank} on last stage missing OutputHeadScratch")
        })?;
    let logits_f32 = head_scratch.logits_f32;
    let ops = &last_stage.tp_model.ops[head_rank];
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &head_shard.output_norm,
        lm_head,
        head_scratch,
        hidden_a_last,
    )
    .context("hybrid batched forward_output_head_decode")?;
    logits_out.clear();
    logits_out.resize(cfg.vocab_size, 0.0f32);
    // SAFETY: logits_f32 is valid for cfg.vocab_size F32 values on `device`;
    // logits_out.as_mut_ptr() is host memory of matching size.
    unsafe {
        <flambeau_backend_hip::HipDevice as flambeau_core::Device>::memcpy_async(
            device,
            stream,
            flambeau_core::CopyDirection::DeviceToHost,
            flambeau_core::DevicePtr(logits_out.as_mut_ptr() as usize),
            logits_f32,
            cfg.vocab_size * 4,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    Ok(())
}

/// **#258 Sampler-D3 Phase B (Hybrid)** — what to do with the F32
/// logits row after the LM head emits it on the head stage's
/// `head_rank` device. Mirrors `tp::LogitsSink`.
pub(crate) enum HybridLogitsSink<'a> {
    /// Run host-side argmax + return the predicted token (greedy).
    HostArgmax,
    /// DtoH `[vocab]` F32 logits into the caller's `Vec<f32>`. Returns 0.
    HostLogits(&'a mut Vec<f32>),
    /// Leave the logits on the head stage's `head_rank` device — caller
    /// consumes `head_scratch.logits_f32` in place via the GPU top-K
    /// sampler before the next forward call clobbers it. Returns 0.
    /// Saves the 600 KB DtoH per token — same lever as the TP path.
    KeepOnDevice,
}

/// One-token hybrid PP-of-TP decode that returns the greedy argmax.
/// See [`forward_one_token_hybrid_logits`] for the variant that
/// downloads the F32 logits row instead.
pub fn forward_one_token_hybrid(
    model: &Qwen3MoEHybridModel,
    scratch: &mut ShardedForwardOneTokenScratchHybrid,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    session: &mut Qwen3MoEHybridSession,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    forward_one_token_hybrid_inner(
        model,
        scratch,
        global_cluster,
        stage_ars,
        session,
        token_id,
        position,
        HybridLogitsSink::HostArgmax,
    )
}

/// Variant of [`forward_one_token_hybrid`] that writes the F32 logits
/// row of the head stage's `head_rank` into `logits_out` (resized to
/// `cfg.vocab_size`) instead of running argmax host-side. Returns
/// `Ok(())`; the sampled token is the caller's job (matches PP/TP).
pub fn forward_one_token_hybrid_logits(
    model: &Qwen3MoEHybridModel,
    scratch: &mut ShardedForwardOneTokenScratchHybrid,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    session: &mut Qwen3MoEHybridSession,
    token_id: u32,
    position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    forward_one_token_hybrid_inner(
        model,
        scratch,
        global_cluster,
        stage_ars,
        session,
        token_id,
        position,
        HybridLogitsSink::HostLogits(logits_out),
    )
    .map(|_| ())
}

/// **#258 Sampler-D3 Phase B (Hybrid)** — variant of
/// [`forward_one_token_hybrid_logits`] that leaves the F32 logits row
/// on the head stage's `head_rank` device for in-place GPU-side
/// sampling. The caller MUST consume
/// `scratch.per_stage[head_stage].per_rank[head_rank].output_head.logits_f32`
/// (e.g. via `topk_softmax_f32` on the same default stream) before
/// the next forward call clobbers it.
///
/// Saves the 600 KB DtoH per token under `FLAMBEAU_GPU_SAMPLER=1`,
/// matching the TP path's improvement.
pub fn forward_one_token_hybrid_keep_logits_on_device(
    model: &Qwen3MoEHybridModel,
    scratch: &mut ShardedForwardOneTokenScratchHybrid,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    session: &mut Qwen3MoEHybridSession,
    token_id: u32,
    position: usize,
) -> Result<()> {
    forward_one_token_hybrid_inner(
        model,
        scratch,
        global_cluster,
        stage_ars,
        session,
        token_id,
        position,
        HybridLogitsSink::KeepOnDevice,
    )
    .map(|_| ())
}

fn forward_one_token_hybrid_inner(
    model: &Qwen3MoEHybridModel,
    scratch: &mut ShardedForwardOneTokenScratchHybrid,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    session: &mut Qwen3MoEHybridSession,
    token_id: u32,
    position: usize,
    sink: HybridLogitsSink<'_>,
) -> Result<u32> {
    let cfg = &model.config;
    let n_stages = model.stages.len();
    let tp_size = model.spec.tp_size as usize;

    // Shape invariants — checked once per call so misuse surfaces here,
    // not deep inside the per-rank kernel-launch loop.
    if scratch.per_stage.len() != n_stages {
        bail!(
            "hybrid scratch.per_stage.len()={} != model.stages.len()={n_stages}",
            scratch.per_stage.len()
        );
    }
    if session.stages.len() != n_stages {
        bail!(
            "hybrid session.stages.len()={} != model.stages.len()={n_stages}",
            session.stages.len()
        );
    }
    if stage_ars.len() != n_stages {
        bail!(
            "stage_ars.len()={} != model.stages.len()={n_stages}",
            stage_ars.len()
        );
    }
    if global_cluster.ranks() != n_stages * tp_size {
        bail!(
            "global_cluster.ranks()={} != pp_size*tp_size={}",
            global_cluster.ranks(),
            n_stages * tp_size
        );
    }

    let world = tp_size as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("hybrid: per-stage tp_size ∈ {{1, 2, 4}} (got {world})");
    }

    let head_stage_idx = scratch.head_stage as usize;
    if head_stage_idx + 1 != n_stages {
        bail!(
            "hybrid: head_stage={head_stage_idx} but expected last stage = {}",
            n_stages - 1
        );
    }

    // ── Stage 0: embed ───────────────────────────────────────────────
    {
        let stage = &model.stages[0];
        if !stage.tp_model.has_token_embd {
            bail!("hybrid stage 0 is missing token_embd; loader bug");
        }
        let stage_scratch = &mut scratch.per_stage[0];
        for r in 0..stage.sub_cluster.ranks() {
            let device = stage.sub_cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            forward_embed_decode_host(
                device,
                stream,
                &stage.tp_model.shards[r].token_embd,
                token_id,
                stage_scratch.per_rank[r].hidden_a,
                cfg.hidden_size,
            )
            .with_context(|| format!("hybrid stage 0 rank {r} embed"))?;
        }
    }

    // ── Per-stage layer loops + hand-offs ────────────────────────────
    for s in 0..n_stages {
        let stage = &model.stages[s];
        let stage_scratch = &mut scratch.per_stage[s];
        let stage_session = &mut session.stages[s];
        let stage_ar = &stage_ars[s];

        // Sanity: stage_session range must equal stage range. The
        // session was built from the same model, so this should be
        // tautological — guard against accidental mis-pairing anyway.
        if stage_session.layer_range != stage.layer_range {
            bail!(
                "hybrid: session stage {s} range {:?} != model range {:?}",
                stage_session.layer_range,
                stage.layer_range
            );
        }

        // Layer loop. `il` is absolute (used to look up weight tensors
        // in `stage.tp_model.shards[r].layers[il]`); cache index is
        // `il - layer_range.start` (each stage's session allocated
        // exactly its slice of caches).
        let range_start = stage.layer_range.start;
        let stage_dev0 = stage.sub_cluster.device(0);
        let stage_stream0 = stage_dev0.default_stream();
        // CN-80B-20 — per-stage SHARED-graph capture/replay, gated on
        // FLAMBEAU_DECODE_GRAPH=1. ONE `hipGraph_t` per stage; every
        // TP-rank stream captures into that same graph via
        // `hipStreamBeginCaptureToGraph`, so cross-rank events
        // (BarP2pAllReduce record/wait) resolve as internal graph
        // edges. Replay = single `launch` on rank-0's stream which
        // issues the whole captured fan-out.
        //
        // Iter 1 limitation: NO slot binding. Captured K/V append
        // destinations and `n_tokens_kv` arg are FROZEN at capture-
        // time. Replay produces TIMING-MEANINGFUL but INCOHERENT
        // output. This isolates the wall-time question (is graph
        // capture worth the implementation cost on hybrid TP?) from
        // the correctness work (iter 2 — slot binding).
        for il in stage.layer_range.clone() {
            let il_cache = il - range_start;
            let is_full_attn = !cfg.is_recurrent(il);
            if is_full_attn {
                forward_full_attn_layer_tp(
                    &stage.tp_model,
                    stage_scratch,
                    &stage.sub_cluster,
                    stage_ar,
                    &mut stage_session.caches,
                    il,
                    il_cache,
                    position,
                    world,
                )
                .with_context(|| format!("hybrid stage {s} full-attn layer {il}"))?;
                if flambeau_backend_hip::profile::is_enabled() {
                    stage_dev0.bind()?;
                    flambeau_backend_hip::profile::mark(
                        "hyb_dec_full_attn",
                        stage_dev0,
                        stage_stream0,
                    )?;
                }
            } else {
                forward_gdn_layer_tp(
                    &stage.tp_model,
                    stage_scratch,
                    &stage.sub_cluster,
                    stage_ar,
                    &mut stage_session.caches,
                    il,
                    il_cache,
                    world,
                )
                .with_context(|| format!("hybrid stage {s} gdn layer {il}"))?;
                if flambeau_backend_hip::profile::is_enabled() {
                    stage_dev0.bind()?;
                    flambeau_backend_hip::profile::mark(
                        "hyb_dec_gdn",
                        stage_dev0,
                        stage_stream0,
                    )?;
                }
            }
        }
        if flambeau_backend_hip::profile::is_enabled() {
            stage_dev0.bind()?;
            flambeau_backend_hip::profile::mark(
                "hyb_dec_post_stage",
                stage_dev0,
                stage_stream0,
            )?;
        }

        // Hand-off: stage s+1's per-rank `hidden_a` ← stage s's rank-0
        // `hidden_a`. We do `tp_size_next` peer_copy_via_host calls,
        // each on the **global** cluster (which spans all ranks).
        if s + 1 < n_stages {
            // Sync the producer's stream so the source `hidden_a` is
            // fully written before we DtoH.
            let prod_dev = model.stages[s].sub_cluster.device(0);
            prod_dev.bind()?;
            prod_dev.default_stream().synchronize()?;

            let src_global_rank = s * tp_size; // stage s, local rank 0
            let src_ptr = scratch.per_stage[s].per_rank[0].hidden_a;
            let bytes = cfg.hidden_size * 2; // F16

            for dst_local in 0..tp_size {
                let dst_global_rank = (s + 1) * tp_size + dst_local;
                let dst_ptr = scratch.per_stage[s + 1].per_rank[dst_local].hidden_a;
                // SAFETY: `src_ptr` is `cfg.hidden_size * 2` bytes on
                // `src_global_rank`'s device (allocated by
                // ShardedForwardOneTokenScratchTp::new for stage s);
                // `dst_ptr` is the same size on `dst_global_rank`'s
                // device (allocated for stage s+1). Both ranks belong
                // to `global_cluster`. No other stream is concurrently
                // writing either region — the producer sync above
                // covers the source, and the destinations are about
                // to be (re-)written by the next stage's first
                // forward kernel.
                unsafe {
                    global_cluster
                        .peer_copy_via_host(
                            dst_ptr,
                            dst_global_rank,
                            src_ptr,
                            src_global_rank,
                            bytes,
                        )
                        .with_context(|| {
                            format!(
                                "hybrid stage {s} → {} hand-off (rank {src_global_rank} \
                                 → {dst_global_rank}, {bytes} B)",
                                s + 1
                            )
                        })?;
                }
            }
        }
    }

    // ── Last stage: output head + argmax / logits download ───────────
    let last = &model.stages[head_stage_idx];
    if !last.tp_model.has_output_head {
        bail!("hybrid last stage is missing output head; loader bug");
    }
    let last_scratch = &mut scratch.per_stage[head_stage_idx];
    let head_rank = last_scratch.head_rank.0 as usize;
    let device = last.sub_cluster.device(head_rank);
    device.bind()?;
    let stream = device.default_stream();
    let head_shard = &last.tp_model.shards[head_rank];
    let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
    let hidden_a = last_scratch.per_rank[head_rank].hidden_a;
    let head_scratch = last_scratch.per_rank[head_rank]
        .output_head
        .as_mut()
        .ok_or_else(|| {
            anyhow!("hybrid: head_rank={head_rank} on last stage missing OutputHeadScratch")
        })?;
    let logits_f32 = head_scratch.logits_f32;
    let ops = &last.tp_model.ops[head_rank];
    if flambeau_backend_hip::profile::is_enabled() {
        flambeau_backend_hip::profile::mark("hyb_dec_pre_lm_head", device, stream)?;
    }
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &head_shard.output_norm,
        lm_head,
        head_scratch,
        hidden_a,
    )
    .context("hybrid forward_output_head_decode")?;
    if flambeau_backend_hip::profile::is_enabled() {
        flambeau_backend_hip::profile::mark("hyb_dec_lm_head", device, stream)?;
    }

    match sink {
        HybridLogitsSink::HostLogits(out) => {
            out.clear();
            out.resize(cfg.vocab_size, 0.0f32);
            // SAFETY: logits_f32 is valid for cfg.vocab_size F32 values on
            // `device`; out.as_mut_ptr() is host memory of matching size.
            unsafe {
                <flambeau_backend_hip::HipDevice as flambeau_core::Device>::memcpy_async(
                    device,
                    stream,
                    flambeau_core::CopyDirection::DeviceToHost,
                    flambeau_core::DevicePtr(out.as_mut_ptr() as usize),
                    logits_f32,
                    cfg.vocab_size * 4,
                )?;
            }
            flambeau_core::Stream::synchronize(stream)?;
            if flambeau_backend_hip::profile::is_enabled() {
                flambeau_backend_hip::profile::mark("hyb_dec_logits_dtoh", device, stream)?;
            }
            Ok(0)
        }
        HybridLogitsSink::HostArgmax => {
            let token = argmax_token_host(device, stream, logits_f32, cfg.vocab_size)
                .context("hybrid argmax_token_host")?;
            if flambeau_backend_hip::profile::is_enabled() {
                flambeau_backend_hip::profile::mark("hyb_dec_argmax", device, stream)?;
            }
            Ok(token)
        }
        HybridLogitsSink::KeepOnDevice => {
            // **#258** — no DtoH, no host argmax, no sync. The next
            // call (gpu_sampler::run_gpu_topk on the same
            // default_stream) will serialise device-side against the
            // output_head writes via stream ordering, exactly like the
            // TP path. CPU-side syncing here would just stall the
            // dispatch loop without benefit.
            Ok(0)
        }
    }
}

// ---------------------------------------------------------------------------
// **P2.9b-i2-D-wire** — batched-decode driver for Hybrid (pp+tp) topology.
// ---------------------------------------------------------------------------

/// Drive `slots.len()` concurrent decode steps through the Hybrid
/// (PP-of-TP) topology with real per-layer batching, returning
/// per-slot `[vocab]` F32 logits.
///
/// Mirrors [`forward_prefill_hybrid_batched_logits`] for control
/// flow but with batched-decode bodies:
/// - Stage 0 embeds N tokens replicated on every rank within the stage.
/// - Per stage: per-rank per-layer:
///   * full-attn → `forward_full_attn_layer_decode_batched_tp`
///   * GDN → per-slot loop calling `forward_gdn_decode_tp`
///   followed by stage-internal AllReduce (`stage_ar`).
/// - Stage-boundary `peer_copy_via_host` of `[N, hidden]` F16 from
///   stage `s` rank 0 to every rank of stage `s+1`'s sub-cluster.
/// - Last stage: output head per-slot on `head_rank`.
///
/// PP-only and TP-only collapses to single-stage / single-rank
/// degenerates of this driver.
#[allow(clippy::too_many_arguments)]
pub fn forward_decode_batched_hybrid(
    model: &Qwen3MoEHybridModel,
    sessions: &mut [&mut Qwen3MoEHybridSession],
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    scratch: &mut ShardedForwardPrefillScratchHybrid,
    slots: &[super::batched::BatchSlot],
    logits_out: &mut [&mut Vec<f32>],
) -> Result<()> {
    use crate::session::LayerCache;
    use flambeau_backend_hip::HipDevice;
    use flambeau_core::DevicePtr;
    use flambeau_ops::hip::norm::rmsnorm_f16;

    let n = slots.len();
    if n == 0 {
        bail!("forward_decode_batched_hybrid: empty slot list");
    }
    if sessions.len() != logits_out.len() {
        bail!(
            "forward_decode_batched_hybrid: sessions({}) != logits_out({})",
            sessions.len(),
            logits_out.len(),
        );
    }
    for s in slots {
        if s.idx >= sessions.len() {
            bail!(
                "forward_decode_batched_hybrid: BatchSlot.idx {} OOB (n={})",
                s.idx,
                sessions.len()
            );
        }
    }

    let cfg = &model.config;
    let n_stages = model.stages.len();
    let tp_size = model.spec.tp_size as usize;
    if stage_ars.len() != n_stages {
        bail!("stage_ars.len()={} != n_stages={n_stages}", stage_ars.len());
    }
    let world = tp_size as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("hybrid batched-decode: per-stage tp_size ∈ {{1, 2, 4}} (got {world})");
    }
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = n * row_bytes;
    let elem_count_l = (n * hidden) as u32;

    // **#275** — drain every stage sub_cluster's per-rank default streams
    // at function entry. The prior call's last `ar_residual_prefill_pub`
    // launched its kernel on the *stage*'s sub_cluster default stream
    // (e.g. stage 1 dev 3). The upcoming `peer_copy_via_host` uses the
    // *global_cluster*'s default stream for the same physical device —
    // they are different `HipStream` handles since each `HipCluster::new`
    // constructs its own `HipDevice` / default-stream pair. Without this
    // sync the prior step's queued AR-residual kernel races the
    // peer_copy HtoD and clobbers `hidden_a` *after* the HtoD lands,
    // producing garbled output from token 3 onwards on N≥2 batched
    // hybrid (pp2tp2) decode.
    for stage in &model.stages {
        for r in 0..stage.sub_cluster.ranks() {
            let device = stage.sub_cluster.device(r);
            device.bind()?;
            Stream::synchronize(device.default_stream())?;
        }
    }

    // Per-slot positions (same across stages: each slot's per-stage
    // KV caches share the same tail per layer).
    let slot_positions: Vec<usize> = slots.iter().map(|s| s.position).collect();

    // 1. Stage 0 embed — replicate token row 0..N across all ranks of
    //    stage 0's sub_cluster.
    {
        let stage0 = &model.stages[0];
        if !stage0.tp_model.has_token_embd {
            bail!("hybrid stage 0 missing token_embd");
        }
        let stage_scratch = &mut scratch.per_stage[0];
        for r in 0..stage0.sub_cluster.ranks() {
            let device = stage0.sub_cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let dst_base = stage_scratch.per_rank[r].hidden_a;
            for (s_pos, slot) in slots.iter().enumerate() {
                super::io::forward_embed_decode_host(
                    device,
                    stream,
                    &stage0.tp_model.shards[r].token_embd,
                    slot.token_id,
                    DevicePtr(dst_base.as_usize() + s_pos * row_bytes),
                    hidden,
                )
                .with_context(|| format!("hybrid stage 0 rank {r} embed slot {s_pos}"))?;
            }
        }
    }

    // 2. Per-stage layer loop with stage-boundary peer_copy.
    for stage_idx in 0..n_stages {
        let stage = &model.stages[stage_idx];
        let stage_scratch = &mut scratch.per_stage[stage_idx];
        let stage_ar = &stage_ars[stage_idx];
        let stage_model = &stage.tp_model;
        let sub_cluster = &stage.sub_cluster;

        // **#275 cycle 6 debug** — `FLAMBEAU_STAGE_ENTRY_DUMP=1` dumps
        // hidden_a row 0 L2 at the START of each stage (post-peer_copy
        // from previous stage). Used to verify stage handoff at N=2.
        if dev_flag("FLAMBEAU_STAGE_ENTRY_DUMP") {
            let entry_dev = sub_cluster.device(0);
            entry_dev.bind()?;
            entry_dev.default_stream().synchronize()?;
            let mut host = vec![half::f16::from_f32(0.0); n * hidden];
            unsafe {
                entry_dev.memcpy_async(
                    entry_dev.default_stream(),
                    flambeau_core::CopyDirection::DeviceToHost,
                    flambeau_core::DevicePtr(host.as_mut_ptr() as usize),
                    stage_scratch.per_rank[0].hidden_a,
                    n * hidden * 2,
                )?;
            }
            entry_dev.default_stream().synchronize()?;
            for s in 0..n {
                let row = &host[s * hidden..(s + 1) * hidden];
                let l2: f64 = row
                    .iter()
                    .map(|v| (v.to_f32() as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let head: Vec<f32> =
                    row[..4.min(row.len())].iter().map(|v| v.to_f32()).collect();
                eprintln!(
                    "[STAGE-ENTRY-DUMP] N={n} stage={stage_idx} slot={s} L2={l2:.4} head={head:?}"
                );
            }
        }
        let il_cache_offset = stage.layer_range.start;
        let kv_replicated = stage_model.tp.kv_replicated();
        let kq_replicated = stage_model.tp.gdn_kq_replicated();

        for il in stage.layer_range.clone() {
            let il_local = il - il_cache_offset;
            let is_full_attn = !cfg.is_recurrent(il);

            // 3a. Per-rank attention forward.
            for r in 0..sub_cluster.ranks() {
                let device = sub_cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &stage_model.shards[r].layers[il];
                let hidden_a = stage_scratch.per_rank[r].hidden_a;
                let partial_attn_out = stage_scratch.per_rank[r].partial_attn_out;
                let layer_scratch = stage_scratch.per_rank[r]
                    .layer
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                let ops = &stage_model.ops[r];

                if is_full_attn {
                    let attn_norm = find_tensor_in_layer(layer_tensors, il, "attn_norm.weight")?;
                    let attn_q = find_tensor_in_layer(layer_tensors, il, "attn_q.weight")?;
                    let attn_k = find_tensor_in_layer(layer_tensors, il, "attn_k.weight")?;
                    let attn_v = find_tensor_in_layer(layer_tensors, il, "attn_v.weight")?;
                    let attn_output = find_tensor_in_layer(layer_tensors, il, "attn_output.weight")?;
                    let attn_q_norm = find_tensor_in_layer(layer_tensors, il, "attn_q_norm.weight")?;
                    let attn_k_norm = find_tensor_in_layer(layer_tensors, il, "attn_k_norm.weight")?;
                    let full = layer_scratch
                        .full_attn
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing FullAttnPrefillScratch"))?;
                    // Gather per-slot KV caches for THIS layer on THIS
                    // stage on THIS rank. Each slot has its own per-stage
                    // session at sessions[s].stages[stage_idx].caches[r][il_local].
                    let sessions_ptr = sessions.as_mut_ptr();
                    let mut slot_kv_caches: Vec<
                        &mut flambeau_runtime::KvCache<flambeau_runtime::F16Contig, HipDevice>,
                    > = Vec::with_capacity(n);
                    for s in 0..n {
                        // SAFETY: indices 0..n distinct; each session is unique.
                        unsafe {
                            let session_ref: &mut Qwen3MoEHybridSession =
                                &mut **sessions_ptr.add(s);
                            match &mut session_ref.stages[stage_idx].caches[r][il_local] {
                                LayerCache::FullAttn(kv) => slot_kv_caches.push(kv),
                                _ => bail!(
                                    "Hybrid batched-decode: slot {s} stage {stage_idx} rank {r} \
                                     layer {il} expected FullAttn cache"
                                ),
                            }
                        }
                    }
                    super::attn_tp::forward_full_attn_layer_decode_batched_tp(
                        ops,
                        stream,
                        device,
                        cfg,
                        attn_norm,
                        attn_q,
                        attn_k,
                        attn_v,
                        attn_output,
                        attn_q_norm,
                        attn_k_norm,
                        &mut slot_kv_caches,
                        full,
                        hidden_a,
                        partial_attn_out,
                        &slot_positions,
                        world,
                        kv_replicated,
                    )
                    .with_context(|| format!("hybrid full-attn stage {stage_idx} layer {il} rank {r}"))?;
                } else {
                    // GDN forward. **#286 batched-GDN**: by default
                    // dispatches a single `forward_gdn_decode_batched_tp`
                    // call over all N slots (matmul-heavy stages
                    // batched at n_tokens=N; per-slot inner loop only
                    // for conv1d + state-step). Set
                    // `FLAMBEAU_GDN_NO_BATCHED=1` to fall back to the
                    // legacy per-slot loop for A/B regression checks.
                    let attn_norm = find_tensor_in_layer(layer_tensors, il, "attn_norm.weight")?;
                    let attn_qkv = find_tensor_in_layer(layer_tensors, il, "attn_qkv.weight")?;
                    let attn_gate = find_tensor_in_layer(layer_tensors, il, "attn_gate.weight")?;
                    let ssm_alpha = find_tensor_in_layer(layer_tensors, il, "ssm_alpha.weight")?;
                    let ssm_beta = find_tensor_in_layer(layer_tensors, il, "ssm_beta.weight")?;
                    let ssm_a = find_tensor_in_layer(layer_tensors, il, "ssm_a")?;
                    let ssm_dt_bias = find_tensor_in_layer(layer_tensors, il, "ssm_dt.bias")?;
                    let ssm_conv1d = find_tensor_in_layer(layer_tensors, il, "ssm_conv1d.weight")?;
                    let ssm_norm = find_tensor_in_layer(layer_tensors, il, "ssm_norm.weight")?;
                    let ssm_out = find_tensor_in_layer(layer_tensors, il, "ssm_out.weight")?;

                    // Gather per-slot &mut GdnLayerState. Indices 0..n
                    // are distinct so the &mut borrows are disjoint.
                    let sessions_ptr = sessions.as_mut_ptr();
                    let mut layer_states: Vec<&mut crate::session::GdnLayerState> =
                        Vec::with_capacity(n);
                    for s in 0..n {
                        // SAFETY: indices 0..n distinct; each session is unique.
                        unsafe {
                            let session_ref: &mut Qwen3MoEHybridSession =
                                &mut **sessions_ptr.add(s);
                            match &mut session_ref.stages[stage_idx].caches[r][il_local] {
                                LayerCache::Gdn(state) => layer_states.push(state),
                                _ => bail!(
                                    "Hybrid batched-decode: slot {s} stage {stage_idx} rank {r} \
                                     layer {il} expected Gdn cache"
                                ),
                            }
                        }
                    }

                    let gdn_batched = stage_scratch.per_rank[r]
                        .gdn_decode_batched
                        .as_mut()
                        .ok_or_else(|| {
                            anyhow!(
                                "rank {r}: missing gdn_decode_batched scratch \
                                 (cfg.gdn was Some at scratch alloc time?)"
                            )
                        })?;
                    super::gdn_tp::forward_gdn_decode_batched_tp(
                        ops, stream, device, cfg,
                        attn_norm, attn_qkv, attn_gate,
                        ssm_alpha, ssm_beta, ssm_a, ssm_dt_bias,
                        ssm_conv1d, ssm_norm, ssm_out,
                        layer_states.as_mut_slice(),
                        gdn_batched,
                        hidden_a, partial_attn_out,
                        n,
                        world, kq_replicated,
                    )
                    .with_context(|| {
                        format!(
                            "hybrid GDN batched stage {stage_idx} layer {il} rank {r}"
                        )
                    })?;
                }
            }

            // **#275 cycle 7 debug** — dump partial_attn_out row 0 BEFORE AR
            // and hidden_a row 0 AFTER AR for the first layer of stage 1.
            // Compares "GDN call wrote wrong row 0" vs "AR mangled row 0
            // at n_tokens=N>1" across N=1 and N=2 dispatches.
            // Dump only layer 32 (first GDN of stage 1) on rank 0.
            let do_pre_ar_dump = dev_flag("FLAMBEAU_AR_DUMP")
                && stage_idx == 1
                && il == 32;
            if do_pre_ar_dump {
                let dump_dev = sub_cluster.device(0);
                dump_dev.bind()?;
                dump_dev.default_stream().synchronize()?;
                let mut host = vec![half::f16::from_f32(0.0); n * hidden];
                unsafe {
                    dump_dev.memcpy_async(
                        dump_dev.default_stream(),
                        flambeau_core::CopyDirection::DeviceToHost,
                        flambeau_core::DevicePtr(host.as_mut_ptr() as usize),
                        stage_scratch.per_rank[0].partial_attn_out,
                        n * hidden * 2,
                    )?;
                }
                dump_dev.default_stream().synchronize()?;
                for s in 0..n {
                    let row = &host[s * hidden..(s + 1) * hidden];
                    let l2: f64 = row
                        .iter()
                        .map(|v| (v.to_f32() as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let head: Vec<f32> =
                        row[..4.min(row.len())].iter().map(|v| v.to_f32()).collect();
                    eprintln!(
                        "[AR-DUMP] N={n} il={il} when=PRE-AR slot={s} L2={l2:.4} head={head:?}"
                    );
                }
            }

            // 3b. Stage-internal AR(hidden_a, partial_attn_out, N*hidden).
            super::tp::ar_residual_prefill_pub(
                stage_ar,
                stage_scratch,
                sub_cluster,
                world,
                elem_count_l,
                super::tp::AttnOrFfnPub::Attn,
            )?;

            // **#275 cycle 7 debug** — post-AR dump.
            if do_pre_ar_dump {
                let dump_dev = sub_cluster.device(0);
                dump_dev.bind()?;
                dump_dev.default_stream().synchronize()?;
                let mut host = vec![half::f16::from_f32(0.0); n * hidden];
                unsafe {
                    dump_dev.memcpy_async(
                        dump_dev.default_stream(),
                        flambeau_core::CopyDirection::DeviceToHost,
                        flambeau_core::DevicePtr(host.as_mut_ptr() as usize),
                        stage_scratch.per_rank[0].hidden_a,
                        n * hidden * 2,
                    )?;
                }
                dump_dev.default_stream().synchronize()?;
                for s in 0..n {
                    let row = &host[s * hidden..(s + 1) * hidden];
                    let l2: f64 = row
                        .iter()
                        .map(|v| (v.to_f32() as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let head: Vec<f32> =
                        row[..4.min(row.len())].iter().map(|v| v.to_f32()).collect();
                    eprintln!(
                        "[AR-DUMP] N={n} il={il} when=POST-AR slot={s} L2={l2:.4} head={head:?}"
                    );
                }
            }

            // 3c. ffn_norm[N] over hidden_a → mid_norm.
            for r in 0..sub_cluster.ranks() {
                let device = sub_cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &stage_model.shards[r].layers[il];
                let ffn_norm = find_tensor_in_layer(layer_tensors, il, "ffn_norm.weight")
                    .or_else(|_| {
                        find_tensor_in_layer(layer_tensors, il, "post_attention_norm.weight")
                    })?;
                let hidden_a = stage_scratch.per_rank[r].hidden_a;
                let layer_scratch = stage_scratch.per_rank[r]
                    .layer
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                let mid_norm = layer_scratch.mid_norm_f16;
                let ops = &stage_model.ops[r];
                rmsnorm_f16(
                    ops,
                    stream,
                    hidden_a,
                    ffn_norm.ptr,
                    mid_norm,
                    n,
                    hidden,
                    cfg.rms_norm_eps,
                )
                .with_context(|| format!("hybrid ffn_norm stage {stage_idx} layer {il}"))?;
            }

            // 3d. Per-rank FFN forward.
            let moe_replicated = !cfg.is_dense_ffn() && stage_model.moe_replicated_at(il);
            let ffn_world = if moe_replicated { 1 } else { world };
            if cfg.is_dense_ffn() {
                for r in 0..sub_cluster.ranks() {
                    let device = sub_cluster.device(r);
                    device.bind()?;
                    let stream = device.default_stream();
                    let layer_tensors = &stage_model.shards[r].layers[il];
                    let ffn_gate = find_tensor_in_layer(layer_tensors, il, "ffn_gate.weight")?;
                    let ffn_up = find_tensor_in_layer(layer_tensors, il, "ffn_up.weight")?;
                    let ffn_down = find_tensor_in_layer(layer_tensors, il, "ffn_down.weight")?;
                    let partial_ffn_out = stage_scratch.per_rank[r].partial_ffn_out;
                    let layer_scratch = stage_scratch.per_rank[r]
                        .layer
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                    let mid_norm = layer_scratch.mid_norm_f16;
                    let dense_scratch = layer_scratch
                        .dense_ffn
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing DenseFfnPrefillScratch"))?;
                    let ops = &stage_model.ops[r];
                    super::dense_ffn_tp::forward_dense_ffn_prefill_tp(
                        ops, stream, cfg, ffn_gate, ffn_up, ffn_down, dense_scratch,
                        mid_norm, partial_ffn_out, n, world,
                    )
                    .with_context(|| format!("hybrid dense ffn stage {stage_idx} layer {il}"))?;
                }
            } else {
                let has_shared = cfg.shared_expert_intermediate_size.is_some()
                    && !dev_flag("FLAMBEAU_TP_SKIP_SHARED");
                for r in 0..sub_cluster.ranks() {
                    let device = sub_cluster.device(r);
                    device.bind()?;
                    let stream = device.default_stream();
                    let layer_tensors = &stage_model.shards[r].layers[il];
                    let ffn_gate_inp =
                        find_tensor_in_layer(layer_tensors, il, "ffn_gate_inp.weight")?;
                    let ffn_gate_exps =
                        find_tensor_in_layer(layer_tensors, il, "ffn_gate_exps.weight")?;
                    let ffn_up_exps =
                        find_tensor_in_layer(layer_tensors, il, "ffn_up_exps.weight")?;
                    let ffn_down_exps =
                        find_tensor_in_layer(layer_tensors, il, "ffn_down_exps.weight")?;
                    let partial_ffn_out = stage_scratch.per_rank[r].partial_ffn_out;
                    let shared_delta_f16 = stage_scratch.per_rank[r]
                        .layer
                        .as_ref()
                        .map(|l| l.shared_delta_f16)
                        .unwrap_or(DevicePtr(0));
                    let layer_scratch = stage_scratch.per_rank[r]
                        .layer
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
                    let mid_norm = layer_scratch.mid_norm_f16;
                    let ops = &stage_model.ops[r];

                    {
                        let moe_scratch = layer_scratch
                            .moe
                            .as_mut()
                            .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                        super::moe::forward_router_prefill(
                            ops, stream, cfg, ffn_gate_inp, moe_scratch, mid_norm, n,
                        )
                        .with_context(|| {
                            format!("hybrid router stage {stage_idx} layer {il}")
                        })?;
                    }
                    if has_shared {
                        let shared_w_gate =
                            find_tensor_in_layer(layer_tensors, il, "ffn_gate_shexp.weight")?;
                        let shared_w_up =
                            find_tensor_in_layer(layer_tensors, il, "ffn_up_shexp.weight")?;
                        let shared_w_down =
                            find_tensor_in_layer(layer_tensors, il, "ffn_down_shexp.weight")?;
                        let shared_w_gate_inp =
                            find_tensor_in_layer(layer_tensors, il, "ffn_gate_inp_shexp.weight")
                                .ok();
                        let shared_scratch = layer_scratch.shared.as_mut().ok_or_else(|| {
                            anyhow!("rank {r}: missing SharedExpertPrefillScratch")
                        })?;
                        super::moe_tp::forward_shared_expert_prefill_tp(
                            ops, stream, cfg, shared_w_gate, shared_w_up, shared_w_down,
                            shared_w_gate_inp, shared_scratch, mid_norm, shared_delta_f16,
                            n, ffn_world,
                        )
                        .with_context(|| {
                            format!("hybrid shared expert stage {stage_idx} layer {il}")
                        })?;
                    }
                    let moe_scratch = layer_scratch
                        .moe
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                    super::moe_tp::forward_moe_ffn_prefill_tp(
                        ops, stream, cfg, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
                        moe_scratch, mid_norm, partial_ffn_out, n, ffn_world,
                    )
                    .with_context(|| {
                        format!("hybrid moe ffn stage {stage_idx} layer {il}")
                    })?;
                    if has_shared {
                        flambeau_ops::hip::mlp::add_f16(
                            ops,
                            stream,
                            partial_ffn_out,
                            shared_delta_f16,
                            partial_ffn_out,
                            n * hidden,
                        )
                        .with_context(|| {
                            format!("hybrid + shared add stage {stage_idx} layer {il}")
                        })?;
                    }
                }
            }

            // 3e. AR-residual on FFN output.
            if ffn_world > 1 {
                super::tp::ar_residual_prefill_pub(
                    stage_ar,
                    stage_scratch,
                    sub_cluster,
                    world,
                    elem_count_l,
                    super::tp::AttnOrFfnPub::Ffn,
                )?;
            } else {
                for r in 0..sub_cluster.ranks() {
                    let device = sub_cluster.device(r);
                    device.bind()?;
                    let stream = device.default_stream();
                    let ops = &stage_model.ops[r];
                    flambeau_ops::hip::mlp::add_f16(
                        ops,
                        stream,
                        stage_scratch.per_rank[r].hidden_a,
                        stage_scratch.per_rank[r].partial_ffn_out,
                        stage_scratch.per_rank[r].hidden_a,
                        n * hidden,
                    )
                    .with_context(|| {
                        format!("hybrid replicated ffn add stage {stage_idx} layer {il}")
                    })?;
                }
            }
        }

        // **#275 cycle 2 debug** — per-stage post-FFN hidden L2 dump
        // gated by FLAMBEAU_BATCHED_DECODE_DUMP=1. Compares N=1 vs N=2
        // dispatch trace to find the first divergent stage. Dumps from
        // stage rank 0's hidden_a (where the AR result lives).
        if dev_flag("FLAMBEAU_BATCHED_DECODE_DUMP") {
            let dump_dev = stage.sub_cluster.device(0);
            dump_dev.bind()?;
            dump_dev.default_stream().synchronize()?;
            let mut host = vec![half::f16::from_f32(0.0); n * hidden];
            let dump_ptr = scratch.per_stage[stage_idx].per_rank[0].hidden_a;
            // SAFETY: dump_ptr is [N, hidden] F16 on dump_dev; sync above.
            unsafe {
                dump_dev.memcpy_async(
                    dump_dev.default_stream(),
                    flambeau_core::CopyDirection::DeviceToHost,
                    flambeau_core::DevicePtr(host.as_mut_ptr() as usize),
                    dump_ptr,
                    n * hidden * 2,
                )?;
            }
            dump_dev.default_stream().synchronize()?;
            for s in 0..n {
                let row = &host[s * hidden..(s + 1) * hidden];
                let l2: f64 = row
                    .iter()
                    .map(|v| {
                        let f = v.to_f32() as f64;
                        f * f
                    })
                    .sum::<f64>()
                    .sqrt();
                let head: Vec<f32> =
                    row[..4.min(row.len())].iter().map(|v| v.to_f32()).collect();
                eprintln!(
                    "[BATCHED-DECODE-DUMP] N={n} stage={stage_idx} slot={s} L2={l2:.4} head={head:?}"
                );
            }
        }

        // **#275 cycle 4 debug** — `FLAMBEAU_LAYER_STATE_DUMP=1` dumps
        // per-layer KV cache + GDN state L2 norms for **every slot** on
        // rank 0 at the END OF EACH STAGE. Used to find the first
        // (layer, slot) pair whose state differs across N>=2 dispatches
        // (#16 35B-A3B/pp2tp2 multi-slot race investigation: slot 0 is
        // bit-deterministic across boots, slot 1 differs — extending
        // the per-slot dump localises which layer first deviates on
        // slot 1 between two boots).
        if dev_flag("FLAMBEAU_LAYER_STATE_DUMP") {
          // SAFETY: same disjointness as the other unsafe split above —
          // indices 0..n are distinct so the &mut borrows stay disjoint.
          let sessions_ptr = sessions.as_mut_ptr();
          for slot_idx in 0..n {
            let session_s: &mut Qwen3MoEHybridSession =
                unsafe { &mut **sessions_ptr.add(slot_idx) };
            let stage_caches = &session_s.stages[stage_idx].caches[0];
            let stage_dev = stage.sub_cluster.device(0);
            stage_dev.bind()?;
            stage_dev.default_stream().synchronize()?;
            for (li, cache) in stage_caches.iter().enumerate() {
                let global_il = stage.layer_range.start + li;
                match cache {
                    LayerCache::FullAttn(kv) => {
                        // Only dump the VALID portion (positions 0..current_tokens).
                        // Uninitialized memory beyond that pollutes L2 with stale
                        // per-session noise that has nothing to do with kernel bugs.
                        let valid_tokens = kv.current_tokens();
                        let per_token_bytes = kv.n_heads() * kv.head_dim() * 2;
                        let valid_bytes = valid_tokens * per_token_bytes;
                        if valid_bytes == 0 {
                            eprintln!(
                                "[STATE-DUMP] N={n} slot={slot_idx} stage={stage_idx} layer={global_il} type=fullattn current_tokens=0 (empty)"
                            );
                            continue;
                        }
                        let mut k_host = vec![0u8; valid_bytes];
                        let mut v_host = vec![0u8; valid_bytes];
                        unsafe {
                            stage_dev.memcpy_async(
                                stage_dev.default_stream(),
                                flambeau_core::CopyDirection::DeviceToHost,
                                flambeau_core::DevicePtr(
                                    k_host.as_mut_ptr() as usize,
                                ),
                                kv.k_buffer(),
                                valid_bytes,
                            )?;
                            stage_dev.memcpy_async(
                                stage_dev.default_stream(),
                                flambeau_core::CopyDirection::DeviceToHost,
                                flambeau_core::DevicePtr(
                                    v_host.as_mut_ptr() as usize,
                                ),
                                kv.v_buffer(),
                                valid_bytes,
                            )?;
                        }
                        stage_dev.default_stream().synchronize()?;
                        let k_f16: &[half::f16] = unsafe {
                            std::slice::from_raw_parts(
                                k_host.as_ptr() as *const half::f16,
                                valid_bytes / 2,
                            )
                        };
                        let v_f16: &[half::f16] = unsafe {
                            std::slice::from_raw_parts(
                                v_host.as_ptr() as *const half::f16,
                                valid_bytes / 2,
                            )
                        };
                        let k_l2: f64 = k_f16
                            .iter()
                            .map(|x| (x.to_f32() as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let v_l2: f64 = v_f16
                            .iter()
                            .map(|x| (x.to_f32() as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        // Also dump LAST-token K row L2 (= the just-appended row)
                        // as a separate metric to detect off-by-one bugs.
                        let last_off = (valid_tokens - 1) * per_token_bytes / 2;
                        let last_k_l2: f64 = k_f16[last_off..last_off + per_token_bytes / 2]
                            .iter()
                            .map(|x| (x.to_f32() as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        eprintln!(
                            "[STATE-DUMP] N={n} slot={slot_idx} stage={stage_idx} layer={global_il} type=fullattn current_tokens={valid_tokens} k_l2={k_l2:.4} v_l2={v_l2:.4} last_k_l2={last_k_l2:.4}"
                        );
                    }
                    LayerCache::Gdn(g) => {
                        let mut state_host = vec![0u8; g.state_bytes];
                        let mut conv_host = vec![0u8; g.conv_history_bytes];
                        unsafe {
                            stage_dev.memcpy_async(
                                stage_dev.default_stream(),
                                flambeau_core::CopyDirection::DeviceToHost,
                                flambeau_core::DevicePtr(
                                    state_host.as_mut_ptr() as usize,
                                ),
                                g.state,
                                g.state_bytes,
                            )?;
                            stage_dev.memcpy_async(
                                stage_dev.default_stream(),
                                flambeau_core::CopyDirection::DeviceToHost,
                                flambeau_core::DevicePtr(
                                    conv_host.as_mut_ptr() as usize,
                                ),
                                g.conv_history,
                                g.conv_history_bytes,
                            )?;
                        }
                        stage_dev.default_stream().synchronize()?;
                        // GDN state is F32, conv_history is F32.
                        let state_f32: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                state_host.as_ptr() as *const f32,
                                g.state_bytes / 4,
                            )
                        };
                        let conv_f32: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                conv_host.as_ptr() as *const f32,
                                g.conv_history_bytes / 4,
                            )
                        };
                        let state_l2: f64 = state_f32
                            .iter()
                            .map(|x| (*x as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let conv_l2: f64 = conv_f32
                            .iter()
                            .map(|x| (*x as f64).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        eprintln!(
                            "[STATE-DUMP] N={n} slot={slot_idx} stage={stage_idx} layer={global_il} type=gdn state_l2={state_l2:.4} conv_l2={conv_l2:.4}"
                        );
                    }
                    LayerCache::FullAttnQ8(_) => {
                        eprintln!(
                            "[STATE-DUMP] N={n} slot={slot_idx} stage={stage_idx} layer={global_il} type=fullattn_q8 (skipped)"
                        );
                    }
                }
            }
          } // end per-slot dump loop (FLAMBEAU_LAYER_STATE_DUMP)
        }

        // 4. Stage-boundary hand-off: peer_copy stage_idx rank 0's
        //    hidden_a to every rank of stage_idx+1.
        if stage_idx + 1 < n_stages {
            let prod_dev = model.stages[stage_idx].sub_cluster.device(0);
            prod_dev.bind()?;
            prod_dev.default_stream().synchronize()?;
            let src_global_rank = stage_idx * tp_size;
            let src_ptr = scratch.per_stage[stage_idx].per_rank[0].hidden_a;
            for dst_local in 0..tp_size {
                let dst_global_rank = (stage_idx + 1) * tp_size + dst_local;
                let dst_ptr = scratch.per_stage[stage_idx + 1].per_rank[dst_local].hidden_a;
                // SAFETY: src/dst are [N, hidden] F16 buffers on
                // their respective devices; producer-side stream
                // synced above.
                unsafe {
                    global_cluster
                        .peer_copy_via_host(
                            dst_ptr,
                            dst_global_rank,
                            src_ptr,
                            src_global_rank,
                            chunk_bytes,
                        )
                        .with_context(|| {
                            format!(
                                "hybrid batched-decode stage {stage_idx} → {} hand-off",
                                stage_idx + 1
                            )
                        })?;
                }
            }
        }
    }

    // 5. Output head per slot on last stage's head_rank.
    let head_stage_idx = scratch.head_stage as usize;
    if head_stage_idx + 1 != n_stages {
        bail!(
            "hybrid batched-decode: head_stage={head_stage_idx} but expected last = {}",
            n_stages - 1
        );
    }
    let last_stage = &model.stages[head_stage_idx];
    if !last_stage.tp_model.has_output_head {
        bail!("hybrid last stage missing output head");
    }
    let last_scratch = &mut scratch.per_stage[head_stage_idx];
    let head_rank = last_scratch.head_rank.0 as usize;
    let device = last_stage.sub_cluster.device(head_rank);
    device.bind()?;
    let stream = device.default_stream();
    let head_shard = &last_stage.tp_model.shards[head_rank];
    let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
    let head_hidden_base = last_scratch.per_rank[head_rank].hidden_a;
    let head_scratch = last_scratch.per_rank[head_rank]
        .output_head
        .as_mut()
        .ok_or_else(|| anyhow!("hybrid: head_rank missing OutputHeadScratch"))?;
    let ops = &last_stage.tp_model.ops[head_rank];

    for (s_pos, slot) in slots.iter().enumerate() {
        let x_final_row = DevicePtr(head_hidden_base.as_usize() + s_pos * row_bytes);
        super::io::forward_output_head_decode(
            ops,
            stream,
            cfg,
            &head_shard.output_norm,
            lm_head,
            head_scratch,
            x_final_row,
        )
        .with_context(|| format!("hybrid batched-decode output head slot {}", slot.idx))?;
        super::io::download_logits_host(
            device,
            stream,
            head_scratch.logits_f32,
            cfg.vocab_size,
            logits_out[slot.idx],
        )
        .with_context(|| {
            format!("hybrid batched-decode logits download slot {}", slot.idx)
        })?;
    }

    Ok(())
}

/// Find a layer-tensor by suffix within a stage's per-rank layer
/// tensor list. Same pattern as `tp::find_by_suffix` (private there).
fn find_tensor_in_layer<'a>(
    tensors: &'a [crate::tp_sharded::TpLayerTensor],
    il: usize,
    suffix: &str,
) -> Result<&'a crate::weights::DeviceTensor> {
    let want = format!("blk.{il}.{suffix}");
    tensors
        .iter()
        .find(|t| t.name.as_ref() == want.as_str())
        .map(|t| &t.tensor)
        .ok_or_else(|| anyhow!("hybrid layer {il}: missing tensor `{suffix}`"))
}

