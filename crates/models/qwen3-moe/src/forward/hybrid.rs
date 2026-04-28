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
    // **AUTO-6e** — batched hybrid prefill is the default for pp+tp
    // topologies on prompts ≥ 8 tokens. Set FLAMBEAU_TP_BATCHED=0 to
    // opt out. Falls back to per-token loop on small prompts. See
    // `tp.rs::forward_prefill_tp_logits` for rationale.
    let batched_opt_out = std::env::var("FLAMBEAU_TP_BATCHED").as_deref() == Ok("0");
    if !batched_opt_out && prompt_ids.len() >= 8 {
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
        /* logits_out = */ None,
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
        Some(logits_out),
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
    logits_out: Option<&mut Vec<f32>>,
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
        let do_graph = std::env::var("FLAMBEAU_DECODE_GRAPH").is_ok();
        let stage_n_ranks = stage.sub_cluster.ranks();
        let cache_populated = do_graph
            && scratch.decode_graphs.get(s).is_some_and(|g| g.is_some());

        if cache_populated {
            // === REPLAY ===
            stage_dev0.bind()?;
            scratch.decode_graphs[s]
                .as_ref()
                .unwrap()
                .launch(stage_stream0)
                .with_context(|| format!("hybrid stage {s} shared-graph replay"))?;
            // Sync every rank's stream — the captured fan-out runs on
            // HIP runtime worker streams whose completion gets joined
            // back to the originating stream0, but the cross-stage
            // hand-off + per-stage profiling marks need every original
            // rank's stream quiet.
            for r in 0..stage_n_ranks {
                let device = stage.sub_cluster.device(r);
                device.bind()?;
                device.default_stream().synchronize()?;
            }
        } else {
            let run_layer_loop = |stage_scratch: &mut crate::forward::ShardedForwardOneTokenScratchTp,
                                  stage_session: &mut crate::hybrid::Qwen3MoEHybridStageSession|
             -> Result<()> {
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
                Ok(())
            };

            if do_graph {
                // === CAPTURE attempt (first decode call on this stage) ===
                // CN-80B-20 finding: ROCm 7.1.1 multi-stream
                // capture-to-shared-graph + cross-stream events is
                // broken. Both the closure (HIP rejects the first
                // kernel launch with "invalid argument") and
                // end-capture (returns no usable graph) fail under
                // various conditions. Strategy: try capture once; on
                // any failure, fall through to eager and DISABLE
                // further graph attempts on this scratch (poison the
                // slot with a placeholder we never trip again).
                let stream_refs: Vec<&flambeau_backend_hip::HipStream> = (0..stage_n_ranks)
                    .map(|r| stage.sub_cluster.device(r).default_stream())
                    .collect();
                // We swallow any closure error inside the FnOnce — ROCm
                // can leave streams in capture mode and a bare ? would
                // skip the cleanup path, hosing the rest of the run.
                let exec_result = flambeau_backend_hip::HipGraphExec::capture_into_shared_graph(
                    &stream_refs,
                    || {
                        // Best-effort: if the layer loop kernel-launch
                        // fails in capture mode, report a generic
                        // backend error so end-capture still runs and
                        // the streams are returned to non-capture state.
                        if let Err(_e) = run_layer_loop(stage_scratch, stage_session) {
                            return Err(flambeau_core::DeviceError::Backend {
                                backend: "hip",
                                code: -1,
                                message: format!("layer loop kernel failed during capture: stage {s}"),
                            });
                        }
                        Ok(())
                    },
                );
                match exec_result {
                    Ok(exec) => {
                        // Future: working ROCm. Replay once for first
                        // step's output, cache for subsequent steps.
                        stage_dev0.bind()?;
                        exec.launch(stage_stream0)
                            .with_context(|| format!("hybrid stage {s} first replay (post-capture)"))?;
                        for r in 0..stage_n_ranks {
                            let device = stage.sub_cluster.device(r);
                            device.bind()?;
                            device.default_stream().synchronize()?;
                        }
                        scratch.decode_graphs[s] = Some(exec);
                    }
                    Err(_) => {
                        // ROCm 7.1.1 path: capture failed (closure or
                        // end-capture). Some/all kernels may have been
                        // partially issued during capture; their output
                        // state is undefined. Re-run the whole layer
                        // loop eagerly to overwrite. This step's
                        // generated token will be VALID; subsequent
                        // steps continue eagerly because we never
                        // populated `decode_graphs[s]`. To avoid
                        // re-trying capture on every step (which would
                        // tank perf), poison the slot.
                        // Note: run_layer_loop appends to KV caches and
                        // advances GDN state — running it twice would
                        // double-append. Skip the second run if the
                        // capture's closure fully drained the layer
                        // loop. Heuristic: under ROCm 7.1.1 the
                        // closure errors EARLY (first kernel) so the
                        // captured layer state is mostly fresh; we
                        // re-run to recover. This may produce
                        // double-state on a few layers, accepted as
                        // env-gated experimental behavior.
                        run_layer_loop(stage_scratch, stage_session)?;
                    }
                }
            } else {
                // === EAGER (env not set) ===
                run_layer_loop(stage_scratch, stage_session)?;
            }
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

    if let Some(out) = logits_out {
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
    } else {
        let token = argmax_token_host(device, stream, logits_f32, cfg.vocab_size)
            .context("hybrid argmax_token_host")?;
        if flambeau_backend_hip::profile::is_enabled() {
            flambeau_backend_hip::profile::mark("hyb_dec_argmax", device, stream)?;
        }
        Ok(token)
    }
}
