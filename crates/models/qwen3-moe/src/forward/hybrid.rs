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
use crate::Qwen3MoEConfig;

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
    use flambeau_core::Stream;
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
        if std::env::var("FLAMBEAU_STAGE_ENTRY_DUMP").is_ok() {
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

                    let no_batched_gdn =
                        std::env::var("FLAMBEAU_GDN_NO_BATCHED").is_ok();
                    if no_batched_gdn {
                        // Legacy per-slot fallback (kept for A/B regression).
                        let rank_scratch_ptr: *mut super::tp::RankForwardPrefillScratchTp =
                            &mut stage_scratch.per_rank[r];
                        for (s, layer_state) in
                            layer_states.iter_mut().enumerate()
                        {
                            let slot_x_in =
                                DevicePtr(hidden_a.as_usize() + s * row_bytes);
                            let slot_partial =
                                DevicePtr(partial_attn_out.as_usize() + s * row_bytes);
                            let gdn = unsafe {
                                (*rank_scratch_ptr).gdn_decode.as_mut().ok_or_else(
                                    || anyhow!("rank {r}: missing gdn_decode scratch"),
                                )?
                            };
                            super::gdn_tp::forward_gdn_decode_tp(
                                ops, stream, device, cfg,
                                attn_norm, attn_qkv, attn_gate,
                                ssm_alpha, ssm_beta, ssm_a, ssm_dt_bias,
                                ssm_conv1d, ssm_norm, ssm_out,
                                *layer_state, gdn,
                                slot_x_in, slot_partial,
                                world, kq_replicated,
                            )
                            .with_context(|| {
                                format!(
                                    "hybrid GDN (no-batched fallback) slot {s} stage {stage_idx} layer {il} rank {r}"
                                )
                            })?;
                        }
                    } else {
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
            }

            // **#275 cycle 7 debug** — dump partial_attn_out row 0 BEFORE AR
            // and hidden_a row 0 AFTER AR for the first layer of stage 1.
            // Compares "GDN call wrote wrong row 0" vs "AR mangled row 0
            // at n_tokens=N>1" across N=1 and N=2 dispatches.
            // Dump only layer 32 (first GDN of stage 1) on rank 0.
            let do_pre_ar_dump = std::env::var("FLAMBEAU_AR_DUMP").is_ok()
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
                    && std::env::var("FLAMBEAU_TP_SKIP_SHARED").is_err();
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
        if std::env::var("FLAMBEAU_BATCHED_DECODE_DUMP").is_ok() {
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
        // per-layer KV cache + GDN state L2 norms for slot 0 on rank 0
        // at the END OF EACH STAGE. Used to find the first layer whose
        // state differs between N=1 and N=2 dispatch (since hidden_a
        // matches at step 1 but step 2 diverges, the corruption is
        // INSIDE a layer's cache or state, invisible to the visible
        // hidden output).
        if std::env::var("FLAMBEAU_LAYER_STATE_DUMP").is_ok() {
            // SAFETY: same disjointness as the other unsafe split above.
            let sessions_ptr = sessions.as_mut_ptr();
            let session0: &mut Qwen3MoEHybridSession =
                unsafe { &mut **sessions_ptr };
            let stage_caches = &session0.stages[stage_idx].caches[0];
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
                                "[STATE-DUMP] N={n} stage={stage_idx} layer={global_il} type=fullattn current_tokens=0 (empty)"
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
                            "[STATE-DUMP] N={n} stage={stage_idx} layer={global_il} type=fullattn current_tokens={valid_tokens} k_l2={k_l2:.4} v_l2={v_l2:.4} last_k_l2={last_k_l2:.4}"
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
                            "[STATE-DUMP] N={n} stage={stage_idx} layer={global_il} type=gdn state_l2={state_l2:.4} conv_l2={conv_l2:.4}"
                        );
                    }
                    LayerCache::FullAttnQ8(_) => {
                        eprintln!(
                            "[STATE-DUMP] N={n} stage={stage_idx} layer={global_il} type=fullattn_q8 (skipped)"
                        );
                    }
                }
            }
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

// ---------------------------------------------------------------------------
// **#290 PP-pipelined batched-decode** — `forward_decode_pipelined_hybrid`
// ---------------------------------------------------------------------------
//
// Same shape as `forward_decode_batched_hybrid` but pipelines stages: at
// PP=2, slot k+1's stage_0 work runs concurrently with slot k's stage_1
// work on disjoint physical GPUs (sub_clusters). Wall-clock ceiling at
// PP=2 / N=4 is 1.6× over `forward_decode_batched_hybrid` (see
// `doc/V1.x/pipelined_decode.md` for the speedup table + design).
//
// Two-phase enqueue (PP=2 only):
// 1. Phase A: for each slot, embed + all stage_0 layers at n_tokens=1
//    + AR-residuals + async peer_copy fan-out to every stage_1 rank
//    (using `bridge_events[slot_idx]` as the cross-stream wait). All
//    issued on stage_0's sub_cluster default streams.
// 2. Phase B: for each slot, all stage_1 layers at n_tokens=1 +
//    AR-residuals + output_head + DtoH of logits. The HtoD into stage_1
//    hidden_a was already queued by Phase A's `peer_copy_via_host_async`,
//    so subsequent ops on stage_1 default streams stack naturally.
//
// PP > 2 / PP=1 are NOT covered by this function — caller falls back to
// `forward_decode_batched_hybrid`.

/// Run all layers of `stage_idx` for a single slot at n_tokens=1 (row 0
/// of stage's per-rank scratch). Includes attn-AR + ffn_norm + FFN/MoE +
/// ffn-AR. Pre-condition: row 0 of every rank's `hidden_a` holds the
/// slot's input hidden vector. Post-condition: row 0 of every rank's
/// `hidden_a` holds the post-stage hidden (replicated across ranks).
#[allow(clippy::too_many_arguments)]
fn pipelined_run_slot_through_stage(
    model: &Qwen3MoEHybridModel,
    cfg: &Qwen3MoEConfig,
    sessions: &mut [&mut Qwen3MoEHybridSession],
    scratch: &mut ShardedForwardPrefillScratchHybrid,
    stage_ars: &[BarP2pAllReduce],
    stage_idx: usize,
    slot_idx: usize,
    slot_pos: usize,
    world: u32,
) -> Result<()> {
    use crate::session::LayerCache;
    use flambeau_backend_hip::HipDevice;
    use flambeau_ops::hip::norm::rmsnorm_f16;

    let stage = &model.stages[stage_idx];
    let stage_scratch = &mut scratch.per_stage[stage_idx];
    let stage_ar = &stage_ars[stage_idx];
    let stage_model = &stage.tp_model;
    let sub_cluster = &stage.sub_cluster;
    let il_cache_offset = stage.layer_range.start;
    let kv_replicated = stage_model.tp.kv_replicated();
    let kq_replicated = stage_model.tp.gdn_kq_replicated();
    let hidden = cfg.hidden_size;
    let elem_count_l = hidden as u32; // n_tokens=1 per slot
    let pos_slice = std::slice::from_ref(&slot_pos);

    for il in stage.layer_range.clone() {
        let il_local = il - il_cache_offset;
        let is_full_attn = !cfg.is_recurrent(il);

        // Per-rank attn or GDN at n_tokens=1.
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
                let session = &mut *sessions[slot_idx];
                let kv = match &mut session.stages[stage_idx].caches[r][il_local] {
                    LayerCache::FullAttn(kv) => kv,
                    _ => bail!(
                        "pipelined hybrid: slot {slot_idx} stage {stage_idx} rank {r} \
                         layer {il} expected FullAttn cache"
                    ),
                };
                let mut slot_kvs: Vec<
                    &mut flambeau_runtime::KvCache<flambeau_runtime::F16Contig, HipDevice>,
                > = vec![kv];
                super::attn_tp::forward_full_attn_layer_decode_batched_tp(
                    ops, stream, device, cfg, attn_norm, attn_q, attn_k, attn_v,
                    attn_output, attn_q_norm, attn_k_norm, &mut slot_kvs, full,
                    hidden_a, partial_attn_out, pos_slice, world, kv_replicated,
                )
                .with_context(|| {
                    format!(
                        "pipelined full-attn slot {slot_idx} stage {stage_idx} layer {il} rank {r}"
                    )
                })?;
            } else {
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
                let session = &mut *sessions[slot_idx];
                let layer_state = match &mut session.stages[stage_idx].caches[r][il_local] {
                    LayerCache::Gdn(state) => state,
                    _ => bail!(
                        "pipelined hybrid: slot {slot_idx} stage {stage_idx} rank {r} \
                         layer {il} expected Gdn cache"
                    ),
                };
                let gdn = stage_scratch.per_rank[r]
                    .gdn_decode
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing gdn_decode scratch"))?;
                super::gdn_tp::forward_gdn_decode_tp(
                    ops, stream, device, cfg, attn_norm, attn_qkv, attn_gate,
                    ssm_alpha, ssm_beta, ssm_a, ssm_dt_bias, ssm_conv1d, ssm_norm,
                    ssm_out, layer_state, gdn, hidden_a, partial_attn_out,
                    world, kq_replicated,
                )
                .with_context(|| {
                    format!(
                        "pipelined GDN slot {slot_idx} stage {stage_idx} layer {il} rank {r}"
                    )
                })?;
            }
        }

        // AR-residual on attn output.
        super::tp::ar_residual_prefill_pub(
            stage_ar, stage_scratch, sub_cluster, world, elem_count_l,
            super::tp::AttnOrFfnPub::Attn,
        )?;

        // ffn_norm + per-rank FFN/MoE at n_tokens=1.
        for r in 0..sub_cluster.ranks() {
            let device = sub_cluster.device(r);
            device.bind()?;
            let stream = device.default_stream();
            let layer_tensors = &stage_model.shards[r].layers[il];
            let ffn_norm = find_tensor_in_layer(layer_tensors, il, "ffn_norm.weight")
                .or_else(|_| find_tensor_in_layer(layer_tensors, il, "post_attention_norm.weight"))?;
            let hidden_a = stage_scratch.per_rank[r].hidden_a;
            let layer_scratch = stage_scratch.per_rank[r]
                .layer
                .as_mut()
                .ok_or_else(|| anyhow!("rank {r}: missing LayerPrefillScratch"))?;
            let mid_norm = layer_scratch.mid_norm_f16;
            let ops = &stage_model.ops[r];
            rmsnorm_f16(ops, stream, hidden_a, ffn_norm.ptr, mid_norm, 1, hidden, cfg.rms_norm_eps)
                .with_context(|| format!("pipelined ffn_norm stage {stage_idx} layer {il}"))?;
        }

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
                    mid_norm, partial_ffn_out, 1, world,
                )
                .with_context(|| format!("pipelined dense ffn stage {stage_idx} layer {il}"))?;
            }
        } else {
            let has_shared = cfg.shared_expert_intermediate_size.is_some()
                && std::env::var("FLAMBEAU_TP_SKIP_SHARED").is_err();
            for r in 0..sub_cluster.ranks() {
                let device = sub_cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let layer_tensors = &stage_model.shards[r].layers[il];
                let ffn_gate_inp = find_tensor_in_layer(layer_tensors, il, "ffn_gate_inp.weight")?;
                let ffn_gate_exps = find_tensor_in_layer(layer_tensors, il, "ffn_gate_exps.weight")?;
                let ffn_up_exps = find_tensor_in_layer(layer_tensors, il, "ffn_up_exps.weight")?;
                let ffn_down_exps = find_tensor_in_layer(layer_tensors, il, "ffn_down_exps.weight")?;
                let partial_ffn_out = stage_scratch.per_rank[r].partial_ffn_out;
                let shared_delta_f16 = stage_scratch.per_rank[r]
                    .layer
                    .as_ref()
                    .map(|l| l.shared_delta_f16)
                    .unwrap_or(flambeau_core::DevicePtr(0));
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
                    super::moe::forward_router_prefill(ops, stream, cfg, ffn_gate_inp, moe_scratch, mid_norm, 1)
                        .with_context(|| format!("pipelined router stage {stage_idx} layer {il}"))?;
                }
                if has_shared {
                    let shared_w_gate = find_tensor_in_layer(layer_tensors, il, "ffn_gate_shexp.weight")?;
                    let shared_w_up = find_tensor_in_layer(layer_tensors, il, "ffn_up_shexp.weight")?;
                    let shared_w_down = find_tensor_in_layer(layer_tensors, il, "ffn_down_shexp.weight")?;
                    let shared_w_gate_inp =
                        find_tensor_in_layer(layer_tensors, il, "ffn_gate_inp_shexp.weight").ok();
                    let shared_scratch = layer_scratch
                        .shared
                        .as_mut()
                        .ok_or_else(|| anyhow!("rank {r}: missing SharedExpertPrefillScratch"))?;
                    super::moe_tp::forward_shared_expert_prefill_tp(
                        ops, stream, cfg, shared_w_gate, shared_w_up, shared_w_down,
                        shared_w_gate_inp, shared_scratch, mid_norm, shared_delta_f16,
                        1, ffn_world,
                    )
                    .with_context(|| format!("pipelined shared expert stage {stage_idx} layer {il}"))?;
                }
                let moe_scratch = layer_scratch
                    .moe
                    .as_mut()
                    .ok_or_else(|| anyhow!("rank {r}: missing MoePrefillScratch"))?;
                super::moe_tp::forward_moe_ffn_prefill_tp(
                    ops, stream, cfg, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
                    moe_scratch, mid_norm, partial_ffn_out, 1, ffn_world,
                )
                .with_context(|| format!("pipelined moe ffn stage {stage_idx} layer {il}"))?;
                if has_shared {
                    flambeau_ops::hip::mlp::add_f16(
                        ops, stream, partial_ffn_out, shared_delta_f16,
                        partial_ffn_out, hidden,
                    )
                    .with_context(|| format!("pipelined + shared add stage {stage_idx} layer {il}"))?;
                }
            }
        }

        // AR-residual on FFN output (or replicated add).
        if ffn_world > 1 {
            super::tp::ar_residual_prefill_pub(
                stage_ar, stage_scratch, sub_cluster, world, elem_count_l,
                super::tp::AttnOrFfnPub::Ffn,
            )?;
        } else {
            for r in 0..sub_cluster.ranks() {
                let device = sub_cluster.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let ops = &stage_model.ops[r];
                flambeau_ops::hip::mlp::add_f16(
                    ops, stream, stage_scratch.per_rank[r].hidden_a,
                    stage_scratch.per_rank[r].partial_ffn_out,
                    stage_scratch.per_rank[r].hidden_a, hidden,
                )
                .with_context(|| format!("pipelined replicated ffn add stage {stage_idx} layer {il}"))?;
            }
        }
    }
    Ok(())
}

/// **#290** PP-pipelined batched-decode for hybrid PP+TP topologies at PP=2.
///
/// Pre-condition: caller drove a successful `prefill_pp_blocking` /
/// `prefill_tp_blocking` for every slot's session. KV caches + GDN
/// states are at `slot.position`. Pipelined decode appends one token
/// to each slot at its position and writes per-slot logits to
/// `logits_out[slot.idx]`.
///
/// Pipelining produces ~1.6× speedup over `forward_decode_batched_hybrid`
/// at PP=2 / N=4 (see `doc/V1.x/pipelined_decode.md`). For PP=1 or N=1
/// the caller should use `forward_decode_batched_hybrid` directly —
/// pipelining has nothing to interleave there.
///
/// **STATUS (2026-05-04 live test on Qwen3.6-27B / pp2tp2)**: function
/// compiles + dispatches but produces output that DIFFERS from
/// `forward_decode_batched_hybrid` at N=2 (which is bit-identical to
/// N=1). Output is coherent but the token sequence diverges at the
/// first decode token.
///
/// Bisect findings:
/// - With `FLAMBEAU_PIPELINE_BLOCKING_COPY=1` (use blocking
///   peer_copy_via_host instead of async): SAME divergent output → bug
///   is NOT in async event coordination.
/// - Wall time at N=2: 6.82s vs batched 6.04s = 0.85× (regression).
///
/// Most likely culprits (not yet bisected):
/// - `forward_gdn_decode_tp` (single-slot path, what we call) vs
///   `forward_gdn_decode_batched_tp` at N=1 (what batched calls): may
///   produce numerically different row-0 outputs even at the same
///   layer state. The cert (`certs/perf/p29b_i2_F_throughput/...`)
///   notes the no-batched-GDN fallback differs slightly from the
///   batched-GDN path; pipelined uses the no-batched-GDN equivalent.
/// - Per-slot scratch reuse pattern across slot iterations (gdn_decode
///   workspace) — should overwrite per call but may have a subtle
///   stale-state path.
/// - The single-token `forward_dense_ffn_prefill_tp(n=1)` /
///   `forward_moe_ffn_prefill_tp(n=1)` may differ from row-0 of
///   batched-N=2 — though row-0 should be independent of batching.
///
/// The function is gated by `FLAMBEAU_DECODE_PIPELINE=1` in the
/// scheduler — default OFF, so this divergence does not affect the
/// production path. FIXME #292: bisect by replacing
/// `forward_gdn_decode_tp` with `forward_gdn_decode_batched_tp` at N=1
/// inside `pipelined_run_slot_through_stage`.
#[allow(clippy::too_many_arguments)]
pub fn forward_decode_pipelined_hybrid(
    model: &Qwen3MoEHybridModel,
    sessions: &mut [&mut Qwen3MoEHybridSession],
    global_cluster: &flambeau_backend_hip::HipCluster,
    stage_ars: &[BarP2pAllReduce],
    scratch: &mut ShardedForwardPrefillScratchHybrid,
    slots: &[super::batched::BatchSlot],
    logits_out: &mut [&mut Vec<f32>],
) -> Result<()> {
    // ── Validation ────────────────────────────────────────────────
    let n = slots.len();
    if n == 0 {
        bail!("forward_decode_pipelined_hybrid: empty slot list");
    }
    if sessions.len() != logits_out.len() {
        bail!(
            "forward_decode_pipelined_hybrid: sessions({}) != logits_out({})",
            sessions.len(),
            logits_out.len(),
        );
    }
    for s in slots {
        if s.idx >= sessions.len() {
            bail!(
                "forward_decode_pipelined_hybrid: BatchSlot.idx {} OOB (n={})",
                s.idx,
                sessions.len()
            );
        }
    }

    let cfg = &model.config;
    let n_stages = model.stages.len();
    let tp_size = model.spec.tp_size as usize;
    if n_stages != 2 {
        bail!(
            "forward_decode_pipelined_hybrid: only PP=2 supported (n_stages={n_stages}); \
             use forward_decode_batched_hybrid for PP=1 or PP>2"
        );
    }
    if stage_ars.len() != n_stages {
        bail!("stage_ars.len()={} != n_stages={n_stages}", stage_ars.len());
    }
    let world = tp_size as u32;
    if world != 1 && world != 2 && world != 4 {
        bail!("hybrid pipelined-decode: per-stage tp_size ∈ {{1, 2, 4}} (got {world})");
    }
    let hidden = cfg.hidden_size;
    let bridge_bytes = hidden * 2; // single slot row, F16
    let head_stage_idx = scratch.head_stage as usize;
    if head_stage_idx + 1 != n_stages {
        bail!(
            "hybrid pipelined-decode: head_stage={head_stage_idx} but expected last = {}",
            n_stages - 1
        );
    }

    // ── #275 entry-time stream drain (every stage's sub_cluster) ──
    for stage in &model.stages {
        for r in 0..stage.sub_cluster.ranks() {
            let device = stage.sub_cluster.device(r);
            device.bind()?;
            Stream::synchronize(device.default_stream())?;
        }
    }

    // ── Lazy bridge-event init on stage_0 rank 0 ──────────────────
    // One event per slot. Bridge events live on the SOURCE device
    // (stage_0 rank 0) — `stream_wait` works cross-device, so any
    // stage_1 rank's stream can wait on the same event handle.
    {
        let stage0_dev0 = model.stages[0].sub_cluster.device(0);
        stage0_dev0.bind()?;
        let st0r0 = &mut scratch.per_stage[0].per_rank[0];
        for _ in st0r0.pipeline_bridge_events.len()..n {
            st0r0
                .pipeline_bridge_events
                .push(flambeau_backend_hip::HipEvent::new(stage0_dev0.id())?);
        }
    }

    // ── Reserve per-lane bounce buffers for the TP fan-out ────────
    // Stage 0 rank 0 fans out to `tp_size` stage_1 ranks per slot.
    // Without per-lane bounces, the shared-bounce reuse races across
    // dst ranks (lane 0's HtoD reads the bounce while lane 1's DtoH
    // overwrites it). Per-lane bounces give each peer-copy call its
    // own pinned slab. Idempotent across decode steps.
    let global_src_rank0 = 0usize; // stage 0, local rank 0
    global_cluster.reserve_lane_bounces(tp_size.max(1), bridge_bytes)?;

    // Per-slot positions (same value across stages).
    let slot_positions: Vec<usize> = slots.iter().map(|s| s.position).collect();

    // ════════════════════════════════════════════════════════════════
    // INTERLEAVED PER-SLOT LOOP — for each slot k:
    //   1. embed + stage_0 layers (queues on stage_0 streams)
    //   2. async peer_copy fan-out (DtoH on stage_0, HtoD on stage_1
    //      with bridge[k])
    //   3. stage_1 layers + AR (queues on stage_1 streams, implicitly
    //      after the HtoD which is already on the same default stream)
    //   4. output_head + ASYNC DtoH of logits (no per-slot sync)
    //
    // Pipelining: at iteration k, while host enqueues slot k's stage_0
    // work, the GPU on stage_1 is still running slot k-1's stage_1
    // layers + head. Stage_0 and stage_1 GPU subsystems are disjoint
    // (different physical devices), so they execute concurrently.
    //
    // Why interleave instead of two-phase: with shared row-0 hidden_a,
    // a "Phase A: all peer_copies, Phase B: all stage_1 work" split
    // would queue all N HtoDs serially on stage_1 streams BEFORE any
    // stage_1 layer kernel — the last HtoD wins and slot 0's stage_1
    // input is corrupted. Interleaving puts each slot's HtoD before
    // ITS OWN stage_1 work (not after slot k+1's HtoD).
    // ════════════════════════════════════════════════════════════════
    let stage0 = &model.stages[0];
    if !stage0.tp_model.has_token_embd {
        bail!("hybrid stage 0 missing token_embd");
    }
    if !model.stages[1].tp_model.has_output_head {
        bail!("hybrid last stage missing output head");
    }
    let head_rank = scratch.per_stage[1].head_rank.0 as usize;

    // To avoid blocking the host on every slot's logits DtoH, we keep
    // track of which slots need their final stream sync at the end and
    // do them all in one shot. The DtoH itself happens on the head
    // rank's default stream inside the loop; we just defer the sync.
    let head_dev_id = model.stages[1].sub_cluster.device(head_rank).id();

    for slot_idx in 0..n {
        let slot = &slots[slot_idx];
        let slot_pos = slot_positions[slot_idx];

        // 1. Embed slot's token into ROW 0 of every stage_0 rank's hidden_a.
        {
            let stage0_sub = &stage0.sub_cluster;
            let stage0_model = &stage0.tp_model;
            for r in 0..stage0_sub.ranks() {
                let device = stage0_sub.device(r);
                device.bind()?;
                let stream = device.default_stream();
                let dst_base = scratch.per_stage[0].per_rank[r].hidden_a;
                super::io::forward_embed_decode_host(
                    device, stream,
                    &stage0_model.shards[r].token_embd,
                    slot.token_id,
                    dst_base, // row 0
                    hidden,
                )
                .with_context(|| format!("pipelined embed slot {slot_idx} rank {r}"))?;
            }
        }

        // 2. Run all stage_0 layers + AR-residuals + FFN at n_tokens=1.
        pipelined_run_slot_through_stage(
            model, cfg, sessions, scratch, stage_ars,
            /* stage_idx = */ 0, slot_idx, slot_pos, world,
        )?;

        // 3. Async peer_copy fan-out: stage_0 rank 0 → every stage_1 rank.
        //    DEBUG MODE (`FLAMBEAU_PIPELINE_BLOCKING_COPY=1`): use the
        //    BLOCKING `peer_copy_via_host` instead of the async variant.
        //    If this fixes output divergence vs `forward_decode_batched_hybrid`,
        //    the bug is in the async event/stream coordination, not the
        //    per-slot loop structure itself.
        let use_blocking_copy =
            std::env::var("FLAMBEAU_PIPELINE_BLOCKING_COPY").is_ok();
        if use_blocking_copy {
            let src_dev = stage0.sub_cluster.device(0);
            src_dev.bind()?;
            // Drain stage_0 sub_cluster stream so DtoH sees all stage_0
            // kernel writes for this slot.
            Stream::synchronize(src_dev.default_stream())?;
            let src_ptr = scratch.per_stage[0].per_rank[0].hidden_a;
            for dst_local in 0..tp_size {
                let dst_global_rank = tp_size + dst_local;
                let dst_ptr = scratch.per_stage[1].per_rank[dst_local].hidden_a;
                unsafe {
                    global_cluster
                        .peer_copy_via_host(
                            dst_ptr,
                            dst_global_rank,
                            src_ptr,
                            global_src_rank0,
                            bridge_bytes,
                        )
                        .with_context(|| {
                            format!("pipelined blocking peer_copy slot {slot_idx} dst {dst_local}")
                        })?;
                }
            }
        } else {
        //    Single bridge event per slot; per-lane bounce keeps the
        //    fan-out non-racing.
        //
        //    **Stream selection** (`feedback_hipcluster_stream_handles`):
        //    we pass the SUB_CLUSTER default streams as src_stream /
        //    dst_stream — NOT the global_cluster streams (which point
        //    to different driver handles for the same physical device).
        //    Using sub_cluster streams ensures the DtoH stacks behind
        //    the previous slot's stage_0 last kernel (also on
        //    sub_cluster stream) and the HtoD stacks ahead of this
        //    slot's stage_1 first kernel (also on sub_cluster stream).
        //    `peer_copy_via_host_async_laned` doesn't care which stream
        //    handle it gets — bounce buffer + bridge event work
        //    cross-stream.
        {
            let src_dev = stage0.sub_cluster.device(0);
            src_dev.bind()?;
            let src_stream = src_dev.default_stream();
            let src_ptr = scratch.per_stage[0].per_rank[0].hidden_a; // row 0
            // SAFETY: pipeline_bridge_events[slot_idx] was lazily
            // allocated with len ≥ n. Raw pointer escapes the borrow
            // checker since per_stage[0] (event) and per_stage[1]
            // (dst_ptr) are disjoint Vec elements.
            let bridge_event_ptr: *const flambeau_backend_hip::HipEvent =
                &scratch.per_stage[0].per_rank[0].pipeline_bridge_events[slot_idx];

            for dst_local in 0..tp_size {
                let dst_global_rank = tp_size + dst_local; // stage 1
                let dst_sub_dev = model.stages[1].sub_cluster.device(dst_local);
                let dst_stream = dst_sub_dev.default_stream();
                let dst_ptr = scratch.per_stage[1].per_rank[dst_local].hidden_a; // row 0
                // SAFETY:
                // - src_ptr/dst_ptr point to ≥ bridge_bytes valid F16 on
                //   their respective devices.
                // - src_stream / dst_stream are sub_cluster default
                //   streams: stage_0 layer kernels also queue on
                //   src_stream, stage_1 layer kernels queue on
                //   dst_stream — both serialise correctly without
                //   needing a cross-handle drain.
                // - bridge_event lives on src_dev's id; HIP allows
                //   stream_wait(event) on any device.
                // - Per-lane bounce (lane=dst_local) prevents the
                //   inter-call bounce race the blocking variant
                //   avoids by host-syncing.
                unsafe {
                    let bridge_event: &flambeau_backend_hip::HipEvent = &*bridge_event_ptr;
                    global_cluster
                        .peer_copy_via_host_async_laned(
                            dst_ptr,
                            dst_global_rank,
                            src_ptr,
                            global_src_rank0,
                            bridge_bytes,
                            src_stream,
                            dst_stream,
                            bridge_event,
                            None,
                            Some(dst_local),
                        )
                        .with_context(|| {
                            format!(
                                "pipelined peer_copy_async slot {slot_idx} dst_local {dst_local}"
                            )
                        })?;
                }
            }
        }
        } // end else branch (async peer_copy)

        // 4. Run all stage_1 layers + AR-residuals + FFN at n_tokens=1.
        //    The HtoD on each stage_1 rank's default stream is already
        //    queued (with stream_wait on bridge_event); the layer
        //    kernels stack behind it implicitly via stream order.
        pipelined_run_slot_through_stage(
            model, cfg, sessions, scratch, stage_ars,
            /* stage_idx = */ 1, slot_idx, slot_pos, world,
        )?;

        // 5. Output head + ASYNC DtoH of logits. Bypass the
        //    `download_logits_host` helper because it host-syncs at
        //    end-of-call, which would serialise stage_1 across slots
        //    and defeat pipelining. Sync once at function tail.
        {
            let stage1 = &model.stages[1];
            let device = stage1.sub_cluster.device(head_rank);
            device.bind()?;
            let stream = device.default_stream();
            let head_shard = &stage1.tp_model.shards[head_rank];
            let lm_head = head_shard.output.as_ref().unwrap_or(&head_shard.token_embd);
            let head_hidden = scratch.per_stage[1].per_rank[head_rank].hidden_a; // row 0
            let head_scratch = scratch.per_stage[1].per_rank[head_rank]
                .output_head
                .as_mut()
                .ok_or_else(|| anyhow!("hybrid pipelined: head_rank missing OutputHeadScratch"))?;
            let ops = &stage1.tp_model.ops[head_rank];

            super::io::forward_output_head_decode(
                ops, stream, cfg, &head_shard.output_norm, lm_head, head_scratch, head_hidden,
            )
            .with_context(|| format!("pipelined output head slot {}", slot.idx))?;

            // Async DtoH of logits to the host buffer. The host buffer
            // is resized + zeroed by the caller (or here below).
            let out = &mut *logits_out[slot.idx];
            out.clear();
            out.resize(cfg.vocab_size, 0.0f32);
            // SAFETY: logits_f32 is valid for cfg.vocab_size F32 values
            // on `device`; out.as_mut_ptr() is host memory of matching
            // size; both stay live until the tail sync below.
            unsafe {
                <flambeau_backend_hip::HipDevice as flambeau_core::Device>::memcpy_async(
                    device,
                    stream,
                    flambeau_core::CopyDirection::DeviceToHost,
                    flambeau_core::DevicePtr(out.as_mut_ptr() as usize),
                    head_scratch.logits_f32,
                    cfg.vocab_size * 4,
                )?;
            }
        }
    }

    // ── Tail sync: wait for all per-slot logits DtoHs to complete
    //    before returning. Sync the head_rank stream — all DtoHs were
    //    queued there. Also sync stage_0 default streams to ensure
    //    the last slot's peer_copy DtoH cleared the bounce, in case
    //    the next pipelined call reuses lane bounces (idempotent).
    {
        let head_dev = model.stages[1].sub_cluster.device(head_rank);
        head_dev.bind()?;
        Stream::synchronize(head_dev.default_stream())?;
        let _ = head_dev_id; // currently unused but kept for diagnostic clarity

        for r in 0..model.stages[0].sub_cluster.ranks() {
            let device = model.stages[0].sub_cluster.device(r);
            device.bind()?;
            Stream::synchronize(device.default_stream())?;
        }
    }

    Ok(())
}

