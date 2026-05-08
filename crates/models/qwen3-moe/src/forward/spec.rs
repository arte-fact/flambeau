//! Speculative-decode drivers (one K=1 macro step at a time).
//!
//! Three flavors live here:
//!
//! - [`forward_speculative_pp_step`] — greedy K=1 spec on PP topology.
//!   Strict-match verify (`accept ⇔ argmax(p_pos0) == draft`).
//! - [`forward_speculative_pp_step_sampling`] — non-greedy spec via
//!   vLLM-canonical rejection sampling (Leviathan et al. 2022). Applies
//!   the full sampling config (temperature / top-k / top-p / min-p /
//!   penalties) to BOTH base `p` and MTP `q` distributions.
//! - [`forward_speculative_tp_step`] — TP topology analog (greedy only).
//!
//! ## Macro-step shape (PP, greedy)
//!
//! Pre-state: cache populated through slot `position - 1`; caller holds
//! `last_token = T@position` (sampled by the prior step) and
//! `h_for_mtp = h@(position - 1)` (the hidden that produced last_token).
//!
//! Per macro:
//!   1. Snapshot GDN state across all ranks.
//!   2. MTP draft: `(h_for_mtp, embed(last_token)) → T_draft`, a guess
//!      at `T@(position + 1)`.
//!   3. Base verify at L=2 with `[last_token, T_draft]` starting at
//!      `position`. K/V extends to slot `position+1`. Pos1 head is
//!      deferred (Lever C); only `p_pos0` is downloaded.
//!   4. `T_actual_pos1 = argmax(p_pos0)`.
//!   5. Verify: `T_draft == T_actual_pos1`?
//!      - **Accept**: lazily run pos1 head (Lever C), commit
//!        `[T_actual_pos1, argmax(p_pos1)]`, advance position by 2.
//!      - **Reject (Lever 1)**: rollback K/V by **1** (slot=position
//!        from `last_token` is correct and stays); restore GDN snapshot;
//!        re-advance GDN by 1 step per recurrent layer using per-layer
//!        x_in snapshots saved during the L=2 verify (~7 ms parallel
//!        across ranks instead of a 50 ms full L=1 PP redo). Commit
//!        `T_actual_pos1` from p_pos0; advance position by 1.
//!
//! The key invariants of the reject path: full-attn K/V at slot=position
//! was written with `last_token` during the L=2 verify and is correct;
//! MoE/FFN outputs at L=2 position 0 are likewise correct; the only
//! piece of state that L=2 advanced incorrectly is each GDN layer's
//! recurrent state (it consumed `[last_token, draft]`; we want only
//! `last_token`). `redo_gdn_only_pp` re-advances exactly that.
//!
//! ## Performance
//!
//! Empirical numbers on Qwen3.6-27B-Q4_0 / 4× MI50 PCIe Mesh<4>:
//!   - greedy A/B at 87.5 % accept: spec 56.11 ms/tok vs baseline
//!     50.14 (+11.9 %); first 16 tokens bit-identical to baseline
//!   - sampling smoke (temp=0.8, top_p=0.9): 90.7 ms/tok at 68.8 %
//!     accept
//!
//! Spec is structurally net-negative on this rig (PCIe-3, no NVLink):
//! L=2 paired-prefill multiplier was ~1.92× (not the projected ~1.5×)
//! because peer-copy/AR sync dominates L=1 wall on PCIe. Default off;
//! opt-in via `FLAMBEAU_SPEC_MTP=path/to/mtp.gguf`.

#![cfg(feature = "hip")]

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;

use flambeau_backend_hip::{BarP2pAllReduce, HipCluster};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_runtime::sampling::{
    build_distribution, sample_from_distribution, Rng, Sampling,
};

use crate::forward::pp::{
    forward_one_token_pp_logits, forward_output_head_at_pp,
    forward_prefill_pp_logits_paired_l2,
    ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use crate::forward::tp::{
    forward_one_token_tp_logits, ShardedForwardOneTokenScratchTp,
};
use crate::mtp::{
    forward_mtp_step_with_lm_head, forward_mtp_step_with_lm_head_async,
    forward_mtp_step_with_lm_head_logits, mtp_logits_argmax_host,
    MtpForwardScratch, MtpHeadWeights,
};
use crate::sharded::{Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use crate::tp_sharded::{Qwen3MoETpModel, Qwen3MoETpSession};
use crate::weights::DeviceTensor;

/// Outcome of one speculative-decode macro step.
#[derive(Debug, Clone)]
pub struct SpecStep {
    /// 1 (reject) or 2 (accept) committed tokens, oldest first.
    pub committed: Vec<u32>,
    /// New base position = `position_before + committed.len()`.
    pub new_position: usize,
    /// Whether MTP's draft was accepted by base.
    pub accepted: bool,
    /// MTP's draft (whatever it was).
    pub draft: u32,
    /// Base's verification of slot position+1 (should equal draft on accept).
    pub verify: u32,
    /// Per-stage wall-clock in milliseconds, for telemetry.
    pub timings_ms: SpecTimings,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpecTimings {
    pub gdn_snapshot: f64,
    pub mtp_draft: f64,
    pub base_l2: f64,
    pub gdn_restore: f64,
    pub base_l1_redo: f64,
}

/// Run one K=1 spec-decode macro step. Caller is responsible for
/// holding `last_token` (the most-recently-committed token, which
/// will be processed at slot `position`) and `h_for_mtp_dev`
/// (post-base-norm hidden of the step that produced `last_token`,
/// living on the **last rank** in `cluster`).
///
/// On the **first** macro step of a session, `h_for_mtp_dev` should
/// come from a preceding `forward_one_token_pp_logits` call with the
/// last prompt token — that primes the MTP input correctly.
///
/// Returns the committed tokens, the new position, and a fresh
/// `h_for_mtp_dev_next` pointer (= the device buffer holding the
/// hidden that produced the LAST committed token). The buffer
/// belongs to one of the prefill/decode scratches and is valid
/// until the next forward call that overwrites it.
#[allow(clippy::too_many_arguments)]
pub fn forward_speculative_pp_step(
    model: &Qwen3MoEShardedModel,
    session: &mut Qwen3MoEShardedSession,
    cluster: &HipCluster,
    decode_scratch: &mut ShardedForwardOneTokenScratch,
    prefill_scratch: &mut ShardedForwardPrefillScratch,
    mtp: &MtpHeadWeights,
    mtp_scratch: &MtpForwardScratch,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    last_token: u32,
    h_for_mtp_dev: DevicePtr,
    position: usize,
) -> Result<(SpecStep, DevicePtr)> {
    let cfg = &model.config;
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    let last_rank = cluster.ranks() - 1;
    let last_device = cluster.device(last_rank);
    let last_ops = &model.shards[last_rank].ops;

    let mut t = SpecTimings::default();
    let now = || std::time::Instant::now();

    // ── 1+2. (Lever B) Issue MTP draft on the last rank's aux stream
    // CONCURRENTLY with `save_gdn_snapshot` issued on per-rank default
    // streams. MTP draft does not depend on GDN state (its only inputs
    // are `h_for_mtp_dev` from the previous macro and `embed(last_token)`),
    // so the work can proceed in parallel on the head device's two
    // streams. This eliminates the ~1.5 ms MTP draft from the critical
    // path — it now hides under the ~2 ms GDN snap wall.
    cluster
        .reserve_aux_streams(1)
        .context("spec: reserve aux streams for Lever B")?;

    // 2a. Embed `last_token` on rank 0 + host bounce to last_device.
    // The bounce sync is unavoidable (rank 0 produces the F16 row,
    // last_device's aux stream consumes it), but it's <0.1 ms.
    let t0 = now();
    let rank0_device = cluster.device(0);
    let token_embd = model.shards[0]
        .token_embd
        .as_ref()
        .ok_or_else(|| anyhow!("rank 0 missing token_embd"))?;
    let mut row_host = vec![half::f16::from_f32(0.0); hidden];
    {
        let rank0_buf = rank0_device.alloc(row_bytes)?;
        rank0_device.bind()?;
        crate::forward::forward_embed_decode_host(
            rank0_device,
            rank0_device.default_stream(),
            token_embd,
            last_token,
            rank0_buf,
            hidden,
        )?;
        // SAFETY: rank0_buf has hidden*2 bytes; row_host has hidden*2 bytes.
        unsafe {
            rank0_device.memcpy_async(
                rank0_device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(row_host.as_mut_ptr() as usize),
                rank0_buf,
                row_bytes,
            )?;
        }
        rank0_device.default_stream().synchronize()?;
        // SAFETY: rank0_buf was just allocated, no aliases.
        unsafe { rank0_device.dealloc(rank0_buf, row_bytes)?; }
    }

    // 2b. Issue MTP-async on the last rank's aux stream. Subsequent
    // host code returns once kernels are queued; logits live in
    // `mtp_scratch.logits_f32` until step 5 syncs and downloads.
    last_device.bind()?;
    let e_token_dev = last_device.alloc(row_bytes)?;
    cluster
        .with_aux_stream(last_rank, 0, |aux| -> flambeau_core::DeviceResult<()> {
            // SAFETY: e_token_dev is fresh; row_host lives through to step 5.
            unsafe {
                last_device.memcpy_async(
                    aux,
                    CopyDirection::HostToDevice,
                    e_token_dev,
                    DevicePtr(row_host.as_ptr() as usize),
                    row_bytes,
                )?;
            }
            forward_mtp_step_with_lm_head_async(
                last_ops, aux, last_device, cfg, mtp, mtp_scratch,
                output_norm_weight, lm_head_weight,
                h_for_mtp_dev, e_token_dev, position + 1, None,
            )
            .map_err(|e| flambeau_core::DeviceError::Backend {
                backend: "hip", code: -1, message: format!("mtp async: {e}"),
            })?;
            Ok(())
        })
        .context("spec: MTP-async on aux stream")?;

    // ── 3. Snapshot GDN on per-rank default streams. Runs concurrently
    // with the MTP draft on the head rank's aux stream.
    let t_snap_start = now();
    session
        .save_gdn_snapshot(cluster)
        .context("spec: save_gdn_snapshot")?;
    t.gdn_snapshot = t_snap_start.elapsed().as_secs_f64() * 1000.0;

    // ── 4. Sync aux stream + download MTP logits + host argmax.
    let mtp_draft = cluster
        .with_aux_stream(last_rank, 0, |aux| -> flambeau_core::DeviceResult<u32> {
            mtp_logits_argmax_host(last_device, aux, mtp_scratch, cfg.vocab_size)
                .map_err(|e| flambeau_core::DeviceError::Backend {
                    backend: "hip", code: -1, message: format!("mtp argmax: {e}"),
                })
        })
        .context("spec: MTP argmax download")?;
    // SAFETY: e_token_dev was just allocated, unique reference here.
    unsafe { last_device.dealloc(e_token_dev, row_bytes)?; }
    t.mtp_draft = t0.elapsed().as_secs_f64() * 1000.0 - t.gdn_snapshot;

    // ── 3. Base verify: batched L=2 paired-logits over
    //    [last_token, mtp_draft] at start_position = position.
    //    MTP-5h-Lever-C — defer pos1 head computation until accept;
    //    saves 3 ms × 12.5 % rejects = 0.4 ms/macro avg.
    let t0 = now();
    let mut step1_logits = Vec::with_capacity(vocab);
    let verify_tokens = [last_token, mtp_draft];
    forward_prefill_pp_logits_paired_l2(
        model,
        session,
        cluster,
        prefill_scratch,
        &verify_tokens,
        position,
        &mut step1_logits,
        None,
    )
    .context("spec: paired-L2 base verify (lazy pos1)")?;
    t.base_l2 = t0.elapsed().as_secs_f64() * 1000.0;

    if step1_logits.len() != vocab {
        return Err(anyhow!(
            "spec: expected {vocab} pos0 logits, got {}",
            step1_logits.len()
        ));
    }

    let argmax = |slice: &[f32]| -> u32 {
        let mut best_i = 0usize;
        let mut best_v = slice[0];
        for (i, &v) in slice.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i as u32
    };
    let t_actual_pos1 = argmax(&step1_logits);
    let accepted = mtp_draft == t_actual_pos1;

    if accepted {
        // MTP-5h-Lever-C — pos1 head deferred; run it now to get the
        // second committed token. Cost (~3 ms) only paid on accept paths.
        let mut step2_logits = Vec::with_capacity(vocab);
        forward_output_head_at_pp(
            model, cluster, prefill_scratch, /*position_in_l2=*/ 1, &mut step2_logits,
        )
        .context("spec: output head at L=2 pos 1 (accept)")?;
        let t_actual_pos2 = argmax(&step2_logits);

        // Cache extends to slot position+1 (correctly). GDN advanced
        // 2 steps with [last_token, draft]; both inputs were correct
        // on this path. State is good. h_for_next = L=2 batch's pos-1
        // hidden (= h@(position+1), which produced t_actual_pos2).
        let h_for_next = prefill_scratch.per_rank[last_rank]
            .hidden_a
            .offset_bytes(row_bytes);
        let mut committed = Vec::with_capacity(2);
        committed.push(t_actual_pos1);
        committed.push(t_actual_pos2);
        Ok((
            SpecStep {
                committed,
                new_position: position + 2,
                accepted: true,
                draft: mtp_draft,
                verify: t_actual_pos1,
                timings_ms: t,
            },
            h_for_next,
        ))
    } else {
        // ── Reject (MTP-5h-1): rollback ONLY the wrong-input K/V slot
        // (slot=position+1, written from `mtp_draft`). Slot=position was
        // written from `last_token` during L=2 verify and is correct,
        // so it stays. Restore GDN to "after position-1", then re-run
        // GDN-only forward per-rank-parallel using the per-layer x_in
        // snapshots saved during L=2 verify. Skips the full L=1 redo
        // (~50 ms → ~7 ms): full-attn K/V at slot=position and the
        // MoE/FFN outputs at L=2 position 0 are already correct, so
        // the only state that needs updating is each recurrent layer's
        // GDN state (advance from snapshot by 1 step with last_token).
        let t0 = now();
        session
            .rollback_full_attn(1)
            .context("spec: rollback_full_attn(1) on reject")?;
        session
            .restore_gdn_snapshot(cluster)
            .context("spec: restore_gdn_snapshot on reject")?;
        t.gdn_restore = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = now();
        session
            .redo_gdn_only_pp(model, cluster, decode_scratch, prefill_scratch)
            .context("spec: redo_gdn_only_pp on reject")?;
        t.base_l1_redo = t0.elapsed().as_secs_f64() * 1000.0;

        // Committed token = argmax of the verify's p_pos0 logits.
        // Same input + same K/V state as a full L=1 redo would see, so
        // identical argmax (under greedy verify).
        let t_redo = t_actual_pos1;

        // h_for_next = h@(position) post-all-layers from the L=2
        // verify, which lives in prefill_scratch.hidden_a[0..hidden].
        let h_for_next = prefill_scratch.per_rank[last_rank].hidden_a;
        let _ = decode_scratch; // unused on this reject path
        let mut committed = Vec::with_capacity(1);
        committed.push(t_redo);
        Ok((
            SpecStep {
                committed,
                new_position: position + 1,
                accepted: false,
                draft: mtp_draft,
                verify: t_redo,
                timings_ms: t,
            },
            h_for_next,
        ))
    }
}

// ===================================================================
// MTP-5g — rejection-sampling spec-decode (non-greedy).
// ===================================================================

fn prob_of(dist: &[(u32, f32)], token: u32) -> f32 {
    dist.iter()
        .find(|&&(id, _)| id == token)
        .map(|&(_, p)| p)
        .unwrap_or(0.0)
}

/// Sample from the residual distribution `max(0, p(x) - q(x))`,
/// normalized over `p`'s support. Used on draft-rejection in vLLM-
/// canonical K=1 rejection sampling.
fn sample_residual(p: &[(u32, f32)], q: &[(u32, f32)], rng: &mut Rng) -> u32 {
    let q_map: HashMap<u32, f32> = q.iter().copied().collect();
    let mut residual: Vec<(u32, f32)> = Vec::with_capacity(p.len());
    let mut sum = 0.0f32;
    for &(id, pv) in p {
        let qv = q_map.get(&id).copied().unwrap_or(0.0);
        let r = (pv - qv).max(0.0);
        if r > 0.0 {
            residual.push((id, r));
            sum += r;
        }
    }
    if sum <= 0.0 {
        // Degenerate — q dominates p everywhere on p's support. Fall
        // back to sampling from p directly (rare, safe).
        return sample_from_distribution(p, rng);
    }
    for (_, r) in residual.iter_mut() {
        *r /= sum;
    }
    sample_from_distribution(&residual, rng)
}

/// MTP-5g — rejection-sampling K=1 spec-decode macro step.
///
/// Same shape as [`forward_speculative_pp_step`], but instead of
/// strict-greedy verify (`accept iff t_d == argmax(p)`), implements
/// vLLM-canonical rejection sampling:
///
/// - Draft: `t_d ~ q(·)` where `q` is MTP's filtered/temperatured
///   distribution under `sampling`.
/// - Verify: compute `p(·)` (base) under the same `sampling`.
/// - Accept iff `u < min(1, p(t_d) / q(t_d))` for `u ~ U[0,1)`.
/// - On accept: commit `t_d` and a second token sampled from
///   `p_pos1(·)` (base at position+1).
/// - On reject: rollback, restore GDN, redo L=1 with `last_token`,
///   commit a token sampled from the **residual**
///   `r(x) = max(0, p_pos0(x) - q(x))` (normalized).
///
/// When `sampling.is_greedy()` is true this collapses to the same
/// behaviour as the strict-greedy path (q is one-hot at draft, p is
/// one-hot at argmax, accept iff they coincide).
///
/// Penalties on `sampling` are NOT applied here — caller should
/// either disable penalties or pass logits with penalties already
/// baked in. Filed as follow-up; the production call site (greedy
/// only today) doesn't depend on penalty-aware spec.
#[allow(clippy::too_many_arguments)]
pub fn forward_speculative_pp_step_sampling(
    model: &Qwen3MoEShardedModel,
    session: &mut Qwen3MoEShardedSession,
    cluster: &HipCluster,
    decode_scratch: &mut ShardedForwardOneTokenScratch,
    prefill_scratch: &mut ShardedForwardPrefillScratch,
    mtp: &MtpHeadWeights,
    mtp_scratch: &MtpForwardScratch,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    last_token: u32,
    h_for_mtp_dev: DevicePtr,
    position: usize,
    sampling: &Sampling,
    rng: &mut Rng,
    // MTP-5g/h — caller's per-turn generated-token slice for
    // penalty application. Pass `&[]` when penalties are inactive.
    history: &[u32],
) -> Result<(SpecStep, DevicePtr)> {
    let cfg = &model.config;
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    let last_rank = cluster.ranks() - 1;
    let last_device = cluster.device(last_rank);
    let last_ops = &model.shards[last_rank].ops;

    let mut t = SpecTimings::default();
    let now = || std::time::Instant::now();

    // ── 1. Snapshot GDN.
    let t0 = now();
    session
        .save_gdn_snapshot(cluster)
        .context("spec-sampling: save_gdn_snapshot")?;
    t.gdn_snapshot = t0.elapsed().as_secs_f64() * 1000.0;

    // ── 2. MTP draft (logits variant). Sample t_d from q.
    let t0 = now();
    let rank0_device = cluster.device(0);
    let token_embd = model.shards[0]
        .token_embd
        .as_ref()
        .ok_or_else(|| anyhow!("rank 0 missing token_embd"))?;
    let mut row_host = vec![half::f16::from_f32(0.0); hidden];
    {
        let rank0_buf = rank0_device.alloc(row_bytes)?;
        rank0_device.bind()?;
        crate::forward::forward_embed_decode_host(
            rank0_device,
            rank0_device.default_stream(),
            token_embd,
            last_token,
            rank0_buf,
            hidden,
        )?;
        unsafe {
            rank0_device.memcpy_async(
                rank0_device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(row_host.as_mut_ptr() as usize),
                rank0_buf,
                row_bytes,
            )?;
        }
        rank0_device.default_stream().synchronize()?;
        unsafe { rank0_device.dealloc(rank0_buf, row_bytes)?; }
    }
    last_device.bind()?;
    let e_token_dev = last_device.alloc(row_bytes)?;
    unsafe {
        last_device.memcpy_async(
            last_device.default_stream(),
            CopyDirection::HostToDevice,
            e_token_dev,
            DevicePtr(row_host.as_ptr() as usize),
            row_bytes,
        )?;
    }
    last_device.default_stream().synchronize()?;

    let mut mtp_logits: Vec<f32> = Vec::with_capacity(vocab);
    let _argmax_token = forward_mtp_step_with_lm_head_logits(
        last_ops,
        last_device.default_stream(),
        last_device,
        cfg,
        mtp,
        mtp_scratch,
        output_norm_weight,
        lm_head_weight,
        h_for_mtp_dev,
        e_token_dev,
        position + 1,
        None,
        &mut mtp_logits,
    )?;
    unsafe { last_device.dealloc(e_token_dev, row_bytes)?; }

    // Build q distribution and sample the draft.
    let q_dist = build_distribution(&mtp_logits, sampling, history);
    let mtp_draft = sample_from_distribution(&q_dist, rng);
    t.mtp_draft = t0.elapsed().as_secs_f64() * 1000.0 - t.gdn_snapshot;

    // ── 3. Paired L=2 verify with [last_token, mtp_draft].
    // MTP-5h-Lever-C — pos1 head deferred to accept branch.
    let t0 = now();
    let mut step1_logits = Vec::with_capacity(vocab);
    let verify_tokens = [last_token, mtp_draft];
    forward_prefill_pp_logits_paired_l2(
        model,
        session,
        cluster,
        prefill_scratch,
        &verify_tokens,
        position,
        &mut step1_logits,
        None,
    )
    .context("spec-sampling: paired-L2 base verify (lazy pos1)")?;
    t.base_l2 = t0.elapsed().as_secs_f64() * 1000.0;

    // Build base pos0 distribution under sampling config.
    let p_pos0 = build_distribution(&step1_logits, sampling, history);

    // Rejection rule (only needs pos0).
    let q_t = prob_of(&q_dist, mtp_draft);
    let p_t = prob_of(&p_pos0, mtp_draft);
    let accept_prob = if q_t <= 0.0 {
        // q sampled it, so q_t > 0 in principle. Numerical 0 → reject.
        0.0
    } else {
        (p_t / q_t).min(1.0)
    };
    let u = rng.next_f32();
    let accepted = u < accept_prob;

    if accepted {
        // MTP-5h-Lever-C — fetch pos1 logits now (only on accept), then
        // build pos1 dist + sample second committed token.
        let mut step2_logits = Vec::with_capacity(vocab);
        forward_output_head_at_pp(
            model, cluster, prefill_scratch, /*position_in_l2=*/ 1, &mut step2_logits,
        )
        .context("spec-sampling: output head at L=2 pos 1 (accept)")?;
        let p_pos1 = build_distribution(&step2_logits, sampling, history);
        let t2 = sample_from_distribution(&p_pos1, rng);
        let h_for_next = prefill_scratch.per_rank[last_rank]
            .hidden_a
            .offset_bytes(row_bytes);
        let mut committed = Vec::with_capacity(2);
        committed.push(mtp_draft);
        committed.push(t2);
        Ok((
            SpecStep {
                committed,
                new_position: position + 2,
                accepted: true,
                draft: mtp_draft,
                verify: mtp_draft,
                timings_ms: t,
            },
            h_for_next,
        ))
    } else {
        // Reject: rollback + restore + redo L=1 + sample from residual.
        let t0 = now();
        session
            .rollback_full_attn(2)
            .context("spec-sampling: rollback_full_attn(2) on reject")?;
        session
            .restore_gdn_snapshot(cluster)
            .context("spec-sampling: restore_gdn_snapshot on reject")?;
        t.gdn_restore = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = now();
        let mut redo_logits: Vec<f32> = Vec::with_capacity(vocab);
        forward_one_token_pp_logits(
            model,
            session,
            cluster,
            decode_scratch,
            last_token,
            position,
            &mut redo_logits,
        )
        .context("spec-sampling: redo L=1 on reject")?;
        t.base_l1_redo = t0.elapsed().as_secs_f64() * 1000.0;

        // Build redo's distribution under the same sampling config and
        // draw from the residual.
        let p_redo = build_distribution(&redo_logits, sampling, history);
        let t_redo = sample_residual(&p_redo, &q_dist, rng);

        let h_for_next = decode_scratch.per_rank[last_rank].hidden_a;
        let mut committed = Vec::with_capacity(1);
        committed.push(t_redo);
        Ok((
            SpecStep {
                committed,
                new_position: position + 1,
                accepted: false,
                draft: mtp_draft,
                verify: t_redo,
                timings_ms: t,
            },
            h_for_next,
        ))
    }
}

// ===================================================================
// MTP-5f-tp2 — TP spec-decode driver (one macro step at a time).
// ===================================================================

/// Run one K=1 spec-decode macro step on a TP-sharded session. TP analog
/// of [`forward_speculative_pp_step`]. MTP head + LM head live on rank 0
/// (replicated `output.weight` in TP layout). Verify uses 2× sequential
/// `forward_one_token_tp_logits` (true L=2 batched TP primitive deferred);
/// reject uses full L=1 redo (Lever-1 GDN-only-redo also deferred).
#[allow(clippy::too_many_arguments)]
pub fn forward_speculative_tp_step(
    model: &Qwen3MoETpModel,
    session: &mut Qwen3MoETpSession,
    cluster: &HipCluster,
    ar: &BarP2pAllReduce,
    decode_scratch: &mut ShardedForwardOneTokenScratchTp,
    mtp: &MtpHeadWeights,
    mtp_scratch: &MtpForwardScratch,
    output_norm_weight: &DeviceTensor,
    lm_head_weight: &DeviceTensor,
    last_token: u32,
    h_for_mtp_dev: DevicePtr,
    position: usize,
) -> Result<(SpecStep, DevicePtr)> {
    let cfg = &model.config;
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    let head_rank = 0usize;
    let head_device = cluster.device(head_rank);
    let head_ops = &model.ops[head_rank];

    let mut t = SpecTimings::default();
    let now = || std::time::Instant::now();

    // ── 1. Snapshot GDN across all ranks.
    let t0 = now();
    session
        .save_gdn_snapshot(cluster)
        .context("spec-tp: save_gdn_snapshot")?;
    t.gdn_snapshot = t0.elapsed().as_secs_f64() * 1000.0;

    // ── 2. MTP draft on rank 0. token_embd is replicated in TP shards
    // so embed runs locally without peer-copy.
    let t0 = now();
    head_device.bind()?;
    let token_embd = &model.shards[head_rank].token_embd;
    let e_token_dev = head_device.alloc(row_bytes)?;
    crate::forward::forward_embed_decode_host(
        head_device,
        head_device.default_stream(),
        token_embd,
        last_token,
        e_token_dev,
        hidden,
    )?;
    let mtp_draft = forward_mtp_step_with_lm_head(
        head_ops,
        head_device.default_stream(),
        head_device,
        cfg,
        mtp,
        mtp_scratch,
        output_norm_weight,
        lm_head_weight,
        h_for_mtp_dev,
        e_token_dev,
        position + 1,
        None,
    )?;
    // SAFETY: e_token_dev was just allocated, unique reference here.
    unsafe { head_device.dealloc(e_token_dev, row_bytes)?; }
    t.mtp_draft = t0.elapsed().as_secs_f64() * 1000.0 - t.gdn_snapshot;

    // ── 3. Base verify: 2× sequential forward_one_token_tp_logits.
    let t0 = now();
    let mut step1_logits = Vec::with_capacity(vocab);
    let mut step2_logits = Vec::with_capacity(vocab);
    forward_one_token_tp_logits(
        model, decode_scratch, cluster, ar, &mut session.caches,
        last_token, position, &mut step1_logits,
    )
    .context("spec-tp: base verify step 1")?;
    forward_one_token_tp_logits(
        model, decode_scratch, cluster, ar, &mut session.caches,
        mtp_draft, position + 1, &mut step2_logits,
    )
    .context("spec-tp: base verify step 2")?;
    t.base_l2 = t0.elapsed().as_secs_f64() * 1000.0;

    if step1_logits.len() != vocab || step2_logits.len() != vocab {
        return Err(anyhow!(
            "spec-tp: expected {vocab} logits per step, got {}, {}",
            step1_logits.len(), step2_logits.len()
        ));
    }

    let argmax_local = |slice: &[f32]| -> u32 {
        let mut best_i = 0usize;
        let mut best_v = slice[0];
        for (i, &v) in slice.iter().enumerate().skip(1) {
            if v > best_v { best_v = v; best_i = i; }
        }
        best_i as u32
    };
    let t_actual_pos1 = argmax_local(&step1_logits);
    let t_actual_pos2 = argmax_local(&step2_logits);
    let accepted = mtp_draft == t_actual_pos1;

    if accepted {
        let h_for_next = decode_scratch.per_rank[head_rank].hidden_a;
        let mut committed = Vec::with_capacity(2);
        committed.push(t_actual_pos1);
        committed.push(t_actual_pos2);
        return Ok((
            SpecStep {
                committed,
                new_position: position + 2,
                accepted: true,
                draft: mtp_draft,
                verify: t_actual_pos1,
                timings_ms: t,
            },
            h_for_next,
        ));
    }

    // Reject: rollback both slots + restore GDN + redo L=1 with last_token.
    let t0 = now();
    session
        .rollback_full_attn(2)
        .context("spec-tp: rollback_full_attn(2) on reject")?;
    session
        .restore_gdn_snapshot(cluster)
        .context("spec-tp: restore_gdn_snapshot on reject")?;
    t.gdn_restore = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = now();
    let mut redo_logits: Vec<f32> = Vec::with_capacity(vocab);
    forward_one_token_tp_logits(
        model, decode_scratch, cluster, ar, &mut session.caches,
        last_token, position, &mut redo_logits,
    )
    .context("spec-tp: redo L=1 on reject")?;
    t.base_l1_redo = t0.elapsed().as_secs_f64() * 1000.0;

    let t_redo = argmax_local(&redo_logits);
    let h_for_next = decode_scratch.per_rank[head_rank].hidden_a;
    let mut committed = Vec::with_capacity(1);
    committed.push(t_redo);
    Ok((
        SpecStep {
            committed,
            new_position: position + 1,
            accepted: false,
            draft: mtp_draft,
            verify: t_redo,
            timings_ms: t,
        },
        h_for_next,
    ))
}
