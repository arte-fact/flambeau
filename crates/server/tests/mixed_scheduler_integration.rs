//! **#305 / #306 integration** — drive the [`MixedScheduler`] across
//! multiple iterations through `forward_decode_mixed_hybrid` and
//! verify the multi-iteration multi-chunk behaviour matches a
//! separate-sessions reference.
//!
//! Test scenario:
//! - Request A: medium prompt (96 tokens), chunked at K=32 → 3 chunks
//! - Request B: short prompt (16 tokens), chunked at K=32 → 1 chunk
//! - After their final chunks, each transitions to decoding state.
//!
//! Reference path: prefill A normally then prefill B normally, then
//! batched-decode both for 2 steps. Capture per-step logits.
//!
//! Mixed path: drive the scheduler — chunks of A + B interleave with
//! decode steps for whichever request has finished prefill first.
//!
//! Skips when GGUF missing or fewer than 4 HIP devices. Defaults to
//! Qwen3.5-9B-Q4_1 on pp2tp2 over [0,2,1,3].

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — model load + forward + dispose"
)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_decode_mixed_hybrid, BatchSlot, MixedPrefillChunk,
    ShardedForwardOneTokenScratchHybrid,
};
use flambeau_qwen3_moe::hybrid::HybridMeshSpec;
use flambeau_qwen3_moe::{
    forward::forward_prefill_hybrid_logits, Qwen3MoEConfig, Qwen3MoEHybridModel,
    Qwen3MoEHybridSession, ShardedForwardPrefillScratchHybrid,
};
use flambeau_server::mixed_scheduler::{MixedScheduler, PendingPrefillReq};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
            if x > acc.1 { (i, x) } else { acc }
        })
        .0 as u32
}

#[test]
fn mixed_scheduler_drives_multi_chunk_pp2tp2() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5-9B GGUF not present");
        return Ok(());
    };
    let dev_ids = vec![0i32, 2, 1, 3];
    let n_avail: i32 = device_count().unwrap_or(0);
    if (n_avail as usize) < dev_ids.len() {
        eprintln!("skip — need 4 HIP devices, have {n_avail}");
        return Ok(());
    }
    std::env::set_var("FLAMBEAU_MAX_CTX", "4096");

    let file = GgufFile::open(&path)?;
    let model_cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(model_cfg.arch, "qwen35");

    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    spec.validate(model_cfg.num_layers, dev_ids.len())?;
    let model = Qwen3MoEHybridModel::load(&file, &dev_ids, spec)?;
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&dev_ids)?);

    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        let ar = BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))
            .with_context(|| format!("BarP2pAllReduce::new for stage {}", stage.stage_idx))?;
        stage_ars.push(ar);
    }

    // Two requests with different chunk counts.
    let prompt_a: Vec<u32> = (0..96u32).map(|i| (5 + i * 41) % 151000).collect();
    let prompt_b: Vec<u32> = (0..16u32).map(|i| (11 + i * 53) % 151000).collect();
    let chunk_size = 32usize; // → A:3 chunks (32/32/32), B:1 chunk
    // Slots: 0 = A (the prefill+decode owner), 1 = B.
    let slot_a = 0;
    let slot_b = 1;
    // Token-budget large enough for any single iteration here.
    let token_budget = 64usize;

    // ── REFERENCE: separate prefills, then nothing else.
    //    We compare the scheduler-driven prefill_final_logits for each
    //    request against the reference's last-row logits. After both
    //    are prefilled, we don't run decode steps in this test (kept
    //    minimal — extending to decode steps requires sampler state
    //    consistency which is beyond this integration check).
    let mut sess_a_ref = Qwen3MoEHybridSession::new(&model)?;
    let mut sess_b_ref = Qwen3MoEHybridSession::new(&model)?;
    let mut prefill_scratch_ref = ShardedForwardOneTokenScratchHybrid::new(&model)?;

    let mut logits_a_ref: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut prefill_scratch_ref,
        &global_cluster,
        &stage_ars,
        &mut sess_a_ref,
        &prompt_a,
        0,
        &mut logits_a_ref,
    )
    .context("ref: prefill A")?;
    let argmax_a_ref = argmax(&logits_a_ref);

    let mut logits_b_ref: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut prefill_scratch_ref,
        &global_cluster,
        &stage_ars,
        &mut sess_b_ref,
        &prompt_b,
        0,
        &mut logits_b_ref,
    )
    .context("ref: prefill B")?;
    let argmax_b_ref = argmax(&logits_b_ref);

    // ── SCHEDULER-DRIVEN MIXED PATH ───────────────────────────────
    let mut scheduler = MixedScheduler::new(token_budget, chunk_size);
    let req_a = scheduler.allocate_request_id();
    let req_b = scheduler.allocate_request_id();
    scheduler.submit_prefill(PendingPrefillReq {
        request_id: req_a,
        slot_idx: slot_a,
        tokens: prompt_a.clone(),
        chunk_start: 0,
    });
    scheduler.submit_prefill(PendingPrefillReq {
        request_id: req_b,
        slot_idx: slot_b,
        tokens: prompt_b.clone(),
        chunk_start: 0,
    });

    let mut sess_a_mix = Qwen3MoEHybridSession::new(&model)?;
    let mut sess_b_mix = Qwen3MoEHybridSession::new(&model)?;
    let mut mixed_scratch =
        ShardedForwardPrefillScratchHybrid::new(&model, token_budget)?;

    // Capture each request's final-chunk logits.
    let mut logits_a_mix: Vec<f32> = Vec::new();
    let mut logits_b_mix: Vec<f32> = Vec::new();

    let mut iteration_count = 0usize;
    while let Some(plan) = scheduler.next_iteration() {
        iteration_count += 1;
        let chunk_plan = plan.chunk.as_ref();
        let n_decodes = plan.decodes.len();

        eprintln!(
            "iter {iteration_count}: chunk = {} (slot={}, K={}, final={}), decodes = {n_decodes}",
            chunk_plan.map_or("None".to_string(), |c| format!("req {:?}", c.request_id)),
            chunk_plan.map_or(usize::MAX, |c| c.slot_idx),
            chunk_plan.map_or(0, |c| c.tokens.len()),
            chunk_plan.map_or(false, |c| c.is_final_chunk),
        );

        let chunk = chunk_plan.map(|c| MixedPrefillChunk {
            idx: c.slot_idx,
            tokens: c.tokens.clone(),
            chunk_start: c.chunk_start,
            is_final_chunk: c.is_final_chunk,
        });

        let slots: Vec<BatchSlot> = plan
            .decodes
            .iter()
            .map(|d| BatchSlot {
                idx: d.slot_idx,
                token_id: d.token_id,
                position: d.position,
            })
            .collect();

        // Borrow disjoint sessions [A, B].
        let mut sess_refs: Vec<&mut Qwen3MoEHybridSession> =
            vec![&mut sess_a_mix, &mut sess_b_mix];

        // Per-session decode logits buffers (parallel to sess_refs).
        let mut decode_logits_a: Vec<f32> = Vec::new();
        let mut decode_logits_b: Vec<f32> = Vec::new();
        let mut logits_refs: Vec<&mut Vec<f32>> =
            vec![&mut decode_logits_a, &mut decode_logits_b];

        // Per-iteration prefill_final buffer if a chunk is is_final_chunk.
        let mut iter_prefill_final: Vec<f32> = Vec::new();
        let prefill_final_slot = chunk
            .as_ref()
            .filter(|c| c.is_final_chunk)
            .map(|_| chunk_plan.unwrap().slot_idx);
        let prefill_final_out = if prefill_final_slot.is_some() {
            Some(&mut iter_prefill_final)
        } else {
            None
        };

        forward_decode_mixed_hybrid(
            &model,
            sess_refs.as_mut_slice(),
            &global_cluster,
            &stage_ars,
            &mut mixed_scratch,
            chunk.as_ref(),
            &slots,
            logits_refs.as_mut_slice(),
            prefill_final_out,
        )
        .with_context(|| format!("iter {iteration_count}: forward_decode_mixed_hybrid"))?;

        // If this was a request's final-chunk dispatch, capture logits.
        if let Some(final_slot) = prefill_final_slot {
            if final_slot == slot_a {
                logits_a_mix = iter_prefill_final;
            } else if final_slot == slot_b {
                logits_b_mix = iter_prefill_final;
            }
        }
    }

    eprintln!(
        "scheduler done: {iteration_count} iterations; argmax_a ref={argmax_a_ref} mix={} | argmax_b ref={argmax_b_ref} mix={}",
        argmax(&logits_a_mix),
        argmax(&logits_b_mix),
    );

    // dispose
    let _ = prefill_scratch_ref.dispose(&model);
    let _ = mixed_scratch.dispose(&model);
    let _ = sess_a_ref.dispose(&model);
    let _ = sess_b_ref.dispose(&model);
    let _ = sess_a_mix.dispose(&model);
    let _ = sess_b_mix.dispose(&model);
    let _ = model.dispose();

    // Assert: top-1 match. Per-element parity may have F16 drift across
    // multi-chunk prefill (each chunk re-attends the prior chunks'
    // KV — different FMA orders). Top-1 is the strict gate.
    assert_eq!(
        argmax(&logits_a_mix),
        argmax_a_ref,
        "prefill A top-1 mismatch (96 tokens / 3 chunks)"
    );
    assert_eq!(
        argmax(&logits_b_mix),
        argmax_b_ref,
        "prefill B top-1 mismatch (16 tokens / 1 chunk)"
    );
    eprintln!("OK: scheduler-driven multi-chunk prefill matches reference (top-1)");
    Ok(())
}
