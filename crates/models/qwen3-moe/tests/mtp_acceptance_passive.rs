//! MTP-4 — passive acceptance-rate measurement on Qwen3.6-27B.
//!
//! At each decode step we:
//!   1. Run the base model: forward_one_token_pp(token_t, position_t)
//!      → token_{t+1} (sampled greedy by argmax).
//!   2. Capture h_t (the pre-output_norm hidden on the last rank).
//!   3. Run MTP probe: forward_mtp_step_with_lm_head(h_t, embed(token_{t+1}),
//!      position_{t+1}) → predicted_{t+2}.
//!   4. Save the prediction. The next loop iteration produces
//!      actual_{t+2}; compare predicted vs actual to count an
//!      acceptance.
//!
//! HARD GATE per CLAUDE.md / MTP-4 task: ≥75% acceptance proceeds to
//! active spec-decode (KV rollback + verify in MTP-4 session 2);
//! <75% files null and we pivot.
//!
//! Skipped when:
//!   - no Qwen3.6-27B-Q4_0.gguf
//!   - no Qwen3.6-27B-mtp.gguf
//!   - fewer than 4 HIP devices (model is 16 GB → needs pp4)
//!
//! `FLAMBEAU_MTP_ACCEPT_STEPS=N` overrides the default 8-step decode
//! window.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel \
              launch over host/device buffers that live for the bounded \
              synchronize that follows."
)]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::mtp::{
    forward_mtp_step_with_lm_head, forward_mtp_step_with_lm_head_kv, load_mtp_head, MtpKvCache,
};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

// Switchable via FLAMBEAU_MTP_BASE env. Default Q4_0 (smaller, faster
// load). Set FLAMBEAU_MTP_BASE=ud_q8_k_xl to test the precision lever.
fn base_path() -> std::path::PathBuf {
    let v = std::env::var("FLAMBEAU_MTP_BASE").unwrap_or_default();
    let p = match v.as_str() {
        "ud_q8_k_xl" => "/artefact/models/Qwen3.6-27B-UD-Q8_K_XL.gguf",
        "q8_0"       => "/artefact/models/Qwen3.6-27B-Q8_0.gguf",
        "q4_1"       => "/artefact/models/Qwen3.6-27B-Q4_1.gguf",
        _            => "/artefact/models/Qwen3.6-27B-Q4_0.gguf",
    };
    PathBuf::from(p)
}
const _UNUSED_BASE_PATH: &str = "/artefact/models/Qwen3.6-27B-Q4_0.gguf";
const MTP_PATH: &str = "/artefact/models/Qwen3.6-27B-mtp.gguf";
const DEFAULT_STEPS: usize = 8;
// Real Qwen3 tokenization of "The capital of France is" (the V1.7.4
// canonical parity prompt). The first MTP-4 attempt used [1,2,3,4]
// arbitrary IDs and got GIGO; with real tokens the base model has
// signal to predict against.
//   The     → 760
//   Ġcapital → 6511
//   Ġof      → 314
//   ĠFrance  → 9338
//   Ġis      → 369
const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];

#[test]
fn mtp_acceptance_passive_qwen36_27b() -> Result<()> {
    let base_path = base_path();
    let mtp_path = PathBuf::from(MTP_PATH);
    if !base_path.exists() {
        eprintln!("skip: {} not present", base_path.display());
        return Ok(());
    }
    if !mtp_path.exists() {
        eprintln!("skip: {MTP_PATH} not present (run convert_qwen36_mtp.py)");
        return Ok(());
    }
    let n_gpus = device_count().unwrap_or(0);
    if n_gpus < 4 {
        eprintln!("skip: need ≥4 HIP devices for pp4 (got {n_gpus})");
        return Ok(());
    }

    let n_steps: usize = std::env::var("FLAMBEAU_MTP_ACCEPT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_STEPS);

    eprintln!("=== MTP-4 passive acceptance ===");
    eprintln!("base: {}", base_path.display());
    eprintln!("mtp:  {MTP_PATH}");
    eprintln!("decode steps: {n_steps}, prompt prefill: {} tokens", PROMPT_IDS.len());

    // ── Load base model on pp{n_gpus}
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    let cluster = HipCluster::new(&(0..n_gpus).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!("loading base across {} ranks ({} layers)…", cluster.ranks(), cfg.num_layers);
    let model = Qwen3MoEShardedModel::load(&base_file, &cluster, &assignment)?;
    eprintln!("base loaded ({:.2} GiB across shards)",
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0));

    let last_rank = (cluster.ranks() - 1) as usize;
    let last_device = cluster.device(last_rank);
    let last_ops = &model.shards[last_rank].ops;

    // ── Load MTP on the last rank (where output_norm + lm_head live)
    eprintln!("loading MTP on rank {last_rank}…");
    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, last_device)?;

    // The base loader put output_norm + output (lm_head) on the last
    // rank. Find them on the shard.
    let last_shard = &model.shards[last_rank];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .ok_or_else(|| anyhow!("last rank missing output_norm"))?;
    let lm_head = last_shard
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("last rank missing output (lm_head)"))?;
    // vLLM convention: base model applies self.norm before returning
    // hidden_states; MTP gets the post-norm value. flambeau's
    // forward_one_token_pp writes pre-norm to scratch.hidden_a, so we
    // re-apply output_norm in forward_mtp_step_with_lm_head.

    // ── Allocate session + scratch
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;

    // Prefill the prompt
    let mut prefill_scratch =
        ShardedForwardPrefillScratch::new(&model, &cluster, PROMPT_IDS.len())?;
    let mut last_token = forward_prefill_pp(
        &model, &mut session, &cluster, &mut prefill_scratch, &PROMPT_IDS, 0,
    )?;
    prefill_scratch.dispose(&cluster).ok();
    eprintln!("prefill done; first sampled token = {last_token}");

    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    // We need:
    //   1. token_embd on rank 0 (already there) to fetch embedding rows for MTP.
    //   2. A scratch slot on the LAST rank to hold the F16 embedding row + a
    //      copy of h_t (since hidden_a gets overwritten on the next forward
    //      pass).
    let hidden = cfg.hidden_size;
    let h_t_saved = last_device.alloc(hidden * 2)?;        // F16 [hidden]
    let e_token_dev = last_device.alloc(hidden * 2)?;      // F16 [hidden]

    // Persistent MTP KV cache (accumulates across decode steps).
    // Sized for prompt prefix + decode steps; 256 slots covers
    // the smoke test by a wide margin.
    const MTP_KV_MAX: usize = 256;
    let kv_row_bytes = cfg.num_kv_heads * cfg.head_dim * 2; // F16
    let mtp_kcache = last_device.alloc(MTP_KV_MAX * kv_row_bytes)?;
    let mtp_vcache = last_device.alloc(MTP_KV_MAX * kv_row_bytes)?;
    let mut mtp_cache_pos: usize = 0;
    let use_kv_accum = std::env::var("FLAMBEAU_MTP_KV_ACCUM")
        .as_deref()
        .map(|s| s != "0" && s != "off")
        .unwrap_or(true);
    if use_kv_accum {
        eprintln!("MTP KV accumulation: ON (persistent cache, {} slots)", MTP_KV_MAX);
    } else {
        eprintln!("MTP KV accumulation: OFF (transient 1-slot per call)");
    }
    // Rank 0 helper buffer for embedding lookup.
    let rank0_device = cluster.device(0);
    let token_embd = model.shards[0]
        .token_embd
        .as_ref()
        .ok_or_else(|| anyhow!("rank 0 missing token_embd"))?;
    let rank0_embed_buf = rank0_device.alloc(hidden * 2)?;

    // Tokens generated, including the prefill's last-emitted one.
    let mut tokens: Vec<u32> = PROMPT_IDS.to_vec();
    tokens.push(last_token);

    // Diagnostic: dump embedding for token 11751 (ĠParis) and compare to
    // Python ref to verify Q4_0 dequant of token_embd is bit-exact.
    // Confirmed match (2026-04-28): flambeau's dequant produces identical
    // F16 values to gguf python's dequant. Embedding lookup ruled out as
    // the source of the 0% acceptance.
    // Reference row[:8]:
    //   [ 0.01465  0.02441  0.00488 -0.01465  0.00488 -0.00488  0.00488  0.0]
    {
        rank0_device.bind()?;
        flambeau_qwen3_moe::forward::forward_embed_decode_host(
            rank0_device, rank0_device.default_stream(),
            token_embd, 11751, rank0_embed_buf, hidden,
        )?;
        let mut buf = vec![half::f16::from_f32(0.0); hidden];
        unsafe {
            rank0_device.memcpy_async(
                rank0_device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(buf.as_mut_ptr() as usize),
                rank0_embed_buf,
                hidden * 2,
            )?;
        }
        rank0_device.default_stream().synchronize()?;
        let f32_first8: Vec<f32> = buf[..8].iter().map(|x| x.to_f32()).collect();
        eprintln!("[diag] embed(11751) flambeau first 8: {f32_first8:?}");
        eprintln!("[diag] embed(11751) python    first 8: [0.01465, 0.02441, 0.00488, -0.01465, 0.00488, -0.00488, 0.00488, 0.0]");
    }

    let mut predicted: Vec<u32> = Vec::with_capacity(n_steps);
    let mut have_h_t = false;
    let mut accepted = 0usize;
    let mut compared = 0usize;

    let t0 = std::time::Instant::now();
    for step in 0..=n_steps {
        let position = PROMPT_IDS.len() + step;

        // ── Run base step. position is the position of `last_token`.
        let next = forward_one_token_pp(
            &model, &mut session, &cluster, &mut decode_scratch, last_token, position,
        )?;
        // After this call, `decode_scratch.per_rank[last_rank].hidden_a`
        // holds the pre-output_norm hidden at position `position` —
        // i.e., the state that produced `next`'s logits.
        let hidden_a_after = decode_scratch.per_rank[last_rank].hidden_a;

        // ── If we have a prediction from the previous step, score it.
        if let Some(&pred) = predicted.last() {
            compared += 1;
            if pred == next {
                accepted += 1;
            }
            eprintln!("  step {step}: predicted={pred} actual={next} {}",
                if pred == next { "✓" } else { "✗" });
        }

        // ── Compute MTP draft for token at position `position+1` (i.e.
        //    the successor of `next`). Inputs: h_t = previously-saved
        //    pre-norm hidden (from the prior base step that PRODUCED
        //    `last_token`). At step 0 we have to use the just-computed
        //    hidden_a (h_position) and skip the comparison until step 1.
        if step < n_steps {
            // Embed `next` onto the last rank.
            // 1) Look up the row on rank 0 → rank0_embed_buf (F16).
            rank0_device.bind()?;
            flambeau_qwen3_moe::forward::forward_embed_decode_host(
                rank0_device,
                rank0_device.default_stream(),
                token_embd,
                next,
                rank0_embed_buf,
                hidden,
            )?;
            // 2) D2H rank 0 → host → D2H rank last (cluster lacks
            //    direct peer-DMA in test infra; staged via host is
            //    fine for the smoke).
            let mut row_host = vec![half::f16::from_f32(0.0); hidden];
            unsafe {
                rank0_device.memcpy_async(
                    rank0_device.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(row_host.as_mut_ptr() as usize),
                    rank0_embed_buf,
                    hidden * 2,
                )?;
            }
            rank0_device.default_stream().synchronize()?;
            last_device.bind()?;
            unsafe {
                last_device.memcpy_async(
                    last_device.default_stream(),
                    CopyDirection::HostToDevice,
                    e_token_dev,
                    DevicePtr(row_host.as_ptr() as usize),
                    hidden * 2,
                )?;
            }
            last_device.default_stream().synchronize()?;

            // For the FIRST step, snapshot h_t from THIS step's
            // hidden_a and continue without scoring (we don't have a
            // prediction yet). For subsequent steps, use the
            // PREVIOUSLY saved h_t (which is from the step that
            // produced `last_token` — that's what we need).
            let h_t_for_mtp = if have_h_t {
                h_t_saved
            } else {
                // Step 0: no previous h. Use this step's hidden_a as
                // a stand-in to keep the chain rolling; we won't
                // score this prediction (it's "predict the token AFTER
                // next, given (h_at_position, embed(next))" which is
                // valid; we just align comparisons one step later).
                hidden_a_after
            };

            let pred_next_next = if use_kv_accum {
                let kv = MtpKvCache {
                    kcache: mtp_kcache,
                    vcache: mtp_vcache,
                    cache_position: mtp_cache_pos,
                    n_tokens_kv: mtp_cache_pos + 1,
                };
                let r = forward_mtp_step_with_lm_head_kv(
                    last_ops,
                    last_device.default_stream(),
                    last_device,
                    &cfg,
                    &mtp,
                    output_norm,
                    lm_head,
                    h_t_for_mtp,
                    e_token_dev,
                    position + 1,
                    Some(kv),
                )?;
                mtp_cache_pos += 1;
                r
            } else {
                forward_mtp_step_with_lm_head(
                    last_ops,
                    last_device.default_stream(),
                    last_device,
                    &cfg,
                    &mtp,
                    output_norm,
                    lm_head,
                    h_t_for_mtp,
                    e_token_dev,
                    position + 1,
                )?
            };
            predicted.push(pred_next_next);

            // ── Snapshot h_t = current hidden_a for next iteration.
            unsafe {
                last_device.memcpy_async(
                    last_device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    h_t_saved,
                    hidden_a_after,
                    hidden * 2,
                )?;
            }
            last_device.default_stream().synchronize()?;
            have_h_t = true;
        }

        tokens.push(next);
        last_token = next;
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let acceptance = if compared == 0 {
        0.0
    } else {
        accepted as f64 / compared as f64
    };
    eprintln!("\n=== MTP-4 acceptance result ===");
    eprintln!("  compared: {compared} steps");
    eprintln!("  accepted: {accepted}");
    eprintln!("  acceptance rate: {:.1}%", acceptance * 100.0);
    eprintln!("  wall: {wall_ms:.0} ms ({:.1} ms/step base+probe)",
        wall_ms / (n_steps + 1) as f64);
    eprintln!("  GATE (≥75%): {}",
        if acceptance >= 0.75 { "PASS" } else { "FAIL — pivot or debug" });

    // Free
    unsafe {
        last_device.dealloc(h_t_saved, hidden * 2)?;
        last_device.dealloc(e_token_dev, hidden * 2)?;
        last_device.dealloc(mtp_kcache, MTP_KV_MAX * kv_row_bytes)?;
        last_device.dealloc(mtp_vcache, MTP_KV_MAX * kv_row_bytes)?;
        rank0_device.dealloc(rank0_embed_buf, hidden * 2)?;
    }
    decode_scratch.dispose(&cluster).ok();
    session.dispose(&cluster).ok();
    model.dispose(&cluster)?;
    cluster.dispose()?;

    // Don't FAIL the test on low acceptance — this is a measurement,
    // not a correctness assertion. The cert documents the result;
    // pass/fail decision goes in MTP-4-decision.
    Ok(())
}

#[allow(dead_code)]
fn _ops_registry_alive(reg: &OpsRegistry) {
    let _ = reg;
}
