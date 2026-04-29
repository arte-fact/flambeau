//! MTP-5e profile — HipEvent-attribution breakdown of baseline vs
//! spec-decode (paired L=2 verify) on Qwen3.6-27B / Mesh<4>.
//!
//! Runs each path through `flambeau_backend_hip::profile::enable` →
//! workload → `flush` and prints per-section ms. Uses the same
//! 5-token prompt + ~32 macro / 60 token target as the perf A/B.
//!
//! Section names baseline (L=1):  `step_start`, `embed_done`,
//!   `stage_start`, `stage_end`, `output_head_start`,
//!   `output_head_done`, `step_logits_downloaded`.
//! Section names spec L=2:        `l2_step_start`, `l2_embed_done`,
//!   `l2_stage_start`, `l2_stage_end`, `l2_output_head_start`,
//!   `l2_output_pos0_done`, `l2_output_pos1_done`.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel \
              launch over host/device buffers that live for the bounded \
              synchronize that follows."
)]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, profile, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp_logits, forward_speculative_pp_step,
    ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::mtp::{load_mtp_head, MtpForwardScratch};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];

fn argmax(slice: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

fn print_breakdown(label: &str, n_runs: usize, stats: Vec<profile::SectionStat>) {
    eprintln!("\n=== {label}: section breakdown over {n_runs} runs ===");
    eprintln!("section                       total_ms    count   mean_ms  per_run_ms");
    let mut total: f32 = 0.0;
    for s in &stats {
        eprintln!(
            "{:<28}  {:>9.2}  {:>7}  {:>8.4}  {:>9.4}",
            s.name, s.total_ms, s.count, s.mean_ms,
            s.total_ms / n_runs as f32
        );
        total += s.total_ms;
    }
    eprintln!(
        "{:<28}  {:>9.2}  {:>7}  {:>8}  {:>9.4}",
        "TOTAL", total, "-", "-", total / n_runs as f32
    );
}

#[test]
fn mtp_spec_decode_profile() -> Result<()> {
    let base_path = PathBuf::from("/artefact/models/Qwen3.6-27B-Q4_0.gguf");
    let mtp_path = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    if !base_path.exists() || !mtp_path.exists() {
        eprintln!("skip: GGUFs not present");
        return Ok(());
    }
    let n_gpus = device_count().unwrap_or(0);
    if n_gpus < 4 {
        eprintln!("skip: need ≥4 HIP devices (got {n_gpus})");
        return Ok(());
    }

    let n_baseline_steps: usize = std::env::var("FLAMBEAU_PROFILE_BASE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let n_spec_macros: usize = std::env::var("FLAMBEAU_PROFILE_SPEC_MACROS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let n_warmup: usize = 4;

    eprintln!("=== MTP-5e profile (baseline {n_baseline_steps} + spec {n_spec_macros} macros) ===");
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    let cluster = HipCluster::new(&(0..n_gpus).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&base_file, &cluster, &assignment)?;

    let last_rank = (cluster.ranks() - 1) as usize;
    let last_device = cluster.device(last_rank);
    let last_shard = &model.shards[last_rank];
    let output_norm = last_shard.output_norm.as_ref().ok_or_else(|| anyhow!("no output_norm"))?;
    let lm_head = last_shard.output.as_ref().ok_or_else(|| anyhow!("no lm_head"))?;

    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, last_device)?;
    last_device.bind()?;
    let mtp_scratch = MtpForwardScratch::new(last_device, &cfg)?;

    // ─────────────────────────────────────────────────────────────
    // Branch A — baseline single-token decode loop (PROFILED).
    // ─────────────────────────────────────────────────────────────
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

        let mut last_token = 0u32;
        for (i, &tok) in PROMPT_IDS.iter().enumerate() {
            let mut logits = Vec::new();
            forward_one_token_pp_logits(
                &model, &mut session, &cluster, &mut decode_scratch, tok, i,
                &mut logits,
            )?;
            if i == PROMPT_IDS.len() - 1 {
                last_token = argmax(&logits);
            }
        }
        // Warm-up steps without the timer (filter out cold-cache effects).
        for step in 0..n_warmup {
            let position = PROMPT_IDS.len() + step;
            let mut logits = Vec::new();
            forward_one_token_pp_logits(
                &model, &mut session, &cluster, &mut decode_scratch,
                last_token, position, &mut logits,
            )?;
            last_token = argmax(&logits);
        }
        // Timed steps.
        profile::enable();
        let t0 = std::time::Instant::now();
        for step in 0..n_baseline_steps {
            let position = PROMPT_IDS.len() + n_warmup + step;
            let mut logits = Vec::new();
            forward_one_token_pp_logits(
                &model, &mut session, &cluster, &mut decode_scratch,
                last_token, position, &mut logits,
            )?;
            last_token = argmax(&logits);
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let stats = profile::flush()?;
        eprintln!(
            "[A] baseline wall: {wall_ms:.0} ms over {n_baseline_steps} steps = {:.2} ms/step",
            wall_ms / n_baseline_steps as f64
        );
        print_breakdown("BASELINE L=1", n_baseline_steps, stats);

        decode_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    // ─────────────────────────────────────────────────────────────
    // Branch B — spec-decode (paired L=2 verify, PROFILED).
    // ─────────────────────────────────────────────────────────────
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
        let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 2)?;

        let mut last_token = 0u32;
        for (i, &tok) in PROMPT_IDS.iter().enumerate() {
            let mut logits = Vec::new();
            forward_one_token_pp_logits(
                &model, &mut session, &cluster, &mut decode_scratch, tok, i,
                &mut logits,
            )?;
            if i == PROMPT_IDS.len() - 1 {
                last_token = argmax(&logits);
            }
        }
        let mut h_for_mtp = decode_scratch.per_rank[last_rank].hidden_a;
        let mut position = PROMPT_IDS.len();
        // Warm-up macros.
        for _ in 0..n_warmup {
            let (s, h_next) = forward_speculative_pp_step(
                &model, &mut session, &cluster, &mut decode_scratch, &mut prefill_scratch,
                &mtp, &mtp_scratch,
                output_norm, lm_head,
                last_token, h_for_mtp, position,
            )?;
            last_token = *s.committed.last().unwrap();
            position = s.new_position;
            h_for_mtp = h_next;
        }
        // Timed macros.
        profile::enable();
        let t0 = std::time::Instant::now();
        let mut accept_count = 0usize;
        let mut tokens_committed = 0usize;
        for _ in 0..n_spec_macros {
            let (s, h_next) = forward_speculative_pp_step(
                &model, &mut session, &cluster, &mut decode_scratch, &mut prefill_scratch,
                &mtp, &mtp_scratch,
                output_norm, lm_head,
                last_token, h_for_mtp, position,
            )?;
            if s.accepted { accept_count += 1; }
            tokens_committed += s.committed.len();
            last_token = *s.committed.last().unwrap();
            position = s.new_position;
            h_for_mtp = h_next;
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let stats = profile::flush()?;
        let accept_pct = 100.0 * accept_count as f64 / n_spec_macros as f64;
        eprintln!(
            "[B] spec wall: {wall_ms:.0} ms over {n_spec_macros} macros = {tokens_committed} tokens, {accept_pct:.1}% accept, {:.2} ms/macro = {:.2} ms/tok",
            wall_ms / n_spec_macros as f64,
            wall_ms / tokens_committed as f64,
        );
        print_breakdown("SPEC L=2 + MTP draft", n_spec_macros, stats);

        decode_scratch.dispose(&cluster).ok();
        prefill_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    last_device.bind()?;
    mtp_scratch.dispose(last_device).ok();
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
