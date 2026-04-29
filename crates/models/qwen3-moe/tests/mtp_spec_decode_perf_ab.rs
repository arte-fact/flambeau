//! MTP-5e perf cert — A/B: baseline single-token decode vs
//! spec-decode (paired L=2 verify) on Qwen3.6-27B / Mesh<4>.
//!
//! Loads model once, runs both branches with the same prompt prime
//! and target token count, prints ms/token + accept rate.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel \
              launch over host/device buffers that live for the bounded \
              synchronize that follows."
)]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, HipCluster};
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

#[test]
fn mtp_spec_decode_perf_ab() -> Result<()> {
    let base_path = PathBuf::from("/artefact/models/Qwen3.6-27B-Q4_0.gguf");
    let mtp_path = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    if !base_path.exists() || !mtp_path.exists() {
        eprintln!("skip: GGUFs not present");
        return Ok(());
    }
    let n_gpus = device_count().unwrap_or(0);
    // FLAMBEAU_PP_RANKS — override the rank count (1..=available).
    // Default to all available. Use this to compare PP2 vs PP4 on the
    // same rig (HIP_VISIBLE_DEVICES gates which physical GPUs are in
    // play; this gates how many of those become PP ranks).
    let n_ranks: i32 = std::env::var("FLAMBEAU_PP_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_gpus);
    if n_ranks < 2 || n_ranks > n_gpus {
        eprintln!("skip: need 2..={n_gpus} HIP devices (FLAMBEAU_PP_RANKS={n_ranks})");
        return Ok(());
    }

    // Target ~60 tokens, the same window the MTP-5c smoke produced.
    let n_tokens: usize = std::env::var("FLAMBEAU_PERF_AB_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    eprintln!("=== MTP-5e perf A/B (target={n_tokens} tokens, pp{n_ranks}) ===");
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    // FLAMBEAU_CTX_CAP — clamp the per-layer KV cache size at load.
    // PP4 with 16 layers/rank fits 32k ctx in 16 GB MI50; PP2 with 32
    // layers/rank needs ≤8k to fit weights+KV. Default clamp 4096
    // covers the perf A/B's ~70-token sequence with headroom.
    let ctx_cap: usize = std::env::var("FLAMBEAU_CTX_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let cluster = HipCluster::new(&(0..n_ranks).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!("loading base across {} ranks…", cluster.ranks());
    let mut model = Qwen3MoEShardedModel::load(&base_file, &cluster, &assignment)?;
    if model.config.context_length > ctx_cap {
        eprintln!("clamping model ctx {} → {}", model.config.context_length, ctx_cap);
        model.config.context_length = ctx_cap;
    }

    let last_rank = (cluster.ranks() - 1) as usize;
    let last_device = cluster.device(last_rank);
    let last_shard = &model.shards[last_rank];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .ok_or_else(|| anyhow!("no output_norm"))?;
    let lm_head = last_shard
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("no lm_head"))?;

    eprintln!("loading MTP on rank {last_rank}…");
    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, last_device)?;
    last_device.bind()?;
    let mtp_scratch = MtpForwardScratch::new(last_device, &model.config)?;

    // ────────────────────────────────────────────────────────────
    // Branch A — baseline single-token decode loop.
    // ────────────────────────────────────────────────────────────
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

        let mut tokens: Vec<u32> = vec![last_token];
        let t0 = std::time::Instant::now();
        for step in 1..n_tokens {
            let position = PROMPT_IDS.len() + step - 1;
            let mut logits = Vec::new();
            forward_one_token_pp_logits(
                &model, &mut session, &cluster, &mut decode_scratch,
                last_token, position, &mut logits,
            )?;
            let next = argmax(&logits);
            tokens.push(next);
            last_token = next;
        }
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        let per_tok = wall / (n_tokens - 1) as f64;
        eprintln!("\n[A] baseline: {} tokens, wall {:.0} ms, {:.2} ms/tok",
                  n_tokens - 1, wall, per_tok);
        eprintln!("    first 16: {:?}", &tokens[..tokens.len().min(16)]);

        decode_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    // ────────────────────────────────────────────────────────────
    // Branch B — spec-decode (paired L=2 verify).
    // ────────────────────────────────────────────────────────────
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
        let mut tokens: Vec<u32> = vec![last_token];
        let mut accept_count = 0usize;
        let mut macro_count = 0usize;

        let t0 = std::time::Instant::now();
        while tokens.len() < n_tokens {
            let (s, h_next) = forward_speculative_pp_step(
                &model, &mut session, &cluster, &mut decode_scratch, &mut prefill_scratch,
                &mtp, &mtp_scratch,
                output_norm, lm_head,
                last_token, h_for_mtp, position,
            )?;
            macro_count += 1;
            if s.accepted { accept_count += 1; }
            for &t in &s.committed {
                tokens.push(t);
                if tokens.len() >= n_tokens { break; }
            }
            last_token = *s.committed.last().unwrap();
            position = s.new_position;
            h_for_mtp = h_next;
        }
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        let n_decoded = tokens.len() - 1;
        let per_tok = wall / n_decoded.max(1) as f64;
        let accept_pct = 100.0 * accept_count as f64 / macro_count.max(1) as f64;
        eprintln!("\n[B] spec-decode: {n_decoded} tokens via {macro_count} macros, \
                   accept {accept_pct:.1}%, wall {wall:.0} ms, {per_tok:.2} ms/tok");
        eprintln!("    first 16: {:?}", &tokens[..tokens.len().min(16)]);

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
