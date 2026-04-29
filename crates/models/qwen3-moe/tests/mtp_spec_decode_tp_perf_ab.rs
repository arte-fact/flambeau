//! MTP-5f-tp2 perf A/B — baseline TP single-token decode vs TP
//! spec-decode on Qwen3.6-27B / tp2 (GPUs 0,1).
//!
//! Mirror of `mtp_spec_decode_perf_ab.rs` but uses Qwen3MoETpModel +
//! BarP2pAllReduce instead of Qwen3MoEShardedModel. Tests whether the
//! TP topology's BAR1 P2P AllReduce (faster than PCIe peer-copy via
//! host) flips spec-decode net-positive.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture"
)]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_tp_logits, forward_speculative_tp_step,
    ShardedForwardOneTokenScratchTp,
};
use flambeau_qwen3_moe::mtp::{load_mtp_head, MtpForwardScratch};
use flambeau_qwen3_moe::tp_sharded::{Qwen3MoETpModel, Qwen3MoETpSession};
use flambeau_qwen3_moe::tp_layout::Qwen35DenseTpLayout;
use flambeau_qwen3_moe::Qwen3MoEConfig;
use std::path::PathBuf;
use std::sync::Arc;

const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];

fn argmax(slice: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_v { best_v = v; best_i = i; }
    }
    best_i as u32
}

#[test]
fn mtp_spec_decode_tp_perf_ab() -> Result<()> {
    let base_path = PathBuf::from("/artefact/models/Qwen3.6-27B-Q4_0.gguf");
    let mtp_path = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    if !base_path.exists() || !mtp_path.exists() {
        eprintln!("skip: GGUFs not present");
        return Ok(());
    }
    let n_gpus = device_count().unwrap_or(0);
    let n_ranks: i32 = std::env::var("FLAMBEAU_TP_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    if n_gpus < n_ranks {
        eprintln!("skip: need ≥{n_ranks} HIP devices (got {n_gpus})");
        return Ok(());
    }

    let n_tokens: usize = std::env::var("FLAMBEAU_PERF_AB_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let ctx_cap: usize = std::env::var("FLAMBEAU_CTX_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    eprintln!("=== MTP-5f-tp2 perf A/B (target={n_tokens} tokens, tp{n_ranks}) ===");
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    let cluster: Arc<HipCluster> =
        Arc::new(HipCluster::new(&(0..n_ranks).collect::<Vec<_>>())?);
    let layout = Qwen35DenseTpLayout::new(&cfg, n_ranks as u32)?;
    eprintln!("loading TP model across {} ranks…", cluster.ranks());
    let mut model = Qwen3MoETpModel::load(&base_file, &cluster, layout)?;
    if model.config.context_length > ctx_cap {
        eprintln!("clamping ctx {} → {}", model.config.context_length, ctx_cap);
        model.config.context_length = ctx_cap;
    }
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))
        .context("BarP2pAllReduce::new")?;

    // MTP head + LM head live on rank 0 (replicated output.weight in TP).
    let head_rank = 0usize;
    let head_device = cluster.device(head_rank);
    let head_shard = &model.shards[head_rank];
    let output_norm = &head_shard.output_norm;
    let lm_head = head_shard
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("no lm_head on rank 0"))?;

    eprintln!("loading MTP on rank {head_rank}…");
    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, head_device)?;
    head_device.bind()?;
    let mtp_scratch = MtpForwardScratch::new(head_device, &model.config)?;

    // ── Branch A: baseline single-token decode loop ──
    {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut decode_scratch =
            ShardedForwardOneTokenScratchTp::new(&model.config, &cluster)?;
        let mut last_token = 0u32;
        for (i, &tok) in PROMPT_IDS.iter().enumerate() {
            let mut logits = Vec::new();
            forward_one_token_tp_logits(
                &model, &mut decode_scratch, &cluster, &ar,
                &mut session.caches, tok, i, &mut logits,
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
            forward_one_token_tp_logits(
                &model, &mut decode_scratch, &cluster, &ar,
                &mut session.caches, last_token, position, &mut logits,
            )?;
            let next = argmax(&logits);
            tokens.push(next);
            last_token = next;
        }
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\n[A] baseline tp{n_ranks}: {} tokens, wall {:.0} ms, {:.2} ms/tok",
            n_tokens - 1, wall, wall / (n_tokens - 1) as f64
        );
        eprintln!("    first 16: {:?}", &tokens[..tokens.len().min(16)]);

        decode_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    // ── Branch B: spec-decode (2× L=1 verify, full L=1 redo on reject) ──
    {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut decode_scratch =
            ShardedForwardOneTokenScratchTp::new(&model.config, &cluster)?;
        let mut last_token = 0u32;
        for (i, &tok) in PROMPT_IDS.iter().enumerate() {
            let mut logits = Vec::new();
            forward_one_token_tp_logits(
                &model, &mut decode_scratch, &cluster, &ar,
                &mut session.caches, tok, i, &mut logits,
            )?;
            if i == PROMPT_IDS.len() - 1 {
                last_token = argmax(&logits);
            }
        }
        let mut h_for_mtp = decode_scratch.per_rank[head_rank].hidden_a;
        let mut position = PROMPT_IDS.len();
        let mut tokens: Vec<u32> = vec![last_token];
        let mut accept_count = 0usize;
        let mut macro_count = 0usize;
        let t0 = std::time::Instant::now();
        while tokens.len() < n_tokens {
            let (s, h_next) = forward_speculative_tp_step(
                &model, &mut session, &cluster, &ar, &mut decode_scratch,
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
        let accept_pct = 100.0 * accept_count as f64 / macro_count.max(1) as f64;
        eprintln!(
            "\n[B] spec tp{n_ranks}: {n_decoded} tokens via {macro_count} macros, \
             accept {accept_pct:.1}%, wall {wall:.0} ms, {:.2} ms/tok",
            wall / n_decoded.max(1) as f64
        );
        eprintln!("    first 16: {:?}", &tokens[..tokens.len().min(16)]);

        decode_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    head_device.bind()?;
    mtp_scratch.dispose(head_device).ok();
    Ok(())
}

use anyhow::Context;
