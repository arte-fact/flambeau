//! V1-BENCH-#114 — TP single-point profiling harness.
//!
//! Sister of `profile_point.rs` (PP-only). Runs exactly one prefill-L
//! plus optional decode-tg cycle on a TP topology, with a warmup pass
//! first. Designed to be wrapped under `rocprofv3 --kernel-trace
//! --stats` (which skips PMC finalize and so doesn't trip the multi-rank
//! SIGABRT documented in V2.30 memory).
//!
//! Env:
//!   FLAMBEAU_PROFILE_GGUF        — path to GGUF (required)
//!   FLAMBEAU_PROFILE_TP_DEVICES  — comma-separated device list (default "0,1")
//!   FLAMBEAU_PROFILE_L           — prefill L (default 512, 0 = skip)
//!   FLAMBEAU_PROFILE_TG          — decode steps (default 0 = skip)
//!   FLAMBEAU_TP_BATCHED          — set to 1 for batched TP prefill
//!                                   (AUTO-6 default; off here unless set)
//!
//! The harness times the prefill + decode wall-clock with `Instant`.
//! Per-kernel attribution comes from the rocprofv3 wrapper, not from
//! this binary.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_tp, forward_prefill_tp_logits, ShardedForwardOneTokenScratchTp,
};
use flambeau_qwen3_moe::{
    Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession,
};
use std::sync::Arc;
use std::time::Instant;

#[test]
fn profile_point_tp() -> Result<()> {
    let Some(path) = std::env::var("FLAMBEAU_PROFILE_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip — set FLAMBEAU_PROFILE_GGUF to a valid path");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let devices: Vec<i32> = std::env::var("FLAMBEAU_PROFILE_TP_DEVICES")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|t| t.trim().parse::<i32>().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![0, 1]);
    if devices.iter().any(|&d| d >= n_available) {
        eprintln!(
            "FLAMBEAU_PROFILE_TP_DEVICES={devices:?} but only {n_available} HIP devices available"
        );
        return Ok(());
    }

    let prefill_l: usize = std::env::var("FLAMBEAU_PROFILE_L")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let decode_tg: usize = std::env::var("FLAMBEAU_PROFILE_TG")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    let file = GgufFile::open(&path)?;
    let mut cfg = Qwen3MoEConfig::from_gguf(&file)?;
    // Honor FLAMBEAU_CTX_CAP so models with multi-100K native ctx don't OOM
    // KV-cache allocation when only a short profile sweep is wanted.
    if let Ok(cap_str) = std::env::var("FLAMBEAU_CTX_CAP") {
        if let Ok(cap) = cap_str.parse::<usize>() {
            if cap > 0 && cap < cfg.context_length {
                eprintln!(
                    "FLAMBEAU_CTX_CAP shrinking cfg.context_length from {} to {cap}",
                    cfg.context_length
                );
                cfg.context_length = cap;
            }
        }
    }
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&devices)?);
    let world = cluster.ranks() as u32;
    let layout = Qwen35DenseTpLayout::new(&cfg, world)?;
    let model = Qwen3MoETpModel::load(&file, &cluster, layout)?;
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))?;

    // Warmup
    {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
        let mut logits = vec![0.0f32; cfg.vocab_size];
        forward_prefill_tp_logits(
            &model, &mut scratch, &cluster, &ar, &mut session.caches,
            &[9419u32], 0, &mut logits,
        )?;
        scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    if prefill_l > 0 {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
        let mut logits = vec![0.0f32; cfg.vocab_size];
        let prompt: Vec<u32> = (0..prefill_l as u32).map(|i| (1 + i * 37) % 151000).collect();
        let t0 = Instant::now();
        forward_prefill_tp_logits(
            &model, &mut scratch, &cluster, &ar, &mut session.caches,
            &prompt, 0, &mut logits,
        )?;
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "TP prefill L={prefill_l} world={world}: {:.2} tok/s ({:.2} ms)",
            prefill_l as f64 / dt, dt * 1000.0,
        );
        scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    if decode_tg > 0 {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
        forward_one_token_tp(
            &model, &mut scratch, &cluster, &ar, &mut session.caches, 9419u32, 0,
        )?;
        let mut last = 9419u32;
        let t0 = Instant::now();
        for pos in 0..decode_tg {
            last = forward_one_token_tp(
                &model, &mut scratch, &cluster, &ar, &mut session.caches,
                last, 1 + pos,
            )?;
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("TP decode tg={decode_tg}: {:.2} tok/s", decode_tg as f64 / dt);
        scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
        let _ = last;
    }

    model.dispose(&cluster).ok();
    drop(ar);
    drop(cluster);
    Ok(())
}
