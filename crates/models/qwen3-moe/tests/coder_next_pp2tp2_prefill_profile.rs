//! V1-BENCH-CN-80B-11a — pp2tp2 prefill section profile.
//!
//! Drives `forward_prefill_hybrid_logits` (which routes to the AUTO-6e
//! batched driver) on Qwen3-Coder-Next-Q4_0 at L=512, with the
//! HipEvent profile timer enabled so the per-section marks added in
//! `forward_prefill_tp_batched_layers` (CN-80B-11a) are captured.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, profile, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::forward_prefill_hybrid_logits;
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen3MoEConfig, Qwen3MoEHybridModel, Qwen3MoEHybridSession,
    ShardedForwardOneTokenScratchHybrid,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const L: usize = 512;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn coder_next_pp2tp2_prefill_profile() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 4 {
        eprintln!("need ≥ 4 HIP devices for pp2tp2 — got {n}, skipping");
        return Ok(());
    }
    let file = GgufFile::open(&path)?;
    let _cfg = Qwen3MoEConfig::from_gguf(&file)?;
    // pp2tp2 = 2 stages × 2 ranks. Devices [0,2,1,3] per V1 rig topology
    // (avoids the 2↔3 BAR1 fault by keeping rank pairs [0,2] / [1,3]).
    let devices = vec![0i32, 2, 1, 3];
    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    let model = Qwen3MoEHybridModel::load(&file, &devices, spec)?;
    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        stage_ars.push(BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))?);
    }
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&devices)?);

    std::env::set_var("FLAMBEAU_TP_BATCHED", "1");

    // Warmup pass (timer disabled) — JIT cache + first-call overhead.
    {
        let mut session = Qwen3MoEHybridSession::new(&model)?;
        let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
        let mut logits = vec![0.0f32; model.config.vocab_size];
        let prompt: Vec<u32> = (0..L as u32).map(|i| (1 + i * 37) % 151000).collect();
        forward_prefill_hybrid_logits(
            &model, &mut scratch, &global_cluster, &stage_ars, &mut session,
            &prompt, 0, &mut logits,
        )?;
        scratch.dispose(&model)?;
        session.dispose(&model)?;
    }

    // Measured pass — timer on.
    profile::enable();
    let mut session = Qwen3MoEHybridSession::new(&model)?;
    let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    let mut logits = vec![0.0f32; model.config.vocab_size];
    let prompt: Vec<u32> = (0..L as u32).map(|i| (1 + i * 37) % 151000).collect();
    let t0 = Instant::now();
    forward_prefill_hybrid_logits(
        &model, &mut scratch, &global_cluster, &stage_ars, &mut session,
        &prompt, 0, &mut logits,
    )?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let stats = profile::flush()?;
    scratch.dispose(&model)?;
    session.dispose(&model)?;

    eprintln!(
        "\n=== Coder-Next pp2tp2 prefill L={L} ===\n  wall: {:.2} ms ({:.2} tok/s)",
        wall_ms,
        L as f64 / (wall_ms / 1000.0),
    );
    eprintln!(
        "{:<22}  {:>10}  {:>10}  {:>10}",
        "section", "total_ms", "count", "mean_ms",
    );
    let mut total = 0.0f32;
    for s in &stats {
        eprintln!(
            "{:<22}  {:>10.3}  {:>10}  {:>10.3}",
            s.name, s.total_ms, s.count, s.mean_ms,
        );
        total += s.total_ms;
    }
    eprintln!("{:<22}  {:>10.3}  attributed", "total", total);

    model.dispose()?;
    Ok(())
}
