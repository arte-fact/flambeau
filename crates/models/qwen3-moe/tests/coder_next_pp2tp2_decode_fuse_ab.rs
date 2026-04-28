//! CN-80B-19c/d — A/B hybrid pp2tp2 decode wall time:
//!   baseline  (FLAMBEAU_VARIANT=baseline) — unfused swiglu+quantize chain
//!   fused     (default)                   — `swiglu_f32_to_q8_1` fused kernel
//!
//! GDN tail (`forward/gdn.rs` step 14+15) and shared-expert decode
//! (`forward/moe.rs` step 4+5) both consume the new fused kernel,
//! saving 1 launch per layer per token across both paths. Coder-Next
//! has 24 GDN layers + 48 shared-expert layers per token forward.
//!
//! Skipped if FLAMBEAU_QWEN3_GGUF unset or fewer than 4 HIP devices.
//! Set `FLAMBEAU_FUSE_AB_ONLY=baseline|fused` to run only one side.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{forward_one_token_hybrid, forward_prefill_hybrid_logits};
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen3MoEConfig, Qwen3MoEHybridModel, Qwen3MoEHybridSession,
    ShardedForwardOneTokenScratchHybrid,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const PROMPT_LEN: usize = 32;
const N_DECODE: usize = 32;
const SEED_TOKEN: u32 = 9419;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn run_decode(
    model: &Qwen3MoEHybridModel,
    global_cluster: &HipCluster,
    stage_ars: &[BarP2pAllReduce],
    label: &str,
    variant: &str,
) -> Result<f64> {
    if variant == "baseline" {
        std::env::set_var("FLAMBEAU_VARIANT", "baseline");
    } else {
        std::env::remove_var("FLAMBEAU_VARIANT");
    }
    // Make sure graph capture stays off — CN-80B-20 path is broken
    // on ROCm 7.1.1 and corrupts streams.
    std::env::remove_var("FLAMBEAU_DECODE_GRAPH");

    let mut session = Qwen3MoEHybridSession::new(model)?;
    let mut scratch = ShardedForwardOneTokenScratchHybrid::new(model)?;
    let mut logits = vec![0.0f32; model.config.vocab_size];

    let prompt: Vec<u32> = (0..PROMPT_LEN as u32)
        .map(|i| (1 + i * 37) % 151000)
        .collect();
    forward_prefill_hybrid_logits(
        model, &mut scratch, global_cluster, stage_ars,
        &mut session, &prompt, 0, &mut logits,
    )?;

    // Two warmup decodes (NOT timed) to amortize JIT cache + first-call overhead.
    let mut tok = SEED_TOKEN;
    tok = forward_one_token_hybrid(
        model, &mut scratch, global_cluster, stage_ars, &mut session,
        tok, prompt.len(),
    )?;
    tok = forward_one_token_hybrid(
        model, &mut scratch, global_cluster, stage_ars, &mut session,
        tok, prompt.len() + 1,
    )?;

    let t0 = Instant::now();
    for step in 0..N_DECODE {
        tok = forward_one_token_hybrid(
            model, &mut scratch, global_cluster, stage_ars, &mut session,
            tok, prompt.len() + 2 + step,
        )?;
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let _ = tok;

    eprintln!(
        "[{label}] wall {:.2} ms, {:.2} tok/s ({} tok)",
        wall_ms,
        N_DECODE as f64 / (wall_ms / 1000.0),
        N_DECODE,
    );

    scratch.dispose(model)?;
    session.dispose(model)?;
    Ok(wall_ms)
}

#[test]
fn coder_next_pp2tp2_decode_fuse_ab() -> Result<()> {
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
    let devices = vec![0i32, 2, 1, 3];
    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    let model = Qwen3MoEHybridModel::load(&file, &devices, spec)?;
    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        stage_ars.push(BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))?);
    }
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&devices)?);

    let only = std::env::var("FLAMBEAU_FUSE_AB_ONLY").ok();
    let do_baseline = only.as_deref() != Some("fused");
    let do_fused = only.as_deref() != Some("baseline");

    let mut baseline = Vec::new();
    if do_baseline {
        for run in 0..2 {
            baseline.push(run_decode(&model, &global_cluster, &stage_ars,
                &format!("baseline run {run}"), "baseline")?);
        }
    }
    let mut fused = Vec::new();
    if do_fused {
        for run in 0..2 {
            fused.push(run_decode(&model, &global_cluster, &stage_ars,
                &format!("fused run {run}"), "fused")?);
        }
    }

    if baseline.is_empty() || fused.is_empty() {
        return Ok(());
    }
    let bmin = baseline.iter().cloned().fold(f64::INFINITY, f64::min);
    let fmin = fused.iter().cloned().fold(f64::INFINITY, f64::min);
    let speedup = bmin / fmin;

    eprintln!("\n=== CN-80B-19c/d hybrid pp2tp2 decode A/B (N={N_DECODE}) ===");
    eprintln!("  baseline min: {:.2} ms ({:.2} tok/s)",
        bmin, N_DECODE as f64 / (bmin / 1000.0));
    eprintln!("  fused    min: {:.2} ms ({:.2} tok/s)",
        fmin, N_DECODE as f64 / (fmin / 1000.0));
    eprintln!("  fused/baseline speedup: {:.3}× ({:+.1}%)",
        speedup, (speedup - 1.0) * 100.0);

    Ok(())
}
