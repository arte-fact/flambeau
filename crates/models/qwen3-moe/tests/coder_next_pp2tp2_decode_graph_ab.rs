//! CN-80B-20 — A/B hybrid pp2tp2 decode wall time:
//!   eager  (FLAMBEAU_DECODE_GRAPH unset) vs
//!   graph  (FLAMBEAU_DECODE_GRAPH=1, single shared `hipGraph_t` per stage
//!           via `hipStreamBeginCaptureToGraph`).
//!
//! Status (ROCm 7.1.1): graph mode is BLOCKED by a runtime bug —
//! multi-stream capture-to-shared-graph with cross-stream events
//! produces no usable graph from end-capture, and capture attempts
//! corrupt the streams for the rest of the process. This test thus
//! produces a clean A/B only with `FLAMBEAU_AB_ONLY=eager`. With
//! the default (eager + graph), the graph runs error out — that's
//! the documented finding, not a flake. See
//! `certs/perf/coder_next_80b_cn80b_20_hybrid_graph_capture.md`.
//!
//! Skipped if FLAMBEAU_QWEN3_GGUF unset or fewer than 4 HIP devices.

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
    use_graph: bool,
) -> Result<f64> {
    if use_graph {
        std::env::set_var("FLAMBEAU_DECODE_GRAPH", "1");
    } else {
        std::env::remove_var("FLAMBEAU_DECODE_GRAPH");
    }

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

    // Warmup decode (NOT timed) so JIT cache + graph capture (if on)
    // are paid before the timed window.
    let mut tok = SEED_TOKEN;
    tok = forward_one_token_hybrid(
        model, &mut scratch, global_cluster, stage_ars, &mut session,
        tok, prompt.len(),
    )?;
    // Second warmup: under graph mode, capture happens here on every
    // stage; replay starts from step 3.
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
fn coder_next_pp2tp2_decode_graph_ab() -> Result<()> {
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
    // pp2tp2: stage-major device order [0,2,1,3] avoids the 2↔3 BAR1
    // fault (project_rig_gpu23_link_fault.md).
    let devices = vec![0i32, 2, 1, 3];
    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    let model = Qwen3MoEHybridModel::load(&file, &devices, spec)?;
    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        stage_ars.push(BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))?);
    }
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&devices)?);

    // Run eager first, graph second. Run each twice and pick the min to
    // shake out variance.
    let only = std::env::var("FLAMBEAU_AB_ONLY").ok();
    let do_eager = only.as_deref() != Some("graph");
    let do_graph = only.as_deref() != Some("eager");
    let mut eager_runs = Vec::new();
    if do_eager {
        for run in 0..2 {
            eager_runs.push(run_decode(&model, &global_cluster, &stage_ars,
                &format!("eager run {run}"), false)?);
        }
    }
    let mut graph_runs = Vec::new();
    if do_graph {
        for run in 0..2 {
            graph_runs.push(run_decode(&model, &global_cluster, &stage_ars,
                &format!("graph run {run}"), true)?);
        }
    }
    if eager_runs.is_empty() || graph_runs.is_empty() {
        return Ok(());
    }

    let eager_min = eager_runs.iter().cloned().fold(f64::INFINITY, f64::min);
    let graph_min = graph_runs.iter().cloned().fold(f64::INFINITY, f64::min);
    let speedup = eager_min / graph_min;

    eprintln!("\n=== CN-80B-20 hybrid pp2tp2 decode A/B (N={N_DECODE}) ===");
    eprintln!("  eager min: {:.2} ms ({:.2} tok/s)",
        eager_min, N_DECODE as f64 / (eager_min / 1000.0));
    eprintln!("  graph min: {:.2} ms ({:.2} tok/s)",
        graph_min, N_DECODE as f64 / (graph_min / 1000.0));
    eprintln!("  graph/eager speedup: {:.3}× ({:+.1}%)",
        speedup, (speedup - 1.0) * 100.0);

    Ok(())
}
