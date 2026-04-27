//! V1-BENCH-CN-80B-5 (perf iter 2) — section-level profile of
//! Coder-Next-80B pp4 decode using the new `flambeau_backend_hip::profile`
//! HipEvent-based section timer (#131).
//!
//! Workaround for the rocprofv3 4-rank SIGABRT (V2.30 + iter-1
//! finding): instead of per-kernel attribution we get per-section
//! attribution at the boundaries marked in `forward_one_token_pp_inner`:
//!
//!   step_start    → embed_done    : per-token embed (rank 0)
//!   embed_done    → stage_start   : peer-copy + embed→stage-0 sync
//!   stage_start   → stage_end     : per-rank layer chain wall
//!   stage_end     → output_head_start : final stage→output sync
//!   output_head_start → step_start (next) : output head + argmax
//!
//! Per-section times are aggregated across N_DECODE decode steps and
//! sorted by total_ms descending, so the hot section surfaces first.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, profile, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;
use std::time::Instant;

const N_DECODE: usize = 32;
const SEED_TOKEN: u32 = 9419;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn coder_next_iter2_profile() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices — got {n}, skipping");
        return Ok(());
    }
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Warmup + prefill (NOT instrumented — kernels JIT cached, prefill
    // is its own fish that #114 / iter-1 already covered analytically).
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
    let seed = forward_prefill_pp(
        &model, &mut session, &cluster, &mut prefill_scratch, &[SEED_TOKEN], 0,
    )?;
    prefill_scratch.dispose(&cluster).ok();

    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
    let mut tok = seed;
    // Warmup decode iter (not timed) so subsequent measured iters are
    // hot.
    tok = forward_one_token_pp(&model, &mut session, &cluster, &mut decode_scratch, tok, 1)?;

    // Enable the section timer THIS thread; runs N_DECODE timed iters.
    profile::enable();
    let t0 = Instant::now();
    for step in 0..N_DECODE {
        tok = forward_one_token_pp(
            &model, &mut session, &cluster, &mut decode_scratch, tok, 2 + step,
        )?;
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let stats = profile::flush()?;
    let _ = tok;

    eprintln!(
        "\n=== Coder-Next pp{} decode tg={N_DECODE} ===",
        cluster.ranks()
    );
    eprintln!(
        "  wall: {:.2} ms  ({:.2} tok/s)",
        wall_ms,
        N_DECODE as f64 / (wall_ms / 1000.0),
    );
    eprintln!(
        "{:<22}  {:>10}  {:>10}  {:>10}",
        "section", "total_ms", "count", "mean_ms",
    );
    let mut total: f32 = 0.0;
    for s in &stats {
        eprintln!(
            "{:<22}  {:>10.3}  {:>10}  {:>10.3}",
            s.name, s.total_ms, s.count, s.mean_ms,
        );
        total += s.total_ms;
    }
    eprintln!(
        "{:<22}  {:>10.3}  (sum of attributed sections; some kernels run \
         async on aux streams and aren't captured)",
        "total_attributed", total,
    );

    decode_scratch.dispose(&cluster).ok();
    session.dispose(&cluster).ok();
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
