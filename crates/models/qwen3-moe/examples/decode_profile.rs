//! Minimal decode-only harness for perf profiling and A/B measurement.
//!
//! Loads the GGUF at `FLAMBEAU_QWEN3_GGUF` across `FLAMBEAU_DEVICES`
//! (comma-separated device IDs, e.g. `0,1`) or `FLAMBEAU_MESH_RANKS`
//! (take first N available), prefills 1 token, warms up 4 decode steps,
//! then runs `FLAMBEAU_PROFILE_STEPS` decode steps (default 32) in a
//! tight loop and exits. Warmup keeps first-launch kernel-load + cache-
//! cold costs out of the measured window.
//!
//! Optional output: `FLAMBEAU_RESULT_JSON=/path/to/out.json` dumps
//! `{devices, steps, wall_secs, tok_per_sec, last_id}` for programmatic
//! A/B comparison.
//!
//! Usage:
//!   FLAMBEAU_QWEN3_GGUF=... FLAMBEAU_DEVICES=0,1 \
//!     target/release/examples/decode_profile

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::var("FLAMBEAU_QWEN3_GGUF")
        .map(std::path::PathBuf::from)
        .expect("FLAMBEAU_QWEN3_GGUF must be set");
    let n_available: i32 = device_count().unwrap_or(0);
    let devices: Vec<i32> = match std::env::var("FLAMBEAU_DEVICES") {
        Ok(s) => s
            .split(',')
            .filter(|t| !t.is_empty())
            .map(|t| t.trim().parse().expect("FLAMBEAU_DEVICES: parse"))
            .collect(),
        Err(_) => {
            let n: i32 = std::env::var("FLAMBEAU_MESH_RANKS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(n_available);
            (0..n).collect()
        }
    };
    let steps: usize = std::env::var("FLAMBEAU_PROFILE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    eprintln!(
        "decode-profile: device_count={n_available}, using devices={devices:?}"
    );
    assert!(
        !devices.is_empty() && devices.iter().all(|&d| d < n_available),
        "invalid device list {devices:?} (have {n_available})"
    );

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&devices)?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!("decode-profile: loading {} ranks on {devices:?}…", cluster.ranks());
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    let mut session = Qwen3MoEShardedSession::new(&model, &cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    // Seed with "Hello" (token 9419 — matches parity cert).
    let mut next = forward_prefill_pp(
        &model,
        &mut session,
        &cluster,
        &mut prefill_scratch,
        &[9419u32],
        0,
    )?;

    // Warm the pipeline before the measured window so first-launch JIT +
    // cache-cold effects land outside the trace.
    for step in 0..4 {
        next = forward_one_token_pp(
            &model,
            &mut session,
            &cluster,
            &mut decode_scratch,
            next,
            1 + step,
        )?;
    }

    eprintln!("decode-profile: running {steps} measured steps…");
    let t0 = Instant::now();
    for step in 0..steps {
        next = forward_one_token_pp(
            &model,
            &mut session,
            &cluster,
            &mut decode_scratch,
            next,
            5 + step,
        )?;
    }
    let dt = t0.elapsed().as_secs_f64();
    let tps = steps as f64 / dt;
    eprintln!(
        "decode-profile: devices={devices:?} {steps} steps in {:.3} s → {:.2} tok/s  (last_id={next})",
        dt, tps,
    );

    if let Ok(out_path) = std::env::var("FLAMBEAU_RESULT_JSON") {
        let json = format!(
            "{{\"devices\":{:?},\"steps\":{},\"wall_secs\":{},\"tok_per_sec\":{},\"last_id\":{}}}\n",
            devices, steps, dt, tps, next,
        );
        std::fs::write(&out_path, json)?;
    }

    decode_scratch.dispose(&cluster)?;
    prefill_scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
