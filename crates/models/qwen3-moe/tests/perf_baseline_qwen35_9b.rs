//! V2.2.c.2 — perf baseline for Qwen3.5-9B-Q4_1 (arch=qwen35 dense-hybrid).
//!
//! First flambeau perf measurement on a model that **fits one MI50**.
//! Exercises the V2.2.b Q4_1 MMVQ kernel + V2.2.c forward_dense_ffn_decode
//! at prefill and decode. Writes
//! `certs/perf/qwen35_9b_q4_1_mesh{N}.json` per run.
//!
//! Rank count is controlled by `FLAMBEAU_MESH_RANKS`; loop over 1/2/4 via
//! a shell loop (one `cargo test` invocation per mesh size).

#![cfg(feature = "hip")]

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

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from("/artefact/models/Qwen3.5-9B-Q4_1.gguf")))
        .filter(|p| p.exists())
}

fn workspace_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

#[test]
fn perf_baseline_qwen35_9b() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let n_requested: i32 = std::env::var("FLAMBEAU_MESH_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    if n_requested <= 0 || n_requested > n_available {
        eprintln!("FLAMBEAU_MESH_RANKS={n_requested} unavailable (have {n_available}) — skipping");
        return Ok(());
    }
    let n = n_requested;

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!(
        "perf-baseline: Qwen3.5-9B-Q4_1 across Mesh<{}> ({} layers)…",
        cluster.ranks(),
        cfg.num_layers
    );
    let load_start = Instant::now();
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    let load_dt = load_start.elapsed();
    eprintln!(
        "load: {:.2}s ({:.2} GiB total)",
        load_dt.as_secs_f64(),
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Warm the pipeline.
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let _ = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &[9419], 0)?;
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();

    // V2.3.c.1: optional opt-out of prefill phase for rocprofv3 decode-only
    // profiling. When set, the test skips the prefill-grid loop and only
    // runs decode — gives a clean per-kernel trace of the decode hot path.
    let decode_only = std::env::var("FLAMBEAU_DECODE_ONLY").is_ok();

    if !decode_only {
    // V2.24.b.1 — optional external ubatch chunking via FLAMBEAU_UBATCH env.
    // V2.25.d — FLAMBEAU_ASYNC_UBATCH=1 additionally routes forward_prefill_pp
    // through the async path (per-rank aux streams + event-bridged peer
    // copies + per-lane scratch). Requires FLAMBEAU_UBATCH set to the
    // ubatch size, otherwise falls back to the sync single-batch path.
    let ubatch: Option<usize> = std::env::var("FLAMBEAU_UBATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&u| u > 0);
    let async_enabled = std::env::var("FLAMBEAU_ASYNC_UBATCH").is_ok();
    let u_lanes: usize = std::env::var("FLAMBEAU_U_LANES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if async_enabled { 2 } else { 1 });
    for &l in &[8usize, 64, 128, 512, 1024] {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let scratch_size = ubatch.map(|u| u.min(l)).unwrap_or(l);
        let mut scratch = flambeau_qwen3_moe::forward::ShardedForwardPrefillScratch::new_with_lanes(
            &model, &cluster, scratch_size, u_lanes,
        )?;
        let tokens: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();

        let t0 = Instant::now();
        let last_id = if async_enabled && u_lanes >= 2 {
            // forward_prefill_pp itself branches into forward_prefill_pp_async
            // when FLAMBEAU_ASYNC_UBATCH is set and u_lanes >= 2.
            forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &tokens, 0)?
        } else if let Some(u) = ubatch {
            let mut pos = 0;
            let mut id = 0;
            for chunk in tokens.chunks(u) {
                id = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, chunk, pos)?;
                pos += chunk.len();
            }
            id
        } else {
            forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &tokens, 0)?
        };
        let dt = t0.elapsed().as_secs_f64();
        let tps = l as f64 / dt;
        eprintln!(
            "  prefill L={l:<4} → {:.2} tok/s  ({:.1} ms total, last_id={last_id})",
            tps, dt * 1000.0
        );
        results.push(("prefill".into(), l, dt, tps));

        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }
    }

    for &tg in &[64usize] {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

        let seed = forward_prefill_pp(
            &model,
            &mut session,
            &cluster,
            &mut prefill_scratch,
            &[9419u32],
            0,
        )?;

        let mut next = seed;
        let t0 = Instant::now();
        for step in 0..tg {
            next = forward_one_token_pp(
                &model,
                &mut session,
                &cluster,
                &mut decode_scratch,
                next,
                1 + step,
            )?;
        }
        let dt = t0.elapsed().as_secs_f64();
        let tps = tg as f64 / dt;
        eprintln!(
            "  decode  tg={tg:<4} → {:.2} tok/s  ({:.1} ms total, last_id={next})",
            tps,
            dt * 1000.0
        );
        results.push(("decode".into(), tg, dt, tps));

        decode_scratch.dispose(&cluster)?;
        prefill_scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    let out_dir = workspace_root().join("certs").join("perf");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(format!("qwen35_9b_q4_1_mesh{}.json", cluster.ranks()));
    let json = serde_json::json!({
        "model_tag": "Qwen3.5-9B-Q4_1",
        "mesh_ranks": cluster.ranks(),
        "gguf_bytes": model.total_bytes(),
        "load_secs": load_dt.as_secs_f64(),
        "runs": results.iter().map(|(phase, n, dt, tps)| serde_json::json!({
            "phase": phase,
            "n_tokens": n,
            "wall_secs": dt,
            "tok_per_sec": tps,
        })).collect::<Vec<_>>(),
        "notes": "V2.2.d candle-port cycle (P8 + fixes 1/3/4) — Q4_1 MMQ DS4 kernel + \
                  Q5_K MMQ wave64 + gdn_alpha_beta batched across L + flash-tile \
                  attention prefill (BR=4 LDS-tiled v2 port). Prefill pp512 M1 \
                  evolved 272.92 (pre-port) → 627.38 (post-fix-1) → 652.85 (post-fix-3) \
                  → 672.01 (post-fix-4). Candle reference on same rig: 935 tok/s M1. \
                  Parity vs llama.cpp: first 3 tokens bit-exact [11, 353, 1044] \
                  (V2.2.c.1 diagnostic preserved). Remaining gap to candle \
                  attributable to Q4_1 MMQ per-call time (2.40 ms/call vs turbo \
                  1.86 ms/call) — queued as Fix 5 (multi-session pipelined-LDS port). \
                  Per-L prefill is fresh-session; decode is 1-token prefill + N greedy. \
                  Regenerate: `FLAMBEAU_MESH_RANKS={1,2,4} cargo test --release -p \
                  flambeau-qwen3-moe --features hip --test perf_baseline_qwen35_9b -- --nocapture`",
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&json)? + "\n")?;
    eprintln!("wrote snapshot → {}", out_path.display());

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
