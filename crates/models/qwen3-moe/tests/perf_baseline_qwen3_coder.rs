//! V2.28.b-i5 — perf baseline for Qwen3-Coder-30B-A3B-Instruct
//! (arch=qwen3moe, pure MoE, UD-Q4_K_XL mixed-quant).

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
    std::env::var("FLAMBEAU_QWEN3_CODER_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            Some(std::path::PathBuf::from(
                "/artefact/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
            ))
        })
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
fn perf_baseline_qwen3_coder() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3-Coder GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let n_requested: i32 = std::env::var("FLAMBEAU_MESH_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    if n_requested <= 0 || n_requested > n_available {
        eprintln!("FLAMBEAU_MESH_RANKS={n_requested} unavailable");
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen3moe");

    let cluster = HipCluster::new(&(0..n_requested).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!(
        "perf-baseline: Qwen3-Coder-30B-A3B-Instruct across Mesh<{}> ({} layers)…",
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

    // Warmup.
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let _ = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &[9419], 0)?;
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    let ubatch: Option<usize> = std::env::var("FLAMBEAU_UBATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&u| u > 0);
    let async_enabled = std::env::var("FLAMBEAU_ASYNC_UBATCH").is_ok();
    let u_lanes: usize = std::env::var("FLAMBEAU_U_LANES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if async_enabled { 2 } else { 1 });

    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();
    let decode_only = std::env::var("FLAMBEAU_DECODE_ONLY").is_ok();

    if !decode_only {
        for &l in &[8usize, 128, 512, 1024, 2048, 4096, 8192] {
            let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
            let scratch_size = ubatch.map(|u| u.min(l)).unwrap_or(l);
            let mut scratch = ShardedForwardPrefillScratch::new_with_lanes(
                &model, &cluster, scratch_size, u_lanes,
            )?;
            let tokens: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();
            let t0 = Instant::now();
            let last_id = if async_enabled && u_lanes >= 2 {
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
            eprintln!("  prefill L={l:<4} → {tps:.2} tok/s  ({:.1} ms, last_id={last_id})", dt * 1000.0);
            results.push(("prefill".into(), l, dt, tps));
            scratch.dispose(&cluster)?;
            session.dispose(&cluster)?;
        }
    }

    for &tg in &[64usize] {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
        let seed = forward_prefill_pp(&model, &mut session, &cluster, &mut prefill_scratch, &[9419u32], 0)?;
        let mut next = seed;
        let t0 = Instant::now();
        for step in 0..tg {
            next = forward_one_token_pp(&model, &mut session, &cluster, &mut decode_scratch, next, 1 + step)?;
        }
        let dt = t0.elapsed().as_secs_f64();
        let tps = tg as f64 / dt;
        eprintln!("  decode  tg={tg:<4} → {tps:.2} tok/s  ({:.1} ms, last_id={next})", dt * 1000.0);
        results.push(("decode".into(), tg, dt, tps));
        decode_scratch.dispose(&cluster)?;
        prefill_scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    let out_dir = workspace_root().join("certs").join("perf");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(format!("qwen3_coder_30b_mesh{}.json", cluster.ranks()));
    let json = serde_json::json!({
        "model_tag": "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL",
        "arch": "qwen3moe",
        "mesh_ranks": cluster.ranks(),
        "gguf_bytes": model.total_bytes(),
        "load_secs": load_dt.as_secs_f64(),
        "u_lanes": u_lanes,
        "ubatch": ubatch,
        "async_enabled": async_enabled,
        "runs": results.iter().map(|(phase, n, dt, tps)| serde_json::json!({
            "phase": phase,
            "n_tokens": n,
            "wall_secs": dt,
            "tok_per_sec": tps,
        })).collect::<Vec<_>>(),
        "notes": "V2.28.b-i5 — first perf baseline for Qwen3-Coder-30B-A3B-Instruct \
                  (arch=qwen3moe). 100 W/GPU power cap. Dense attention + Q5_K \
                  indexed-MoE MMVQ (V2.28.b-i2). Regenerate: \
                  `FLAMBEAU_MESH_RANKS=4 cargo test --release -p flambeau-qwen3-moe \
                  --features hip --test perf_baseline_qwen3_coder -- --nocapture`",
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&json)? + "\n")?;
    eprintln!("wrote snapshot → {}", out_path.display());

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
