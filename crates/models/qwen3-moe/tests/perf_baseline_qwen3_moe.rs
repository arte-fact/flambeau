//! V1.7.6 perf baseline for real Qwen3.6-35B-A3B on the available rig.
//!
//! Measures prefill throughput (pp tok/s) at a few chunk sizes and decode
//! throughput (tg tok/s) at fixed n_predict, then prints both as a one-line
//! summary per (mesh, phase, L). Writes a JSON snapshot to
//! `certs/perf/qwen3_6_35b_a3b_ud_q4_k_s_mesh{N}.json` for future regression.
//!
//! Rank count is controlled by `FLAMBEAU_MESH_RANKS` (default = all available
//! devices). Mesh\<2\> fits Qwen3.6-35B at ~9.7 GiB/rank. Mesh\<1\> needs a
//! smaller gated model (Qwen3.5-9B is arch `qwen35` dense hybrid — separate
//! loader work, tracked outside this test).
//!
//! Skipped when `FLAMBEAU_QWEN3_GGUF` is unset or `device_count() < ranks`.

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
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

fn card_vram_bytes(card: usize) -> Option<u64> {
    let s = std::fs::read_to_string(format!(
        "/sys/class/drm/card{card}/device/mem_info_vram_total"
    ))
    .ok()?;
    s.trim().parse::<u64>().ok()
}

fn workspace_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

#[test]
fn perf_baseline_qwen3_moe_mesh_all() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping perf baseline");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let n_requested: i32 = std::env::var("FLAMBEAU_MESH_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_available);
    if n_requested <= 0 || n_requested > n_available {
        eprintln!(
            "FLAMBEAU_MESH_RANKS={n_requested} unavailable (have {n_available} HIP devices) — skipping"
        );
        return Ok(());
    }
    let n = n_requested;

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let layout_bytes =
        flambeau_qwen3_moe::ModelLayout::from_gguf(&file, &cfg)?.total_bytes() as usize;
    let replicated = if cfg.tied_lm_head {
        file.info("token_embd.weight")
            .map(|i| i.size_in_bytes() as usize)
            .unwrap_or(0)
    } else {
        0
    };
    let need = layout_bytes + replicated;
    let total_vram: u64 = (0..n as usize).filter_map(card_vram_bytes).sum();
    if total_vram > 0 && (need as u64) + 2 * 1024 * 1024 * 1024 > total_vram {
        eprintln!(
            "skipping: model needs {:.2} GiB, {}-rank cluster has {:.2} GiB",
            need as f64 / (1024.0 * 1024.0 * 1024.0),
            n,
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        return Ok(());
    }

    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!(
        "perf-baseline: loading Qwen3.6-35B across {} ranks ({:.2} GiB weights)…",
        cluster.ranks(),
        need as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    let load_start = Instant::now();
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    let load_dt = load_start.elapsed();
    eprintln!("load: {:.2}s", load_dt.as_secs_f64());

    // Warm the pipeline once so first-launch kernel-load costs don't skew
    // the L=1 prefill timing. Tiny compute; KV/GDN state disposed with the
    // session immediately after.
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let _ = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &[9419], 0)?;
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();

    // Profiling modes:
    //   FLAMBEAU_PREFILL_ONLY=1  — only run prefill-grid loop, skip decode
    //   FLAMBEAU_PREFILL_L=<N>   — restrict prefill-grid to a single L
    let prefill_only = std::env::var("FLAMBEAU_PREFILL_ONLY").is_ok();
    let prefill_single_l: Option<usize> = std::env::var("FLAMBEAU_PREFILL_L")
        .ok()
        .and_then(|s| s.parse().ok());
    let prefill_grid: Vec<usize> = match prefill_single_l {
        Some(l) => vec![l],
        None => vec![8, 64, 128, 512, 1024],
    };

    // Prefill throughput at a few chunk sizes. Each run is a fresh session
    // (KV/GDN state starts zeroed) so per-L numbers aren't cross-contaminated
    // by history length effects.
    for l in &prefill_grid {
        let l = *l;
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, l)?;
        let tokens: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();

        let t0 = Instant::now();
        let _ = forward_prefill_pp(
            &model,
            &mut session,
            &cluster,
            &mut scratch,
            &tokens,
            0,
        )?;
        let dt = t0.elapsed().as_secs_f64();
        let tps = l as f64 / dt;
        eprintln!("  prefill L={l:<4} → {:.2} tok/s  ({:.1} ms total)", tps, dt * 1000.0);
        results.push(("prefill".into(), l, dt, tps));

        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    if prefill_only {
        // Skip decode in profiling mode.
        model.dispose(&cluster)?;
        cluster.dispose()?;
        return Ok(());
    }

    // Decode throughput: prefill 1 token, then N decode steps feeding the
    // argmax back. Matches how `flambeau serve` will run.
    for &tg in &[64usize] {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

        // Seed with token 9419 ("Hello") to match the parity cert's input.
        let seed = forward_prefill_pp(
            &model,
            &mut session,
            &cluster,
            &mut prefill_scratch,
            &[9419u32],
            0,
        )?;

        let mut next = seed;
        // Warm GPU out of idle DPM state before timing — first 8-16 decode
        // steps after `session.new()` can be up to 2× slower while clocks
        // ramp. 16-step warmup eliminates that.
        let warmup = 16usize;
        for step in 0..warmup {
            next = forward_one_token_pp(
                &model,
                &mut session,
                &cluster,
                &mut decode_scratch,
                next,
                /*position=*/ 1 + step,
            )?;
        }
        let t0 = Instant::now();
        for step in warmup..(warmup + tg) {
            next = forward_one_token_pp(
                &model,
                &mut session,
                &cluster,
                &mut decode_scratch,
                next,
                /*position=*/ 1 + step,
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

    // Snapshot to disk for future regression diffs.
    let out_dir = workspace_root().join("certs").join("perf");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(format!(
        "qwen3_6_35b_a3b_ud_q4_k_s_mesh{}.json",
        cluster.ranks()
    ));
    let json = serde_json::json!({
        "model_tag": "Qwen3.6-35B-A3B-UD-Q4_K_S",
        "mesh_ranks": cluster.ranks(),
        "gguf_bytes": need,
        "load_secs": load_dt.as_secs_f64(),
        "runs": results.iter().map(|(phase, n, dt, tps)| {
            serde_json::json!({
                "phase": phase,
                "n_tokens": n,
                "wall_secs": dt,
                "tok_per_sec": tps,
            })
        }).collect::<Vec<_>>(),
        "notes": "V2.10.b (Q4_K gate_up_tile8 inline-accumulator refactor — Scratch 156 → 48 B, kernel −15.5 %; V2.9.b down kernels at (64, 2)). Per-L prefill numbers are fresh-session (no history); decode is 1-token prefill + N greedy steps. tok/s excludes load time. Regenerate: `FLAMBEAU_QWEN3_GGUF=... cargo test --release -p flambeau-qwen3-moe --features hip --test perf_baseline_qwen3_moe -- --nocapture`",
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&json)? + "\n")?;
    eprintln!("wrote snapshot → {}", out_path.display());

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
