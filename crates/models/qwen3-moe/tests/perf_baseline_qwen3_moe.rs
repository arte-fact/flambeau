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
    //   FLAMBEAU_TG_LEN=<N>      — override decode length (default 64)
    //   FLAMBEAU_LONG_TEXT=<N>   — long-text scenario: prefill N tokens, then
    //                              decode FLAMBEAU_TG_LEN tokens continuing
    //                              from position N (real chat latency).
    //                              Disables the prefill grid + short-decode
    //                              loop.
    let prefill_only = std::env::var("FLAMBEAU_PREFILL_ONLY").is_ok();
    let prefill_single_l: Option<usize> = std::env::var("FLAMBEAU_PREFILL_L")
        .ok()
        .and_then(|s| s.parse().ok());
    let long_text_prompt: Option<usize> = std::env::var("FLAMBEAU_LONG_TEXT")
        .ok()
        .and_then(|s| s.parse().ok());
    let tg_len: usize = std::env::var("FLAMBEAU_TG_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let prefill_grid: Vec<usize> = match (long_text_prompt, prefill_single_l) {
        (Some(_), _) => vec![],          // long-text handled below
        (None, Some(l)) => vec![l],
        (None, None) => vec![8, 64, 128, 512, 1024],
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

    // Long-text mode: prefill a big prompt, then decode FLAMBEAU_TG_LEN
    // tokens CONTINUING from position=prompt_len. Reports TTFT + per-step
    // decode latency histogram so we can see if decode slows down as the
    // KV cache fills up (= real chat scenario).
    if let Some(prompt_len) = long_text_prompt {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut prefill_scratch =
            ShardedForwardPrefillScratch::new(&model, &cluster, prompt_len)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

        let prompt_tokens: Vec<u32> =
            (0..prompt_len as u32).map(|i| (1 + i * 37) % 151000).collect();

        let t_prefill = Instant::now();
        let seed = forward_prefill_pp(
            &model,
            &mut session,
            &cluster,
            &mut prefill_scratch,
            &prompt_tokens,
            0,
        )?;
        let ttft = t_prefill.elapsed().as_secs_f64();
        let prefill_tps = prompt_len as f64 / ttft;
        eprintln!(
            "  long-text prefill L={prompt_len:<5} TTFT={:.2}s  ({:.2} tok/s)",
            ttft, prefill_tps
        );

        // Warm GPU for per-step timing. Warmup decode steps start at
        // position=prompt_len.
        let warmup = 16usize;
        let mut next = seed;
        for step in 0..warmup {
            next = forward_one_token_pp(
                &model,
                &mut session,
                &cluster,
                &mut decode_scratch,
                next,
                prompt_len + step,
            )?;
        }

        // Timed decode. Record per-step wall-clock so we can see degradation.
        let mut per_step = Vec::with_capacity(tg_len);
        for step in 0..tg_len {
            let t0 = Instant::now();
            next = forward_one_token_pp(
                &model,
                &mut session,
                &cluster,
                &mut decode_scratch,
                next,
                prompt_len + warmup + step,
            )?;
            per_step.push(t0.elapsed().as_secs_f64());
        }
        let total_decode: f64 = per_step.iter().sum();
        let tpot = total_decode / tg_len as f64;
        let decode_tps = tg_len as f64 / total_decode;
        eprintln!(
            "  long-text decode tg={tg_len:<4} total={:.2}s  TPOT={:.1} ms  ({:.2} tok/s)",
            total_decode, tpot * 1000.0, decode_tps
        );

        // Per-step latency summary: min / median / max + slice-of-4 means
        // (early, early-mid, late-mid, late). Detects KV-cache-growth
        // slowdown over the decode window.
        let mut sorted = per_step.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_ms = sorted[0] * 1000.0;
        let med_ms = sorted[sorted.len() / 2] * 1000.0;
        let max_ms = sorted[sorted.len() - 1] * 1000.0;
        eprintln!(
            "  per-step min/med/max = {:.1} / {:.1} / {:.1} ms",
            min_ms, med_ms, max_ms
        );
        let q = tg_len / 4;
        if q >= 1 {
            for (i, label) in ["early", "e-mid", "l-mid", "late"].iter().enumerate() {
                let slice = &per_step[i * q..(i + 1) * q];
                let s: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
                eprintln!("    {label}: {:.1} ms/tok ({:.1} tok/s)", s * 1000.0, 1.0 / s);
            }
        }

        results.push(("long-prefill".into(), prompt_len, ttft, prefill_tps));
        results.push(("long-decode".into(), tg_len, total_decode, decode_tps));

        decode_scratch.dispose(&cluster)?;
        prefill_scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;

        model.dispose(&cluster)?;
        cluster.dispose()?;
        return Ok(());
    }

    // Decode throughput: prefill 1 token, then N decode steps feeding the
    // argmax back. Matches how `flambeau serve` will run.
    for &tg in &[tg_len] {
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
