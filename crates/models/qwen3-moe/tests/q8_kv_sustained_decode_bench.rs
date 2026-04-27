//! V1-BENCH-#118 — Q8 KV sustained-decode bench.
//!
//! Per CLAUDE.md rule #6 the Q8 KV layout stays opt-in (FLAMBEAU_KV=q8);
//! this bench measures the HBM-bandwidth advantage so users have a
//! measured reason to opt in for long-context workloads.
//!
//! For each `ctx_size` the harness:
//!   1. Prefills `ctx_size` synthetic tokens (cache fills to that depth).
//!   2. Runs `TG_STEPS` decode steps measured under wall-clock (HIP
//!      synchronize at start + end).
//!   3. Repeats under both `FLAMBEAU_KV=f16` and `FLAMBEAU_KV=q8` on the
//!      same model in the same process.
//!
//! Reports tok/s per (kv_layout, ctx_size) and the Q8 / F16 ratio.
//! Q8's edge grows with ctx because attention_decode is HBM-bound on
//! the K/V cache fetch — shorter cache rows = less to read per token.
//!
//! Q8 prefill currently runs per-token (#116 fallback) — measured here
//! as a separate "prefill_warmup" timing for transparency. The user-
//! visible regime this bench targets is sustained generation after a
//! one-time prefill, where the prefill cost is amortised across many
//! decode tokens.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::{Device, Stream};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;
use std::time::Instant;

const TG_STEPS: usize = 64;
const SEED_TOKEN: u32 = 9419;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn parse_ctx_lengths() -> Vec<usize> {
    std::env::var("FLAMBEAU_BENCH_CTX_LENGTHS")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|t| t.trim().parse::<usize>().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![512, 2048, 8192])
}

#[derive(Debug)]
struct Run {
    layout: &'static str,
    ctx: usize,
    prefill_secs: f64,
    decode_secs: f64,
    tg_per_sec: f64,
}

fn run_one(
    kv_env: &'static str,
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    cfg: &Qwen3MoEConfig,
    ctx: usize,
) -> Result<Run> {
    // SAFETY: single-threaded test; this is the only way to thread the
    // KV layout into Qwen3MoEShardedSession::new without changing the
    // ctor signature for #118 — see #117's quality cert which uses the
    // same pattern.
    unsafe {
        std::env::set_var("FLAMBEAU_KV", kv_env);
    }

    let prompt: Vec<u32> = (0..ctx as u32).map(|i| (1 + i * 37) % 151000).collect();

    // 1. Prefill warmup — pre-fill the KV cache to `ctx` tokens.
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(model, cluster, ctx)?;
    let prefill_t = Instant::now();
    forward_prefill_pp(model, &mut session, cluster, &mut prefill_scratch, &prompt, 0)?;
    // Force a sync — last layer's argmax download isn't a full sync of
    // every rank's compute stream; insure barrier so the Instant captures
    // wall time, not pipeline issue time.
    for r in 0..cluster.ranks() {
        cluster.device(r).bind()?;
        cluster.device(r).default_stream().synchronize()?;
    }
    let prefill_secs = prefill_t.elapsed().as_secs_f64();
    prefill_scratch.dispose(cluster).ok();

    // 2. Sustained decode — TG_STEPS tokens.
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(model, cluster)?;
    let mut next = SEED_TOKEN;
    // Warmup one decode step (kernels JIT'd, caches warm).
    next = forward_one_token_pp(model, &mut session, cluster, &mut decode_scratch, next, ctx)?;
    for r in 0..cluster.ranks() {
        cluster.device(r).default_stream().synchronize()?;
    }
    let decode_t = Instant::now();
    for step in 0..TG_STEPS {
        next = forward_one_token_pp(
            model, &mut session, cluster, &mut decode_scratch, next, ctx + 1 + step,
        )?;
    }
    for r in 0..cluster.ranks() {
        cluster.device(r).default_stream().synchronize()?;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    let _ = next;
    decode_scratch.dispose(cluster).ok();
    session.dispose(cluster).ok();

    Ok(Run {
        layout: kv_env,
        ctx,
        prefill_secs,
        decode_secs,
        tg_per_sec: TG_STEPS as f64 / decode_secs,
    })
}

#[test]
fn q8_kv_sustained_decode_bench() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices — got {n}, skipping");
        return Ok(());
    }
    let ctx_lengths = parse_ctx_lengths();

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    let model_tag = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
    let mut runs: Vec<Run> = Vec::with_capacity(ctx_lengths.len() * 2);
    for &ctx in &ctx_lengths {
        eprintln!("\n=== ctx={ctx} ===");
        for layout in ["f16", "q8"] {
            let r = run_one(layout, &model, &cluster, &cfg, ctx)?;
            eprintln!(
                "  {} ctx={:>5}  prefill={:>6.2}s  decode={:>5.2}s  tg64={:>6.2} tok/s",
                r.layout, r.ctx, r.prefill_secs, r.decode_secs, r.tg_per_sec,
            );
            runs.push(r);
        }
    }

    // Pretty-print the F16 vs Q8 ratio per ctx + write a JSON cert.
    eprintln!("\n=== F16 vs Q8 sustained-decode summary ===");
    eprintln!(
        "{:>6}  {:>10}  {:>10}  {:>10}",
        "ctx", "f16 tg/s", "q8 tg/s", "q8/f16",
    );
    let mut summary_rows: Vec<serde_json::Value> = Vec::with_capacity(ctx_lengths.len());
    for &ctx in &ctx_lengths {
        let f16 = runs.iter().find(|r| r.layout == "f16" && r.ctx == ctx).unwrap();
        let q8 = runs.iter().find(|r| r.layout == "q8" && r.ctx == ctx).unwrap();
        let ratio = q8.tg_per_sec / f16.tg_per_sec;
        eprintln!(
            "{:>6}  {:>10.2}  {:>10.2}  {:>9.3}×",
            ctx, f16.tg_per_sec, q8.tg_per_sec, ratio,
        );
        summary_rows.push(serde_json::json!({
            "ctx": ctx,
            "f16_tg_per_sec": f16.tg_per_sec,
            "q8_tg_per_sec": q8.tg_per_sec,
            "q8_over_f16": ratio,
            "f16_prefill_secs": f16.prefill_secs,
            "q8_prefill_secs": q8.prefill_secs,
            "f16_decode_secs": f16.decode_secs,
            "q8_decode_secs": q8.decode_secs,
        }));
    }

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("workspace root above CARGO_MANIFEST_DIR")
        .to_path_buf();
    let cert_path = workspace_root
        .join("certs/perf/q8_kv_sustained_decode")
        .join(format!("{model_tag}.json"));
    std::fs::create_dir_all(cert_path.parent().unwrap())?;

    let cert = serde_json::json!({
        "schema_version": 1,
        "kind": "q8_kv_sustained_decode_bench",
        "model": model_tag,
        "ranks": n,
        "tg_steps": TG_STEPS,
        "seed_token": SEED_TOKEN,
        "ctx_lengths": ctx_lengths,
        "rig": "threadreaper-gfx906",
        "results": summary_rows,
    });
    std::fs::write(&cert_path, serde_json::to_string_pretty(&cert)? + "\n")?;
    eprintln!("\nWrote {}", cert_path.display());

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
