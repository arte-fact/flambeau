//! TP-2e — naive-AR perf baseline for Qwen3.5-27B-Q4_1 (arch=qwen35) at world=4.
//!
//! Captures the pre-TP-3a baseline: 2 AllReduce launches per layer
//! (post-attn + post-FFN), no AR overlap with compute (caller syncs
//! all producer streams before the AR launch). Writes
//! `certs/perf/qwen35_27b_q4_1_tp4_decode_naive.json`.
//!
//! Numbers from the plan §4 + TP-0b's measured AR latency:
//!   - 64 layers × 2 AR/layer × 27 µs ≈ 3.5 ms of pure AR per token.
//!   - 60 tok/s budget = 16.6 ms/token; AR alone is 21% of budget.
//!   - mi50grad-equivalent reference: 56.3 tok/s on Qwen3.5-27B-GPTQ-Int4.
//!   - Plan target post-TP-3a: ≥ 50 tok/s. This naive-AR baseline is
//!     expected at 30–40 tok/s.
//!
//! Requires:
//!   - 4 HIP devices visible
//!   - `peer_access_full() == true` (TP-0a passing)
//!   - `/artefact/models/Qwen3.5-27B-Q4_1.gguf` (env override:
//!     `FLAMBEAU_QWEN3_GGUF`)
//!
//! Skips cleanly when any of the above are missing.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — load + forward + dispose; per-call unsafety \
              is documented in the corresponding TP-1c / TP-2a / TP-2d files."
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_qwen3_moe::forward::{forward_one_token_tp, ShardedForwardOneTokenScratchTp};
use flambeau_qwen3_moe::{
    Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession,
};
use flambeau_quant::GgufFile;

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-27B-Q4_1.gguf";
const WORLD: u32 = 4;
const TG: usize = 64;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_PATH);
            p.exists().then_some(p)
        })
}

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

#[test]
fn perf_baseline_qwen35_27b_q4_1_tp4_naive_ar() -> Result<()> {
    let n_available = device_count().unwrap_or(0);
    if n_available < WORLD as i32 {
        eprintln!("[skip] need {WORLD} HIP devices for tp4 perf baseline (have {n_available})");
        return Ok(());
    }
    let Some(path) = gguf_path() else {
        eprintln!(
            "[skip] no GGUF: set FLAMBEAU_QWEN3_GGUF or place Qwen3.5-27B-Q4_1.gguf at {DEFAULT_PATH}"
        );
        return Ok(());
    };

    eprintln!("loading {}", path.display());
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    if cfg.arch != "qwen35" {
        eprintln!("[skip] expected arch=qwen35, got {}", cfg.arch);
        return Ok(());
    }
    let tp = Qwen35DenseTpLayout::new(&cfg, WORLD)?;
    let cluster = Arc::new(HipCluster::new(&[0, 1, 2, 3])?);
    if !cluster.peer_access_full() {
        eprintln!("[skip] BAR1 peer-access matrix not fully connected");
        return Ok(());
    }

    // Load (TP-1c).
    let load_start = Instant::now();
    let model = Qwen3MoETpModel::load(&file, &cluster, tp)?;
    let load_dt = load_start.elapsed();
    eprintln!(
        "  load: {:.2}s ({:.2} GiB total)",
        load_dt.as_secs_f64(),
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Per-rank scratch (TP-2a) + per-rank session (TP-2e session).
    let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
    let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))?;

    // Warm pass — first token absorbs JIT / module-load costs.
    let warm_token: u32 = 9419;
    let _ = forward_one_token_tp(&model, &mut scratch, &cluster, &ar, &mut session.caches, warm_token, 0)?;

    // Timed decode loop.
    let mut next: u32 = warm_token;
    let t0 = Instant::now();
    for step in 0..TG {
        next = forward_one_token_tp(
            &model,
            &mut scratch,
            &cluster,
            &ar,
            &mut session.caches,
            next,
            1 + step,
        )?;
    }
    let dt = t0.elapsed().as_secs_f64();
    let tps = TG as f64 / dt;
    eprintln!(
        "  decode tg={TG} → {tps:.2} tok/s ({:.1} ms total, last_id={next})",
        dt * 1000.0
    );

    // Cert.
    let cert_dir = workspace_root().join("certs/perf");
    std::fs::create_dir_all(&cert_dir).ok();
    let cert_path = cert_dir.join("qwen35_27b_q4_1_tp4_decode_naive.json");
    let cert = format!(
        "{{\n  \"schema_version\": 1,\n  \"kind\": \"tp_decode_perf_naive_ar\",\n  \
         \"model\": \"Qwen3.5-27B-Q4_1\",\n  \"arch\": \"qwen35\",\n  \"world\": {WORLD},\n  \
         \"topology\": \"Tp\",\n  \"ar_pattern\": \"naive\",\n  \
         \"ar_per_layer\": 2,\n  \"layers\": {},\n  \
         \"ar_per_token\": {},\n  \
         \"tg\": {TG},\n  \"decode_tok_per_s\": {tps:.4},\n  \
         \"decode_total_ms\": {:.4},\n  \
         \"warm_token_id\": {warm_token},\n  \
         \"last_token_id\": {next},\n  \
         \"plan_baseline_min_tok_s\": 30.0,\n  \
         \"plan_baseline_max_tok_s\": 40.0,\n  \
         \"plan_post_tp_3a_target_tok_s\": 50.0,\n  \
         \"mi50grad_reference_tok_s\": 56.3,\n  \
         \"captured_via\": \"cargo test --release -p flambeau-qwen3-moe --features hip --test perf_baseline_qwen35_27b_tp -- --nocapture\",\n  \
         \"captured_at\": \"2026-04-26\",\n  \
         \"notes\": \"Naive-AR: 2 BarP2pAllReduce::residual_tp4 launches per layer (post-attn + post-ffn). All producer streams synchronously synced before each AR launch. TP-3a deferred-AR collapses these to 1 AR/layer.\"\n}}\n",
        cfg.num_layers,
        2 * cfg.num_layers,
        dt * 1000.0,
    );
    std::fs::write(&cert_path, cert)?;
    eprintln!("  wrote {}", cert_path.display());

    // Cleanup.
    session.dispose(&cluster)?;
    scratch.dispose(&cluster)?;
    model.dispose(&cluster)?;
    Ok(())
}
