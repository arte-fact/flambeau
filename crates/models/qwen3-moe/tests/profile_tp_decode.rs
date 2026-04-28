//! CN-80B-22 — TP decode per-layer-type profiling.
//!
//! Runs prefill at `FLAMBEAU_PROFILE_L` then `FLAMBEAU_PROFILE_TG` decode
//! steps with `flambeau_backend_hip::profile` enabled. Prints aggregate
//! per-section ms (full-attn vs GDN vs post-layer head). Designed for
//! Coder-Next on pp2tp2 / tp2.
//!
//! Env:
//!   FLAMBEAU_PROFILE_GGUF       — required, model path
//!   FLAMBEAU_PROFILE_TP_DEVICES — comma-separated, default 0,2,1,3
//!   FLAMBEAU_PROFILE_TP_SIZE    — TP world (1, 2, or 4); default 2
//!   FLAMBEAU_PROFILE_L          — prefill length (default 2048)
//!   FLAMBEAU_PROFILE_TG         — decode steps (default 64)
//!
//! Run:
//!   FLAMBEAU_PROFILE_GGUF=/artefact/models/Qwen3-Coder-Next-Q4_0.gguf \
//!   FLAMBEAU_PROFILE_TP_DEVICES=0,2,1,3 FLAMBEAU_PROFILE_TP_SIZE=2 \
//!   FLAMBEAU_PROFILE_L=2048 FLAMBEAU_PROFILE_TG=64 \
//!   cargo test --release -p flambeau-qwen3-moe --features hip \
//!     --test profile_tp_decode profile_tp_decode -- --ignored --nocapture

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_tp, forward_prefill_tp_logits, ShardedForwardOneTokenScratchTp,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession, Qwen35DenseTpLayout};
use std::sync::Arc;
use std::time::Instant;

const SEED_TOKEN: u32 = 1;

#[test]
#[ignore = "Profiling harness — runs only with FLAMBEAU_PROFILE_GGUF set."]
fn profile_tp_decode() -> Result<()> {
    let Some(path) = std::env::var("FLAMBEAU_PROFILE_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip — set FLAMBEAU_PROFILE_GGUF to a valid path");
        return Ok(());
    };

    let n_available: i32 = device_count().unwrap_or(0);
    let devices: Vec<i32> = std::env::var("FLAMBEAU_PROFILE_TP_DEVICES")
        .unwrap_or_else(|_| "0,2,1,3".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if devices.is_empty() || devices.iter().any(|&d| d >= n_available) {
        eprintln!(
            "skip — devices {devices:?} not all in [0,{n_available})"
        );
        return Ok(());
    }
    let tp_size: usize = std::env::var("FLAMBEAU_PROFILE_TP_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    if tp_size > devices.len() {
        eprintln!("skip — tp_size {tp_size} > devices {}", devices.len());
        return Ok(());
    }

    let prefill_l: usize = std::env::var("FLAMBEAU_PROFILE_L")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let decode_tg: usize = std::env::var("FLAMBEAU_PROFILE_TG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    // Use the first `tp_size` devices for a pure-TP cluster.
    let tp_devices: Vec<i32> = devices.iter().take(tp_size).copied().collect();

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    eprintln!(
        "=== profile_tp_decode === arch={} hidden={} layers={} vocab={} \
         tp_devices={:?} L={} tg={} path={}",
        cfg.arch,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.vocab_size,
        tp_devices,
        prefill_l,
        decode_tg,
        path.display()
    );

    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&tp_devices)?);
    let world = cluster.ranks() as u32;
    let layout = Qwen35DenseTpLayout::new(&cfg, world)?;
    let model = Qwen3MoETpModel::load(&file, &cluster, layout)?;
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))?;

    // Force batched prefill (default-on per CN-80B-15) and pin behaviour.
    std::env::set_var("FLAMBEAU_TP_BATCHED", "1");

    // Prefill — not profiled here; we want decode-only attribution.
    let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
    let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
    let prompt: Vec<u32> = vec![SEED_TOKEN; prefill_l];
    let mut logits = vec![0.0f32; cfg.vocab_size];

    let pf_start = Instant::now();
    forward_prefill_tp_logits(
        &model,
        &mut scratch,
        &cluster,
        &ar,
        &mut session.caches,
        &prompt,
        0,
        &mut logits,
    )?;
    let pf_secs = pf_start.elapsed().as_secs_f64();
    eprintln!("prefill L={prefill_l}: {pf_secs:.3}s ({:.1} tok/s)", prefill_l as f64 / pf_secs);

    // Warm-up decode (first call has cold-cache effects).
    forward_one_token_tp(
        &model,
        &mut scratch,
        &cluster,
        &ar,
        &mut session.caches,
        SEED_TOKEN,
        prefill_l,
    )?;

    // Now profile the next `decode_tg` steps.
    flambeau_backend_hip::profile::enable();
    let dec_start = Instant::now();
    let mut last = SEED_TOKEN;
    for i in 0..decode_tg {
        last = forward_one_token_tp(
            &model,
            &mut scratch,
            &cluster,
            &ar,
            &mut session.caches,
            last,
            prefill_l + 1 + i,
        )?;
    }
    let dec_secs = dec_start.elapsed().as_secs_f64();
    let stats = flambeau_backend_hip::profile::flush()?;

    eprintln!(
        "\ndecode tg={decode_tg} from ctx={prefill_l}: {dec_secs:.3}s ({:.1} tok/s)",
        decode_tg as f64 / dec_secs
    );
    eprintln!(
        "  per-token: {:.2} ms",
        dec_secs * 1000.0 / decode_tg as f64
    );

    eprintln!("\n=== per-section breakdown (sorted by total ms) ===");
    eprintln!(
        "{:<24}  {:>10}  {:>8}  {:>10}  {:>10}",
        "section", "total ms", "count", "mean ms", "ms/token"
    );
    let total_recorded: f32 = stats.iter().map(|s| s.total_ms).sum();
    for s in &stats {
        let per_tok = s.total_ms / decode_tg as f32;
        eprintln!(
            "{:<24}  {:>10.2}  {:>8}  {:>10.4}  {:>10.3}",
            s.name, s.total_ms, s.count, s.mean_ms, per_tok
        );
    }
    eprintln!("---");
    eprintln!(
        "{:<24}  {:>10.2}  {:>8}  {:>10}  {:>10.3}",
        "TOTAL_RECORDED",
        total_recorded,
        "-",
        "-",
        total_recorded / decode_tg as f32
    );

    scratch.dispose(&cluster).ok();
    session.dispose(&cluster).ok();
    model.dispose(&cluster).ok();
    drop(ar);
    drop(cluster);
    Ok(())
}
