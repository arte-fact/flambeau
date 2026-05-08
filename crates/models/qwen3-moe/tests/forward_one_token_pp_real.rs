//! V1.7.5.J / V1.7.5.E — real-weight pipeline-parallel forward smoke.
//!
//! Loads the real Qwen3.6 GGUF via `Qwen3MoEShardedModel::load` across
//! `device_count()` ranks and runs `forward_one_token_pp` once. A success
//! is a finite `u32` lane index in `[0, vocab)` — we don't cert argmax
//! here (that's V1.7.4), only that the full real-weight PP path executes
//! without HIP errors or dtype mismatches.
//!
//! Skips when `FLAMBEAU_QWEN3_GGUF` is unset, fewer than 2 HIP devices
//! are available, or the cluster doesn't have enough aggregate VRAM.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{forward_one_token_pp, ShardedForwardOneTokenScratch};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;

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

#[test]
fn forward_one_token_pp_real_qwen3_moe() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping forward_one_token_pp_real_qwen3_moe");
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices for real-weight PP — got {n}, skipping");
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let layout = flambeau_qwen3_moe::ModelLayout::from_gguf(&file, &cfg)?;
    let layout_bytes = layout.total_bytes() as usize;

    let total_vram: u64 = (0..n as usize).filter_map(card_vram_bytes).sum();
    let token_embd_bytes = file
        .info("token_embd.weight")
        .map(|i| i.size_in_bytes() as usize)
        .unwrap_or(0);
    let replicated = if cfg.tied_lm_head { token_embd_bytes } else { 0 };
    let need = layout_bytes + replicated;
    if total_vram > 0 && (need as u64) + 2 * 1024 * 1024 * 1024 > total_vram {
        eprintln!(
            "skipping real-weight PP: model needs {:.2} GiB, cluster has {:.2} GiB",
            need as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        return Ok(());
    }

    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);

    eprintln!(
        "loading Qwen3 MoE across {} ranks ({} layers, {:.2} GiB)…",
        cluster.ranks(),
        cfg.num_layers,
        need as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    eprintln!(
        "load ok: {:.2} GiB across {} shards",
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        model.shards.len(),
    );

    let mut session = Qwen3MoEShardedSession::new(&model, &cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    // BOS-adjacent token id — any id in-range suffices; we're only checking
    // the path executes. Qwen3 vocab size is 151936; id 1 is safe.
    // Override via FLAMBEAU_BENCH_WARM_TOKEN for parity comparisons (B5 bisect:
    // set to 9419 to match the cross-model bench seed).
    let token_id: u32 = std::env::var("FLAMBEAU_BENCH_WARM_TOKEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let next = forward_one_token_pp(
        &model,
        &mut session,
        &cluster,
        &mut scratch,
        token_id,
        /*position=*/ 0,
    )?;

    assert!(
        (next as usize) < cfg.vocab_size,
        "next token {next} out of vocab range {}",
        cfg.vocab_size
    );
    eprintln!("forward_one_token_pp real-weight decode → next = {next}");

    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
