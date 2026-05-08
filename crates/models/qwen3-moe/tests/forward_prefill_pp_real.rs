//! V1.7.5.D real-weight smoke — pipeline-parallel prefill against the
//! real Qwen3.6-35B-A3B-UD-Q4_K_S GGUF across `device_count()` ranks.
//!
//! Two gates:
//!   1. PP prefill executes end-to-end without HIP errors.
//!   2. Argmax on the final token matches what llama.cpp produces from
//!      the same prompt — for seed 9419 ("Hello") at position 0 the
//!      expected greedy argmax is 11 (',').
//!
//! Skips when `FLAMBEAU_QWEN3_GGUF` unset, < 2 HIP devices, or cluster
//! doesn't have enough VRAM.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{forward_prefill_pp, ShardedForwardPrefillScratch};
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
fn forward_prefill_pp_real_qwen3_moe_hello() -> Result<()> {
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
            "skipping real-weight PP prefill: need {:.2} GiB, have {:.2} GiB",
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

    // Case 1: L=1 prefill on seed token 9419 should match the decode-path
    // parity cert (llama.cpp's argmax for position 0 = 11 ',').
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let tokens = vec![9419u32];
        let next = forward_prefill_pp(
            &model, &mut session, &cluster, &mut scratch, &tokens, 0,
        )?;
        eprintln!("prefill L=1 → next = {next} (llama.cpp reference: 11)");
        assert_eq!(next, 11, "PP prefill L=1 parity: expected 11 (','), got {next}");
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    // Case 2: L=2 prefill on [9419, 11]. The single-device decode parity
    // cert records llama.cpp's full 8-token greedy as [11, 271, 40, …]
    // for seed 9419. Prefilling [9419, 11] in one shot should produce the
    // same argmax as llama.cpp at position 1 — i.e. token #2 in the
    // decode sequence = 271 ('\n\n'). Fresh session so KV/GDN state
    // starts clean.
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 2)?;
        let tokens = vec![9419u32, 11];
        let next = forward_prefill_pp(
            &model, &mut session, &cluster, &mut scratch, &tokens, 0,
        )?;
        eprintln!("prefill L=2 → next = {next} (llama.cpp reference: 271)");
        assert_eq!(
            next, 271,
            "PP prefill L=2 parity: expected 271 ('\\n\\n'), got {next}"
        );
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
