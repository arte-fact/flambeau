//! V1.7.5.A smoke test — load a real GGUF across N ranks and assert each
//! rank holds only its assigned layers + the correct globals.
//!
//! Skips when `FLAMBEAU_QWEN3_GGUF` is unset OR device_count < 2 (the
//! single-device case is already covered by `load_weights.rs`). With the
//! 20 GB Qwen3.6-31B model across 4×16 GB cards, each rank gets ~5 GB of
//! weights — the whole point of V1.7.5 is to unblock this load.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel};
use flambeau_runtime::{LayerAssignment, RankId};

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
fn load_sharded_qwen3_moe_across_all_ranks() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping load_sharded_qwen3_moe_across_all_ranks");
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices for sharded smoke — got {n}, skipping");
        return Ok(());
    }
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let expected_layout_bytes =
        flambeau_qwen3_moe::ModelLayout::from_gguf(&file, &cfg)?.total_bytes() as usize;

    // VRAM pre-check: sum of VRAM across the ranks we intend to use must
    // be comfortably larger than the model. With tied LM head, rank 0 +
    // last rank each carry a copy of token_embd — budget extra.
    let total_vram: u64 = (0..n as usize)
        .filter_map(card_vram_bytes)
        .sum();
    let token_embd_bytes = file
        .info("token_embd.weight")
        .map(|i| i.size_in_bytes() as usize)
        .unwrap_or(0);
    let replicated_bytes = if cfg.tied_lm_head { token_embd_bytes } else { 0 };
    let need_bytes = expected_layout_bytes + replicated_bytes;
    if total_vram > 0 && (need_bytes as u64) + 1024 * 1024 * 1024 > total_vram {
        eprintln!(
            "skipping sharded upload: model needs {:.2} GiB (incl. replicated embd); \
             cluster has {:.2} GiB total VRAM — increase the cluster or try a smaller GGUF",
            need_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        return Ok(());
    }

    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);

    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Invariants:
    //   1. One shard per rank.
    //   2. Each shard holds exactly the layers assigned to its rank.
    //   3. Rank 0's shard holds `token_embd` (always); last rank's shard
    //      holds `output_norm` and (iff tied) `token_embd`.
    //   4. No layer lives on more than one rank.
    assert_eq!(model.shards.len(), cluster.ranks());

    let mut seen_layers: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (rank_idx, shard) in model.shards.iter().enumerate() {
        let expected_layers = assignment.layers_on(RankId(rank_idx as u32));
        let got_layers: Vec<usize> =
            shard.layers.iter().map(|l| l.layer_idx).collect();
        assert_eq!(
            got_layers, expected_layers,
            "rank {rank_idx}: expected layers {expected_layers:?}, got {got_layers:?}"
        );
        for &il in &got_layers {
            assert!(
                seen_layers.insert(il),
                "layer {il} appeared on more than one rank"
            );
        }
        let is_last = rank_idx == cluster.ranks() - 1;
        if rank_idx == 0 {
            assert!(shard.token_embd.is_some(), "rank 0 must hold token_embd");
        }
        if is_last {
            assert!(shard.output_norm.is_some(), "last rank must hold output_norm");
            if cfg.tied_lm_head {
                assert!(
                    shard.token_embd.is_some(),
                    "last rank must hold tied token_embd for the LM head"
                );
            } else {
                assert!(
                    shard.output.is_some(),
                    "last rank must hold explicit output.weight when not tied"
                );
            }
        } else if rank_idx != 0 {
            // Middle ranks hold no globals.
            assert!(
                shard.token_embd.is_none(),
                "rank {rank_idx} must not hold token_embd"
            );
            assert!(
                shard.output_norm.is_none(),
                "rank {rank_idx} must not hold output_norm"
            );
            assert!(
                shard.output.is_none(),
                "rank {rank_idx} must not hold output.weight"
            );
        }
        eprintln!(
            "rank {rank_idx}: {} layers, {:.2} GiB on-device",
            got_layers.len(),
            shard.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    assert_eq!(
        seen_layers.len(),
        cfg.num_layers,
        "some layers weren't assigned to any rank"
    );

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
