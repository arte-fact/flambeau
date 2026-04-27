//! TP-1c smoke — load Qwen3.5-27B-Q4_1 onto Mesh<4> through the
//! TP-aware `Qwen3MoETpModel::load` path. Runtime-skipped without 4
//! HIP devices or the GGUF.
//!
//! Invariants (mirrors TP-1b's host-only cert but exercises the
//! full alloc + upload pipeline):
//!   1. Per-rank shard byte total matches TP-1b's predicted
//!      `bytes_per_rank` (6.39 GiB on Qwen3.5-27B-Q4_1, world=4).
//!   2. Inter-rank spread = 0 bytes (V1 layout is symmetric).
//!   3. Every shard carries `cfg.num_layers` per-layer entries.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture: GGUF load + dispose only; dispose's unsafe \
              free is documented in tp_sharded.rs."
)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_qwen3_moe::{Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel};
use flambeau_quant::GgufFile;

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-27B-Q4_1.gguf";
const WORLD: u32 = 4;

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

#[test]
fn qwen35_27b_q4_1_tp4_load_smoke() -> Result<()> {
    match device_count() {
        Ok(n) if n >= 4 => (),
        Ok(n) => {
            eprintln!("[skip] need >= 4 HIP devices for tp4 load smoke (have {n})");
            return Ok(());
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return Ok(());
        }
    }
    let Some(path) = gguf_path() else {
        eprintln!(
            "[skip] no GGUF: set FLAMBEAU_QWEN3_GGUF or place Qwen3.5-27B-Q4_1.gguf at {DEFAULT_PATH}"
        );
        return Ok(());
    };

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let tp = Qwen35DenseTpLayout::new(&cfg, WORLD)?;
    let cluster = Arc::new(HipCluster::new(&[0, 1, 2, 3])?);
    if !cluster.peer_access_full() {
        eprintln!("[skip] BAR1 peer-access matrix not fully connected");
        return Ok(());
    }

    let model = Qwen3MoETpModel::load(&file, &cluster, tp)?;

    // Invariant 1+2: identical per-rank totals.
    let totals: Vec<usize> = (0..WORLD as usize)
        .map(|r| model.rank_bytes(r).expect("shard exists"))
        .collect();
    let min = *totals.iter().min().unwrap();
    let max = *totals.iter().max().unwrap();
    eprintln!(
        "Qwen3.5-27B-Q4_1 TP load: per-rank = [{:.2}, {:.2}, {:.2}, {:.2}] GiB (spread {} B)",
        totals[0] as f64 / 1024.0 / 1024.0 / 1024.0,
        totals[1] as f64 / 1024.0 / 1024.0 / 1024.0,
        totals[2] as f64 / 1024.0 / 1024.0 / 1024.0,
        totals[3] as f64 / 1024.0 / 1024.0 / 1024.0,
        max - min,
    );
    assert_eq!(min, max, "inter-rank spread > 0 — V1 TP layout is symmetric");

    // Invariant 3: every shard has `cfg.num_layers` entries.
    for (r, shard) in model.shards.iter().enumerate() {
        assert_eq!(
            shard.layers.len(),
            cfg.num_layers,
            "rank {r}: expected {} layers, got {}",
            cfg.num_layers,
            shard.layers.len()
        );
    }

    // Cleanup so the rig's VRAM frees before the next test.
    model.dispose(&cluster)?;
    Ok(())
}
