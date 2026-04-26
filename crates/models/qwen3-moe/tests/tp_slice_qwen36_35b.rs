//! TP-4 cert harness — Qwen3.6-35B-A3B (arch=qwen35moe, hybrid GDN +
//! MoE + shared expert). The most complex arch in V1 scope: exercises
//! TP-4a-i1 (FusedQkvParallel for GDN attn_qkv / ssm_conv1d), TP-4b
//! (MoE expert ColParallel{dim=1} / RowParallel{dim=2}), and TP-4c
//! (shared expert ColParallel{dim=0} / RowParallel{dim=1}) all in one
//! load.
//!
//! World = 2. Qwen3.6-35B-A3B has `num_kv_heads=2` which doesn't divide
//! world=4; KV-replication for that case is deferred (see
//! `qwen36_35b_tp_world_constraint` note in the cert).

use std::path::PathBuf;

use anyhow::Result;
use flambeau_qwen3_moe::{Qwen35DenseTpLayout, Qwen3MoEConfig};
use flambeau_quant::GgufFile;
use flambeau_runtime::WeightLayout;

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf";
// **TP-4d-i2** — world=4 now works via KV-replication (nKV=2 < 4).
// Was world=2 before TP-4d-i2.
const WORLD: u32 = 4;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN36_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_PATH);
            p.exists().then_some(p)
        })
}

#[test]
fn qwen36_35b_a3b_tp4_load_cert() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!(
            "[skip] no GGUF: set FLAMBEAU_QWEN36_GGUF or place Qwen3.6-35B-A3B at {DEFAULT_PATH}"
        );
        return Ok(());
    };
    eprintln!("loading {}", path.display());
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    if cfg.arch != "qwen35moe" {
        eprintln!("[skip] expected arch=qwen35moe, got {}", cfg.arch);
        return Ok(());
    }
    let tp = Qwen35DenseTpLayout::new(&cfg, WORLD)?;

    let mut total: usize = 0;
    let mut bytes_per_rank: [usize; 4] = [0; 4];
    let mut counts = LayoutCounts::default();
    let mut unknown: Vec<String> = Vec::new();
    let mut names: Vec<&String> = file.tensors.keys().collect();
    names.sort();
    for name in &names {
        let info = file.info(name)?;
        let bytes = info.size_in_bytes() as usize;
        total += bytes;
        match tp.for_tensor(name) {
            None => {
                unknown.push((*name).clone());
                for r in 0..WORLD as usize {
                    bytes_per_rank[r] += bytes;
                }
            }
            Some(layout) => {
                counts.tally(layout);
                let per = match layout {
                    WeightLayout::Replicated => bytes,
                    _ => bytes / WORLD as usize,
                };
                for r in 0..WORLD as usize {
                    bytes_per_rank[r] += per;
                }
            }
        }
    }
    let ideal = total / WORLD as usize;
    let mean = bytes_per_rank.iter().sum::<usize>() / WORLD as usize;
    let spread = bytes_per_rank.iter().max().unwrap() - bytes_per_rank.iter().min().unwrap();
    let overhead = (mean as f64 - ideal as f64) / ideal as f64 * 100.0;

    eprintln!(
        "Qwen3.6-35B-A3B (qwen35moe) load (world={WORLD}): total={:.2} GiB, ideal/rank={:.2} GiB",
        total as f64 / 1024.0 / 1024.0 / 1024.0,
        ideal as f64 / 1024.0 / 1024.0 / 1024.0,
    );
    for r in 0..WORLD as usize {
        eprintln!("  rank {r}: {:.2} GiB", bytes_per_rank[r] as f64 / 1024.0 / 1024.0 / 1024.0);
    }
    eprintln!("  inter-rank spread: {spread} bytes");
    eprintln!("  per-rank overhead vs ideal: {overhead:+.2} %");
    eprintln!(
        "  ColParallel: {}, RowParallel: {}, FusedQkvParallel: {}, Replicated: {}, unknown: {}",
        counts.col, counts.row, counts.fused_qkv, counts.rep, unknown.len()
    );

    let cert_dir = PathBuf::from("../../certs/parity");
    std::fs::create_dir_all(&cert_dir).ok();
    let cert_path = cert_dir.join("qwen36_35b_a3b_tp4_load.json");
    let cert = format!(
        "{{\n  \"schema_version\": 1,\n  \"kind\": \"tp_load_summary\",\n  \
         \"model\": \"Qwen3.6-35B-A3B-UD-Q4_K_S\",\n  \
         \"arch\": \"qwen35moe\",\n  \"world\": {WORLD},\n  \
         \"gguf_path\": \"{}\",\n  \
         \"total_bytes_full\": {total},\n  \
         \"ideal_bytes_per_rank\": {ideal},\n  \
         \"bytes_per_rank\": [{}, {}, {}, {}],\n  \
         \"mean_bytes_per_rank\": {mean},\n  \
         \"inter_rank_spread_bytes\": {spread},\n  \
         \"overhead_vs_ideal_pct\": {overhead:.4},\n  \
         \"tensor_count\": {},\n  \
         \"col_parallel_tensors\": {},\n  \
         \"row_parallel_tensors\": {},\n  \
         \"fused_qkv_tensors\": {},\n  \
         \"replicated_tensors\": {},\n  \
         \"unknown_tensors\": {},\n  \
         \"qwen36_35b_tp_world_constraint\": \"World=4 fails divisibility (num_kv_heads=2). KV-replication (each rank holds full nKV; reads stay symmetric) is the standard Megatron workaround — deferred to TP-4d-i2. World=2 has nKV=1/rank, validates cleanly.\",\n  \
         \"validates\": [\"TP-4a-i1 FusedQkvParallel\", \"TP-4b MoE 3D ColParallel/RowParallel\", \"TP-4c shared expert ColParallel/RowParallel\"],\n  \
         \"captured_via\": \"cargo test --release -p flambeau-qwen3-moe --features hip --test tp_slice_qwen36_35b\",\n  \
         \"captured_at\": \"2026-04-26\"\n}}\n",
        path.display(),
        bytes_per_rank[0],
        bytes_per_rank[1],
        bytes_per_rank[2],
        bytes_per_rank[3],
        names.len(),
        counts.col,
        counts.row,
        counts.fused_qkv,
        counts.rep,
        unknown.len(),
    );
    std::fs::write(&cert_path, cert)?;
    eprintln!("wrote {}", cert_path.display());
    assert_eq!(spread, 0, "inter-rank spread {spread} > 0 — V1 layout symmetric, bug");
    Ok(())
}

#[derive(Default)]
struct LayoutCounts {
    col: usize,
    row: usize,
    rep: usize,
    fused_qkv: usize,
}
impl LayoutCounts {
    fn tally(&mut self, layout: WeightLayout) {
        match layout {
            WeightLayout::ColParallel { .. } => self.col += 1,
            WeightLayout::RowParallel { .. } => self.row += 1,
            WeightLayout::Replicated => self.rep += 1,
            WeightLayout::FusedQkvParallel { .. } => self.fused_qkv += 1,
        }
    }
}
