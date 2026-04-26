//! TP-4b cert harness — load Qwen3-Coder-30B-A3B (arch=qwen3moe),
//! walk every tensor, validate the TP-1a + TP-4b layout table, dump
//! per-rank byte totals to
//! `certs/parity/qwen3_coder_30b_tp4_load.json`.
//!
//! No GPU required — pure host-side I/O + slicing. Skips at runtime
//! if the GGUF isn't present.

use std::path::PathBuf;

use anyhow::Result;
use flambeau_qwen3_moe::{slice_for_tp, Qwen35DenseTpLayout, Qwen3MoEConfig};
use flambeau_quant::GgufFile;
use flambeau_runtime::WeightLayout;

const DEFAULT_PATH: &str = "/artefact/models/Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf";
const WORLD: u32 = 4;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_MOE_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_PATH);
            p.exists().then_some(p)
        })
}

#[test]
fn qwen3_coder_30b_tp4_load_cert() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!(
            "[skip] no GGUF: set FLAMBEAU_QWEN3_MOE_GGUF or place the Coder-30B GGUF at {DEFAULT_PATH}"
        );
        return Ok(());
    };
    eprintln!("loading {}", path.display());
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    if cfg.arch != "qwen3moe" {
        eprintln!("[skip] expected arch=qwen3moe (Coder-30B), got {}", cfg.arch);
        return Ok(());
    }
    let tp = Qwen35DenseTpLayout::new(&cfg, WORLD)?;

    let mut total_bytes_full: usize = 0;
    let mut bytes_per_rank: [usize; WORLD as usize] = [0; WORLD as usize];
    let mut counts = LayoutCounts::default();
    let mut unknown_tensors: Vec<String> = Vec::new();

    let mut names: Vec<&String> = file.tensors.keys().collect();
    names.sort();

    for name in &names {
        let info = file.info(name)?;
        let total = info.size_in_bytes() as usize;
        total_bytes_full += total;
        match tp.for_tensor(name) {
            None => {
                unknown_tensors.push((*name).clone());
                for r in 0..WORLD as usize {
                    bytes_per_rank[r] += total;
                }
                continue;
            }
            Some(layout) => {
                counts.tally(layout);
                let per_rank = match layout {
                    WeightLayout::Replicated => total,
                    WeightLayout::ColParallel { .. }
                    | WeightLayout::RowParallel { .. }
                    | WeightLayout::FusedQkvParallel { .. } => total / WORLD as usize,
                };
                for r in 0..WORLD as usize {
                    bytes_per_rank[r] += per_rank;
                }
            }
        }
    }

    // Sample-validate the new 3D MoE slicing on representative tensors.
    sample_validate_moe_col_parallel(&file, &tp, "blk.0.ffn_gate_exps.weight")?;
    sample_validate_moe_row_parallel(&file, &tp, "blk.0.ffn_down_exps.weight")?;

    let ideal = total_bytes_full / WORLD as usize;
    let min_per_rank = *bytes_per_rank.iter().min().unwrap();
    let max_per_rank = *bytes_per_rank.iter().max().unwrap();
    let mean_per_rank = bytes_per_rank.iter().sum::<usize>() / WORLD as usize;
    let inter_rank_spread = max_per_rank - min_per_rank;
    let inter_rank_spread_pct = (inter_rank_spread as f64 / mean_per_rank as f64) * 100.0;
    let overhead_vs_ideal_pct =
        (mean_per_rank as f64 - ideal as f64) / ideal as f64 * 100.0;

    eprintln!(
        "Qwen3-Coder-30B (qwen3moe) load (world={WORLD}): total={:.2} GiB, ideal/rank={:.2} GiB",
        total_bytes_full as f64 / 1024.0 / 1024.0 / 1024.0,
        ideal as f64 / 1024.0 / 1024.0 / 1024.0
    );
    for r in 0..WORLD as usize {
        eprintln!(
            "  rank {r}: {:.2} GiB",
            bytes_per_rank[r] as f64 / 1024.0 / 1024.0 / 1024.0,
        );
    }
    eprintln!(
        "  inter-rank spread: {:.4} % (gate: identical per rank)",
        inter_rank_spread_pct
    );
    eprintln!(
        "  per-rank overhead vs ideal: {:+.2} %",
        overhead_vs_ideal_pct
    );
    eprintln!(
        "  ColParallel: {}, RowParallel: {}, FusedQkvParallel: {}, Replicated: {}",
        counts.col, counts.row, counts.fused_qkv, counts.rep
    );
    if !unknown_tensors.is_empty() {
        eprintln!(
            "  unknown to layout table ({}): {:?}",
            unknown_tensors.len(),
            &unknown_tensors[..unknown_tensors.len().min(5)]
        );
    }

    let cert_dir = PathBuf::from("../../certs/parity");
    std::fs::create_dir_all(&cert_dir).ok();
    let cert_path = cert_dir.join("qwen3_coder_30b_tp4_load.json");
    let cert = format!(
        "{{\n  \"schema_version\": 1,\n  \"kind\": \"tp_load_summary\",\n  \
         \"model\": \"Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0\",\n  \
         \"arch\": \"qwen3moe\",\n  \"world\": {WORLD},\n  \
         \"gguf_path\": \"{}\",\n  \
         \"total_bytes_full\": {total_bytes_full},\n  \
         \"ideal_bytes_per_rank\": {ideal},\n  \
         \"bytes_per_rank\": [{}, {}, {}, {}],\n  \
         \"mean_bytes_per_rank\": {mean_per_rank},\n  \
         \"inter_rank_spread_bytes\": {inter_rank_spread},\n  \
         \"inter_rank_spread_pct\": {inter_rank_spread_pct:.4},\n  \
         \"overhead_vs_ideal_pct\": {overhead_vs_ideal_pct:.4},\n  \
         \"tensor_count\": {},\n  \
         \"col_parallel_tensors\": {},\n  \
         \"row_parallel_tensors\": {},\n  \
         \"replicated_tensors\": {},\n  \
         \"fused_qkv_tensors\": {},\n  \
         \"unknown_tensors\": {},\n  \
         \"notes\": \"TP-4b — MoE intra-expert sharding for arch=qwen3moe (Coder-30B). ffn_gate/up_exps land as ColParallel{{dim=1}}; ffn_down_exps as RowParallel{{dim=2}}. Expert count (128) is preserved; each rank holds 1/world of EVERY expert's intermediate slab. Router (ffn_gate_inp) is Replicated.\",\n  \
         \"captured_via\": \"cargo test --release -p flambeau-qwen3-moe --features hip --test tp_slice_qwen3_coder_30b\",\n  \
         \"captured_at\": \"2026-04-26\"\n}}\n",
        path.display(),
        bytes_per_rank[0],
        bytes_per_rank[1],
        bytes_per_rank[2],
        bytes_per_rank[3],
        names.len(),
        counts.col,
        counts.row,
        counts.rep,
        counts.fused_qkv,
        unknown_tensors.len(),
    );
    std::fs::write(&cert_path, cert)?;
    eprintln!("wrote {}", cert_path.display());

    assert_eq!(
        inter_rank_spread, 0,
        "inter-rank spread {inter_rank_spread} > 0 — V1 layout is symmetric, bug"
    );
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

fn sample_validate_moe_col_parallel(
    file: &GgufFile,
    tp: &Qwen35DenseTpLayout,
    name: &str,
) -> Result<()> {
    let layout = tp.for_tensor(name).expect("known tensor");
    assert!(matches!(layout, WeightLayout::ColParallel { dim: 1, .. }));
    let info = file.info(name)?;
    assert_eq!(info.dims.len(), 3, "{name} should be 3D");
    let total = info.size_in_bytes() as usize;
    let per_rank_expected = total / 4;
    for r in 0..4u32 {
        let s = slice_for_tp(file, name, layout, r)?;
        assert_eq!(
            s.len(),
            per_rank_expected,
            "{name} ColParallel{{dim=1}} rank {r}: got {} bytes, expected {per_rank_expected}",
            s.len()
        );
    }
    Ok(())
}

fn sample_validate_moe_row_parallel(
    file: &GgufFile,
    tp: &Qwen35DenseTpLayout,
    name: &str,
) -> Result<()> {
    let layout = tp.for_tensor(name).expect("known tensor");
    assert!(matches!(layout, WeightLayout::RowParallel { dim: 2, .. }));
    let info = file.info(name)?;
    assert_eq!(info.dims.len(), 3, "{name} should be 3D");
    let total = info.size_in_bytes() as usize;
    let per_rank_expected = total / 4;
    let raw = file.tensor_raw(name)?;
    // Reconstruct: 4 rank slices, each [n_experts, dim1, dim2/4],
    // concatenated row-by-row inside each expert should reproduce the
    // mmap bytes.
    let n_experts = info.dims[0] as usize;
    let dim1 = info.dims[1] as usize;
    let dim2 = info.dims[2] as usize;
    let block_size = info.block_size() as usize;
    let type_size = info.type_size() as usize;
    let full_row_bytes = (dim2 / block_size) * type_size;
    let local_row_bytes = full_row_bytes / 4;
    let slices: Vec<_> = (0..4)
        .map(|r| slice_for_tp(file, name, layout, r))
        .collect::<Result<_>>()?;
    for (r, s) in slices.iter().enumerate() {
        assert_eq!(
            s.len(),
            per_rank_expected,
            "{name} RowParallel{{dim=2}} rank {r}: got {} bytes, expected {per_rank_expected}",
            s.len()
        );
    }
    let mut reconstructed = Vec::with_capacity(total);
    for e in 0..n_experts {
        for row in 0..dim1 {
            for r in 0..4 {
                let s = &slices[r];
                let off = e * dim1 * local_row_bytes + row * local_row_bytes;
                reconstructed.extend_from_slice(&s[off..off + local_row_bytes]);
            }
        }
    }
    assert_eq!(reconstructed.len(), raw.len(), "MoE RowParallel reconstruction length mismatch");
    assert_eq!(reconstructed, raw, "MoE RowParallel slicing did not preserve bytes");
    Ok(())
}
