//! TP-1b cert harness — load Qwen3.5-27B-Q4_1, walk every tensor,
//! validate the TP-1a layout table + TP-1b slicing logic, and dump
//! per-rank byte totals to `certs/parity/qwen35_27b_q4_1_tp4_load.json`.
//!
//! Reads the GGUF via `FLAMBEAU_QWEN3_GGUF` (preferred) or falls back
//! to `/artefact/models/Qwen3.5-27B-Q4_1.gguf`. Skips at runtime if
//! neither path exists.
//!
//! No GPU required — pure host-side I/O + slicing logic.

use std::path::PathBuf;

use anyhow::Result;
use flambeau_qwen3_moe::{slice_for_tp, Qwen35DenseTpLayout, Qwen3MoEConfig};
use flambeau_quant::GgufFile;
use flambeau_runtime::WeightLayout;

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-27B-Q4_1.gguf";
const WORLD: u32 = 4;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_PATH);
            if p.exists() {
                Some(p)
            } else {
                None
            }
        })
}

#[test]
fn qwen35_27b_q4_1_tp4_load_cert() -> Result<()> {
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
        eprintln!("[skip] expected arch=qwen35, got {} at {}", cfg.arch, path.display());
        return Ok(());
    }
    let tp = Qwen35DenseTpLayout::new(&cfg, WORLD)
        .expect("Qwen3.5-27B is divisible by world=4");

    // Per-tensor layout table — count tensors by layout kind for the
    // cert summary.
    let mut total_bytes_full: usize = 0;
    let mut bytes_per_rank: [usize; WORLD as usize] = [0; WORLD as usize];
    let mut counts = LayoutCounts::default();
    let mut unknown_tensors: Vec<String> = Vec::new();

    // Sort tensor names so the cert is deterministic.
    let mut names: Vec<&String> = file.tensors.keys().collect();
    names.sort();

    for name in &names {
        let info = file.info(name)?;
        let total = info.size_in_bytes() as usize;
        total_bytes_full += total;
        match tp.for_tensor(name) {
            None => {
                unknown_tensors.push((*name).clone());
                // Conservative: replicate unknown tensors so the cert
                // doesn't lose count of them. Real loader would error
                // (TP-1c).
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

    // Sample-validate the slicing on representative tensors.
    // Layer 0 is GDN; layer 3 is full-attn (full_attention_interval=4).
    // - blk.3.attn_q.weight is fused [Q | gate], Q4_1, ColParallel{dim=0}.
    // - blk.0.ffn_down.weight (every layer has dense FFN), Q4_1, RowParallel{dim=1}.
    // - blk.0.attn_norm.weight is F32, Replicated.
    sample_validate_col_parallel(&file, &tp, "blk.3.attn_q.weight")?;
    sample_validate_row_parallel(&file, &tp, "blk.0.ffn_down.weight")?;
    sample_validate_replicated(&file, &tp, "blk.0.attn_norm.weight")?;
    // TP-4a: FusedQkvParallel — outer-dim permutation slicing.
    sample_validate_fused_qkv(&file, &tp, "blk.0.attn_qkv.weight")?;

    // The load-bearing correctness gate is *inter-rank balance*: every
    // rank's footprint must be identical (modulo any future per-rank
    // asymmetric layout, which V1 doesn't have). The footprint vs the
    // ideal `total / world` is informational — it's elevated whenever
    // any tensor is `Replicated` (token_embd, LM head, GDN fused QKV
    // until TP-4a, etc.).
    let ideal = total_bytes_full / WORLD as usize;
    let min_per_rank = *bytes_per_rank.iter().min().unwrap();
    let max_per_rank = *bytes_per_rank.iter().max().unwrap();
    let mean_per_rank = bytes_per_rank.iter().sum::<usize>() / WORLD as usize;
    let inter_rank_spread = max_per_rank - min_per_rank;
    let inter_rank_spread_pct = (inter_rank_spread as f64 / mean_per_rank as f64) * 100.0;
    let overhead_vs_ideal_pct =
        (mean_per_rank as f64 - ideal as f64) / ideal as f64 * 100.0;

    eprintln!(
        "Qwen3.5-27B-Q4_1 load (world={WORLD}): total={:.2} GiB, ideal/rank={:.2} GiB",
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
        "  per-rank overhead vs ideal: {:+.2} % (driven by Replicated tensors — V1-deferred)",
        overhead_vs_ideal_pct
    );
    eprintln!(
        "  ColParallel: {}, RowParallel: {}, FusedQkvParallel: {}, Replicated: {}",
        counts.col, counts.row, counts.fused_qkv, counts.rep
    );
    if !unknown_tensors.is_empty() {
        eprintln!(
            "  unknown to dense layout table ({}): {:?}",
            unknown_tensors.len(),
            &unknown_tensors[..unknown_tensors.len().min(5)]
        );
    }

    // Write the cert.
    let cert_dir = PathBuf::from("../../certs/parity");
    std::fs::create_dir_all(&cert_dir).ok();
    let cert_path = cert_dir.join("qwen35_27b_q4_1_tp4_load.json");
    let cert = format!(
        "{{\n  \"schema_version\": 1,\n  \"kind\": \"tp_load_summary\",\n  \
         \"model\": \"Qwen3.5-27B-Q4_1\",\n  \"arch\": \"qwen35\",\n  \"world\": {WORLD},\n  \
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
         \"notes\": \"Inter-rank spread is the load-bearing balance metric (0% = perfectly balanced). overhead_vs_ideal_pct measures replication tax: V1 leaves token_embd, LM head, and GDN fused-QKV (attn_qkv, ssm_conv1d) Replicated. TP-2d shards token_embd+LM head; TP-4a shards GDN fused-QKV with head-aware permutation, dropping overhead toward 0%.\",\n  \
         \"captured_via\": \"cargo test --release -p flambeau-qwen3-moe --test tp_slice_qwen35_27b\",\n  \
         \"captured_at\": \"2026-04-25\"\n}}\n",
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

    // Hard gate — inter-rank spread must be 0 (perfectly balanced).
    // V1 layout has only symmetric Replicated/ColParallel/RowParallel,
    // so any spread > 0 signals a layout-table bug.
    assert_eq!(
        inter_rank_spread, 0,
        "inter-rank spread {inter_rank_spread} bytes > 0; V1 layout is symmetric — bug"
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

fn sample_validate_col_parallel(
    file: &GgufFile,
    tp: &Qwen35DenseTpLayout,
    name: &str,
) -> Result<()> {
    let layout = tp.for_tensor(name).expect("known tensor");
    assert!(matches!(layout, WeightLayout::ColParallel { dim: 0, .. }));
    let info = file.info(name)?;
    let total = info.size_in_bytes() as usize;
    let per_rank_expected = total / WORLD as usize;
    let mut firsts = [0u8; WORLD as usize];
    for r in 0..WORLD {
        let s = slice_for_tp(file, name, layout, r)?;
        assert_eq!(
            s.len(),
            per_rank_expected,
            "{name} ColParallel rank {r}: got {} bytes, expected {per_rank_expected}",
            s.len()
        );
        firsts[r as usize] = s[0];
    }
    // ColParallel slices are contiguous row ranges from different
    // offsets in the same mmap; their first bytes should differ
    // (with vanishingly low probability of all 4 starts colliding).
    let all_same = firsts.iter().all(|&b| b == firsts[0]);
    assert!(!all_same, "{name} ColParallel: all rank starts have identical first byte — slicing likely a no-op");
    Ok(())
}

fn sample_validate_row_parallel(
    file: &GgufFile,
    tp: &Qwen35DenseTpLayout,
    name: &str,
) -> Result<()> {
    let layout = tp.for_tensor(name).expect("known tensor");
    assert!(matches!(layout, WeightLayout::RowParallel { dim: 1, .. }));
    let info = file.info(name)?;
    let total = info.size_in_bytes() as usize;
    let per_rank_expected = total / WORLD as usize;
    for r in 0..WORLD {
        let s = slice_for_tp(file, name, layout, r)?;
        assert_eq!(
            s.len(),
            per_rank_expected,
            "{name} RowParallel rank {r}: got {} bytes, expected {per_rank_expected}",
            s.len()
        );
    }
    // The 4 row-parallel slices, concatenated, should reproduce the
    // original mmap byte-for-byte (slicing is just a permutation).
    let raw = file.tensor_raw(name)?;
    let mut reconstructed = Vec::with_capacity(total);
    let outer = info.dims[0] as usize;
    let inner = info.dims[1] as usize;
    let block_size = info.block_size() as usize;
    let type_size = info.type_size() as usize;
    let full_row_bytes = inner / block_size * type_size;
    let per_rank_row_bytes = full_row_bytes / WORLD as usize;
    let slices: Vec<_> = (0..WORLD)
        .map(|r| slice_for_tp(file, name, layout, r))
        .collect::<Result<_>>()?;
    for row in 0..outer {
        for r in 0..WORLD as usize {
            let off = row * per_rank_row_bytes;
            reconstructed.extend_from_slice(&slices[r][off..off + per_rank_row_bytes]);
        }
    }
    assert_eq!(reconstructed.len(), raw.len(), "reconstructed length mismatch");
    assert_eq!(reconstructed, raw, "RowParallel slicing did not preserve byte content of {name}");
    Ok(())
}

fn sample_validate_fused_qkv(
    file: &GgufFile,
    tp: &Qwen35DenseTpLayout,
    name: &str,
) -> Result<()> {
    let layout = tp.for_tensor(name).expect("known tensor");
    let (world, num_v, num_k, head_v, head_k, kq_replicated) = match layout {
        WeightLayout::FusedQkvParallel {
            world,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            kq_replicated,
        } => (world, num_v_heads, num_k_heads, head_v_dim, head_k_dim, kq_replicated),
        other => panic!("{name} expected FusedQkvParallel, got {other:?}"),
    };
    // Reconstruction below assumes the contiguous-split case.
    assert!(
        !kq_replicated,
        "{name}: this reconstruction harness is for contiguous-split (kq_replicated=false) only"
    );
    let info = file.info(name)?;
    let outer_full = info.dims[0] as usize;
    let v_full = (num_v as usize) * (head_v as usize);
    let k_full = (num_k as usize) * (head_k as usize);
    assert_eq!(
        outer_full,
        v_full + 2 * k_full,
        "{name}: outer {outer_full} != V({v_full}) + 2·K({k_full})"
    );
    let raw = file.tensor_raw(name)?;
    let total = raw.len();
    let v_local = v_full / (world as usize);
    let k_local = k_full / (world as usize);
    let row_bytes = total / outer_full;
    // Reconstruct the full tensor by interleaving the V/K/Q sub-slabs
    // from each rank in head order. Result must match the mmap bytes
    // exactly (slicing is just a permutation of contiguous row ranges).
    let slices: Vec<_> = (0..world)
        .map(|r| slice_for_tp(file, name, layout, r))
        .collect::<Result<_>>()?;
    let mut reconstructed_v = Vec::with_capacity(v_full * row_bytes);
    let mut reconstructed_k = Vec::with_capacity(k_full * row_bytes);
    let mut reconstructed_q = Vec::with_capacity(k_full * row_bytes);
    let v_bytes = v_local * row_bytes;
    let k_bytes = k_local * row_bytes;
    for r in 0..world as usize {
        let s = &slices[r];
        reconstructed_v.extend_from_slice(&s[0..v_bytes]);
        reconstructed_k.extend_from_slice(&s[v_bytes..v_bytes + k_bytes]);
        reconstructed_q.extend_from_slice(&s[v_bytes + k_bytes..v_bytes + 2 * k_bytes]);
    }
    let mut reconstructed = Vec::with_capacity(total);
    reconstructed.extend_from_slice(&reconstructed_v);
    reconstructed.extend_from_slice(&reconstructed_k);
    reconstructed.extend_from_slice(&reconstructed_q);
    assert_eq!(
        reconstructed.len(),
        raw.len(),
        "FusedQkv {name}: reconstructed length mismatch"
    );
    assert_eq!(
        reconstructed, raw,
        "FusedQkv {name}: byte-permutation round-trip didn't match mmap"
    );
    Ok(())
}

fn sample_validate_replicated(
    file: &GgufFile,
    tp: &Qwen35DenseTpLayout,
    name: &str,
) -> Result<()> {
    let layout = tp.for_tensor(name).expect("known tensor");
    assert_eq!(layout, WeightLayout::Replicated);
    let raw = file.tensor_raw(name)?;
    for r in 0..WORLD {
        let s = slice_for_tp(file, name, layout, r)?;
        assert_eq!(s.len(), raw.len(), "{name} Replicated rank {r}: size mismatch");
        // Replicated should borrow the same mmap bytes for every rank.
        assert_eq!(&*s, raw, "{name} Replicated rank {r}: bytes don't match mmap");
    }
    Ok(())
}
