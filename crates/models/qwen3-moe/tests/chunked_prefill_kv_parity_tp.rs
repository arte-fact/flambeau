//! **Phase A2-TP** — chunked-vs-single-shot KV parity test for TP topology.
//!
//! Mirrors `chunked_prefill_kv_parity.rs` (PP) but for tensor-parallel.
//! Runs forward_prefill_tp_logits with the same prompt:
//!   - Single-shot at L=256 (m=256 → MMQ kernel bucket on Q4_1)
//!   - Two manual calls at L=128 each (m=128 → SAME MMQ bucket per Phase A)
//!
//! Per-rank LayerCache vectors are snapshotted via the existing helper.
//! The PP test proved the recursive chunker is bit-exact when chunks
//! stay in the same m-bucket. If TP behaves identically, this test
//! passes; if it doesn't, the diff message localises the divergence
//! (full-attn K/V or GDN state per-rank).
//!
//! Skips when: GGUF missing, fewer than 2 HIP devices, no peer access
//! between the chosen pair (default [0,1]).

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — model load + forward + dispose"
)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_prefill_tp_logits, ShardedForwardOneTokenScratchTp,
};
use flambeau_qwen3_moe::{
    snapshot_layer_caches_to_host, LayerCacheSnapshot, Qwen35DenseTpLayout, Qwen3MoEConfig,
    Qwen3MoETpModel, Qwen3MoETpSession,
};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

fn snapshot_tp_session(
    session: &Qwen3MoETpSession,
    cluster: &HipCluster,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    let mut per_rank = Vec::with_capacity(cluster.ranks());
    for rank in 0..cluster.ranks() {
        let device = cluster.device(rank);
        let layers = snapshot_layer_caches_to_host(&session.caches[rank], device)?;
        per_rank.push(layers);
    }
    Ok(per_rank)
}

fn describe_diff(a: &LayerCacheSnapshot, b: &LayerCacheSnapshot) -> Option<String> {
    match (a, b) {
        (
            LayerCacheSnapshot::FullAttn { k: ka, v: va, current_tokens: cta },
            LayerCacheSnapshot::FullAttn { k: kb, v: vb, current_tokens: ctb },
        ) => {
            if cta != ctb {
                return Some(format!(
                    "FullAttn current_tokens differ: single={cta} chunked={ctb}"
                ));
            }
            if ka != kb {
                let first = (0..ka.len()).find(|i| ka[*i] != kb[*i]).unwrap_or(0);
                return Some(format!(
                    "FullAttn K differs ({} bytes); first byte differing at offset {first}",
                    ka.len()
                ));
            }
            if va != vb {
                let first = (0..va.len()).find(|i| va[*i] != vb[*i]).unwrap_or(0);
                return Some(format!(
                    "FullAttn V differs ({} bytes); first byte differing at offset {first}",
                    va.len()
                ));
            }
            None
        }
        (
            LayerCacheSnapshot::Gdn { state: sa, conv_history: ca },
            LayerCacheSnapshot::Gdn { state: sb, conv_history: cb },
        ) => {
            if sa != sb {
                let first = (0..sa.len()).find(|i| sa[*i] != sb[*i]).unwrap_or(0);
                return Some(format!(
                    "GDN state differs ({} bytes); first byte differing at offset {first}",
                    sa.len()
                ));
            }
            if ca != cb {
                let first = (0..ca.len()).find(|i| ca[*i] != cb[*i]).unwrap_or(0);
                return Some(format!(
                    "GDN conv_history differs ({} bytes); first byte differing at offset {first}",
                    ca.len()
                ));
            }
            None
        }
        _ => Some("layer-type mismatch".into()),
    }
}

fn compare_tp_snapshots(
    a: &[Vec<LayerCacheSnapshot>],
    b: &[Vec<LayerCacheSnapshot>],
) -> usize {
    let mut diffs = 0;
    for (rank_idx, (sr, cr)) in a.iter().zip(b.iter()).enumerate() {
        for (layer_idx, (sa, cb)) in sr.iter().zip(cr.iter()).enumerate() {
            if let Some(msg) = describe_diff(sa, cb) {
                diffs += 1;
                if diffs <= 4 {
                    eprintln!("  rank {rank_idx} layer {layer_idx}: {msg}");
                }
            }
        }
    }
    if diffs > 4 {
        eprintln!("  ... ({} more diffs not shown)", diffs - 4);
    }
    eprintln!("  total layer diffs: {diffs}");
    diffs
}

#[test]
fn chunked_prefill_kv_parity_tp2() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    if n_available < 2 {
        eprintln!("skip — need 2 HIP devices, have {n_available}");
        return Ok(());
    }

    std::env::set_var("FLAMBEAU_MAX_CTX", "8192");

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let world: u32 = 2;
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&[0i32, 1])?);
    let layout = Qwen35DenseTpLayout::new(&cfg, world)?;
    let model = Qwen3MoETpModel::load(&file, &cluster, layout)?;
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))?;

    // Same m-bucket (>=128 → MMQ on Q4_1) for both runs.
    // Try larger L to see if many-chunk runs diverge.
    let l: usize = std::env::var("FLAMBEAU_TEST_L")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let chunk: usize = std::env::var("FLAMBEAU_TEST_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let prompt: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();

    // ---- single-shot ----
    let mut sess_a = Qwen3MoETpSession::new(&model, &cluster)?;
    let mut decode_a = ShardedForwardOneTokenScratchTp::new(&model.config, &cluster)?;
    let mut sink: Vec<f32> = Vec::new();
    forward_prefill_tp_logits(
        &model,
        &mut decode_a,
        &cluster,
        &ar,
        &mut sess_a.caches,
        &prompt,
        0,
        &mut sink,
    )?;
    let snap_single = snapshot_tp_session(&sess_a, &cluster)?;
    decode_a.dispose(&cluster)?;
    sess_a.dispose(&cluster)?;

    // ---- chunked: 2 manual calls of L=128 ----
    let mut sess_b = Qwen3MoETpSession::new(&model, &cluster)?;
    let mut decode_b = ShardedForwardOneTokenScratchTp::new(&model.config, &cluster)?;
    let mut chunk_err: Option<anyhow::Error> = None;
    let mut start = 0usize;
    while start < l {
        let end = (start + chunk).min(l);
        let mut ssink: Vec<f32> = Vec::new();
        if let Err(e) = forward_prefill_tp_logits(
            &model,
            &mut decode_b,
            &cluster,
            &ar,
            &mut sess_b.caches,
            &prompt[start..end],
            start,
            &mut ssink,
        ) {
            chunk_err = Some(e);
            break;
        }
        start = end;
    }
    let snap_chunked = if chunk_err.is_none() {
        snapshot_tp_session(&sess_b, &cluster).ok()
    } else {
        None
    };
    let _ = decode_b.dispose(&cluster);
    let _ = sess_b.dispose(&cluster);

    let _ = ar; // disposed automatically when dropped (noop here)
    if let Some(e) = chunk_err {
        return Err(e);
    }
    let snap_chunked = snap_chunked.expect("snapshot taken on success");

    // Try to dispose model & cluster cleanly (best-effort; SEGVs in
    // some test envs are unrelated to the parity check).
    let _ = model.dispose(&cluster);

    eprintln!("--- single-shot L=256 vs chunked 2×L=128 (TP2) ---");
    let diffs = compare_tp_snapshots(&snap_single, &snap_chunked);
    if diffs > 0 {
        panic!(
            "TP chunked KV parity FAILED ({diffs} layer diffs across both ranks). \
             First diff localises the divergence (FullAttn K/V or GDN state)."
        );
    }
    eprintln!("OK: TP chunked KV state matches single-shot byte-for-byte");
    Ok(())
}
