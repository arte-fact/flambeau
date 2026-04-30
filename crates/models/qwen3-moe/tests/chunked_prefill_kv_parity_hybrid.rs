//! **Phase B4a-Hybrid** — chunked-vs-single-shot KV parity for the
//! Hybrid (pp+tp) topology.
//!
//! Mirrors the PP and TP parity tests. Default: pp2tp2 over devices
//! [0,2,1,3] (intra-die pairs per the rig topology memory).
//!
//! Skips when GGUF missing, fewer than 4 HIP devices, or BAR1 peer
//! access is unhealthy on a stage sub-cluster.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — model load + forward + dispose"
)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_prefill_hybrid_logits, ShardedForwardOneTokenScratchHybrid,
};
use flambeau_qwen3_moe::hybrid::HybridMeshSpec;
use flambeau_qwen3_moe::{
    snapshot_layer_caches_to_host, LayerCacheSnapshot, Qwen3MoEConfig, Qwen3MoEHybridModel,
    Qwen3MoEHybridSession,
};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

fn device_ids() -> Vec<i32> {
    // Default to intra-die pairs per the rig topology memory note.
    vec![0, 2, 1, 3]
}


#[test]
fn chunked_prefill_kv_parity_hybrid_pp2tp2() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    let dev_ids = device_ids();
    let n_available: i32 = device_count().unwrap_or(0);
    if (n_available as usize) < dev_ids.len() {
        eprintln!("skip — need 4 HIP devices, have {n_available}");
        return Ok(());
    }

    std::env::set_var("FLAMBEAU_MAX_CTX", "8192");

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    spec.validate(cfg.num_layers, dev_ids.len())?;

    let model = Qwen3MoEHybridModel::load(&file, &dev_ids, spec)?;
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&dev_ids)?);

    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        let ar = BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster)).with_context(|| {
            format!("BarP2pAllReduce::new for stage {}", stage.stage_idx)
        })?;
        stage_ars.push(ar);
    }

    let l: usize = std::env::var("FLAMBEAU_TEST_L")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let chunk: usize = std::env::var("FLAMBEAU_TEST_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let prompt: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();

    // Helper: per-stage per-rank snapshot using each stage's sub_cluster.
    let snap = |session: &Qwen3MoEHybridSession,
                model: &Qwen3MoEHybridModel|
     -> Result<Vec<Vec<Vec<LayerCacheSnapshot>>>> {
        let mut out = Vec::with_capacity(session.stages.len());
        for (stage_idx, ssess) in session.stages.iter().enumerate() {
            let mstage = &model.stages[stage_idx];
            let mut per_rank = Vec::with_capacity(ssess.caches.len());
            for (rank_idx, layer_caches) in ssess.caches.iter().enumerate() {
                let device = mstage.sub_cluster.device(rank_idx);
                let layers = snapshot_layer_caches_to_host(layer_caches, device)?;
                per_rank.push(layers);
            }
            out.push(per_rank);
        }
        Ok(out)
    };

    // ---- single-shot ----
    let mut sess_a = Qwen3MoEHybridSession::new(&model)?;
    let mut decode_a = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    let mut sink: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut decode_a,
        &global_cluster,
        &stage_ars,
        &mut sess_a,
        &prompt,
        0,
        &mut sink,
    )?;
    let snap_single = snap(&sess_a, &model)?;
    decode_a.dispose(&model)?;
    sess_a.dispose(&model)?;

    // ---- chunked: 2+ manual calls ----
    let mut sess_b = Qwen3MoEHybridSession::new(&model)?;
    let mut decode_b = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    let mut start = 0usize;
    let mut chunk_err: Option<anyhow::Error> = None;
    while start < l {
        let end = (start + chunk).min(l);
        let mut ssink: Vec<f32> = Vec::new();
        if let Err(e) = forward_prefill_hybrid_logits(
            &model,
            &mut decode_b,
            &global_cluster,
            &stage_ars,
            &mut sess_b,
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
        snap(&sess_b, &model).ok()
    } else {
        None
    };
    let _ = decode_b.dispose(&model);
    let _ = sess_b.dispose(&model);

    if let Some(e) = chunk_err {
        let _ = model.dispose();
        return Err(e);
    }
    let snap_chunked = snap_chunked.unwrap();
    let _ = model.dispose();

    eprintln!("--- single-shot L={l} vs chunked chunks={} (Hybrid pp2tp2) ---",
        (l + chunk - 1) / chunk);
    let mut diffs = 0usize;
    for (s_idx, (sa, sb)) in snap_single.iter().zip(snap_chunked.iter()).enumerate() {
        for (r_idx, (ra, rb)) in sa.iter().zip(sb.iter()).enumerate() {
            for (l_idx, (la, lb)) in ra.iter().zip(rb.iter()).enumerate() {
                if let Some(msg) = describe_diff(la, lb) {
                    diffs += 1;
                    if diffs <= 4 {
                        eprintln!("  stage {s_idx} rank {r_idx} layer {l_idx}: {msg}");
                    }
                }
            }
        }
    }
    if diffs > 4 {
        eprintln!("  ... ({} more diffs not shown)", diffs - 4);
    }
    eprintln!("  total layer diffs: {diffs}");
    if diffs > 0 {
        panic!("Hybrid chunked KV parity FAILED ({diffs} layer diffs)");
    }
    eprintln!("OK: Hybrid chunked KV state matches single-shot byte-for-byte");
    Ok(())
}

fn describe_diff(a: &LayerCacheSnapshot, b: &LayerCacheSnapshot) -> Option<String> {
    match (a, b) {
        (LayerCacheSnapshot::FullAttn { k: ka, v: va, current_tokens: cta },
         LayerCacheSnapshot::FullAttn { k: kb, v: vb, current_tokens: ctb }) => {
            if cta != ctb {
                return Some(format!("FullAttn current_tokens: single={cta} chunked={ctb}"));
            }
            if ka != kb {
                let i = (0..ka.len()).find(|i| ka[*i] != kb[*i]).unwrap_or(0);
                return Some(format!("FullAttn K differs ({} bytes); first at {i}", ka.len()));
            }
            if va != vb {
                let i = (0..va.len()).find(|i| va[*i] != vb[*i]).unwrap_or(0);
                return Some(format!("FullAttn V differs ({} bytes); first at {i}", va.len()));
            }
            None
        }
        (LayerCacheSnapshot::Gdn { state: sa, conv_history: ca },
         LayerCacheSnapshot::Gdn { state: sb, conv_history: cb }) => {
            if sa != sb {
                let i = (0..sa.len()).find(|i| sa[*i] != sb[*i]).unwrap_or(0);
                return Some(format!("GDN state differs ({} bytes); first at {i}", sa.len()));
            }
            if ca != cb {
                let i = (0..ca.len()).find(|i| ca[*i] != cb[*i]).unwrap_or(0);
                return Some(format!("GDN conv_history differs ({} bytes); first at {i}", ca.len()));
            }
            None
        }
        _ => Some("layer-type mismatch".into()),
    }
}
