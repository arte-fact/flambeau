//! Phase 10c-H smoke: real-GGUF 26B-A4B-Q8_0 on pp2tp2 (hip:0,2,1,3).
//! Exercises the hybrid MoE composer — F32 attention output path on
//! full-attn layers + F32 MoE cascade + per-stage AR — through one
//! forward step + a 16-token greedy decode. Skipped when the GGUF is
//! absent, fewer than 4 HIP devices, or peer access incomplete on the
//! sub-clusters.

#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::Arc;

use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_blocks::HybridCluster;
use flambeau_gemma4::{Gemma4Config, Gemma4HybridDriver, ModelLayout};
use flambeau_quant::GgufFile;

const MODELS_DIR: &str = "/artefact/models";
const MAX_TOKENS: usize = 64;
const N_DECODE: usize = 16;
const PP_SIZE: usize = 2;
const TP_SIZE: usize = 2;

fn open_or_skip(name: &str) -> Option<Arc<GgufFile>> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok().map(Arc::new)
}

/// pp2tp2 on hip:0,2,1,3 (MEMORY.md `never_tp4_use_pp2tp2`). Stage 0 =
/// hip:0,2 (global ranks 0,1); stage 1 = hip:1,3 (global ranks 2,3).
fn clusters_or_skip() -> Option<(Vec<Arc<HipCluster>>, Arc<HipCluster>)> {
    let n = device_count().ok()?;
    let total = PP_SIZE * TP_SIZE;
    if (n as usize) < total {
        eprintln!("skipping hybrid MoE smoke — need {total} HIP devices but only {n}");
        return None;
    }
    // Production mesh is `hip:0,2,1,3` (MEMORY.md `never_tp4_use_pp2tp2`)
    // to dodge the {2,3} link fault. Each sub-cluster needs full peer
    // access (intra-stage TP AR uses BAR1 P2P); the 4-way global
    // cluster only needs the stage-boundary edge (rank 0 of stage 1 ↔
    // rank 0 of stage 0), so we do NOT require `peer_access_full()` on
    // the global — that check is over-strict for hybrid.
    let mesh = [0i32, 2, 1, 3];
    let stage_0_ids = [mesh[0], mesh[1]];
    let stage_1_ids = [mesh[2], mesh[3]];
    let sub0 = Arc::new(HipCluster::new(&stage_0_ids).ok()?);
    let sub1 = Arc::new(HipCluster::new(&stage_1_ids).ok()?);
    if !sub0.peer_access_full() || !sub1.peer_access_full() {
        eprintln!("skipping — sub-cluster peer access incomplete on hip:0,2 or hip:1,3");
        return None;
    }
    let global = Arc::new(HipCluster::new(&mesh).ok()?);
    eprintln!("  using mesh hip:0,2,1,3 (global peer_full={})", global.peer_access_full());
    Some((vec![sub0, sub1], global))
}

#[test]
fn smoke_26b_a4b_q8_0_pp2tp2() {
    let Some(file) = open_or_skip("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    let Some((sub_clusters, global)) = clusters_or_skip() else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    assert!(cfg.moe.is_some(), "26B-A4B should have cfg.moe = Some");
    let vocab = cfg.vocab_size;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let hc = HybridCluster::new(sub_clusters, global, TP_SIZE).expect("HybridCluster");
    let mut driver = match Gemma4HybridDriver::upload(&file, cfg, layout, hc, MAX_TOKENS) {
        Ok(d) => d,
        Err(e) => {
            let full = format!("{e:#}");
            if full.contains("out of memory") || full.contains("OutOfMemory") {
                eprintln!("skipping — 26B-A4B-Q8_0 pp2tp2 OOM: {full}");
                return;
            }
            panic!("Gemma4HybridDriver::upload: {full}");
        }
    };

    // Greedy decode 16 tokens from BOS=2 at position 0.
    let mut tok = 2u32;
    let mut pos = 0usize;
    let mut ids = Vec::with_capacity(N_DECODE);
    for step in 0..N_DECODE {
        tok = driver
            .forward_one_token(tok, pos)
            .unwrap_or_else(|e| panic!("decode step {step}: {e:#}"));
        assert!(
            (tok as usize) < vocab,
            "step {step}: argmax {tok} oob (vocab={vocab})"
        );
        ids.push(tok);
        pos += 1;
    }

    eprintln!("\n=== SMOKE | 26B-A4B-Q8_0 pp2tp2 (hip:0,2,1,3) ===");
    eprintln!("  generated ids: {ids:?}");
    let first = ids[0];
    if ids.iter().all(|&t| t == first) {
        panic!(
            "all {} decoded tokens identical ({first}); hybrid MoE forward producing constant logits",
            ids.len()
        );
    }
    eprintln!("  [OK] non-degenerate output");

    driver.dispose().expect("dispose");
}
