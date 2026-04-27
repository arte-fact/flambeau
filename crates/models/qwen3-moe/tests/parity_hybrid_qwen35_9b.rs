//! AUTO-4d — hybrid PP-of-TP parity smoke for Qwen3.5-9B-Q4_1 at
//! `pp_size=2, tp_size=2` on Mesh<4>.
//!
//! Reuses the V2.2.c.1 Mesh<1> reference token sequence
//! (`certs/parity/qwen35_9b_q4_1_decode.json`): hybrid decomposition is
//! algebraically identical to the dense forward — same MMVQ kernels in
//! the same order, same residual stream — so the greedy argmax should
//! match bit-for-bit at every step the existing Mesh<1> cert already
//! covers. Drift would point at the inter-stage hand-off or the
//! per-stage AR, not at the kernel layer.
//!
//! This test is gated on (1) the GGUF being present and (2) ≥ 4 HIP
//! devices being visible. It also bails cleanly when the rig has a
//! known-bad peer pair across the configured device list — the bracket
//! probe in `feedback_bracket_before_audit.md` is the operator's
//! responsibility before invoking `pp+tp`.

#![cfg(feature = "hip")]

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_core::Device;
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::forward_one_token_hybrid;
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen3MoEConfig, Qwen3MoEHybridModel, Qwen3MoEHybridSession,
    ShardedForwardOneTokenScratchHybrid,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct ParityCert {
    model_tag: String,
    seed_token_id: u32,
    expected_token_ids: Vec<u32>,
    #[serde(default)]
    #[expect(dead_code, reason = "second-oracle field; cert JSON retains candle run for diffs")]
    candle_token_ids: Vec<u32>,
    temperature: f32,
    #[serde(default = "default_expect_pass")]
    expect_pass: bool,
    #[expect(dead_code, reason = "cert provenance; never read by the test itself")]
    notes: String,
}

fn default_expect_pass() -> bool {
    true
}

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from("/artefact/models/Qwen3.5-9B-Q4_1.gguf")))
        .filter(|p| p.exists())
}

fn workspace_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

fn load_cert() -> Result<Option<ParityCert>> {
    let cert_path = workspace_root()
        .join("certs")
        .join("parity")
        .join("qwen35_9b_q4_1_decode.json");
    if !cert_path.exists() {
        eprintln!("parity cert {} not found — skipping", cert_path.display());
        return Ok(None);
    }
    let bytes = std::fs::read(&cert_path)?;
    let cert: ParityCert = serde_json::from_slice(&bytes)?;
    if cert.temperature != 0.0 {
        bail!("parity cert temperature must be 0.0 for greedy parity");
    }
    Ok(Some(cert))
}

/// Stage-major device ids, overridable via `FLAMBEAU_HYBRID_DEVICES`
/// (comma-separated). Default: `0,1,2,3` — stage 0 owns {0,1}, stage 1
/// owns {2,3}. The user picks an order whose intra-stage pairs all
/// have a healthy BAR1 peer-access matrix (per the rig-23-link-fault
/// memory).
fn device_ids() -> Vec<i32> {
    if let Ok(s) = std::env::var("FLAMBEAU_HYBRID_DEVICES") {
        return s
            .split(',')
            .filter_map(|p| p.trim().parse::<i32>().ok())
            .collect();
    }
    vec![0, 1, 2, 3]
}

#[test]
#[ignore = "AUTO-4d hybrid parity smoke — runs only with FLAMBEAU_HYBRID=1; \
             requires 4× HIP + a healthy BAR1 peer matrix on each stage's \
             tp pair. Default device order is 0,1,2,3 (stage0={0,1}, \
             stage1={2,3}); override with FLAMBEAU_HYBRID_DEVICES."]
fn qwen35_hybrid_pp2tp2_matches_llama_cpp() -> Result<()> {
    if std::env::var("FLAMBEAU_HYBRID").ok().as_deref() != Some("1") {
        eprintln!("skip — FLAMBEAU_HYBRID=1 not set");
        return Ok(());
    }
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5-9B GGUF not present");
        return Ok(());
    };
    let Some(cert) = load_cert()? else {
        return Ok(());
    };
    let dev_ids = device_ids();
    if (device_count().unwrap_or(0) as usize) < dev_ids.len() {
        eprintln!("skip — fewer than {} HIP devices visible", dev_ids.len());
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let spec = HybridMeshSpec {
        pp_size: 2,
        tp_size: 2,
    };
    spec.validate(cfg.num_layers, dev_ids.len())?;

    eprintln!(
        "hybrid parity [{}]: pp={}, tp={}, devices={:?}, layers={}",
        cert.model_tag, spec.pp_size, spec.tp_size, dev_ids, cfg.num_layers,
    );

    // Construct the per-stage sub-clusters FIRST (via model load) and
    // the global cluster SECOND. Reverse order leaves the per-stage
    // canAccessPeer probe reporting 0 on this rig (observed on
    // 2026-04-27 — `peer_access_pairs` test in `flambeau-backend-hip`
    // shows ascending pairs work fresh but fail when a 4-device cluster
    // already exists). Hypothesis: HIP runtime peer-access state is
    // path-dependent across new HipDevice contexts.
    let model = Qwen3MoEHybridModel::load(&file, &dev_ids, spec)?;
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&dev_ids)?);

    // Diagnostic — print each stage's sub-cluster peer matrix so a
    // failed AR construction below tells us *which* edge is bad.
    for stage in &model.stages {
        let mat = stage.sub_cluster.peer_access_matrix();
        let ranks: Vec<i32> = (0..stage.sub_cluster.ranks())
            .map(|r| stage.sub_cluster.device(r).id())
            .collect();
        eprintln!(
            "  stage {} sub-cluster devices={:?} peer_access_matrix:",
            stage.stage_idx, ranks,
        );
        for (r, row) in mat.iter().enumerate() {
            let cells: Vec<String> = row.iter().map(|b| if *b { "1" } else { "0" }.to_string()).collect();
            eprintln!("    rank {r}: [{}]", cells.join(", "));
        }
        eprintln!(
            "    peer_access_full = {}",
            stage.sub_cluster.peer_access_full()
        );
    }

    // Per-stage AllReduce, each on its own sub-cluster (bit-identical
    // to a pure-TP world=tp_size run inside the stage).
    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        let ar = BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster)).with_context(|| {
            format!(
                "BarP2pAllReduce::new for stage {} (devices for that stage need fully-\
                 connected BAR1 peer access)",
                stage.stage_idx
            )
        })?;
        stage_ars.push(ar);
    }

    let mut session = Qwen3MoEHybridSession::new(&model)?;
    let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;

    let n_predict = cert.expected_token_ids.len();
    let mut got: Vec<u32> = Vec::with_capacity(n_predict);
    let mut current = cert.seed_token_id;
    for pos in 0..n_predict {
        let next = forward_one_token_hybrid(
            &model,
            &mut scratch,
            &global_cluster,
            &stage_ars,
            &mut session,
            current,
            pos,
        )
        .with_context(|| format!("hybrid decode step pos={pos}"))?;
        got.push(next);
        current = next;
    }

    eprintln!("  got       = {got:?}");
    eprintln!("  llama.cpp = {:?}", cert.expected_token_ids);

    // Dispose in reverse-construction order.
    scratch.dispose(&model)?;
    session.dispose(&model)?;
    drop(stage_ars);
    model.dispose()?;
    drop(global_cluster);

    let matches = got == cert.expected_token_ids;
    if cert.expect_pass {
        assert!(
            matches,
            "hybrid parity mismatch: got {got:?}, expected {:?}",
            cert.expected_token_ids
        );
    } else if matches {
        eprintln!(
            "note: cert.expect_pass=false but the run matched — flip the cert \
             to expect_pass=true once the rig stabilises."
        );
    }

    Ok(())
}
