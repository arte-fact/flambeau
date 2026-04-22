//! V2.2.c.1 — parity cert for Qwen3.5-9B-Q4_1 (arch=qwen35 dense-hybrid).
//!
//! Mirrors `parity_vs_llama_cpp.rs` but targets the dense-FFN forward path
//! and Q4_1 MMVQ kernel landed in V2.2.b/c. Qwen3.5-9B fits one MI50, so
//! the test runs on Mesh<1> — the first flambeau parity gate against a
//! model that doesn't need multi-device.
//!
//! Regenerate reference tokens per the note in
//! `certs/parity/qwen35_9b_q4_1_decode.json`.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{forward_one_token_pp, ShardedForwardOneTokenScratch};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use serde::Deserialize;

#[derive(Deserialize)]
struct ParityCert {
    model_tag: String,
    seed_token_id: u32,
    expected_token_ids: Vec<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    candle_token_ids: Vec<u32>,
    temperature: f32,
    #[serde(default = "default_expect_pass")]
    expect_pass: bool,
    #[allow(dead_code)]
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

#[test]
fn qwen35_decode_greedy_matches_llama_cpp() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    let Some(cert) = load_cert()? else {
        return Ok(());
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("skip — no HIP device");
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let cluster = HipCluster::new(&[0])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, 1);

    eprintln!(
        "parity run [{}]: loading on Mesh<1> ({} layers)…",
        cert.model_tag, cfg.num_layers,
    );
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    let n_predict = cert.expected_token_ids.len();
    let mut got: Vec<u32> = Vec::with_capacity(n_predict);
    let mut current = cert.seed_token_id;
    for pos in 0..n_predict {
        let next = forward_one_token_pp(
            &model,
            &mut session,
            &cluster,
            &mut scratch,
            current,
            pos,
        )
        .with_context(|| format!("decode step pos={pos}"))?;
        got.push(next);
        current = next;
    }

    eprintln!("  got       = {got:?}");
    eprintln!("  llama.cpp = {:?}", cert.expected_token_ids);

    let matches = got == cert.expected_token_ids;
    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;

    if !matches {
        let mut first_div = None;
        for (i, (g, e)) in got.iter().zip(&cert.expected_token_ids).enumerate() {
            if g != e {
                first_div = Some((i, *g, *e));
                break;
            }
        }
        let msg = if let Some((i, g, e)) = first_div {
            format!(
                "greedy-parity divergence at step {i}: flambeau={g}, llama.cpp={e}\n  got      = {got:?}\n  expected = {:?}",
                cert.expected_token_ids
            )
        } else {
            format!(
                "greedy-parity length mismatch: got {} tokens, expected {}",
                got.len(),
                cert.expected_token_ids.len(),
            )
        };
        if cert.expect_pass {
            bail!("{msg}");
        } else {
            eprintln!("DIAGNOSTIC (expect_pass=false): {msg}");
        }
    }
    Ok(())
}
