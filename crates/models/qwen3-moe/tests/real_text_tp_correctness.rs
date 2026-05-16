//! Real-text TP correctness check — load Qwen3.5-9B-Q4_1 on TP2, prefill
//! "The capital of France is", and assert the first sampled token decodes
//! to text containing "Paris".
//!
//! Why this test exists: the typed-role TP upload path (commit 75861ae,
//! "qwen3-moe S4a") shipped uploads that passed the synthetic-token
//! chunked-prefill KV parity test but produced garbage on real prompts.
//! The KV parity test only checks per-rank consistency of cache state,
//! not absolute correctness of weight values. This test locks in that
//! the legacy `upload_tp_with_layout` produces output matching llama.cpp's
//! " Paris" continuation.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — model load + forward + dispose"
)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::{load_from_gguf, GgufFile};
use flambeau_qwen3_moe::forward::{forward_prefill_tp_logits, ShardedForwardOneTokenScratchTp};
use flambeau_qwen3_moe::{
    Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession,
};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";
const PROMPT: &str = "The capital of France is";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

#[test]
fn real_text_tp2_qwen35_9b_q4_1_paris() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5-9B-Q4_1 GGUF not present");
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
    let tokenizer = load_from_gguf(&file)?;
    let prompt_ids = tokenizer.encode(PROMPT)?;
    assert!(
        !prompt_ids.is_empty(),
        "tokenizer encoded `{PROMPT}` to zero tokens"
    );

    let world: u32 = 2;
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&[0i32, 1])?);
    let layout = Qwen35DenseTpLayout::new(&cfg, world)?;
    let model = Qwen3MoETpModel::load(&file, &cluster, layout)?;
    let tp = flambeau_blocks::TpCluster::from_arc(Arc::clone(&cluster))?;

    let mut sess = Qwen3MoETpSession::new(
        &model,
        &cluster,
        flambeau_qwen3_moe::session::KvLayout::F16,
    )?;
    let mut scratch = ShardedForwardOneTokenScratchTp::new(&model.config, &cluster)?;
    let mut logits: Vec<f32> = Vec::new();
    forward_prefill_tp_logits(
        &model,
        &mut scratch,
        &tp,
        &mut sess.caches,
        &prompt_ids,
        0,
        &mut logits,
    )?;

    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "logits len {} != vocab {}",
        logits.len(),
        cfg.vocab_size
    );
    let next = argmax(&logits);
    let decoded = tokenizer.decode(&[next])?;
    eprintln!("TP2 next-token after `{PROMPT}` -> id={next} text={decoded:?}");

    let _ = scratch.dispose(&cluster);
    let _ = sess.dispose(&cluster);
    let _ = model.dispose(&cluster);

    assert!(
        decoded.contains("Paris"),
        "TP2 next-token after `{PROMPT}` did not contain `Paris` — got id={next} text={decoded:?}. \
         This is the same failure mode that bypassed typed-role TP upload (see #118)."
    );
    Ok(())
}
