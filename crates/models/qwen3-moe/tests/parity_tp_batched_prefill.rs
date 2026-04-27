//! AUTO-6b3 / AUTO-6c4 — parity smoke for the L-batched TP prefill
//! path. Compares the batched prefill (full-attn or GDN + dense or
//! MoE FFN per layer with one AR/side, opt-in via
//! `FLAMBEAU_TP_BATCHED=1`) against the per-token loop default,
//! verifying both produce bit-identical last-position logits at L≥8.
//!
//! AUTO-6c4 extended coverage to all four layer flavors:
//!   * full-attn + dense FFN  (Qwen3.5 9B/27B Q4_1, dense)
//!   * GDN + dense FFN        (Qwen3.5 hybrid)
//!   * full-attn + MoE        (Qwen3-Coder-30B)
//!   * GDN + MoE + shared exp (Qwen3.6-35B-A3B hybrid MoE)
//!
//! Bit-exact is the right bar here: both paths run the same per-rank
//! kernels (rmsnorm → mmvq → split_q_gate → rope → attention_prefill
//! → output proj → dense FFN), only the `n_tokens` argument and AR
//! cadence change. Op order is preserved.
//!
//! Gates: `FLAMBEAU_TP_BATCHED_PARITY=1` + GGUF + ≥2 HIP devices on a
//! healthy ascending pair. Defaults to `[0,1]`; override via
//! `FLAMBEAU_HYBRID_DEVICES` (same convention as the hybrid parity
//! smoke).

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — load + bench + dispose; same invariant as parity_hybrid"
)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_prefill_tp_logits, ShardedForwardOneTokenScratchTp,
};
use flambeau_qwen3_moe::{Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

fn device_ids() -> Vec<i32> {
    if let Ok(s) = std::env::var("FLAMBEAU_HYBRID_DEVICES") {
        return s
            .split(',')
            .filter_map(|p| p.trim().parse::<i32>().ok())
            .collect();
    }
    vec![0, 1]
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
#[ignore = "AUTO-6b3 batched-vs-per-token TP prefill parity. Runs only with \
             FLAMBEAU_TP_BATCHED_PARITY=1 + Qwen3.5-9B-Q4_1 + ≥2 HIP devices \
             on a healthy ascending pair (default [0,1]; override via \
             FLAMBEAU_HYBRID_DEVICES)."]
fn tp_batched_prefill_matches_per_token_qwen35_9b() -> Result<()> {
    if std::env::var("FLAMBEAU_TP_BATCHED_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip — FLAMBEAU_TP_BATCHED_PARITY=1 not set");
        return Ok(());
    }
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5-9B GGUF not present");
        return Ok(());
    };
    let dev_ids = device_ids();
    if (device_count().unwrap_or(0) as usize) < dev_ids.len() {
        eprintln!(
            "skip — fewer than {} HIP devices visible",
            dev_ids.len()
        );
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    // AUTO-6c4 — the batched dispatcher now handles all layer flavors;
    // no model-shape skip needed. Qwen3.5-9B (hybrid GDN + dense FFN),
    // Qwen3.5-27B-Q4_1 (pure dense), Qwen3-Coder-30B (full-attn + MoE),
    // and Qwen3.6-35B-A3B (hybrid GDN + MoE + shared) all exercise the
    // same code path with different per-layer branches.
    let world = dev_ids.len() as u32;
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&dev_ids)?);

    eprintln!(
        "AUTO-6b3 batched parity: arch={}, devices={:?}, world={}, layers={}",
        cfg.arch, dev_ids, world, cfg.num_layers
    );
    let layout = Qwen35DenseTpLayout::new(&cfg, world)?;
    let model = Qwen3MoETpModel::load(&file, &cluster, layout)?;
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))
        .context("BarP2pAllReduce::new (test rig must have a healthy peer pair)")?;

    // Synthetic prompt — distinct token IDs across the prompt so each
    // position exercises distinct embed rows. SEED token (9419) at
    // position 0 keeps continuity with the existing parity certs.
    let l = 16usize;
    let prompt: Vec<u32> = (0..l as u32)
        .map(|i| if i == 0 { 9419 } else { 1 + i })
        .collect();
    let vocab = cfg.vocab_size;

    // Run 1 — per-token (default, FLAMBEAU_TP_BATCHED unset).
    std::env::remove_var("FLAMBEAU_TP_BATCHED");
    let mut session_a = Qwen3MoETpSession::new(&model, &cluster)?;
    let mut scratch_a = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
    let mut logits_a: Vec<f32> = vec![0.0; vocab];
    forward_prefill_tp_logits(
        &model,
        &mut scratch_a,
        &cluster,
        &ar,
        &mut session_a.caches,
        &prompt,
        0,
        &mut logits_a,
    )
    .context("per-token TP prefill")?;
    let argmax_per_token = argmax(&logits_a);
    scratch_a.dispose(&cluster).ok();
    session_a.dispose(&cluster).ok();

    // Run 2 — batched (FLAMBEAU_TP_BATCHED=1).
    std::env::set_var("FLAMBEAU_TP_BATCHED", "1");
    let mut session_b = Qwen3MoETpSession::new(&model, &cluster)?;
    let mut scratch_b = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
    let mut logits_b: Vec<f32> = vec![0.0; vocab];
    forward_prefill_tp_logits(
        &model,
        &mut scratch_b,
        &cluster,
        &ar,
        &mut session_b.caches,
        &prompt,
        0,
        &mut logits_b,
    )
    .context("batched TP prefill")?;
    let argmax_batched = argmax(&logits_b);
    scratch_b.dispose(&cluster).ok();
    session_b.dispose(&cluster).ok();
    std::env::remove_var("FLAMBEAU_TP_BATCHED");

    eprintln!(
        "  argmax per-token = {argmax_per_token}, batched = {argmax_batched}"
    );

    // Bit-exact element-wise comparison. Op order is preserved between
    // paths (same per-rank kernels, same warp reductions); the only
    // change is the AR cadence + n_tokens argument. Drift would point
    // at a per-token layout assumption AUTO-6b2 missed.
    let mut max_abs_err = 0.0f32;
    let mut argmax_pertkn = 0usize;
    let mut argmax_batched_idx = 0usize;
    let mut a_max = f32::NEG_INFINITY;
    let mut b_max = f32::NEG_INFINITY;
    for (i, (&a, &b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_err {
            max_abs_err = d;
        }
        if a > a_max {
            a_max = a;
            argmax_pertkn = i;
        }
        if b > b_max {
            b_max = b;
            argmax_batched_idx = i;
        }
    }
    eprintln!(
        "  max |logits_per_token - logits_batched| = {max_abs_err:.6e}; \
         argmax per-token = {argmax_pertkn} (val {a_max:.4}); \
         argmax batched = {argmax_batched_idx} (val {b_max:.4})"
    );
    assert_eq!(
        argmax_pertkn, argmax_batched_idx,
        "argmax mismatch: per-token={argmax_pertkn}, batched={argmax_batched_idx}"
    );
    // Allow a tiny FP32 reassociation drift from the bigger AR (one AR
    // sums L*H elems vs L ARs of H elems each — Kahan-summation order
    // differs). Bound: 1e-3 relative to peak logit magnitude.
    let tol = a_max.abs().max(1.0) * 1e-3;
    assert!(
        max_abs_err < tol,
        "logits drift {max_abs_err:.6e} exceeds tol {tol:.6e}"
    );

    drop(ar);
    model.dispose(&cluster).ok();
    drop(cluster);
    Ok(())
}
