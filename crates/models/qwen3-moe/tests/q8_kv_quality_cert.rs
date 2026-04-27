//! V1-BENCH-#117 — Q8 KV cache quality cert.
//!
//! Differential test: drive the same prompt through forward_one_token_pp
//! under FLAMBEAU_KV=f16 and FLAMBEAU_KV=q8, capturing the full F32 logits
//! at each decode step. Compute, position by position:
//!
//!   * top-1 agreement (does Q8 pick the same argmax as F16?)
//!   * top-5 agreement (is F16's top-1 in Q8's top-5?)
//!   * KL(softmax(F16) || softmax(Q8))
//!   * KL(softmax(Q8) || softmax(F16))
//!   * cosine similarity on the raw logit vectors
//!
//! Aggregates across all positions, writes a JSON cert under
//! `certs/quality/<model>_q8_kv.json`.
//!
//! Gate (KL-based, approximates the CLAUDE.md / ROADMAP §turbo-quant
//! "delta-ppl ≤ 0.5 %" target — per-token KL(F16 || Q8) is the
//! information-theoretic equivalent and avoids the harshness of a strict
//! top-1 agreement bar that fires on near-tie logit swaps):
//!
//!   * mean_KL ≤ 0.005   (~0.5 % delta-ppl)
//!   * max_KL  ≤ 0.05    (no single position diverges catastrophically)
//!   * top-5 agreement = 100 %  (the F16 argmax is always in Q8's top-5,
//!     i.e. quant noise never pushes a "right answer" out of contention)
//!
//! Top-1 agreement is reported as informational, not gating.

#![cfg(feature = "hip")]

use anyhow::{Context, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp_logits, ShardedForwardOneTokenScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

const SEED_TOKEN: u32 = 9419; // "Hello" in Qwen tokenizer
const N_DECODE: usize = 32;   // decode steps to capture per prompt

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

#[derive(Debug, Default)]
struct PerStepStats {
    top1_match: bool,
    top5_match: bool,
    kl_f_q: f64,
    kl_q_f: f64,
    cosine: f64,
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&x| ((x - m) as f64).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

fn argmax(logits: &[f32]) -> u32 {
    let (i, _) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    i as u32
}

fn topk_indices(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.iter().take(k).map(|&i| i as u32).collect()
}

fn compare_logits(f16: &[f32], q8: &[f32]) -> PerStepStats {
    let f16_top1 = argmax(f16);
    let q8_top5 = topk_indices(q8, 5);
    let f16_p = softmax(f16);
    let q8_p = softmax(q8);

    // KL(p || q) = sum_i p_i * log(p_i / q_i). Clamp q_i to avoid -inf
    // when one distribution puts ~0 mass where the other puts > 0.
    const EPS: f64 = 1e-12;
    let kl = |p: &[f64], q: &[f64]| -> f64 {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| if pi > EPS { pi * (pi.max(EPS) / qi.max(EPS)).ln() } else { 0.0 })
            .sum()
    };

    let dot: f64 = f16.iter().zip(q8.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
    let na: f64 = f16.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = q8.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };

    PerStepStats {
        top1_match: argmax(q8) == f16_top1,
        top5_match: q8_top5.contains(&f16_top1),
        kl_f_q: kl(&f16_p, &q8_p),
        kl_q_f: kl(&q8_p, &f16_p),
        cosine,
    }
}

/// Run `n_steps` decode steps under the given KV layout (selected via
/// FLAMBEAU_KV env). Returns logits[step][vocab].
fn run_decode_capture(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    cfg: &Qwen3MoEConfig,
    kv_env: &str,
    seed_token: u32,
    n_steps: usize,
) -> Result<Vec<Vec<f32>>> {
    // SAFETY: this test is single-threaded by construction (#[test] runs
    // serially on this binary). The env mutation is the only way to thread
    // the layout into Qwen3MoEShardedSession::new without changing the API
    // surface for #117 — promotion + a typed ctor param lands in #118.
    unsafe {
        std::env::set_var("FLAMBEAU_KV", kv_env);
    }
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut scratch = ShardedForwardOneTokenScratch::new(model, cluster)?;
    let mut logits_out: Vec<f32> = vec![0.0; cfg.vocab_size];
    let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(n_steps);

    let mut tok = seed_token;
    for pos in 0..n_steps {
        forward_one_token_pp_logits(
            model, &mut session, cluster, &mut scratch, tok, pos, &mut logits_out,
        )
        .with_context(|| format!("forward_one_token_pp_logits step={pos} kv={kv_env}"))?;
        all_logits.push(logits_out.clone());
        // Use F16's argmax to drive both runs through the same trajectory
        // — without this, after the first divergence the Q8 run wanders
        // into a different prompt context and KL becomes meaningless.
        // Caller injects the F16 next-token; for the F16 pass we just use
        // its own argmax.
        tok = argmax(&logits_out);
    }
    scratch.dispose(cluster).ok();
    session.dispose(cluster).ok();
    Ok(all_logits)
}

/// Same as [`run_decode_capture`] but the Q8 run is forced to use F16's
/// trajectory token-by-token. This isolates per-step quantisation noise
/// from compound divergence after a top-1 disagreement.
fn run_q8_locked_to_f16(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    cfg: &Qwen3MoEConfig,
    seed_token: u32,
    f16_trajectory: &[u32],
) -> Result<Vec<Vec<f32>>> {
    unsafe {
        std::env::set_var("FLAMBEAU_KV", "q8");
    }
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut scratch = ShardedForwardOneTokenScratch::new(model, cluster)?;
    let mut logits_out: Vec<f32> = vec![0.0; cfg.vocab_size];
    let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(f16_trajectory.len());

    let mut tok = seed_token;
    for (pos, &next_tok) in f16_trajectory.iter().enumerate() {
        forward_one_token_pp_logits(
            model, &mut session, cluster, &mut scratch, tok, pos, &mut logits_out,
        )
        .with_context(|| format!("forward_one_token_pp_logits Q8 step={pos}"))?;
        all_logits.push(logits_out.clone());
        // Lock to F16's trajectory: feed the F16-chosen next token, not Q8's.
        tok = next_tok;
    }
    scratch.dispose(cluster).ok();
    session.dispose(cluster).ok();
    Ok(all_logits)
}

#[test]
fn q8_kv_quality_cert() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices — got {n}, skipping");
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    eprintln!("=== F16 KV trajectory ===");
    let f16_logits = run_decode_capture(&model, &cluster, &cfg, "f16", SEED_TOKEN, N_DECODE)?;
    let f16_traj: Vec<u32> = f16_logits.iter().map(|l| argmax(l)).collect();
    eprintln!("F16 trajectory: {:?}", &f16_traj);

    eprintln!("=== Q8 KV (locked to F16 trajectory) ===");
    let q8_logits = run_q8_locked_to_f16(&model, &cluster, &cfg, SEED_TOKEN, &f16_traj)?;
    let q8_traj: Vec<u32> = q8_logits.iter().map(|l| argmax(l)).collect();
    eprintln!("Q8  argmaxes:    {:?}", &q8_traj);

    // Per-step diff
    let stats: Vec<PerStepStats> = f16_logits
        .iter()
        .zip(q8_logits.iter())
        .map(|(f, q)| compare_logits(f, q))
        .collect();
    let n_steps = stats.len() as f64;
    let top1_agree = stats.iter().filter(|s| s.top1_match).count() as f64 / n_steps;
    let top5_agree = stats.iter().filter(|s| s.top5_match).count() as f64 / n_steps;
    let mean_kl_fq = stats.iter().map(|s| s.kl_f_q).sum::<f64>() / n_steps;
    let mean_kl_qf = stats.iter().map(|s| s.kl_q_f).sum::<f64>() / n_steps;
    let max_kl = stats.iter().map(|s| s.kl_f_q.max(s.kl_q_f)).fold(0.0f64, f64::max);
    let mean_cos = stats.iter().map(|s| s.cosine).sum::<f64>() / n_steps;

    let model_tag = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    // Resolve relative to the workspace root (cargo test runs the binary
    // with cwd = crate dir, but our cert canon lives at workspace-root).
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("workspace root above CARGO_MANIFEST_DIR")
        .to_path_buf();
    let cert_path = workspace_root
        .join("certs/quality")
        .join(format!("{model_tag}_q8_kv.json"));
    std::fs::create_dir_all(cert_path.parent().unwrap())?;

    let cert = serde_json::json!({
        "schema_version": 1,
        "kind": "q8_kv_quality_cert",
        "model": model_tag,
        "n_decode_steps": N_DECODE,
        "seed_token": SEED_TOKEN,
        "method": "differential: F16 trajectory drives both runs; Q8 logits captured under F16-chosen next tokens (per-step quant noise, no compound divergence).",
        "f16_trajectory": f16_traj,
        "q8_trajectory_locked": q8_traj,
        "metrics": {
            "top1_agreement": top1_agree,
            "top5_agreement": top5_agree,
            "mean_kl_f16_q8": mean_kl_fq,
            "mean_kl_q8_f16": mean_kl_qf,
            "max_kl_per_step": max_kl,
            "mean_cosine_similarity": mean_cos,
        },
        "gate": {
            "mean_kl_max": 0.005,
            "max_kl_per_step_max": 0.05,
            "top5_agreement_min": 1.0,
        },
        "pass": mean_kl_fq <= 0.005 && max_kl <= 0.05 && top5_agree >= 1.0,
        "rig": "threadreaper-gfx906",
    });
    std::fs::write(&cert_path, serde_json::to_string_pretty(&cert)? + "\n")?;
    eprintln!("Wrote {}", cert_path.display());

    eprintln!(
        "top1 agree: {:.2} %, top5 agree: {:.2} %, mean_KL: {:.4}/{:.4}, max_KL: {:.4}, cos: {:.4}",
        top1_agree * 100.0, top5_agree * 100.0, mean_kl_fq, mean_kl_qf, max_kl, mean_cos,
    );

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
