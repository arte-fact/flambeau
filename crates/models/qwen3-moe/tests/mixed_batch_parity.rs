//! **#307 — Sarathi mixed-batch parity**.
//!
//! Validates `forward_decode_mixed_hybrid` against a separate-sessions
//! reference. Two parallel runs:
//!
//! - **reference**: session A prefilled then decoded one step in
//!   isolation (gives `logits_a_dec_ref`); session B prefilled in
//!   isolation (gives `logits_b_pre_ref`, the last-row logits).
//! - **mixed**: same end states reached via one mixed call —
//!   `forward_decode_mixed_hybrid` with `chunk = prompt_B` (final
//!   chunk) and `slots = [BatchSlot { idx: A, ... }]`.
//!
//! Asserts: per-stream logits within F16 relative tolerance and
//! top-1 token-id parity.
//!
//! Skips when GGUF missing or fewer than 4 HIP devices. Defaults to
//! Qwen3.5-9B-Q4_1 on pp2tp2 over devices [0,2,1,3].
//!
//! **Known constraint**: each test reloads the model + cluster +
//! BarP2pAllReduce. Running multiple tests in the same process can hit
//! a stale BAR P2P matrix on the second test (`BarP2pAllReduce::new`
//! requires fully-connected peer access; ROCm doesn't fully reset it
//! across cluster lifecycles). Run tests one at a time:
//!     cargo test ... --test mixed_batch_parity mixed_batch_parity_pp2tp2_k32_n1 -- --nocapture
//! or use --test-threads=1 with each test in its own cargo invocation.

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
    forward_decode_batched_hybrid, forward_decode_mixed_hybrid,
    forward_prefill_hybrid_logits, BatchSlot, MixedPrefillChunk,
    ShardedForwardOneTokenScratchHybrid,
};
use flambeau_qwen3_moe::hybrid::HybridMeshSpec;
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEHybridModel, Qwen3MoEHybridSession,
    ShardedForwardPrefillScratchHybrid,
};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

/// Hybrid abs+rel error check on F16-rounded logits.
///
/// Element passes if `|a-b| <= abs_tol` OR `|a-b| / max(|a|,|b|) <= rel_tol`.
/// Pure-relative is brittle for logits near zero (denominator → 0).
fn close_enough(label: &str, a: &[f32], b: &[f32], abs_tol: f32, rel_tol: f32) -> Result<()> {
    if a.len() != b.len() {
        anyhow::bail!("{label}: len mismatch {} vs {}", a.len(), b.len());
    }
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_idx = 0usize;
    let mut violations = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        let s = x.abs().max(y.abs()).max(1e-6);
        let r = d / s;
        if d > abs_tol && r > rel_tol {
            violations += 1;
        }
        if d > max_abs {
            max_abs = d;
            max_rel = r;
            max_idx = i;
        }
    }
    eprintln!(
        "[{label}] max_abs={max_abs:.3e} (rel={max_rel:.3e}) at idx={max_idx} (a={}, b={}); \
         violations={violations}",
        a[max_idx], b[max_idx]
    );
    if violations > 0 {
        anyhow::bail!(
            "{label}: {violations} elements exceed abs_tol={abs_tol:.1e} AND rel_tol={rel_tol:.1e}",
        );
    }
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
            if x > acc.1 { (i, x) } else { acc }
        })
        .0 as u32
}

struct ParityCfg {
    label: &'static str,
    /// K — prefill chunk length.
    chunk_k: usize,
    /// N decode-slot count.
    n_decode: usize,
    /// Per-slot pre-decode prompt length.
    decode_prompt_len: usize,
}

fn run_parity(cfg: &ParityCfg) -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip [{}] — Qwen3.5-9B GGUF not present", cfg.label);
        return Ok(());
    };
    let dev_ids = vec![0i32, 2, 1, 3];
    let n_avail: i32 = device_count().unwrap_or(0);
    if (n_avail as usize) < dev_ids.len() {
        eprintln!("skip [{}] — need 4 HIP devices, have {n_avail}", cfg.label);
        return Ok(());
    }

    std::env::set_var("FLAMBEAU_MAX_CTX", "4096");

    let file = GgufFile::open(&path)?;
    let model_cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(model_cfg.arch, "qwen35");

    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    spec.validate(model_cfg.num_layers, dev_ids.len())?;

    let model = Qwen3MoEHybridModel::load(&file, &dev_ids, spec)?;
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&dev_ids)?);

    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        let ar = BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))
            .with_context(|| format!("BarP2pAllReduce::new for stage {}", stage.stage_idx))?;
        stage_ars.push(ar);
    }

    let k = cfg.chunk_k;
    let n_dec = cfg.n_decode;
    let t = k + n_dec;

    // Build N decode-session prompts (each length cfg.decode_prompt_len, shifted seeds).
    // Sessions are 0..n_dec (decoders), n_dec (chunk owner B). Layout matches the test below.
    let decode_prompts: Vec<Vec<u32>> = (0..n_dec)
        .map(|s| {
            (0..cfg.decode_prompt_len as u32)
                .map(move |i| ((s as u32 * 37 + 5) + i * 41) % 151000)
                .collect::<Vec<u32>>()
        })
        .collect();
    let prompt_b: Vec<u32> = (0..k as u32).map(|i| (11 + i * 53) % 151000).collect();
    let decode_tokens: Vec<u32> = (0..n_dec).map(|s| (17 + s as u32 * 19) % 151000).collect();
    let pos_per_slot: Vec<usize> = (0..n_dec).map(|_| cfg.decode_prompt_len).collect();

    eprintln!(
        "mixed-batch parity [{}]: K={k} N={n_dec} T={t} | decode-prompt-len={}",
        cfg.label, cfg.decode_prompt_len
    );

    // ── REFERENCE PATH ────────────────────────────────────────────
    let mut sessions_ref: Vec<Qwen3MoEHybridSession> = (0..(n_dec + 1))
        .map(|_| Qwen3MoEHybridSession::new(&model))
        .collect::<Result<_>>()?;
    let mut prefill_scratch_ref = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    let mut decode_scratch_ref = ShardedForwardPrefillScratchHybrid::new(&model, t)?;

    // (R1) prefill each decode session.
    for s in 0..n_dec {
        let mut sink: Vec<f32> = Vec::new();
        forward_prefill_hybrid_logits(
            &model,
            &mut prefill_scratch_ref,
            &global_cluster,
            &stage_ars,
            &mut sessions_ref[s],
            &decode_prompts[s],
            0,
            &mut sink,
        )
        .with_context(|| format!("ref: prefill decode-session {s}"))?;
    }

    // (R2) batched decode of all N decode-slots in one call.
    let mut logits_decode_ref: Vec<Vec<f32>> = (0..(n_dec + 1)).map(|_| Vec::new()).collect();
    {
        let mut sess_refs: Vec<&mut Qwen3MoEHybridSession> =
            sessions_ref.iter_mut().collect();
        let slots_ref: Vec<BatchSlot> = (0..n_dec)
            .map(|s| BatchSlot {
                idx: s,
                token_id: decode_tokens[s],
                position: pos_per_slot[s],
            })
            .collect();
        let mut logits_refs: Vec<&mut Vec<f32>> = logits_decode_ref.iter_mut().collect();
        forward_decode_batched_hybrid(
            &model,
            sess_refs.as_mut_slice(),
            &global_cluster,
            &stage_ars,
            &mut decode_scratch_ref,
            &slots_ref,
            logits_refs.as_mut_slice(),
        )
        .context("ref: decode batch")?;
    }
    let argmax_decode_ref: Vec<u32> =
        (0..n_dec).map(|s| argmax(&logits_decode_ref[s])).collect();

    // (R3) prefill B → last-row logits.
    let mut logits_b_pre_ref: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut prefill_scratch_ref,
        &global_cluster,
        &stage_ars,
        &mut sessions_ref[n_dec],
        &prompt_b,
        0,
        &mut logits_b_pre_ref,
    )
    .context("ref: prefill B")?;
    let argmax_b_pre_ref = argmax(&logits_b_pre_ref);

    // ── MIXED PATH ────────────────────────────────────────────────
    let mut sessions_mix: Vec<Qwen3MoEHybridSession> = (0..(n_dec + 1))
        .map(|_| Qwen3MoEHybridSession::new(&model))
        .collect::<Result<_>>()?;
    let mut prefill_scratch_mix = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    let mut mixed_scratch = ShardedForwardPrefillScratchHybrid::new(&model, t)?;

    // (M1) prefill each decode session (same as R1).
    for s in 0..n_dec {
        let mut sink: Vec<f32> = Vec::new();
        forward_prefill_hybrid_logits(
            &model,
            &mut prefill_scratch_mix,
            &global_cluster,
            &stage_ars,
            &mut sessions_mix[s],
            &decode_prompts[s],
            0,
            &mut sink,
        )
        .with_context(|| format!("mix: prefill decode-session {s}"))?;
    }

    // (M2) one mixed call: chunk B + N decodes.
    let chunk = MixedPrefillChunk {
        idx: n_dec,
        tokens: prompt_b.clone(),
        chunk_start: 0,
        is_final_chunk: true,
    };
    let slots_mix: Vec<BatchSlot> = (0..n_dec)
        .map(|s| BatchSlot {
            idx: s,
            token_id: decode_tokens[s],
            position: pos_per_slot[s],
        })
        .collect();
    let mut logits_decode_mix: Vec<Vec<f32>> = (0..(n_dec + 1)).map(|_| Vec::new()).collect();
    let mut prefill_final_b_mix: Vec<f32> = Vec::new();
    {
        let mut sess_refs: Vec<&mut Qwen3MoEHybridSession> =
            sessions_mix.iter_mut().collect();
        let mut logits_refs: Vec<&mut Vec<f32>> = logits_decode_mix.iter_mut().collect();
        forward_decode_mixed_hybrid(
            &model,
            sess_refs.as_mut_slice(),
            &global_cluster,
            &stage_ars,
            &mut mixed_scratch,
            Some(&chunk),
            &slots_mix,
            logits_refs.as_mut_slice(),
            Some(&mut prefill_final_b_mix),
        )
        .context("mix: forward_decode_mixed_hybrid")?;
    }
    let argmax_decode_mix: Vec<u32> =
        (0..n_dec).map(|s| argmax(&logits_decode_mix[s])).collect();
    let argmax_b_pre_mix = argmax(&prefill_final_b_mix);

    eprintln!(
        "[{}] argmax: decode ref={:?} mix={:?} | B.pre ref={argmax_b_pre_ref} mix={argmax_b_pre_mix}",
        cfg.label, argmax_decode_ref, argmax_decode_mix
    );

    // ── DISPOSE ──────────────────────────────────────────────────
    let _ = prefill_scratch_ref.dispose(&model);
    let _ = decode_scratch_ref.dispose(&model);
    let _ = prefill_scratch_mix.dispose(&model);
    let _ = mixed_scratch.dispose(&model);
    for s in sessions_ref {
        let _ = s.dispose(&model);
    }
    for s in sessions_mix {
        let _ = s.dispose(&model);
    }
    let _ = model.dispose();

    // ── ASSERT ───────────────────────────────────────────────────
    for s in 0..n_dec {
        close_enough(
            &format!("[{}] decode-{s} logits", cfg.label),
            &logits_decode_ref[s],
            &logits_decode_mix[s],
            0.5,  // abs_tol — F16 logit-magnitude noise budget
            1e-2, // rel_tol — for large-magnitude logits
        )?;
        if argmax_decode_ref[s] != argmax_decode_mix[s] {
            anyhow::bail!(
                "[{}] decode-{s} top-1 mismatch: ref={} mix={}",
                cfg.label,
                argmax_decode_ref[s],
                argmax_decode_mix[s],
            );
        }
    }
    close_enough(
        &format!("[{}] prefill-B last-row logits", cfg.label),
        &logits_b_pre_ref,
        &prefill_final_b_mix,
        0.5,
        1e-2,
    )?;
    if argmax_b_pre_ref != argmax_b_pre_mix {
        anyhow::bail!(
            "[{}] prefill-B top-1 mismatch: ref={argmax_b_pre_ref} mix={argmax_b_pre_mix}",
            cfg.label,
        );
    }
    eprintln!("OK [{}]: parity matches reference (per-element + top-1)", cfg.label);
    Ok(())
}

#[test]
fn mixed_batch_parity_pp2tp2_k32_n1() -> Result<()> {
    run_parity(&ParityCfg {
        label: "K=32/N=1",
        chunk_k: 32,
        n_decode: 1,
        decode_prompt_len: 24,
    })
}

#[test]
fn mixed_batch_parity_pp2tp2_k32_n2() -> Result<()> {
    run_parity(&ParityCfg {
        label: "K=32/N=2",
        chunk_k: 32,
        n_decode: 2,
        decode_prompt_len: 24,
    })
}

#[test]
fn mixed_batch_parity_pp2tp2_k128_n4() -> Result<()> {
    run_parity(&ParityCfg {
        label: "K=128/N=4",
        chunk_k: 128,
        n_decode: 4,
        decode_prompt_len: 32,
    })
}
