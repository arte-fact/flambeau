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

/// Relative-error check on F16-rounded logits.
fn close_enough(label: &str, a: &[f32], b: &[f32], rel_tol: f32) -> Result<()> {
    if a.len() != b.len() {
        anyhow::bail!("{label}: len mismatch {} vs {}", a.len(), b.len());
    }
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        let s = x.abs().max(y.abs()).max(1e-6);
        let r = d / s;
        if r > max_rel {
            max_rel = r;
            max_abs = d;
            max_idx = i;
        }
    }
    eprintln!(
        "[{label}] max_rel={max_rel:.3e} (abs={max_abs:.3e}) at idx={max_idx} (a={}, b={})",
        a[max_idx], b[max_idx]
    );
    if max_rel > rel_tol {
        anyhow::bail!(
            "{label}: max_rel {max_rel:.3e} exceeds tol {rel_tol:.1e}",
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

#[test]
fn mixed_batch_parity_pp2tp2() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5-9B GGUF not present");
        return Ok(());
    };
    let dev_ids = vec![0i32, 2, 1, 3];
    let n_avail: i32 = device_count().unwrap_or(0);
    if (n_avail as usize) < dev_ids.len() {
        eprintln!("skip — need 4 HIP devices, have {n_avail}");
        return Ok(());
    }

    std::env::set_var("FLAMBEAU_MAX_CTX", "4096");

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let spec = HybridMeshSpec { pp_size: 2, tp_size: 2 };
    spec.validate(cfg.num_layers, dev_ids.len())?;

    let model = Qwen3MoEHybridModel::load(&file, &dev_ids, spec)?;
    let global_cluster: Arc<HipCluster> = Arc::new(HipCluster::new(&dev_ids)?);

    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        let ar = BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))
            .with_context(|| format!("BarP2pAllReduce::new for stage {}", stage.stage_idx))?;
        stage_ars.push(ar);
    }

    // Test config: K = 32 prefill tokens for chunk B; A's prompt is 24
    // tokens (different length, exercises per-token positions).
    let prompt_a: Vec<u32> = (0..24u32).map(|i| (5 + i * 41) % 151000).collect();
    let prompt_b: Vec<u32> = (0..32u32).map(|i| (11 + i * 53) % 151000).collect();
    let k = prompt_b.len();
    let n_dec = 1usize;
    let t = k + n_dec;
    // Decode token for slot A — choose any in-vocab id.
    let decode_token_a: u32 = 17;
    let pos_a = prompt_a.len();

    eprintln!(
        "mixed-batch parity: K={k} N={n_dec} T={t} | A.prompt={} B.prompt={}",
        prompt_a.len(),
        prompt_b.len()
    );

    // ── REFERENCE PATH ────────────────────────────────────────────
    // sess_A_ref: prefill prompt_A; one decode step → logits_a_dec_ref
    // sess_B_ref: prefill prompt_B → logits_b_pre_ref (last-row)

    let mut sess_a_ref = Qwen3MoEHybridSession::new(&model)?;
    let mut sess_b_ref = Qwen3MoEHybridSession::new(&model)?;
    let mut prefill_scratch_ref = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    // Decode scratch sized for max_tokens >= max(K_b, T) = T
    let mut decode_scratch_ref = ShardedForwardPrefillScratchHybrid::new(&model, t)?;

    // (R1) prefill A
    let mut sink_a: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut prefill_scratch_ref,
        &global_cluster,
        &stage_ars,
        &mut sess_a_ref,
        &prompt_a,
        0,
        &mut sink_a,
    )
    .context("ref: prefill A")?;

    // (R2) decode A one step → logits_a_dec_ref
    let mut logits_a_dec_ref: Vec<f32> = Vec::new();
    {
        let mut sessions: Vec<&mut Qwen3MoEHybridSession> = vec![&mut sess_a_ref];
        let mut logits_refs: Vec<&mut Vec<f32>> = vec![&mut logits_a_dec_ref];
        forward_decode_batched_hybrid(
            &model,
            sessions.as_mut_slice(),
            &global_cluster,
            &stage_ars,
            &mut decode_scratch_ref,
            &[BatchSlot {
                idx: 0,
                token_id: decode_token_a,
                position: pos_a,
            }],
            logits_refs.as_mut_slice(),
        )
        .context("ref: decode A")?;
    }
    let argmax_a_dec_ref = argmax(&logits_a_dec_ref);

    // (R3) prefill B → logits_b_pre_ref (last-row)
    let mut logits_b_pre_ref: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut prefill_scratch_ref,
        &global_cluster,
        &stage_ars,
        &mut sess_b_ref,
        &prompt_b,
        0,
        &mut logits_b_pre_ref,
    )
    .context("ref: prefill B")?;
    let argmax_b_pre_ref = argmax(&logits_b_pre_ref);

    // ── MIXED PATH ────────────────────────────────────────────────
    let mut sess_a_mix = Qwen3MoEHybridSession::new(&model)?;
    let mut sess_b_mix = Qwen3MoEHybridSession::new(&model)?;
    let mut prefill_scratch_mix = ShardedForwardOneTokenScratchHybrid::new(&model)?;
    let mut mixed_scratch = ShardedForwardPrefillScratchHybrid::new(&model, t)?;

    // (M1) prefill A — same as R1 on the mix copy of A.
    let mut sink_a_mix: Vec<f32> = Vec::new();
    forward_prefill_hybrid_logits(
        &model,
        &mut prefill_scratch_mix,
        &global_cluster,
        &stage_ars,
        &mut sess_a_mix,
        &prompt_a,
        0,
        &mut sink_a_mix,
    )
    .context("mix: prefill A")?;

    // (M2) mixed call: chunk = prompt_B (final), slots = [decode A].
    let chunk = MixedPrefillChunk {
        idx: 1,
        tokens: prompt_b.clone(),
        chunk_start: 0,
        is_final_chunk: true,
    };
    let slots = [BatchSlot {
        idx: 0,
        token_id: decode_token_a,
        position: pos_a,
    }];
    let mut decode_logits_a_mix: Vec<f32> = Vec::new();
    let mut decode_logits_b_unused: Vec<f32> = Vec::new();
    let mut prefill_final_b_mix: Vec<f32> = Vec::new();
    {
        let mut sessions: Vec<&mut Qwen3MoEHybridSession> =
            vec![&mut sess_a_mix, &mut sess_b_mix];
        // decode_logits_out is parallel to sessions; only slot.idx
        // entries are written. B has no decode slot in this dispatch
        // so its entry stays untouched.
        let mut logits_refs: Vec<&mut Vec<f32>> =
            vec![&mut decode_logits_a_mix, &mut decode_logits_b_unused];
        forward_decode_mixed_hybrid(
            &model,
            sessions.as_mut_slice(),
            &global_cluster,
            &stage_ars,
            &mut mixed_scratch,
            Some(&chunk),
            &slots,
            logits_refs.as_mut_slice(),
            Some(&mut prefill_final_b_mix),
        )
        .context("mix: forward_decode_mixed_hybrid")?;
    }
    let argmax_a_dec_mix = argmax(&decode_logits_a_mix);
    let argmax_b_pre_mix = argmax(&prefill_final_b_mix);

    eprintln!(
        "argmax: A.dec ref={argmax_a_dec_ref} mix={argmax_a_dec_mix} | \
         B.pre ref={argmax_b_pre_ref} mix={argmax_b_pre_mix}"
    );

    // ── DISPOSE ──────────────────────────────────────────────────
    let _ = prefill_scratch_ref.dispose(&model);
    let _ = decode_scratch_ref.dispose(&model);
    let _ = prefill_scratch_mix.dispose(&model);
    let _ = mixed_scratch.dispose(&model);
    let _ = sess_a_ref.dispose(&model);
    let _ = sess_b_ref.dispose(&model);
    let _ = sess_a_mix.dispose(&model);
    let _ = sess_b_mix.dispose(&model);
    let _ = model.dispose();

    // ── ASSERT ───────────────────────────────────────────────────
    // F16 rounding tolerance ~1e-2 relative, somewhat loose. Pure
    // bit-id parity is unlikely (FMA contraction differs across two
    // attn calls vs one). Top-1 argmax match is the strict gate.
    close_enough(
        "decode-A logits",
        &logits_a_dec_ref,
        &decode_logits_a_mix,
        1e-2,
    )?;
    close_enough(
        "prefill-B last-row logits",
        &logits_b_pre_ref,
        &prefill_final_b_mix,
        1e-2,
    )?;
    if argmax_a_dec_ref != argmax_a_dec_mix {
        anyhow::bail!(
            "decode-A top-1 mismatch: ref={argmax_a_dec_ref} mix={argmax_a_dec_mix}",
        );
    }
    if argmax_b_pre_ref != argmax_b_pre_mix {
        anyhow::bail!(
            "prefill-B top-1 mismatch: ref={argmax_b_pre_ref} mix={argmax_b_pre_mix}",
        );
    }
    eprintln!("OK: mixed-batch parity matches reference (per-element + top-1)");
    Ok(())
}
