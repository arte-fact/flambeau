//! **#308 microbench** — does Sarathi mixed-batch dispatch beat
//! sequential `prefill_B then decode_batched_N`?
//!
//! Compares two wall-clock paths on the same workload:
//! - **sequential**: `forward_prefill_hybrid_logits(K)` then
//!   `forward_decode_batched_hybrid(N)`.
//! - **mixed**: one `forward_decode_mixed_hybrid` call with
//!   chunk = K + slots = N.
//!
//! Reports median wall over warmups + N reps, plus aggregate
//! throughput (tokens / wall) for both paths.
//!
//! Skips when GGUF missing or fewer than 4 HIP devices. Defaults to
//! Qwen3.5-9B-Q4_1 on pp2tp2.
//!
//! **Run alone** — multi-test in one process hits the
//! BarP2pAllReduce cluster-state-leak constraint (see
//! mixed_batch_parity.rs header).

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — model load + forward + dispose"
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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

fn median_ms(samples: &mut Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[test]
fn mixed_batch_microbench_pp2tp2_k512_n4() -> Result<()> {
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

    // Bench config: K=512 prefill chunk + N=4 concurrent decodes.
    // K=512 is Sarathi's sweet-spot; N=4 matches the v2 cert
    // 4-concurrent-user workload.
    let k = std::env::var("FLAMBEAU_BENCH_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512usize);
    let n_dec = std::env::var("FLAMBEAU_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize);
    let warmups: usize = 2;
    let reps: usize = 5;
    let decode_prompt_len: usize = 64;
    let t = k + n_dec;

    let prompt_b: Vec<u32> = (0..k as u32).map(|i| (11 + i * 53) % 151000).collect();
    let decode_prompts: Vec<Vec<u32>> = (0..n_dec)
        .map(|s| {
            (0..decode_prompt_len as u32)
                .map(move |i| ((s as u32 * 37 + 5) + i * 41) % 151000)
                .collect::<Vec<u32>>()
        })
        .collect();
    let decode_tokens: Vec<u32> = (0..n_dec).map(|s| (17 + s as u32 * 19) % 151000).collect();
    let pos_per_slot: Vec<usize> = (0..n_dec).map(|_| decode_prompt_len).collect();

    eprintln!("microbench: K={k} N={n_dec} T={t} | warmups={warmups} reps={reps}");

    // Helper: build fresh sessions + scratch for one trial.
    let build_sessions = || -> Result<(
        Vec<Qwen3MoEHybridSession>,
        ShardedForwardOneTokenScratchHybrid,
        ShardedForwardPrefillScratchHybrid,
    )> {
        let mut sessions: Vec<Qwen3MoEHybridSession> = (0..(n_dec + 1))
            .map(|_| Qwen3MoEHybridSession::new(&model))
            .collect::<Result<_>>()?;
        let mut prefill_scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
        let decode_scratch = ShardedForwardPrefillScratchHybrid::new(&model, t)?;

        // Pre-prefill the decode sessions (cost not in either path's timing).
        for s in 0..n_dec {
            let mut sink: Vec<f32> = Vec::new();
            forward_prefill_hybrid_logits(
                &model,
                &mut prefill_scratch,
                &global_cluster,
                &stage_ars,
                &mut sessions[s],
                &decode_prompts[s],
                0,
                &mut sink,
            )
            .with_context(|| format!("setup: prefill decode-session {s}"))?;
        }
        Ok((sessions, prefill_scratch, decode_scratch))
    };

    // ── PATH A: SEQUENTIAL (prefill B then decode batch) ─────────
    let mut seq_samples_ms: Vec<f64> = Vec::with_capacity(reps);
    for trial in 0..(warmups + reps) {
        let (mut sessions, mut prefill_scratch, mut decode_scratch) = build_sessions()?;

        let t_start = Instant::now();
        // Step 1: prefill B.
        let mut logits_b: Vec<f32> = Vec::new();
        forward_prefill_hybrid_logits(
            &model,
            &mut prefill_scratch,
            &global_cluster,
            &stage_ars,
            &mut sessions[n_dec],
            &prompt_b,
            0,
            &mut logits_b,
        )
        .context("seq: prefill B")?;
        // Step 2: batched decode of N decode-slots.
        {
            let mut logits_decode: Vec<Vec<f32>> = (0..(n_dec + 1)).map(|_| Vec::new()).collect();
            let mut sess_refs: Vec<&mut Qwen3MoEHybridSession> = sessions.iter_mut().collect();
            let slots: Vec<BatchSlot> = (0..n_dec)
                .map(|s| BatchSlot {
                    idx: s,
                    token_id: decode_tokens[s],
                    position: pos_per_slot[s],
                })
                .collect();
            let mut logits_refs: Vec<&mut Vec<f32>> = logits_decode.iter_mut().collect();
            forward_decode_batched_hybrid(
                &model,
                sess_refs.as_mut_slice(),
                &global_cluster,
                &stage_ars,
                &mut decode_scratch,
                &slots,
                logits_refs.as_mut_slice(),
            )
            .context("seq: decode batch")?;
        }
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if trial >= warmups {
            seq_samples_ms.push(elapsed_ms);
            eprintln!("seq trial {trial}: {elapsed_ms:.2} ms");
        }

        // dispose
        let _ = prefill_scratch.dispose(&model);
        let _ = decode_scratch.dispose(&model);
        for s in sessions {
            let _ = s.dispose(&model);
        }
    }
    let seq_med_ms = median_ms(&mut seq_samples_ms.clone());

    // ── PATH B: MIXED (one mixed call) ───────────────────────────
    let mut mix_samples_ms: Vec<f64> = Vec::with_capacity(reps);
    for trial in 0..(warmups + reps) {
        let (mut sessions, mut prefill_scratch, mut decode_scratch) = build_sessions()?;

        let t_start = Instant::now();
        let chunk = MixedPrefillChunk {
            idx: n_dec,
            tokens: prompt_b.clone(),
            chunk_start: 0,
            is_final_chunk: true,
        };
        let slots: Vec<BatchSlot> = (0..n_dec)
            .map(|s| BatchSlot {
                idx: s,
                token_id: decode_tokens[s],
                position: pos_per_slot[s],
            })
            .collect();
        let mut logits_decode: Vec<Vec<f32>> = (0..(n_dec + 1)).map(|_| Vec::new()).collect();
        let mut prefill_final: Vec<f32> = Vec::new();
        {
            let mut sess_refs: Vec<&mut Qwen3MoEHybridSession> = sessions.iter_mut().collect();
            let mut logits_refs: Vec<&mut Vec<f32>> = logits_decode.iter_mut().collect();
            forward_decode_mixed_hybrid(
                &model,
                sess_refs.as_mut_slice(),
                &global_cluster,
                &stage_ars,
                &mut decode_scratch,
                Some(&chunk),
                &slots,
                logits_refs.as_mut_slice(),
                Some(&mut prefill_final),
            )
            .context("mix: forward_decode_mixed_hybrid")?;
        }
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if trial >= warmups {
            mix_samples_ms.push(elapsed_ms);
            eprintln!("mix trial {trial}: {elapsed_ms:.2} ms");
        }

        // dispose
        let _ = prefill_scratch.dispose(&model);
        let _ = decode_scratch.dispose(&model);
        for s in sessions {
            let _ = s.dispose(&model);
        }
    }
    let mix_med_ms = median_ms(&mut mix_samples_ms.clone());

    let _ = model.dispose();

    // ── REPORT ───────────────────────────────────────────────────
    let tokens_processed = (k + n_dec) as f64;
    let seq_throughput = tokens_processed / (seq_med_ms / 1000.0);
    let mix_throughput = tokens_processed / (mix_med_ms / 1000.0);
    let speedup = seq_med_ms / mix_med_ms;

    eprintln!();
    eprintln!("=== mixed-batch microbench K={k} N={n_dec} pp2tp2 ===");
    eprintln!("  sequential median: {seq_med_ms:.2} ms ({seq_throughput:.1} tok/s aggregate)");
    eprintln!("  mixed      median: {mix_med_ms:.2} ms ({mix_throughput:.1} tok/s aggregate)");
    eprintln!("  speedup (seq/mix): {speedup:.3}x");
    eprintln!("  seq samples ms: {seq_samples_ms:?}");
    eprintln!("  mix samples ms: {mix_samples_ms:?}");

    Ok(())
}
