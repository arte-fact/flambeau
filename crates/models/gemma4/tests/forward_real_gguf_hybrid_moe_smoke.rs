//! Phase 10c-H smoke: real-GGUF 26B-A4B-Q8_0 on pp2tp2 (hip:0,2,1,3).
//! Exercises the hybrid MoE composer — F32 attention output path on
//! full-attn layers + F32 MoE cascade + per-stage AR — through a
//! 16-token greedy decode. Skipped when the GGUF is absent or fewer
//! than 4 HIP devices with full per-sub-cluster peer access are
//! available.

#![cfg(feature = "hip")]

mod common;

use flambeau_blocks::HybridCluster;
use flambeau_gemma4::{Gemma4Config, Gemma4HybridDriver, ModelLayout};

const MAX_TOKENS: usize = 64;
const N_DECODE: usize = 16;
const TP_SIZE: usize = 2;

#[test]
fn smoke_26b_a4b_q8_0_pp2tp2() {
    let Some(file) = common::open_or_skip("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    // Production mesh `hip:0,2,1,3` (MEMORY.md `never_tp4_use_pp2tp2`).
    let stage_ids: [&[i32]; 2] = [&[0, 2], &[1, 3]];
    let Some((subs, global, _mesh)) = common::hybrid_clusters_or_skip(&stage_ids) else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    assert!(cfg.moe.is_some(), "26B-A4B should have cfg.moe = Some");
    let vocab = cfg.vocab_size;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let hc = HybridCluster::new(subs, global, TP_SIZE).expect("HybridCluster");
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

    // Real prompt + coherence keyword — "not all same" + in-vocab is
    // too weak (see MEMORY.md `parity_vs_argmax_in_vocab`).
    let prompt_ids = common::tokenize_prompt(&file, common::PROMPT).expect("tokenize");
    let ids = common::greedy_decode(&mut driver, &prompt_ids, N_DECODE)
        .unwrap_or_else(|e| panic!("greedy decode: {e:#}"));
    for (step, &t) in ids.iter().enumerate() {
        assert!(
            (t as usize) < vocab,
            "step {step}: argmax {t} oob (vocab={vocab})"
        );
    }
    let tokenizer = flambeau_quant::load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&ids).unwrap_or_default();
    eprintln!("\n=== SMOKE | 26B-A4B-Q8_0 pp2tp2 (hip:0,2,1,3) ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  generated ids: {ids:?}");
    eprintln!("  flambeau text: {fb_text:?}");
    assert!(
        fb_text.to_lowercase().contains("paris"),
        "hybrid MoE pp2tp2 decode of 'The capital of France is' did NOT contain 'Paris'. \
         Got: {fb_text:?}"
    );
    eprintln!("  [OK] coherent output (contains 'Paris')");

    driver.dispose().expect("dispose");
}
