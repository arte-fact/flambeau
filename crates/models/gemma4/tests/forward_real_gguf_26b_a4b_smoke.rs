//! Phase 10c-G bisect: real-GGUF single-device smoke for gemma4
//! 26B-A4B-Q8_0. Diagnostic — confirms whether the F16 overflow at
//! layer 5 (caught in `smoke_26b_a4b_q8_0_tp2`) is TP-specific or a
//! gemma4-general MoE bug.
//!
//! Skipped when the GGUF is absent or no HIP device.

#![cfg(feature = "hip")]

mod common;

use flambeau_gemma4::{
    forward_one_token, Gemma4Config, Gemma4DeviceWeights, Gemma4Session, ModelLayout,
};

use common::{device_or_skip, open_or_skip};

const MAX_TOKENS: usize = 64;

#[test]
fn real_gguf_26b_a4b_q8_0_single_device_smoke() {
    let Some(file) = open_or_skip("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    let Some(device) = device_or_skip() else {
        return;
    };
    device.bind().expect("bind");

    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    assert!(cfg.moe.is_some(), "26B-A4B should have cfg.moe = Some");
    let vocab = cfg.vocab_size;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let weights = Gemma4DeviceWeights::upload(&file, &cfg, &layout, &device)
        .expect("upload (real-GGUF 26B-A4B-Q8_0)");

    let mut session =
        Gemma4Session::new_with_gguf(&device, weights, cfg.clone(), layout, MAX_TOKENS, file)
            .expect("session");

    let mut tok = 2u32;
    let mut pos = 0usize;
    let mut ids = Vec::with_capacity(16);
    for step in 0..16 {
        let next = forward_one_token(&mut session, &device, tok, pos)
            .unwrap_or_else(|e| panic!("decode step {step}: {e}"));
        assert!(
            (next as usize) < vocab,
            "step {step}: argmax {next} oob (vocab={vocab})"
        );
        ids.push(next);
        tok = next;
        pos += 1;
    }
    eprintln!("\n=== SMOKE | 26B-A4B-Q8_0 single-device ===");
    eprintln!("  generated ids: {ids:?}");
    let first = ids[0];
    let all_same = ids.iter().all(|&t| t == first);
    if all_same {
        eprintln!(
            "  [WARN] all {} decoded tokens identical ({first}) — gemma4 single-device \
             MoE forward likely producing constant logits (not a TP-specific bug)",
            ids.len()
        );
    } else {
        eprintln!("  [OK] non-degenerate output");
    }

    session.dispose(&device).expect("dispose");
}
