//! #22 — real-GGUF single-device smoke for gemma4 E4B-Q4_0. Loads
//! per-layer-embd weights + globals, builds the per-layer table for
//! each input token from the GGUF mmap, applies the sqrt(n_embd)
//! embedding scale, runs 6 decode tokens, verifies argmax stays in
//! vocab. Skipped when the GGUF is absent or no HIP device.

#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::Arc;

use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_gemma4::{
    forward_one_token, Gemma4Config, Gemma4DeviceWeights, Gemma4Session, ModelLayout,
};
use flambeau_quant::GgufFile;

const MODELS_DIR: &str = "/artefact/models";
const MAX_TOKENS: usize = 64;

fn open_or_skip(name: &str) -> Option<Arc<GgufFile>> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok().map(Arc::new)
}

fn device_or_skip() -> Option<HipDevice> {
    let n = device_count().ok()?;
    if n < 1 {
        eprintln!("skipping — no HIP devices");
        return None;
    }
    HipDevice::new(0).ok()
}

#[test]
fn real_gguf_e4b_q4_0_single_device_smoke() {
    let Some(file) = open_or_skip("gemma-4-E4B-it-Q4_0.gguf") else {
        return;
    };
    let Some(device) = device_or_skip() else {
        return;
    };
    device.bind().expect("bind");

    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    assert!(
        cfg.per_layer_embed.is_some(),
        "E4B should have per_layer_embed in cfg"
    );
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let weights = Gemma4DeviceWeights::upload(&file, &cfg, &layout, &device)
        .expect("upload (real-GGUF E4B-Q4_0)");
    assert!(
        weights.per_layer_embd_globals.is_some(),
        "E4B upload should populate per_layer_embd_globals"
    );
    assert_eq!(weights.layers.len(), cfg.num_layers);
    for (i, lw) in weights.layers.iter().enumerate() {
        assert!(
            lw.per_layer_embed.is_some(),
            "layer {i} should have per-layer-embd weights uploaded"
        );
    }

    let mut session = Gemma4Session::new_with_gguf(
        &device,
        weights,
        cfg.clone(),
        layout,
        MAX_TOKENS,
        file,
    )
    .expect("session");

    // Decode 6 tokens starting from a small synthetic prompt suffix.
    // BOS=2, then a couple of arbitrary in-vocab tokens.
    let mut tok = 2u32;
    let mut pos = 0usize;
    for step in 0..6 {
        let next = forward_one_token(&mut session, &device, tok, pos)
            .unwrap_or_else(|e| panic!("decode step {step}: {e}"));
        assert!(
            (next as usize) < cfg.vocab_size,
            "step {step}: argmax {next} oob (vocab={})",
            cfg.vocab_size
        );
        tok = next;
        pos += 1;
    }

    session.dispose(&device).expect("dispose");
}
