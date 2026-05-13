//! #25 + #26 — real-GGUF PP upload + decode smoke. Loads
//! gemma-4-31B-it-Q4_0.gguf across 2 ranks via
//! [`Gemma4PpDriver::upload`], validates stage ownership / KV-cache
//! shape, prefills a short prompt, decodes a handful of tokens, then
//! disposes cleanly. Single test (not split) because two simultaneous
//! 31B uploads exhaust the 2× 16 GB MI50 VRAM pool. Skipped when the
//! GGUF is absent or fewer than 2 HIP devices are present.

#![cfg(feature = "hip")]

use std::path::Path;

use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_gemma4::{partition_layers, Gemma4Config, Gemma4PpDriver, ModelLayout};
use flambeau_quant::GgufFile;

const MODELS_DIR: &str = "/artefact/models";
const N_RANKS: usize = 2;
const MAX_TOKENS: usize = 64;

fn open_or_skip(name: &str) -> Option<GgufFile> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok()
}

fn cluster_or_skip() -> Option<HipCluster> {
    let n = device_count().ok()?;
    if (n as usize) < N_RANKS {
        eprintln!(
            "skipping real-GGUF PP smoke — need {N_RANKS} HIP devices but only {n}"
        );
        return None;
    }
    let ids: Vec<i32> = (0..N_RANKS as i32).collect();
    HipCluster::new(&ids).ok()
}

#[test]
fn real_gguf_pp_upload_and_decode_31b_q4_0() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    let Some(cluster) = cluster_or_skip() else {
        return;
    };

    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let layer_to_rank = partition_layers(N_RANKS, &layout).expect("partition");

    let mut driver = Gemma4PpDriver::upload(
        &file,
        cfg.clone(),
        layout,
        layer_to_rank,
        cluster,
        MAX_TOKENS,
    )
    .expect("upload");

    // Stage ownership invariants.
    assert_eq!(driver.stages.len(), N_RANKS);
    let total_layers: usize = driver
        .stages
        .iter()
        .map(|s| s.global_layer_indices.len())
        .sum();
    assert_eq!(total_layers, cfg.num_layers, "uploaded layer total");

    let s0 = &driver.stages[0];
    assert!(s0.token_embd.is_some(), "rank 0 owns token_embd");
    assert!(s0.token_embd_dims.is_some(), "rank 0 token_embd_dims");
    let last = &driver.stages[N_RANKS - 1];
    assert!(last.output_norm.is_some(), "last rank owns output_norm");
    assert!(last.output.is_some(), "last rank owns LM-head replica");
    assert!(
        last.output_head_scratch.is_some(),
        "last rank owns output-head scratch"
    );
    for (rank, stage) in driver.stages.iter().enumerate() {
        assert_eq!(
            stage.kv_caches.len(),
            stage.global_layer_indices.len(),
            "rank {rank} kv_caches len"
        );
        for (li, kv) in stage.kv_caches.iter().enumerate() {
            assert!(
                kv.is_some(),
                "rank {rank} local layer {li} expected KV (no shared-KV on 31B)"
            );
        }
    }

    // Prefill + decode. Exercises the head_dim=512 attention kernels
    // (flash_tile d512 + oracle decode/prefill MAX 512) — gemma4 31B
    // sets key_length=512 on every full-attn layer.
    let prompt: Vec<u32> = vec![2, 105, 23, 7, 35, 1234];
    let prompt_len = prompt.len();
    let first = driver.forward_prefill(&prompt, 0).expect("prefill");
    assert!(
        (first as usize) < cfg.vocab_size,
        "prefill argmax oob: {first}"
    );

    let mut tok = first;
    let mut pos = prompt_len;
    for step in 0..6 {
        let next = driver
            .forward_one_token(tok, pos)
            .unwrap_or_else(|e| panic!("decode step {step}: {e}"));
        assert!(
            (next as usize) < cfg.vocab_size,
            "decode step {step}: argmax {next} oob"
        );
        tok = next;
        pos += 1;
    }

    let total = prompt_len + 6;
    let r0_first = driver.stages[0].kv_caches[0]
        .as_ref()
        .expect("rank 0 first KV")
        .current_tokens();
    assert_eq!(r0_first, total, "rank 0 first KV current_tokens");

    driver.dispose().expect("dispose");
}
