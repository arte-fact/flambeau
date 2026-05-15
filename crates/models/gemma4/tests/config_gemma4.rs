//! S4 — parse Gemma 4 config off the on-disk GGUFs in /artefact/models.
//!
//! Skipped (with stderr note) when the files are absent — keeps CI lint
//! and sandbox `cargo check` green.

use std::path::Path;

use flambeau_gemma4::{Gemma4Config, Gemma4Variant};
use flambeau_quant::GgufFile;

const MODELS_DIR: &str = "/artefact/models";

fn open_if_exists(name: &str) -> Option<GgufFile> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok()
}

#[test]
fn parse_e4b_q4_0() {
    let Some(file) = open_if_exists("gemma-4-E4B-it-Q4_0.gguf") else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("parse E4B-Q4_0");
    assert_eq!(cfg.variant, Gemma4Variant::E4B);
    assert_eq!(cfg.num_layers, 42);
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_heads, 8);
    // `key_length` / `key_length_swa` are per-head; E4B is uniform
    // n_kv=2 across all layers but the per-head dim differs.
    assert!(cfg.num_kv_heads.iter().all(|&v| v == 2));
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.swa.head_dim_swa, 256);
    assert_eq!(cfg.swa.sliding_window, 512);
    assert_eq!(cfg.shared_kv_layers, 18);
    assert_eq!(cfg.feed_forward_length, 10240);
    assert!(cfg.moe.is_none());
    let per = cfg.per_layer_embed.expect("E4B has per-layer embed");
    assert_eq!(per.n_embd_per_layer, 256);
    assert_eq!(cfg.final_logit_softcap, 30.0);
    assert!(cfg.tied_lm_head);
    assert_eq!(cfg.swa.swa_layers.len(), cfg.num_layers);
    // At least one full-attn layer (rope_freqs tensor exists in the file).
    assert!(cfg.swa.swa_layers.iter().any(|b| !b));
    // Tail shared-KV count matches metadata.
    let no_kv = (0..cfg.num_layers).filter(|&il| !cfg.has_kv(il)).count();
    assert_eq!(no_kv, cfg.shared_kv_layers);
}

#[test]
fn parse_26b_a4b_q8_0() {
    let Some(file) = open_if_exists("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("parse 26B-A4B-Q8_0");
    assert_eq!(cfg.variant, Gemma4Variant::Moe26BA4B);
    assert_eq!(cfg.num_layers, 30);
    assert_eq!(cfg.hidden_size, 2816);
    assert_eq!(cfg.num_heads, 16);
    // Non-uniform n_kv_heads — pattern (5 SWA × n_kv=8, 1 full × n_kv=2)×5.
    // SWA layers use head_dim_swa=256, full-attn use head_dim=512.
    let n_kv_swa = cfg.num_kv_heads.iter().filter(|&&v| v == 8).count();
    let n_kv_full = cfg.num_kv_heads.iter().filter(|&&v| v == 2).count();
    assert_eq!(n_kv_swa + n_kv_full, cfg.num_layers);
    assert_eq!(n_kv_full, 5);
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.swa.head_dim_swa, 256);
    assert_eq!(cfg.swa.sliding_window, 1024);
    assert_eq!(cfg.shared_kv_layers, 0);
    let m = cfg.moe.expect("MoE variant must carry moe");
    assert_eq!(m.num_experts, 128);
    assert_eq!(m.num_experts_per_tok, 8);
    assert_eq!(m.moe_intermediate_size, 704);
    assert_eq!(cfg.feed_forward_length, 2112);
    assert!(cfg.per_layer_embed.is_none());
    assert_eq!(cfg.final_logit_softcap, 30.0);
}

#[test]
fn parse_31b_q4_0() {
    let Some(file) = open_if_exists("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("parse 31B-Q4_0");
    assert_eq!(cfg.variant, Gemma4Variant::Dense31B);
    assert_eq!(cfg.num_layers, 60);
    assert_eq!(cfg.hidden_size, 5376);
    assert_eq!(cfg.num_heads, 32);
    // 31B-Q4_0: 50 SWA layers × n_kv=16, 10 full-attn × n_kv=4
    // (every 6th layer is full-attn).
    let n_kv_swa = cfg.num_kv_heads.iter().filter(|&&v| v == 16).count();
    let n_kv_full = cfg.num_kv_heads.iter().filter(|&&v| v == 4).count();
    assert_eq!(n_kv_swa + n_kv_full, cfg.num_layers);
    assert_eq!(n_kv_full, 10);
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.swa.head_dim_swa, 256);
    assert_eq!(cfg.swa.sliding_window, 1024);
    assert_eq!(cfg.shared_kv_layers, 0);
    assert!(cfg.moe.is_none());
    assert_eq!(cfg.feed_forward_length, 21504);
    assert!(cfg.per_layer_embed.is_none());
    assert_eq!(cfg.final_logit_softcap, 30.0);
}

#[test]
fn parse_26b_a4b_ud_xl() {
    // Risk check from S1: this is the Unsloth UD-Q8_K_XL with mixed
    // quants (Q8_K + BF16). The Phase-4 IQ kernel coverage memory
    // entry predicts native load. We exercise the config path here;
    // weight resolution is the load_weights smoke.
    let Some(file) = open_if_exists("gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf") else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("parse UD-Q8_K_XL");
    assert_eq!(cfg.variant, Gemma4Variant::Moe26BA4B);
    assert!(cfg.moe.is_some());
}
