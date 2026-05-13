//! S4 — resolve every expected tensor and validate per-shape against
//! the parsed config. No device upload.

use std::path::Path;

use flambeau_gemma4::{
    resolve_weights, validate_shapes, Gemma4Config, ModelLayout,
};
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

fn check(name: &str) {
    let Some(file) = open_if_exists(name) else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("parse cfg");
    let mut layout = ModelLayout::from_config(&cfg);

    // Resolve shared-KV pairings (no-op for variants with shared_kv_layers=0).
    let resolved = layout.resolve_kv_sharing();
    assert_eq!(resolved, layout.num_shared_kv_layers());

    let weights = resolve_weights(&file, &cfg, &layout)
        .unwrap_or_else(|e| panic!("resolve {name}: {e}"));
    validate_shapes(&cfg, &layout, &weights)
        .unwrap_or_else(|e| panic!("shape check {name}: {e}"));

    // Cross-checks.
    assert_eq!(weights.layers.len(), cfg.num_layers);
    let moe_layers = weights.layers.iter().filter(|l| l.moe.is_some()).count();
    let pe_layers = weights
        .layers
        .iter()
        .filter(|l| l.per_layer_embed.is_some())
        .count();
    if cfg.variant.is_moe() {
        assert_eq!(moe_layers, cfg.num_layers);
    } else {
        assert_eq!(moe_layers, 0);
    }
    if cfg.per_layer_embed.is_some() {
        assert_eq!(pe_layers, cfg.num_layers);
    } else {
        assert_eq!(pe_layers, 0);
    }
}

#[test]
fn load_e4b_q4_0() {
    check("gemma-4-E4B-it-Q4_0.gguf");
}

#[test]
fn load_31b_q4_0() {
    check("gemma-4-31B-it-Q4_0.gguf");
}

#[test]
fn load_26b_a4b_q8_0() {
    check("gemma-4-26B-A4B-it-Q8_0.gguf");
}

#[test]
fn load_26b_a4b_ud_xl() {
    check("gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf");
}

#[test]
fn load_31b_q8_0() {
    check("gemma-4-31B-it-Q8_0.gguf");
}

#[test]
fn e4b_kv_sharing_resolves() {
    let Some(file) = open_if_exists("gemma-4-E4B-it-Q4_0.gguf") else {
        return;
    };
    let cfg = Gemma4Config::from_gguf(&file).expect("parse");
    let mut layout = ModelLayout::from_config(&cfg);
    let resolved = layout.resolve_kv_sharing();
    assert_eq!(resolved, 18, "E4B has 18 shared-KV tail layers");
    // Every tail layer must point at an earlier layer of the same SWA type.
    for spec in &layout.layers {
        if spec.has_kv {
            assert!(spec.kv_share_src.is_none());
        } else {
            let src = spec.kv_share_src.unwrap_or_else(|| {
                panic!("tail layer {} did not resolve kv_share_src", spec.index)
            });
            assert!(src < spec.index);
            let src_spec = &layout.layers[src];
            assert!(src_spec.has_kv);
            assert_eq!(src_spec.is_swa, spec.is_swa);
        }
    }
}
