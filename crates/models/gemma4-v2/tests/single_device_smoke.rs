//! Real-GGUF smoke on gemma-4-31B-it-Q4_0.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::ForwardCtx;
use flambeau_gemma4_v2::{forward_one_token, load_from_gguf};
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgufFile;

const MODEL_PATH: &str = "/artefact/models/gemma-4-31B-it-Q4_0.gguf";

#[test]
fn gemma4_31b_forward_one_token_produces_finite_non_constant_logits() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let device = HipDevice::new(0).expect("HIP device 0");
    device.bind().expect("bind");

    let mut model = match load_from_gguf(&file, &device, None) {
        Ok(m) => m,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("out of memory") {
                eprintln!(
                    "SKIP: {MODEL_PATH} doesn't fit on a single GPU ({msg}). \
                     Needs the TP-sharded loader, which isn't in this first cut."
                );
                return;
            }
            panic!("load_from_gguf: {msg}");
        }
    };
    let cfg = &model.config;
    let swa_count = cfg.attn.iter().filter(|a| a.window_size > 0).count();
    eprintln!(
        "gemma4-v2 loaded: hidden={} layers={} (full {}, swa {}) heads={} vocab={} softcap={}",
        cfg.hidden,
        cfg.num_layers,
        cfg.num_layers - swa_count,
        swa_count,
        cfg.num_heads,
        cfg.vocab_size,
        cfg.final_logit_softcap,
    );

    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let max_seq_len = 64.min(cfg.context_length);

    // Pool widths: take the max across layers so the scratch fits any.
    let q_width = cfg.attn.iter().map(|a| cfg.num_heads * a.head_dim).max().unwrap();
    let kv_width = cfg
        .attn
        .iter()
        .zip(cfg.num_kv_heads.iter())
        .map(|(a, &nkv)| nkv * a.head_dim)
        .max()
        .unwrap();

    let scratch_cfg = ScratchConfig {
        hidden: cfg.hidden,
        intermediate: cfg.intermediate,
        q_width,
        kv_width,
        vocab: cfg.vocab_size,
        max_seq_len,
        num_layers: cfg.num_layers,
        max_experts: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
    };
    let mut pool = ScratchPool::new(&device, scratch_cfg).expect("ScratchPool::new");

    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        forward_one_token(&model, &mut ctx, 1, 0).expect("forward_one_token");
        let logits = ctx.logits();
        assert_eq!(logits.len(), cfg.vocab_size);
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        let mut argmax = 0_usize;
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logits[{i}] = {l} not finite");
            if l > max {
                max = l;
                argmax = i;
            }
            if l < min {
                min = l;
            }
        }
        eprintln!(
            "gemma4-v2 token=1 pos=0 logits: min={min:.4} max={max:.4} argmax={argmax}"
        );
        assert!(max - min > 1e-2, "logits collapsed");
        if cfg.final_logit_softcap > 0.0 {
            let cap = cfg.final_logit_softcap;
            assert!(
                max <= cap + 1e-3 && min >= -cap - 1e-3,
                "softcap not applied: min={min} max={max} cap={cap}"
            );
        }
    }

    pool.dispose(&device).expect("pool dispose");
    model.dispose(&device).expect("model dispose");
}

#[test]
fn gemma4_31b_config_parses_with_swa_alternation_and_softcap() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let cfg = flambeau_gemma4_v2::Gemma4V2Config::from_gguf(&file).expect("config");
    let swa_count = cfg.attn.iter().filter(|a| a.window_size > 0).count();
    let global_count = cfg.num_layers - swa_count;
    eprintln!(
        "gemma4-31B config: hidden={} layers={} (full {}, swa {}) heads={} vocab={} softcap={} tied_lm_head={}",
        cfg.hidden,
        cfg.num_layers,
        global_count,
        swa_count,
        cfg.num_heads,
        cfg.vocab_size,
        cfg.final_logit_softcap,
        cfg.tied_lm_head,
    );
    assert!(swa_count > 0, "expected SWA layers in gemma4 pattern");
    assert!(global_count > 0, "expected at least one global-attn layer");
    assert!(cfg.final_logit_softcap > 0.0, "gemma4 should set softcap");
    assert!(cfg.tied_lm_head, "gemma4 ties LM head to token_embd");
    // SWA + global layers should differ on at least head_dim or rope_theta.
    let swa_dims = cfg.attn.iter().find(|a| a.window_size > 0).unwrap();
    let global_dims = cfg.attn.iter().find(|a| a.window_size == 0).unwrap();
    assert!(
        swa_dims.head_dim != global_dims.head_dim
            || swa_dims.rope_theta != global_dims.rope_theta,
        "SWA / global layers should differ in head_dim or rope_theta"
    );
}
