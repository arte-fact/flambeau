//! Real-GGUF smoke for SingleDeviceForwardCtx. Skips if the model
//! file isn't present locally.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::ForwardCtx;
use flambeau_ops::OpsRegistry;
use flambeau_qwen3_v2::{forward_one_token, load_from_gguf};
use flambeau_quant::GgufFile;

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";
const QWEN35_9B: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

#[test]
fn forward_one_token_produces_finite_non_constant_logits() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    let device = HipDevice::new(0).expect("HIP device 0");
    device.bind().expect("bind");

    let mut model = load_from_gguf(&file, &device).expect("load_from_gguf");
    eprintln!(
        "qwen3-v2 loaded: hidden={} layers={} heads={}/{} head_dim={} vocab={} rotated_dims={} rope_theta={}",
        model.config.hidden,
        model.config.num_layers,
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.vocab_size,
        model.config.rotated_dims,
        model.config.rope_theta,
    );

    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();

    let max_seq_len = 64.min(model.config.context_length);
    let cfg = ScratchConfig {
        hidden: model.config.hidden,
        intermediate: model.config.intermediate,
        q_width: model.config.n_heads * model.config.head_dim,
        kv_width: model.config.n_kv_heads * model.config.head_dim,
        vocab: model.config.vocab_size,
        max_seq_len,
        num_layers: model.config.num_layers,
    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");

    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

        if let Err(e) = forward_one_token(&model, &mut ctx, 1, 0) {
            eprintln!("forward_one_token FAILED, error chain:");
            for (i, cause) in e.chain().enumerate() {
                eprintln!("  [{i}] {cause}");
            }
            panic!("forward_one_token failed");
        }

        let logits = ctx.logits();
        assert_eq!(
            logits.len(),
            model.config.vocab_size,
            "logits len mismatch"
        );

        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        let mut argmax: usize = 0;
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
            "qwen3-v2 token=1 pos=0 logits: min={min:.4} max={max:.4} argmax={argmax}"
        );
        assert!(
            max - min > 1e-2,
            "logits collapsed to a constant (min={min}, max={max})"
        );
    }

    pool.dispose(&device).expect("pool dispose");
    model.dispose(&device).expect("model dispose");
}

/// qwen35 is hybrid GDN+full-attn — out of scope until the GDN
/// composite lands. Loader should reject with a clear arch error.
#[test]
fn qwen35_9b_rejected_with_clear_arch_error() {
    let path = PathBuf::from(QWEN35_9B);
    if !path.exists() {
        eprintln!("SKIP: {QWEN35_9B} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let device = HipDevice::new(0).expect("HIP device 0");
    device.bind().expect("bind");

    match flambeau_qwen3_v2::load_from_gguf(&file, &device) {
        Ok(_) => panic!("qwen3-v2 loader accepted a qwen35 GGUF — expected arch rejection"),
        Err(e) => {
            let msg = format!("{e:#}");
            eprintln!("qwen3-v2 correctly rejected qwen35: {msg}");
            assert!(
                msg.contains("qwen3") && (msg.contains("qwen35") || msg.contains("WrongArchitecture")),
                "expected arch-mismatch error, got: {msg}"
            );
        }
    }
}
