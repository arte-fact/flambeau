//! Real-GGUF smoke for qwen35-v2 on Qwen3.5-9B-Q4_1. Skips when the
//! GGUF isn't present locally.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::ForwardCtx;
use flambeau_ops::OpsRegistry;
use flambeau_qwen35_v2::{forward_one_token, load_from_gguf};
use flambeau_quant::GgufFile;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

#[test]
fn qwen35_9b_forward_one_token_produces_finite_non_constant_logits() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let device = HipDevice::new(0).expect("HIP device 0");
    device.bind().expect("bind");

    let mut model = load_from_gguf(&file, &device, None).expect("load_from_gguf");
    let cfg = &model.config;
    let n_gdn = (0..cfg.num_layers).filter(|&li| cfg.is_recurrent(li)).count();
    eprintln!(
        "qwen35-v2 loaded: hidden={} layers={} (full-attn {}, gdn {}) heads={}/{} head_dim={} rope_theta={} vocab={} full_attn_interval={}",
        cfg.hidden,
        cfg.num_layers,
        cfg.num_layers - n_gdn,
        n_gdn,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_theta,
        cfg.vocab_size,
        cfg.full_attention_interval,
    );

    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let max_seq_len = 64.min(cfg.context_length);

    let scratch_cfg = ScratchConfig {
        hidden: cfg.hidden,
        intermediate: cfg.intermediate,
        q_width: cfg.n_heads * cfg.head_dim,
        kv_width: cfg.n_kv_heads * cfg.head_dim,
        vocab: cfg.vocab_size,
        max_seq_len,
        num_layers: cfg.num_layers,
        max_experts: 0,
            max_experts_per_tok: 0,
        gdn: Some(cfg.gdn),
        per_layer_kv_widths: None,
        attn_q_gated: true,
        shared_intermediate: 0,
    };
    let mut pool = ScratchPool::new(&device, scratch_cfg).expect("ScratchPool::new");

    // Zero per-layer GDN state + conv history.
    let g = cfg.gdn;
    let state_n = g.num_v_heads * g.head_k_dim * g.head_v_dim;
    let conv_n = (g.conv_kernel - 1) * g.conv_channels;
    let state_zero = vec![0.0_f32; state_n];
    let conv_zero = vec![0.0_f32; conv_n];
    for ls in &pool.gdn_state {
        unsafe {
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    ls.state,
                    DevicePtr(state_zero.as_ptr() as usize),
                    state_n * 4,
                )
                .expect("zero state");
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    ls.conv_history,
                    DevicePtr(conv_zero.as_ptr() as usize),
                    conv_n * 4,
                )
                .expect("zero conv_history");
        }
    }
    flambeau_core::Stream::synchronize(stream).expect("sync zero-init");

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
            "qwen35-v2 token=1 pos=0 logits: min={min:.4} max={max:.4} argmax={argmax}"
        );
        assert!(max - min > 1e-2, "logits collapsed (min={min}, max={max})");
    }

    pool.dispose(&device).expect("pool dispose");
    model.dispose(&device).expect("model dispose");
}
