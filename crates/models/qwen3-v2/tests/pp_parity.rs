//! Pipeline-parallel parity vs single-device on Qwen3-Embedding-0.6B.
//!
//! Each rank runs ITS slice of layers using the same `forward_one_token`
//! function and the same `Qwen3V2Model` weight handle (shared across
//! ranks for test simplicity; a true multi-device load would shard
//! weights per rank but isn't required to validate the trait surface).
//!
//! PP rank `r` reads its inbound residual from the shared `peer_buffer`
//! at `embed` (rank 0 ignores) and writes its outbound residual at
//! `output_head` (last rank ignores → does the real LM head).
//!
//! Expected outcome: logits bit-equal vs single-device. The host F16
//! roundtrip introduces zero precision loss.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::pp::PpForwardCtx;
use flambeau_forward::ForwardCtx;
use flambeau_ops::OpsRegistry;
use flambeau_qwen3_v2::{forward_one_token, load_from_gguf};
use flambeau_quant::GgufFile;
use half::f16;

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

#[test]
fn pp_size_2_logits_match_single_device() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let device = HipDevice::new(0).expect("HIP device 0");
    device.bind().expect("bind");
    let mut model = load_from_gguf(&file, &device).expect("load_from_gguf");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let max_seq_len = 64.min(model.config.context_length);

    let h = model.config.hidden;
    let q_width = model.config.n_heads * model.config.head_dim;
    let kv_width = model.config.n_kv_heads * model.config.head_dim;
    let vocab = model.config.vocab_size;

    // -----------------------------------------------------------------
    // Single-device baseline.
    // -----------------------------------------------------------------
    let baseline_logits: Vec<f32> = {
        let cfg_sd = ScratchConfig {
            hidden: h,
            intermediate: model.config.intermediate,
            q_width,
            kv_width,
            vocab,
            max_seq_len,
            num_layers: model.config.num_layers,
        };
        let mut pool_sd = ScratchPool::new(&device, cfg_sd).expect("ScratchPool::new SD");
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool_sd);
        forward_one_token(&model, &mut ctx, 1, 0).expect("SD forward");
        let logits = ctx.logits().to_vec();
        drop(ctx);
        pool_sd.dispose(&device).expect("pool_sd dispose");
        logits
    };

    // -----------------------------------------------------------------
    // pp_size = 2.
    // -----------------------------------------------------------------
    let split = model.config.num_layers / 2;
    eprintln!(
        "PP layout: rank 0 = layers [0..{split}); rank 1 = layers [{split}..{}).",
        model.config.num_layers
    );

    // Each rank owns a slice → KV cache count = slice size.
    let cfg_r0 = ScratchConfig {
        hidden: h,
        intermediate: model.config.intermediate,
        q_width,
        kv_width,
        vocab,
        max_seq_len,
        num_layers: split,
    };
    let cfg_r1 = ScratchConfig {
        hidden: h,
        intermediate: model.config.intermediate,
        q_width,
        kv_width,
        vocab,
        max_seq_len,
        num_layers: model.config.num_layers - split,
    };
    let mut pool_r0 = ScratchPool::new(&device, cfg_r0).expect("pool r0");
    let mut pool_r1 = ScratchPool::new(&device, cfg_r1).expect("pool r1");

    let mut peer_buffer: Vec<f16> = vec![f16::ZERO; h];

    {
        let mut ctx_r0 = PpForwardCtx::new(
            &device,
            stream,
            &reg,
            &mut pool_r0,
            0,
            2,
            0,
            split,
            &mut peer_buffer,
        );
        forward_one_token(&model, &mut ctx_r0, 1, 0).expect("PP rank 0 forward");
        // ctx_r0 drops; peer_buffer now holds rank 0's post-layer-split residual.
    }

    let pp_logits: Vec<f32> = {
        let mut ctx_r1 = PpForwardCtx::new(
            &device,
            stream,
            &reg,
            &mut pool_r1,
            1,
            2,
            split,
            model.config.num_layers,
            &mut peer_buffer,
        );
        forward_one_token(&model, &mut ctx_r1, 1, 0).expect("PP rank 1 forward");
        ctx_r1.logits().to_vec()
    };

    pool_r0.dispose(&device).expect("pool r0 dispose");
    pool_r1.dispose(&device).expect("pool r1 dispose");
    model.dispose(&device).expect("model dispose");

    // -----------------------------------------------------------------
    // Parity assertion.
    // -----------------------------------------------------------------
    assert_eq!(pp_logits.len(), baseline_logits.len(), "logits len mismatch");
    let mut max_abs_diff = 0.0_f32;
    let mut worst_idx = 0;
    for (i, (&a, &b)) in pp_logits.iter().zip(baseline_logits.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            worst_idx = i;
        }
    }
    let sd_argmax = baseline_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let pp_argmax = pp_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "PP vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst_idx} \
         (SD={:.4} PP={:.4})   argmax SD={sd_argmax} PP={pp_argmax}",
        baseline_logits[worst_idx],
        pp_logits[worst_idx]
    );
    assert_eq!(pp_argmax, sd_argmax, "argmax differs SD vs PP");
    // Host F16 roundtrip is lossless; expect bit-equal logits.
    assert_eq!(
        max_abs_diff, 0.0,
        "PP logits diverge from SD by {max_abs_diff:.6} at idx {worst_idx}"
    );
}
