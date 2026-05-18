//! tp_size=1 wrapping test for TpForwardCtx. The AR hook is trivially
//! skipped (n_ranks=1) so behaviour collapses to SingleDevice — logits
//! must match bit-equal.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::tp::{TpForwardCtx, TpHooks};
use flambeau_forward::ForwardCtx;
use flambeau_ops::OpsRegistry;
use flambeau_qwen3_v2::{forward_one_token, load_from_gguf};
use flambeau_quant::GgufFile;

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

#[test]
fn tp_size_1_logits_match_single_device() {
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

    let cfg = ScratchConfig {
        hidden: model.config.hidden,
        intermediate: model.config.intermediate,
        q_width: model.config.n_heads * model.config.head_dim,
        kv_width: model.config.n_kv_heads * model.config.head_dim,
        vocab: model.config.vocab_size,
        max_seq_len,
        num_layers: model.config.num_layers,
            max_experts: 0,
    };

    // Baseline.
    let baseline: Vec<f32> = {
        let mut pool = ScratchPool::new(&device, cfg).expect("SD pool");
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        forward_one_token(&model, &mut ctx, 1, 0).expect("SD forward");
        let v = ctx.logits().to_vec();
        drop(ctx);
        pool.dispose(&device).expect("SD pool dispose");
        v
    };

    // tp_size=1.
    let tp: Vec<f32> = {
        let mut pool = ScratchPool::new(&device, cfg).expect("TP pool");
        let hooks = TpHooks {
            rank: 0,
            n_ranks: 1,
            ar_callback: Box::new(|_r, _n, _buf, _n_elems, _device, _stream| {
                unreachable!("tp_size=1 should short-circuit ar_callback")
            }),
        };
        let mut ctx = TpForwardCtx::new(&device, stream, &reg, &mut pool, hooks);
        forward_one_token(&model, &mut ctx, 1, 0).expect("TP forward");
        let v = ctx.logits().to_vec();
        drop(ctx);
        pool.dispose(&device).expect("TP pool dispose");
        v
    };

    model.dispose(&device).expect("model dispose");

    assert_eq!(tp.len(), baseline.len());
    let mut max_abs_diff = 0.0_f32;
    let mut worst = 0;
    for (i, (&a, &b)) in tp.iter().zip(baseline.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            worst = i;
        }
    }
    let sd_argmax = baseline
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let tp_argmax = tp
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "tp_size=1 vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst}  argmax SD={sd_argmax} TP={tp_argmax}"
    );
    assert_eq!(tp_argmax, sd_argmax);
    assert_eq!(max_abs_diff, 0.0, "tp_size=1 should match SD bit-equal");
}
