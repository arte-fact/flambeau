//! Dense attention + FFN block snapshot — single layer, single
//! token. Refreshing the snapshot is intentional; the diff makes the
//! math change reviewable.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights,
    ModelLayout, RopeVariant,
};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_ops::OpsRegistry;

const VOCAB: usize = 64;
const HIDDEN: usize = 128;
const INTERMEDIATE: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 4;
const RMS_EPS: f32 = 1e-5;

const REFERENCE_LOGITS_HEAD16: [f32; 16] = [
    -0.27348816, -0.6983838, -0.4945717, -0.20793961,
    -0.41295117, 0.44879973, -0.12378508, 0.9048374,
    0.12987937, 0.9078851, 0.19723168, 0.42451006,
    0.074222, -0.29097384, -0.09338939, -0.86321926,
];
const REFERENCE_ARGMAX: usize = 56;
const TOL: f32 = 5e-3;

#[test]
fn parity_snapshot_dense_single_token() {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;

    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

    let embd = EmbeddingWeights {
        token_embd: allocs.upload_f16(&det_signal(VOCAB * HIDDEN, 1)),
        vocab_size: VOCAB,
        hidden: HIDDEN,
        post_scale: None,
    };
    let lm_head = LmHeadWeights {
        output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
        final_logit_softcap: None,
        vocab_size: VOCAB,
        hidden: HIDDEN,
        rms_eps: RMS_EPS,
    };
    let attn = AttnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, 101), q_width, HIDDEN),
        attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 102), kv_width, HIDDEN),
        attn_v: Some(allocs.upload_q8_0(
            &det_signal(kv_width * HIDDEN, 103),
            kv_width,
            HIDDEN,
        )),
        attn_output: allocs.upload_q8_0(
            &det_signal(HIDDEN * q_width, 104),
            HIDDEN,
            q_width,
        ),
        attn_q_norm: None,
        attn_k_norm: None,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        rotated_dims: HEAD_DIM,
        rope_theta: 10000.0,
        rope_variant: RopeVariant::NeoxSplit,
        window_size: 0,
        rms_eps: RMS_EPS,
        softmax_scale: None,
        attn_q_gated: false,
    };
    let ffn = FfnWeights {
        ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        ffn_gate: allocs.upload_q8_0(
            &det_signal(INTERMEDIATE * HIDDEN, 105),
            INTERMEDIATE,
            HIDDEN,
        ),
        ffn_up: allocs.upload_q8_0(
            &det_signal(INTERMEDIATE * HIDDEN, 106),
            INTERMEDIATE,
            HIDDEN,
        ),
        ffn_down: allocs.upload_q8_0(
            &det_signal(HIDDEN * INTERMEDIATE, 107),
            HIDDEN,
            INTERMEDIATE,
        ),
        activation: Activation::SwiGLU,
        rms_eps: RMS_EPS,
    };

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        q_width,
        kv_width,
        vocab: VOCAB,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: 1,
        max_experts: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
            shared_intermediate: 0,
    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
    let layout = ModelLayout {
        num_layers: 1,
        hidden: HIDDEN,
        kv_max_seq_len: MAX_SEQ_LEN,
    };

    let logits: Vec<f32>;
    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        let resid_in = ctx.embed(&embd, 7).expect("embed");
        let attn_delta = ctx.standard_attn(&resid_in, &attn, 0, 0).expect("standard_attn");
        let resid_mid = ctx.residual_add(resid_in, attn_delta).expect("attn residual");
        let ffn_delta = ctx.dense_ffn(&resid_mid, &ffn).expect("dense_ffn");
        let resid_out = ctx.residual_add(resid_mid, ffn_delta).expect("ffn residual");
        ctx.output_head(&resid_out, &lm_head).expect("output_head");
        logits = ctx.logits().to_vec();
    }
    pool.dispose(&device).expect("pool dispose");

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let head: Vec<f32> = logits.iter().take(16).copied().collect();
    eprintln!("parity-snapshot dense: argmax={argmax} head={head:?}");

    assert_eq!(argmax, REFERENCE_ARGMAX, "argmax drift");
    for (i, (got, want)) in head.iter().zip(REFERENCE_LOGITS_HEAD16.iter()).enumerate() {
        let d = (got - want).abs();
        assert!(d < TOL, "logits[{i}] = {got} drifted from {want} by {d}");
    }
}
