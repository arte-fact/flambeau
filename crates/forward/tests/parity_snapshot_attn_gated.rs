//! Gated full-attention snapshot — exercises the
//! `attn_q_gated=true` branch in `standard_attn_local` (fused Q+gate
//! projection, `split_q_gate_f16`, `sigmoid_mul_f16`). Used by
//! qwen3.5 / qwen3.6 / qwen3-Next; complements
//! `parity_snapshot_dense.rs` which covers the non-gated branch.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    AttnWeights, EmbeddingWeights, ForwardCtx, LmHeadWeights, ModelLayout, RopeVariant,
};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_ops::OpsRegistry;

const VOCAB: usize = 64;
const HIDDEN: usize = 128;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 4;
const RMS_EPS: f32 = 1e-5;

const REFERENCE_LOGITS_HEAD16: [f32; 16] = [
    -0.09114712,
    0.13210723,
    -0.35692906,
    0.27114975,
    -0.47151846,
    0.32448775,
    -0.38120058,
    0.26533583,
    -0.14050573,
    0.10671204,
    0.14263627,
    -0.10818739,
    0.35757548,
    -0.2975785,
    0.41441876,
    -0.3999702,
];
const REFERENCE_ARGMAX: usize = 25;
const TOL: f32 = 5e-3;

#[test]
fn parity_snapshot_attn_gated_single_token() {
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
    // Gated: attn_q on disk is `[2 * q_width, hidden]` in head-
    // interleaved `[head_i_Q | head_i_gate]` layout.
    let attn = AttnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_q: allocs.upload_q8_0(&det_signal(2 * q_width * HIDDEN, 301), 2 * q_width, HIDDEN),
        attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 302), kv_width, HIDDEN),
        attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 303), kv_width, HIDDEN)),
        attn_v_unit_norm_w: None,
        attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, 304), HIDDEN, q_width),
        attn_q_norm: Some(allocs.upload_f16(&det_signal(HEAD_DIM, 305))),
        attn_k_norm: Some(allocs.upload_f16(&det_signal(HEAD_DIM, 306))),
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        rotated_dims: HEAD_DIM,
        rope_theta: 10000.0,
        rope_variant: RopeVariant::NeoxSplit,
        window_size: 0,
        rms_eps: RMS_EPS,
        softmax_scale: None,
        attn_q_gated: true,
        kv_share_src: None,
        post_attn_norm: None,
    };

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: 0,
        q_width,
        kv_width,
        vocab: VOCAB,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: 1,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: true,
        shared_intermediate: 0,
        max_prefill_tokens: 1,
        max_slots: 1,
        per_layer_embd: 0,
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
        let resid_in = ctx.embed(&embd, &[7u32]).expect("embed");
        let delta = ctx
            .standard_attn(&resid_in, &attn, 0, &[0], &[0], None)
            .expect("standard_attn")
            .expect("delta");
        let resid_out = ctx.residual_add(resid_in, delta, 1).expect("residual");
        ctx.output_head(&resid_out, &lm_head, &[0])
            .expect("output_head");
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
    eprintln!("parity-snapshot attn_gated: argmax={argmax} head={head:?}");

    assert_eq!(argmax, REFERENCE_ARGMAX, "argmax drift");
    for (i, (got, want)) in head.iter().zip(REFERENCE_LOGITS_HEAD16.iter()).enumerate() {
        let d = (got - want).abs();
        assert!(d < TOL, "logits[{i}] = {got} drifted from {want} by {d}");
    }
}
