//! Dense (qwen-like) one-token forward on deterministic synth
//! weights. Asserts the output is finite + non-constant.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_ops::OpsRegistry;

#[test]
fn synth_dense_one_token_forward() {
    const VOCAB: usize = 64;
    const HIDDEN: usize = 128;
    const INTERMEDIATE: usize = 256;
    const NUM_LAYERS: usize = 2;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const MAX_SEQ_LEN: usize = 16;
    const RMS_EPS: f32 = 1e-5;

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
    let lm_head_weights = LmHeadWeights {
        output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
        final_logit_softcap: None,
        vocab_size: VOCAB,
        hidden: HIDDEN,
        rms_eps: RMS_EPS,
    };

    let mut attn_weights: Vec<AttnWeights> = Vec::with_capacity(NUM_LAYERS);
    let mut ffn_weights: Vec<FfnWeights> = Vec::with_capacity(NUM_LAYERS);
    for li in 0..NUM_LAYERS {
        let seed = 100 + (li as u32) * 10;
        attn_weights.push(AttnWeights {
            attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, seed + 1), q_width, HIDDEN),
            attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 2), kv_width, HIDDEN),
            attn_v: Some(allocs.upload_q8_0(
                &det_signal(kv_width * HIDDEN, seed + 3),
                kv_width,
                HIDDEN,
            )),
            attn_v_unit_norm_w: None,
            attn_output: allocs.upload_q8_0(
                &det_signal(HIDDEN * q_width, seed + 4),
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
            rope_variant: flambeau_forward::ctx::RopeVariant::NeoxSplit,
            window_size: 0,
            rms_eps: RMS_EPS,
            softmax_scale: None,
            attn_q_gated: false,
            kv_share_src: None,
            post_attn_norm: None,
        });
        ffn_weights.push(FfnWeights {
            ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            ffn_gate: allocs.upload_q8_0(
                &det_signal(INTERMEDIATE * HIDDEN, seed + 5),
                INTERMEDIATE,
                HIDDEN,
            ),
            ffn_up: allocs.upload_q8_0(
                &det_signal(INTERMEDIATE * HIDDEN, seed + 6),
                INTERMEDIATE,
                HIDDEN,
            ),
            ffn_down: allocs.upload_q8_0(
                &det_signal(HIDDEN * INTERMEDIATE, seed + 7),
                HIDDEN,
                INTERMEDIATE,
            ),
            activation: Activation::SwiGLU,
            rms_eps: RMS_EPS,
            post_ffn_norm: None,
        });
    }

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        q_width,
        kv_width,
        vocab: VOCAB,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: NUM_LAYERS,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: 1,
        max_slots: 1,
        per_layer_embd: 0,
            paged_kv: None,
            kv_layout: flambeau_forward::core::KvLayout::F16Contig,
            per_layer_kv_layouts: None,
            per_layer_kv_depths: None,    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
    let layout = ModelLayout {
        num_layers: NUM_LAYERS,
        hidden: HIDDEN,
        kv_max_seq_len: MAX_SEQ_LEN,
    };

    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        let token_id: u32 = 7;
        let position: usize = 0;

        let mut resid = ctx.embed(&embd, &[token_id]).expect("embed");
        let layers: Vec<usize> = ctx.layer_range(&layout).collect();
        assert_eq!(layers, vec![0, 1]);

        for li in layers {
            let normed = ctx
                .rmsnorm(&resid, &attn_weights[li].attn_norm, RMS_EPS, 1)
                .expect("attn rmsnorm");
            let delta = ctx
                .standard_attn(&normed, &attn_weights[li], li, &[position], &[0], None)
                .expect("standard_attn")
                .expect("delta");
            resid = ctx
                .residual_add(resid, delta, 1)
                .expect("attn residual_add");

            let normed = ctx
                .rmsnorm(&resid, &ffn_weights[li].ffn_norm, RMS_EPS, 1)
                .expect("ffn rmsnorm");
            let delta = ctx
                .dense_ffn(&normed, &ffn_weights[li], 1, None)
                .expect("dense_ffn")
                .expect("delta");
            resid = ctx
                .residual_add(resid, delta, 1)
                .expect("ffn residual_add");
        }

        ctx.output_head(&resid, &lm_head_weights, &[0])
            .expect("output_head");
        let logits = ctx.logits();
        assert_eq!(logits.len(), VOCAB);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logits[{i}] = {l} not finite");
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max - min > 1e-3);
    }

    pool.dispose(&device).expect("pool dispose");
}
