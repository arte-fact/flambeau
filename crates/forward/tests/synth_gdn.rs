//! GDN one-token forward on deterministic synth weights.
//! `DeltaNetLayer` is hardwired to head_k_dim = head_v_dim = 128.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    EmbeddingWeights, ForwardCtx, GdnDims, GdnWeights, LmHeadWeights, ModelLayout,
};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_ops::OpsRegistry;

#[test]
fn synth_gdn_one_token_forward() {
    const HIDDEN: usize = 256;
    const VOCAB: usize = 64;
    const NUM_LAYERS: usize = 2;
    const HEAD_K_DIM: usize = 128;
    const HEAD_V_DIM: usize = 128;
    const NUM_V_HEADS: usize = 2;
    const NUM_K_HEADS: usize = 2;
    const CONV_KERNEL: usize = 4;
    const RMS_EPS: f32 = 1e-5;

    let d_inner = NUM_V_HEADS * HEAD_V_DIM;
    let conv_channels = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM;

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

    let dims = GdnDims {
        d_inner,
        num_v_heads: NUM_V_HEADS,
        num_k_heads: NUM_K_HEADS,
        head_k_dim: HEAD_K_DIM,
        head_v_dim: HEAD_V_DIM,
        conv_channels,
        conv_kernel: CONV_KERNEL,
    };

    let mut gdn_weights: Vec<GdnWeights> = Vec::with_capacity(NUM_LAYERS);
    for li in 0..NUM_LAYERS {
        let seed = 200 + (li as u32) * 30;
        gdn_weights.push(GdnWeights {
            attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            attn_qkv: allocs.upload_q8_0(
                &det_signal(conv_channels * HIDDEN, seed + 1),
                conv_channels,
                HIDDEN,
            ),
            attn_gate: allocs.upload_q8_0(
                &det_signal(d_inner * HIDDEN, seed + 2),
                d_inner,
                HIDDEN,
            ),
            ssm_alpha: allocs.upload_q8_0(
                &det_signal(NUM_V_HEADS * HIDDEN, seed + 3),
                NUM_V_HEADS,
                HIDDEN,
            ),
            ssm_beta: allocs.upload_q8_0(
                &det_signal(NUM_V_HEADS * HIDDEN, seed + 4),
                NUM_V_HEADS,
                HIDDEN,
            ),
            ssm_out: allocs.upload_q8_0(
                &det_signal(HIDDEN * d_inner, seed + 5),
                HIDDEN,
                d_inner,
            ),
            ssm_dt_bias: allocs.upload_f32(&det_signal(NUM_V_HEADS, seed + 6)),
            ssm_a: allocs.upload_f32(&det_signal(NUM_V_HEADS, seed + 7)),
            ssm_conv1d: allocs.upload_f32(&det_signal(CONV_KERNEL * conv_channels, seed + 8)),
            ssm_norm_w: allocs.upload_f32(&vec![1.0_f32; HEAD_V_DIM]),
            dims,
            rms_eps: RMS_EPS,
            rep_inner_layout: false,
        });
    }

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: 0,
        q_width: 0,
        kv_width: 0,
        vocab: VOCAB,
        max_seq_len: 1,
        num_layers: NUM_LAYERS,
        max_experts: 0,
        gdn: Some(dims),
        per_layer_kv_widths: None,
        attn_q_gated: false,
    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");

    // Recurrent state + conv history must be zero at position=0
    // (the pool zeros them too; the explicit sync below keeps the
    // synth test self-contained).
    for ls in &pool.gdn_state {
        let state_n = NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;
        let conv_n = (CONV_KERNEL - 1) * conv_channels;
        let state_zero = vec![0.0_f32; state_n];
        let conv_zero = vec![0.0_f32; conv_n];
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
    stream.synchronize().expect("sync zero-init");

    let layout = ModelLayout {
        num_layers: NUM_LAYERS,
        hidden: HIDDEN,
        kv_max_seq_len: 1,
    };

    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        let mut resid = ctx.embed(&embd, 7).expect("embed");
        let layers: Vec<usize> = ctx.layer_range(&layout).collect();
        for li in layers {
            let normed = ctx
                .rmsnorm(&resid, &gdn_weights[li].attn_norm, RMS_EPS)
                .expect("attn_norm");
            let delta = ctx
                .gdn_layer(&normed, &gdn_weights[li], li)
                .expect("gdn_layer");
            resid = ctx.residual_add(resid, delta).expect("residual_add");
        }
        ctx.output_head(&resid, &lm_head).expect("output_head");
        let logits = ctx.logits();
        assert_eq!(logits.len(), VOCAB);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logits[{i}] = {l} not finite");
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max - min > 1e-3, "GDN logits collapsed");
    }

    pool.dispose(&device).expect("pool dispose");
}
