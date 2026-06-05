//! GDN block snapshot — asserts forward output matches the committed
//! `REFERENCE_*` constants. Refreshing the snapshot is intentional;
//! the diff makes the math change reviewable.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    EmbeddingWeights, ForwardCtx, GdnDims, GdnWeights, LmHeadWeights,
};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_model_ops::Tensor;
use flambeau_ops::OpsRegistry;

fn upload_f32_tensor(allocs: &mut DeviceAllocs, host: &[f32]) -> Tensor<flambeau_model_ops::F32> {
    let (ptr, _) = allocs.upload(host);
    unsafe { Tensor::<flambeau_model_ops::F32>::from_raw(ptr, host.len()) }
}

const HIDDEN: usize = 256;
const VOCAB: usize = 64;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const NUM_V_HEADS: usize = 2;
const NUM_K_HEADS: usize = 2;
const CONV_KERNEL: usize = 4;
const RMS_EPS: f32 = 1e-5;

/// Reference for `forward_one_token(token_id=7, position=0)` on the
/// synthetic GDN model below, captured against the post-#222 build.
/// Tolerance covers F16/F32 rounding across the recurrent step and
/// the LM-head matmul.
const REFERENCE_LOGITS_HEAD16: [f32; 16] = [
    6.895144, -7.6692233, -20.527489, -24.147297, -15.382086, 2.195806, 19.785957, 28.057117,
    22.347519, 5.3705873, -14.10443, -26.103209, -24.811516, -11.56735, 6.1491146, 19.138206,
];
const REFERENCE_ARGMAX: usize = 7;
const TOL: f32 = 5e-3;

#[test]
fn parity_snapshot_gdn_single_token() {
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
    let gdn = GdnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_qkv: allocs.upload_q8_0(
            &det_signal(conv_channels * HIDDEN, 201),
            conv_channels,
            HIDDEN,
        ),
        attn_gate: allocs.upload_q8_0(&det_signal(d_inner * HIDDEN, 202), d_inner, HIDDEN),
        ssm_alpha: allocs.upload_q8_0(&det_signal(NUM_V_HEADS * HIDDEN, 203), NUM_V_HEADS, HIDDEN),
        ssm_beta: allocs.upload_q8_0(&det_signal(NUM_V_HEADS * HIDDEN, 204), NUM_V_HEADS, HIDDEN),
        ssm_out: allocs.upload_q8_0(&det_signal(HIDDEN * d_inner, 205), HIDDEN, d_inner),
        ssm_dt_bias: upload_f32_tensor(&mut allocs, &det_signal(NUM_V_HEADS, 206)),
        ssm_a: upload_f32_tensor(&mut allocs, &det_signal(NUM_V_HEADS, 207)),
        ssm_conv1d: upload_f32_tensor(&mut allocs, &det_signal(CONV_KERNEL * conv_channels, 208)),
        ssm_norm_w: upload_f32_tensor(&mut allocs, &vec![1.0_f32; HEAD_V_DIM]),
        dims,
        rms_eps: RMS_EPS,
        rep_inner_layout: false,
    };

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: 0,
        q_width: 0,
        kv_width: 0,
        vocab: VOCAB,
        max_seq_len: 1,
        num_layers: 1,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: Some(dims),
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

    let logits: Vec<f32>;
    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        let resid_in = ctx.embed(&embd, &[7u32]).expect("embed");
        let delta = ctx
            .gdn_layer(&resid_in, &gdn, 0, &[0], None)
            .expect("gdn_layer")
            .expect("gdn_layer delta");
        let resid_out = ctx.residual_add(resid_in, delta, 1).expect("residual_add");
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
    eprintln!("parity-snapshot gdn: argmax={argmax} head={head:?}");

    assert_eq!(
        argmax, REFERENCE_ARGMAX,
        "argmax drift — block math changed since snapshot was minted"
    );
    for (i, (got, want)) in head.iter().zip(REFERENCE_LOGITS_HEAD16.iter()).enumerate() {
        let d = (got - want).abs();
        assert!(
            d < TOL,
            "logits[{i}] = {got} drifted from snapshot {want} by {d} (tol {TOL})"
        );
    }
}
