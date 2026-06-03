//! Multi-Session-per-process leak: documents what we know.
//!
//! Single-Session-per-process usage works correctly (production
//! server pattern). The tests below show the failure surface for
//! multi-Session-per-process workflows.
//!
//! Findings:
//! * Two identical GDN sessions back-to-back (`multi_session_leak_double_gdn`):
//!   bit-equal output. NO leak.
//! * Dense session → GDN session (`multi_session_leak_dense_then_gdn`):
//!   GDN produces NaN. THE leak.
//! * Leaking the dense session's `DeviceAllocs` (so its HBM
//!   addresses stay live → can't be recycled by the GDN session's
//!   hipMalloc) fixes the NaN. Bit-equal to the GDN-only baseline.
//! * Universally zeroing every freshly-allocated buffer (in
//!   `HipDevice::alloc`) does NOT fix it. So the bug isn't
//!   uninitialised memory reads — it's address-reuse-specific.
//! * `hipDeviceSynchronize` in `HipDevice::Drop` doesn't fix it.
//! * Reproduces identically on rocm 7.1.1 AND 7.2.1 — not a
//!   version-specific regression.
//!
//! Empirical conclusion: hipMalloc returning HBM addresses that
//! were freed earlier in the process triggers a HIP-driver-level
//! correctness issue specific to our kernel set on gfx906/MI50.
//! Persists across rocm 7.1.1 → 7.2.1.
//!
//! Production-safe: servers run one long-lived Session per process.
//! Tests use one-binary-per-test-file (`crates/forward/tests/synth_*.rs`)
//! so each test gets its own process and can't repro this.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    EmbeddingWeights, ForwardCtx, GdnDims, GdnWeights, LmHeadWeights, ModelLayout,
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
const NUM_LAYERS: usize = 2;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const NUM_V_HEADS: usize = 2;
const NUM_K_HEADS: usize = 2;
const CONV_KERNEL: usize = 4;
const RMS_EPS: f32 = 1e-5;

fn run_gdn_forward() -> Vec<f32> {
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
            attn_gate: allocs.upload_q8_0(&det_signal(d_inner * HIDDEN, seed + 2), d_inner, HIDDEN),
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
            ssm_out: allocs.upload_q8_0(&det_signal(HIDDEN * d_inner, seed + 5), HIDDEN, d_inner),
            ssm_dt_bias: upload_f32_tensor(&mut allocs, &det_signal(NUM_V_HEADS, seed + 6)),
            ssm_a: upload_f32_tensor(&mut allocs, &det_signal(NUM_V_HEADS, seed + 7)),
            ssm_conv1d: upload_f32_tensor(&mut allocs, &det_signal(CONV_KERNEL * conv_channels, seed + 8)),
            ssm_norm_w: upload_f32_tensor(&mut allocs, &vec![1.0_f32; HEAD_V_DIM]),
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
    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
    let layout = ModelLayout {
        num_layers: NUM_LAYERS,
        hidden: HIDDEN,
        kv_max_seq_len: 1,
    };

    let logits_vec: Vec<f32>;
    {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        let mut resid = ctx.embed(&embd, &[7u32]).expect("embed");
        let layers: Vec<usize> = ctx.layer_range(&layout).collect();
        for li in layers {
            let normed = ctx
                .rmsnorm(&resid, &gdn_weights[li].attn_norm, RMS_EPS, 1)
                .expect("attn_norm");
            let delta = ctx
                .gdn_layer(&normed, &gdn_weights[li], li, &[0], None)
                .expect("gdn_layer")
                .expect("delta");
            resid = ctx.residual_add(resid, delta, 1).expect("residual_add");
        }
        ctx.output_head(&resid, &lm_head, &[0])
            .expect("output_head");
        logits_vec = ctx.logits().to_vec();
    }
    pool.dispose(&device).expect("pool dispose");
    logits_vec
}

#[test]
fn multi_session_leak_double_gdn() {
    let a = run_gdn_forward();
    let b = run_gdn_forward();
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("double-GDN max_abs_diff = {max_diff}");
    assert!(
        a[0].is_finite() && b[0].is_finite(),
        "either call produced NaN"
    );
    assert_eq!(max_diff, 0.0, "two GDN sessions should be bit-equal");
}

// Ignored: documented multi-Session leak. Single-Session usage works
// (synth_dense, synth_gdn, synth_moe each in own binary all pass).
// Production server uses one long-lived Session per process. See
// the module header for the investigation summary.
#[ignore]
#[test]
fn multi_session_leak_dense_then_gdn() {
    use flambeau_forward::ctx::{Activation, AttnWeights, FfnWeights};
    const D_HIDDEN: usize = 128;
    const D_VOCAB: usize = 64;
    const D_INTERMEDIATE: usize = 256;
    const D_LAYERS: usize = 2;
    const D_N_HEADS: usize = 4;
    const D_N_KV_HEADS: usize = 2;
    const D_HEAD_DIM: usize = 64;
    const D_MAX_SEQ: usize = 16;
    const D_RMS_EPS: f32 = 1e-5;

    let q_width = D_N_HEADS * D_HEAD_DIM;
    let kv_width = D_N_KV_HEADS * D_HEAD_DIM;

    {
        let device = HipDevice::new(0).expect("HipDevice 0");
        device.bind().expect("bind");
        let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
        let stream = device.default_stream();
        let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

        let embd = EmbeddingWeights {
            token_embd: allocs.upload_f16(&det_signal(D_VOCAB * D_HIDDEN, 1)),
            vocab_size: D_VOCAB,
            hidden: D_HIDDEN,
            post_scale: None,
        };
        let lm_head_weights = LmHeadWeights {
            output_norm: allocs.upload_f16(&vec![1.0_f32; D_HIDDEN]),
            lm_head: allocs.upload_q8_0(&det_signal(D_VOCAB * D_HIDDEN, 9), D_VOCAB, D_HIDDEN),
            final_logit_softcap: None,
            vocab_size: D_VOCAB,
            hidden: D_HIDDEN,
            rms_eps: D_RMS_EPS,
        };

        let mut attn_weights: Vec<AttnWeights> = Vec::with_capacity(D_LAYERS);
        let mut ffn_weights: Vec<FfnWeights> = Vec::with_capacity(D_LAYERS);
        for li in 0..D_LAYERS {
            let seed = 100 + (li as u32) * 10;
            attn_weights.push(AttnWeights {
                attn_norm: allocs.upload_f16(&vec![1.0_f32; D_HIDDEN]),
                attn_q: allocs.upload_q8_0(
                    &det_signal(q_width * D_HIDDEN, seed + 1),
                    q_width,
                    D_HIDDEN,
                ),
                attn_k: allocs.upload_q8_0(
                    &det_signal(kv_width * D_HIDDEN, seed + 2),
                    kv_width,
                    D_HIDDEN,
                ),
                attn_v: Some(allocs.upload_q8_0(
                    &det_signal(kv_width * D_HIDDEN, seed + 3),
                    kv_width,
                    D_HIDDEN,
                )),
                attn_v_unit_norm_w: None,
                attn_output: allocs.upload_q8_0(
                    &det_signal(D_HIDDEN * q_width, seed + 4),
                    D_HIDDEN,
                    q_width,
                ),
                attn_q_norm: None,
                attn_k_norm: None,
                n_heads: D_N_HEADS,
                n_kv_heads: D_N_KV_HEADS,
                head_dim: D_HEAD_DIM,
                rotated_dims: D_HEAD_DIM,
                rope_theta: 10000.0,
                rope_variant: flambeau_forward::ctx::RopeVariant::NeoxSplit,
                window_size: 0,
                rms_eps: D_RMS_EPS,
                softmax_scale: None,
                attn_q_gated: false,
                kv_share_src: None,
                post_attn_norm: None,
            });
            ffn_weights.push(FfnWeights {
                ffn_norm: allocs.upload_f16(&vec![1.0_f32; D_HIDDEN]),
                ffn_gate: allocs.upload_q8_0(
                    &det_signal(D_INTERMEDIATE * D_HIDDEN, seed + 5),
                    D_INTERMEDIATE,
                    D_HIDDEN,
                ),
                ffn_up: allocs.upload_q8_0(
                    &det_signal(D_INTERMEDIATE * D_HIDDEN, seed + 6),
                    D_INTERMEDIATE,
                    D_HIDDEN,
                ),
                ffn_down: allocs.upload_q8_0(
                    &det_signal(D_HIDDEN * D_INTERMEDIATE, seed + 7),
                    D_HIDDEN,
                    D_INTERMEDIATE,
                ),
                activation: Activation::SwiGLU,
                rms_eps: D_RMS_EPS,
                post_ffn_norm: None,
            });
        }

        let cfg = ScratchConfig {
            hidden: D_HIDDEN,
            intermediate: D_INTERMEDIATE,
            q_width,
            kv_width,
            vocab: D_VOCAB,
            max_seq_len: D_MAX_SEQ,
            num_layers: D_LAYERS,
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
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
        let layout = ModelLayout {
            num_layers: D_LAYERS,
            hidden: D_HIDDEN,
            kv_max_seq_len: D_MAX_SEQ,
        };

        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        let mut resid = ctx.embed(&embd, &[7u32]).expect("embed");
        let layers: Vec<usize> = ctx.layer_range(&layout).collect();
        for li in layers {
            let normed = ctx
                .rmsnorm(&resid, &attn_weights[li].attn_norm, D_RMS_EPS, 1)
                .expect("attn rmsnorm");
            let delta = ctx
                .standard_attn(&normed, &attn_weights[li], li, &[0], &[0], None)
                .expect("standard_attn")
                .expect("delta");
            resid = ctx
                .residual_add(resid, delta, 1)
                .expect("attn residual_add");
            let normed = ctx
                .rmsnorm(&resid, &ffn_weights[li].ffn_norm, D_RMS_EPS, 1)
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
        drop(ctx);
        pool.dispose(&device).expect("pool dispose");
        eprintln!("[dense warm-up] done");
    }

    let logits = run_gdn_forward();
    assert!(logits[0].is_finite(), "GDN logits NaN after dense warm-up");
}
