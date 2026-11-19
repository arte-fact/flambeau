//! Cross-backend MoE parity with **Q4_K experts** — exercises the full MoE
//! composite (router → topk → fused indexed-MoE gate+up → down → weighted
//! combine) through `ForwardEngine` on HIP gfx906 and CUDA sm_86, asserting the
//! two backends agree. The Q4_K experts drive
//! `indexed_moe_mmvq_q4_k_gate_up_r4_dp4a` (the kernel ported + certified this
//! cycle) and `indexed_moe_mmvq_q4_k_r2_dp4a` on the CUDA side. Self-validating:
//! HIP is the reference, computed in the same run — no hard-coded snapshot.

#![cfg(all(feature = "hip", feature = "cuda"))]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend::{Backend, CudaBackend, HipBackend};
use flambeau_backend_cuda::CudaDevice;
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, ForwardCtx, LmHeadWeights, ModelLayout, MoeWeights,
    RopeVariant,
};
use flambeau_forward::{ForwardEngine, ScratchPool};

const VOCAB: usize = 64;
const HIDDEN: usize = 256; // Q4_K needs K % 256 == 0
const INTERMEDIATE: usize = 512;
const NUM_LAYERS: usize = 2;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 16;
const N_EXPERTS: usize = 4;
const EXPERTS_PER_TOK: usize = 2;
const RMS_EPS: f32 = 1e-5;

fn run<B: Backend>(eng_dev: &B::Device, reg: &B::Registry, allocs: &mut DeviceAllocs<B::Device>) -> Vec<f32> {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;

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

    let mut attn_weights: Vec<AttnWeights> = Vec::with_capacity(NUM_LAYERS);
    let mut moe_weights: Vec<MoeWeights> = Vec::with_capacity(NUM_LAYERS);
    for li in 0..NUM_LAYERS {
        let seed = 100 + (li as u32) * 100;
        attn_weights.push(AttnWeights {
            attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, seed + 1), q_width, HIDDEN),
            attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 2), kv_width, HIDDEN),
            attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 3), kv_width, HIDDEN)),
            attn_v_unit_norm_w: None,
            attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, seed + 4), HIDDEN, q_width),
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
            kv_share_src: None,
            post_attn_norm: None,
        });

        // Stacked [n_experts * inter, hidden] expert tensors, Q4_K. All N handles
        // share the base ptr; the kernel addresses experts by row stride.
        let mut gate_stacked: Vec<f32> = Vec::new();
        let mut up_stacked: Vec<f32> = Vec::new();
        let mut down_stacked: Vec<f32> = Vec::new();
        for e in 0..N_EXPERTS as u32 {
            let s = seed + 100 + e * 7;
            gate_stacked.extend_from_slice(&det_signal(INTERMEDIATE * HIDDEN, s + 1));
            up_stacked.extend_from_slice(&det_signal(INTERMEDIATE * HIDDEN, s + 2));
            down_stacked.extend_from_slice(&det_signal(HIDDEN * INTERMEDIATE, s + 3));
        }
        let gate_base = allocs.upload_q4_k(&gate_stacked, N_EXPERTS * INTERMEDIATE, HIDDEN);
        let up_base = allocs.upload_q4_k(&up_stacked, N_EXPERTS * INTERMEDIATE, HIDDEN);
        let down_base = allocs.upload_q4_k(&down_stacked, N_EXPERTS * HIDDEN, INTERMEDIATE);
        moe_weights.push(MoeWeights {
            ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            router: allocs.upload_q8_0(&det_signal(N_EXPERTS * HIDDEN, seed + 50), N_EXPERTS, HIDDEN),
            experts_gate: (0..N_EXPERTS).map(|_| gate_base).collect(),
            experts_up: (0..N_EXPERTS).map(|_| up_base).collect(),
            experts_down: (0..N_EXPERTS).map(|_| down_base).collect(),
            n_experts: N_EXPERTS,
            experts_per_tok: EXPERTS_PER_TOK,
            activation: Activation::SwiGLU,
            rms_eps: RMS_EPS,
            shared: None,
            post_ffn_norm: None,
            pre_router_weight_f16: None,
            pre_ffw_norm_2_f16: None,
            post_ffw_norm_1_f32: None,
            post_ffw_norm_2_f32: None,
            post_ffn_norm_f32: None,
            expert_down_scale_f32: None,
        });
    }

    let cfg = ScratchConfig {
        mtp: None,
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        q_width,
        kv_width,
        vocab: VOCAB,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: NUM_LAYERS,
        max_experts: N_EXPERTS,
        max_experts_per_tok: EXPERTS_PER_TOK,
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
        per_layer_kv_depths: None,
    };
    let mut pool = ScratchPool::new(eng_dev, cfg).expect("ScratchPool::new");
    let layout = ModelLayout {
        num_layers: NUM_LAYERS,
        hidden: HIDDEN,
        kv_max_seq_len: MAX_SEQ_LEN,
    };

    let stream = eng_dev.default_stream();
    let logits: Vec<f32>;
    {
        let mut ctx = ForwardEngine::<B>::new_solo(eng_dev, stream, reg, &mut pool);
        let mut resid = ctx.embed(&embd, &[7u32]).expect("embed");
        let layers: Vec<usize> = ctx.layer_range(&layout).collect();
        for li in layers {
            let normed = ctx.rmsnorm(&resid, &attn_weights[li].attn_norm, RMS_EPS, 1).expect("attn rmsnorm");
            let delta = ctx
                .standard_attn(&normed, &attn_weights[li], li, &[0], &[0], None)
                .expect("standard_attn")
                .expect("delta");
            resid = ctx.residual_add(resid, delta, 1).expect("attn residual");
            let normed = ctx.rmsnorm(&resid, &moe_weights[li].ffn_norm, RMS_EPS, 1).expect("moe rmsnorm");
            let delta = ctx.moe_ffn(&normed, &moe_weights[li], 1, None).expect("moe_ffn").expect("delta");
            resid = ctx.residual_add(resid, delta, 1).expect("moe residual");
        }
        ctx.output_head(&resid, &lm_head, &[0]).expect("output_head");
        logits = ctx.logits().to_vec();
    }
    pool.dispose(eng_dev).expect("pool dispose");
    logits
}

#[test]
fn moe_q4k_cuda_matches_hip() {
    let hip_dev = HipDevice::new(0).expect("HipDevice 0");
    hip_dev.bind().expect("hip bind");
    let hip_reg = flambeau_ops::OpsRegistry::new(&hip_dev).expect("OpsRegistry");
    let mut hip_allocs = DeviceAllocs::new(HipDevice::new(0).expect("hip alias"));
    let hip_logits = run::<HipBackend>(&hip_dev, &hip_reg, &mut hip_allocs);

    let cuda_dev = CudaDevice::new(0).expect("CudaDevice 0");
    cuda_dev.bind().expect("cuda bind");
    let cuda_reg = flambeau_ops::CudaOpsRegistry::new(&cuda_dev).expect("CudaOpsRegistry");
    let mut cuda_allocs = DeviceAllocs::new(CudaDevice::new(0).expect("cuda alias"));
    let cuda_logits = run::<CudaBackend>(&cuda_dev, &cuda_reg, &mut cuda_allocs);

    assert_eq!(hip_logits.len(), VOCAB);
    assert_eq!(cuda_logits.len(), VOCAB);
    let max_abs = hip_logits
        .iter()
        .zip(&cuda_logits)
        .map(|(h, c)| (h - c).abs())
        .fold(0.0_f32, f32::max);
    let dot: f32 = hip_logits.iter().zip(&cuda_logits).map(|(h, c)| h * c).sum();
    let nh: f32 = hip_logits.iter().map(|h| h * h).sum::<f32>().sqrt();
    let nc: f32 = cuda_logits.iter().map(|c| c * c).sum::<f32>().sqrt();
    let cosine = dot / (nh * nc);
    eprintln!("MoE Q4_K cross-backend: max_abs={max_abs:.6} logit-cosine={cosine:.7}");

    assert!(cuda_logits.iter().all(|l| l.is_finite()), "CUDA logits not finite");
    assert!(cosine >= 0.9999, "logit-cosine {cosine} < 0.9999 (CUDA vs HIP MoE Q4_K)");
    assert!(max_abs < 5e-3, "max abs logit diff {max_abs} too large");
}
