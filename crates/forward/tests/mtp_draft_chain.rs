//! Greedy MTP drafting: the device-chained draw (Q4_K table, the next step
//! embeds the previous argmax slot, one sync) must propose exactly the tokens
//! of the per-step draw (F16 table from the same bytes, one 4-byte readback
//! per step, the path that table keeps) and of the host draw (logits back,
//! adjustments + argmax on the host). Same kernels; only where the token
//! travels between steps changes.

#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::sync::Arc;

use common::{det_signal, DeviceAllocs};
use flambeau_backend::Backend;
use flambeau_core::Device;
use flambeau_forward::core::scratch::MtpScratch;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingTable, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, MtpDraft,
    MtpWeights, QuantWeight, RopeVariant,
};
use flambeau_forward::{ForwardEngine, ScratchPool};
use flambeau_quant::dequant::dequantize_to_vec;
use flambeau_quant::quantize_k::quantize_row_q4_k;
use flambeau_quant::GgmlDType;
use flambeau_runtime::{MtpArgmax, MtpSampler};

const VOCAB: usize = 512;
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 512;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 32;
const N_DRAFT: usize = 4;
const RMS_EPS: f32 = 1e-5;
const ADDS: [(u32, f32); 2] = [(5, 0.25), (300, -0.5)];

fn host_draw(logits: &[f32]) -> (u32, Vec<(u32, f32)>) {
    let mut l = logits.to_vec();
    for (id, v) in ADDS {
        l[id as usize] += v;
    }
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > best_v {
            best = i as u32;
            best_v = v;
        }
    }
    (best, vec![(best, 1.0)])
}

fn device_sampler() -> MtpSampler {
    MtpSampler {
        draw: Box::new(|_| panic!("device plan: no host draw expected")),
        argmax: Some(Arc::new(MtpArgmax {
            adds: ADDS.to_vec(),
            penalty_counts: Vec::new(),
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
        })),
    }
}

fn run<B: Backend>(dev: &B::Device, reg: &B::Registry, allocs: &mut DeviceAllocs<B::Device>) {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;

    let signal = det_signal(VOCAB * HIDDEN, 3);
    let mut raw = Vec::new();
    for r in 0..VOCAB {
        quantize_row_q4_k(&signal[r * HIDDEN..(r + 1) * HIDDEN], &mut raw);
    }
    let dequant = dequantize_to_vec(GgmlDType::Q4K, &raw, VOCAB * HIDDEN).expect("host dequant");
    let f16_embd = EmbeddingWeights {
        token_embd: allocs.upload_f16(&dequant).into(),
        vocab_size: VOCAB,
        hidden: HIDDEN,
        post_scale: None,
    };
    let (q4k_ptr, _) = allocs.upload(&raw);
    let q4k_embd = EmbeddingWeights {
        token_embd: EmbeddingTable::Quant(QuantWeight {
            ptr: q4k_ptr,
            dtype: flambeau_core::op::QDtype::Q4_K,
            n_elems: VOCAB * HIDDEN,
        }),
        vocab_size: VOCAB,
        hidden: HIDDEN,
        post_scale: None,
    };
    assert!(q4k_embd.gathers_device_ids() && !f16_embd.gathers_device_ids());

    let lm_head = LmHeadWeights {
        output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
        final_logit_softcap: None,
        vocab_size: VOCAB,
        hidden: HIDDEN,
        rms_eps: RMS_EPS,
    };
    let seed = 200;
    let mtp = MtpWeights {
        eh_proj_e: allocs.upload_f16(&det_signal(HIDDEN * HIDDEN, seed)),
        eh_proj_h: allocs.upload_f16(&det_signal(HIDDEN * HIDDEN, seed + 1)),
        enorm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        hnorm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        shared_head_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn: AttnWeights {
            attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, seed + 2), q_width, HIDDEN),
            attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 3), kv_width, HIDDEN),
            attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 4), kv_width, HIDDEN)),
            attn_v_unit_norm_w: None,
            attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, seed + 5), HIDDEN, q_width),
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
        },
        ffn: FfnWeights {
            ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            post_ffn_norm: None,
            ffn_gate: allocs.upload_q8_0(&det_signal(INTERMEDIATE * HIDDEN, seed + 6), INTERMEDIATE, HIDDEN),
            ffn_up: allocs.upload_q8_0(&det_signal(INTERMEDIATE * HIDDEN, seed + 7), INTERMEDIATE, HIDDEN),
            ffn_down: allocs.upload_q8_0(&det_signal(HIDDEN * INTERMEDIATE, seed + 8), HIDDEN, INTERMEDIATE),
            activation: Activation::SwiGLU,
            rms_eps: RMS_EPS,
        },
        rms_eps: RMS_EPS,
    };
    let anchor_h = allocs.upload_f16(&det_signal(HIDDEN, 77));

    let cfg = ScratchConfig {
        mtp: Some(MtpScratch { kv_width, max_draft: N_DRAFT }),
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        q_width,
        kv_width,
        vocab: VOCAB,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: 2,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: N_DRAFT + 1,
        max_slots: 1,
        per_layer_embd: 0,
        paged_kv: None,
        kv_layout: flambeau_forward::core::KvLayout::F16Contig,
        per_layer_kv_layouts: None,
        per_layer_kv_depths: None,
    };
    let mut pool = ScratchPool::new(dev, cfg).expect("ScratchPool::new");
    let stream = dev.default_stream();
    {
        let mut ctx = ForwardEngine::<B>::new_solo(dev, stream, reg, &mut pool);
        let mut draft = |embd: &EmbeddingWeights, mut sampler: MtpSampler| -> Vec<u32> {
            let mut drafts: Vec<MtpDraft> = Vec::new();
            ctx.mtp_draft(&mtp, &lm_head, embd, 17, &anchor_h, 0, 0, 1, N_DRAFT, &mut sampler, &mut drafts)
                .expect("mtp_draft");
            drafts.iter().map(|d| d.token).collect()
        };
        let chained = draft(&q4k_embd, device_sampler());
        let per_step = draft(&f16_embd, device_sampler());
        let host = draft(&q4k_embd, MtpSampler { draw: Box::new(host_draw), argmax: None });
        assert_eq!(chained.len(), N_DRAFT);
        assert_eq!(chained, per_step, "chained vs per-step device draw");
        assert_eq!(chained, host, "chained vs host draw");
        eprintln!("drafts {chained:?}");
    }
    pool.dispose(dev).expect("pool dispose");
}

#[cfg(feature = "hip")]
#[test]
fn mtp_draft_chain_matches_per_step_hip() {
    use flambeau_backend::HipBackend;
    use flambeau_backend_hip::HipDevice;
    let dev = HipDevice::new(0).expect("HipDevice 0");
    dev.bind().expect("hip bind");
    let reg = flambeau_ops::OpsRegistry::new(&dev).expect("OpsRegistry");
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("hip alias"));
    run::<HipBackend>(&dev, &reg, &mut allocs);
}

#[cfg(feature = "cuda")]
#[test]
fn mtp_draft_chain_matches_per_step_cuda() {
    use flambeau_backend::CudaBackend;
    use flambeau_backend_cuda::CudaDevice;
    let dev = CudaDevice::new(0).expect("CudaDevice 0");
    dev.bind().expect("cuda bind");
    let reg = flambeau_ops::CudaOpsRegistry::new(&dev).expect("CudaOpsRegistry");
    let mut allocs = DeviceAllocs::new(CudaDevice::new(0).expect("cuda alias"));
    run::<CudaBackend>(&dev, &reg, &mut allocs);
}
