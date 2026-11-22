//! `embed` through `ForwardEngine` with a Q4_K `token_embd` kept on device
//! must give the bits of the F16 table the loader would have uploaded from the
//! same bytes (host dequant + f16 cast), for a single token and a batch.

#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend::Backend;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{EmbeddingTable, EmbeddingWeights, ForwardCtx};
use flambeau_forward::{ForwardEngine, ScratchPool};
use flambeau_quant::dequant::dequantize_to_vec;
use flambeau_quant::quantize_k::quantize_row_q4_k;
use flambeau_quant::GgmlDType;
use half::f16;

const VOCAB: usize = 64;
const HIDDEN: usize = 512;
const MAX_PREFILL: usize = 8;

fn read_back<D: Device>(dev: &D, ptr: DevicePtr, n: usize) -> Vec<u16> {
    let mut out = vec![0u16; n];
    // SAFETY: `ptr` holds `n` F16 written by the embed pass; synced below.
    unsafe {
        dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(out.as_mut_ptr() as usize), ptr, n * 2)
            .expect("DtoH");
    }
    dev.default_stream().synchronize().expect("sync");
    out
}

fn run<B: Backend>(dev: &B::Device, reg: &B::Registry, allocs: &mut DeviceAllocs<B::Device>) {
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
    let (ptr, _) = allocs.upload(&raw);
    let q4k_embd = EmbeddingWeights {
        token_embd: EmbeddingTable::Quant(flambeau_forward::ctx::QuantWeight {
            ptr,
            dtype: flambeau_core::op::QDtype::Q4_K,
            n_elems: VOCAB * HIDDEN,
        }),
        vocab_size: VOCAB,
        hidden: HIDDEN,
        post_scale: None,
    };

    let cfg = ScratchConfig {
        mtp: None,
        hidden: HIDDEN,
        intermediate: HIDDEN,
        q_width: HIDDEN,
        kv_width: HIDDEN,
        vocab: VOCAB,
        max_seq_len: 16,
        num_layers: 1,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: MAX_PREFILL,
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
        for tokens in [&[7u32][..], &[VOCAB as u32 - 1, 0, 7, 7, 31][..], &[5u32; MAX_PREFILL][..]] {
            let n = tokens.len() * HIDDEN;
            let want = ctx.embed(&f16_embd, tokens).expect("embed F16");
            let want = read_back(dev, want.ptr, n);
            let got = ctx.embed(&q4k_embd, tokens).expect("embed Q4_K");
            let got = read_back(dev, got.ptr, n);
            let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            assert_eq!(bad, 0, "{bad} differing F16 elements for {} tokens", tokens.len());
            assert!(want.iter().any(|&b| f16::from_bits(b).to_f32() != 0.0));
        }
    }
    pool.dispose(dev).expect("pool dispose");
}

#[cfg(feature = "hip")]
#[test]
fn embed_q4k_matches_f16_hip() {
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
fn embed_q4k_matches_f16_cuda() {
    use flambeau_backend::CudaBackend;
    use flambeau_backend_cuda::CudaDevice;
    let dev = CudaDevice::new(0).expect("CudaDevice 0");
    dev.bind().expect("cuda bind");
    let reg = flambeau_ops::CudaOpsRegistry::new(&dev).expect("CudaOpsRegistry");
    let mut allocs = DeviceAllocs::new(CudaDevice::new(0).expect("cuda alias"));
    run::<CudaBackend>(&dev, &reg, &mut allocs);
}
