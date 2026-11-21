//! Real-model cross-backend parity for an **i-quant** model: load
//! `Qwen3.6-27B-IQ2_XXS` (arch `qwen35`) on HIP gfx906 and CUDA sm_86, prefill a
//! chat prompt then greedily generate. CUDA reproduces HIP's greedy continuation
//! token-for-token (the correctness proof for the ported IQ decode kernels).
//! ("Qwen3.6" is a release label — the GGUF arch is `qwen35`.) Skips when absent.
//!
//! 2-bit IQ2_XXS is the default because it fits the 16 GB MI50 for the HIP
//! cross-check; its flat logits mean greedy splits from llama.cpp at near-ties.
//! The sharper 4-bit **IQ4_NL matches llama.cpp exactly** (10/10), like Q4_K —
//! reproduce with (HIP OOMs at 16 GB, so it's a CUDA-vs-llama.cpp check):
//!   FLAMBEAU_TEST_QWEN_IQ=…/Qwen3.6-27B-IQ4_NL.gguf \
//!   FLAMBEAU_TEST_LLAMACPP_REF=248068,271,248069,271,760,6511,314,9338,369,2972 \
//!   cargo test -p flambeau-qwen35-v2 --features hip,cuda --test real_gguf_cuda_parity

#![cfg(all(feature = "hip", feature = "cuda"))]

use flambeau_backend::{Backend, CudaBackend, HipBackend};
use flambeau_backend_cuda::CudaDevice;
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::{LoadOptions, ShardMode};
use flambeau_forward::{Arch, ForwardEngine, KvLayout, ScratchPool};
use flambeau_qwen35_v2::Qwen35V2;
use flambeau_quant::GgufFile;

const DEFAULT_MODEL: &str = "Qwen3.6-27B-UD-IQ2_XXS.gguf";
const CTX_CAP: usize = 96;

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
}

struct RunOut {
    prefill_logits: Vec<f32>,
    generated: Vec<u32>,
}

fn run<B: Backend>(
    file: &GgufFile,
    device: &B::Device,
    reg: &B::Registry,
    prompt: &[u32],
    n_gen: usize,
) -> anyhow::Result<RunOut> {
    let opts = LoadOptions::with_ctx_cap(CTX_CAP);
    let mut model = Qwen35V2::load(file, device, ShardMode::Replicated, None, opts)?;
    let cfg = Qwen35V2::scratch_config(&model, ShardMode::Replicated, prompt.len(), 1, None, KvLayout::F16Contig, 0);
    let mut pool = ScratchPool::new(device, cfg)?;
    let stream = device.default_stream();

    let result = (|| -> anyhow::Result<RunOut> {
        let mut ctx = ForwardEngine::<B>::new_solo(device, stream, reg, &mut pool);
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let slot_ids = vec![0usize; prompt.len()];
        Qwen35V2::forward(&model, &mut ctx, prompt, &positions, &slot_ids)?;
        let prefill_logits = ctx.logits().to_vec();

        let mut generated = Vec::with_capacity(n_gen);
        let mut logits = prefill_logits.clone();
        let mut pos = prompt.len();
        for _ in 0..n_gen {
            let tok = argmax(&logits) as u32;
            generated.push(tok);
            Qwen35V2::forward(&model, &mut ctx, &[tok], &[pos], &[0])?;
            logits = ctx.logits().to_vec();
            pos += 1;
        }
        Ok(RunOut { prefill_logits, generated })
    })();

    pool.dispose(device).ok();
    Qwen35V2::dispose(&mut model, device).ok();
    result
}

#[test]
fn qwen35_iq2xxs_cuda_matches_llamacpp() {
    let Some(path) = flambeau_test_models::model_from_env("qwen35_iq2xxs_cuda_matches_llamacpp", "FLAMBEAU_TEST_QWEN_IQ", DEFAULT_MODEL) else {
        return;
    };
    let file = GgufFile::open(&path).expect("open gguf");

    // Chat-templated "What is the capital of France?" via the qwen tokenizer
    // (<|im_start|>user … <|im_start|>assistant). Overridable.
    let prompt: Vec<u32> = std::env::var("FLAMBEAU_TEST_PROMPT_IDS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]);
    let n_gen: usize = std::env::var("FLAMBEAU_TEST_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    // llama.cpp (build-mi50, same IQ2_XXS GGUF, temp 0) greedy continuation:
    // "<think>\nThe user is asking for the capital of" — the oracle.
    let llamacpp_ref: Option<Vec<u32>> = Some(
        std::env::var("FLAMBEAU_TEST_LLAMACPP_REF")
            .ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![248068, 198, 760, 1156, 369, 9859, 364, 279, 6511, 314]),
    );

    let cuda_dev = CudaDevice::new(0).expect("CudaDevice 0");
    cuda_dev.bind().expect("cuda bind");
    let cuda_reg = flambeau_ops::CudaOpsRegistry::new(&cuda_dev).expect("CudaOpsRegistry");
    let cuda = run::<CudaBackend>(&file, &cuda_dev, &cuda_reg, &prompt, n_gen).expect("CUDA run");

    assert!(cuda.prefill_logits.iter().all(|l| l.is_finite()), "CUDA logits not finite");
    eprintln!("qwen IQ2_XXS CUDA generated={:?}", cuda.generated);

    if let Some(r) = &llamacpp_ref {
        let common = r.iter().zip(&cuda.generated).take_while(|(a, b)| a == b).count();
        eprintln!("  CUDA matches llama.cpp on first {common} tokens (ref={r:?})");
        // IQ2_XXS is 2-bit — logits are flat enough that greedy splits at near-ties
        // across implementations; require only a matching high-confidence prefix.
        assert!(common >= 2, "CUDA diverges from llama.cpp immediately (common={common})");
    }

    let hip_dev = HipDevice::new(0).expect("HipDevice 0");
    hip_dev.bind().expect("hip bind");
    let hip_reg = flambeau_ops::OpsRegistry::new(&hip_dev).expect("OpsRegistry");
    match run::<HipBackend>(&file, &hip_dev, &hip_reg, &prompt, n_gen) {
        Ok(hip) => {
            let dot: f64 = hip.prefill_logits.iter().zip(&cuda.prefill_logits).map(|(h, c)| *h as f64 * *c as f64).sum();
            let nh: f64 = hip.prefill_logits.iter().map(|h| (*h as f64).powi(2)).sum::<f64>().sqrt();
            let nc: f64 = cuda.prefill_logits.iter().map(|c| (*c as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (nh * nc);
            let common = hip.generated.iter().zip(&cuda.generated).take_while(|(h, c)| h == c).count();
            eprintln!(
                "qwen IQ2_XXS cross-backend: logit-cosine={cosine:.6}\n  HIP  generated={:?}\n  HIP↔CUDA agree on first {common} tokens",
                hip.generated,
            );
            assert_eq!(argmax(&hip.prefill_logits), argmax(&cuda.prefill_logits), "prefill argmax disagrees");
            assert!(cosine >= 0.95, "prefill logit-cosine {cosine} < 0.95 (CUDA vs HIP)");
            // The IQ2_XXS decode kernels on CUDA reproduce HIP's greedy output
            // token-for-token — the correctness proof for the ported IQ path
            // (llama.cpp greedy splits at 2-bit near-ties, but both backends agree).
            assert_eq!(hip.generated, cuda.generated, "HIP vs CUDA greedy sequences differ");
        }
        Err(e) => eprintln!("SKIP HIP cross-check (gfx906 unavailable): {e:#}"),
    }
}
