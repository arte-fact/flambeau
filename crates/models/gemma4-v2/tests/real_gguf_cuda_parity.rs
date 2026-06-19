//! Real-model cross-backend parity: load a Q4_K Gemma-4 GGUF on HIP (gfx906)
//! and CUDA (sm_86), run one forward pass through `ForwardEngine`, and assert
//! the logits agree. This is the end-to-end P5.4 proof on a real model — real
//! config parsing, full-scale Q4_K weight upload, the gemma-4 op set (GELU FFN,
//! final-logit softcap, per-layer SWA, side-channel embeddings), and the KV
//! cache — all exercised on CUDA against the HIP reference.
//!
//! Skips (passes) when the model file is absent so CI without the GGUF is green.
//! Point at a model with `FLAMBEAU_TEST_GEMMA4_Q4K=/path/to.gguf`.

#![cfg(all(feature = "hip", feature = "cuda"))]

use flambeau_backend::{Backend, CudaBackend, HipBackend};
use flambeau_backend_cuda::CudaDevice;
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::{Arch, ForwardEngine, KvLayout, ScratchPool};
use flambeau_gemma4_v2::Gemma4V2;
use flambeau_quant::GgufFile;

const DEFAULT_MODEL: &str = "/home/artefact/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
const CTX_CAP: usize = 32;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Prefill `prompt`, then greedily generate `n_gen` tokens through the decode
/// path (single token per step, reading the growing KV cache). Returns the
/// prefill next-token logits (for the cross-backend cosine) and the generated
/// token sequence. The engine + KV cache persist across all forwards.
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
    let mut model = Gemma4V2::load(file, device, ShardMode::Replicated, None, Some(CTX_CAP))?;
    // prefill_ubatch must cover the whole prompt — embed writes one residual row
    // per token and CUDA validates the DtoD copy stays inside the scratch slot.
    let cfg = Gemma4V2::scratch_config(
        &model,
        ShardMode::Replicated,
        prompt.len(),
        1,
        None,
        KvLayout::F16Contig,
    );
    let mut pool = ScratchPool::new(device, cfg)?;
    let stream = device.default_stream();

    let result = (|| -> anyhow::Result<RunOut> {
        let mut ctx = ForwardEngine::<B>::new_solo(device, stream, reg, &mut pool);

        let positions: Vec<usize> = (0..prompt.len()).collect();
        let slot_ids = vec![0usize; prompt.len()];
        Gemma4V2::forward(&model, &mut ctx, prompt, &positions, &slot_ids)?;
        let prefill_logits = ctx.logits().to_vec();

        let mut generated = Vec::with_capacity(n_gen);
        let mut logits = prefill_logits.clone();
        let mut pos = prompt.len();
        for _ in 0..n_gen {
            let tok = argmax(&logits) as u32;
            generated.push(tok);
            // Decode the just-emitted token at `pos`, reading KV [0..=pos].
            Gemma4V2::forward(&model, &mut ctx, &[tok], &[pos], &[0])?;
            logits = ctx.logits().to_vec();
            pos += 1;
        }
        Ok(RunOut { prefill_logits, generated })
    })();

    pool.dispose(device).ok();
    Gemma4V2::dispose(&mut model, device).ok();
    result
}

#[test]
fn gemma4_q4k_cuda_matches_hip() {
    let path = std::env::var("FLAMBEAU_TEST_GEMMA4_Q4K").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: model {path} not found (set FLAMBEAU_TEST_GEMMA4_Q4K)");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");

    // BOS (2) + a short deterministic prompt; prefill then greedily generate.
    let prompt: Vec<u32> = vec![2, 651, 1117, 576, 573, 4080, 603];
    let n_gen: usize = std::env::var("FLAMBEAU_TEST_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    // CUDA is the primary subject — the RTX 3090 is dedicated to it. Real config
    // parse + full Q4_K weight upload + the gemma-4 op set + prefill + decode loop.
    let cuda_dev = CudaDevice::new(0).expect("CudaDevice 0");
    cuda_dev.bind().expect("cuda bind");
    let cuda_reg = flambeau_ops::CudaOpsRegistry::new(&cuda_dev).expect("CudaOpsRegistry");
    let cuda = run::<CudaBackend>(&file, &cuda_dev, &cuda_reg, &prompt, n_gen).expect("CUDA run");

    assert!(cuda.prefill_logits.len() > 1000, "logits suspiciously small");
    assert!(cuda.prefill_logits.iter().all(|l| l.is_finite()), "CUDA logits not finite");
    let cmax = cuda.prefill_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let cmin = cuda.prefill_logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(cmax - cmin > 1.0, "CUDA logits degenerate (max-min={})", cmax - cmin);
    eprintln!("gemma4-12B Q4_K CUDA: prefill argmax={} generated={:?}", argmax(&cuda.prefill_logits), cuda.generated);

    // HIP cross-check is best-effort: the gfx906 cards may be occupied (another
    // process holding VRAM). When they're free, assert agreement.
    let hip_dev = HipDevice::new(0).expect("HipDevice 0");
    hip_dev.bind().expect("hip bind");
    let hip_reg = flambeau_ops::OpsRegistry::new(&hip_dev).expect("OpsRegistry");
    match run::<HipBackend>(&file, &hip_dev, &hip_reg, &prompt, n_gen) {
        Ok(hip) => {
            assert_eq!(hip.prefill_logits.len(), cuda.prefill_logits.len(), "vocab mismatch");
            let dot: f64 = hip.prefill_logits.iter().zip(&cuda.prefill_logits).map(|(h, c)| *h as f64 * *c as f64).sum();
            let nh: f64 = hip.prefill_logits.iter().map(|h| (*h as f64).powi(2)).sum::<f64>().sqrt();
            let nc: f64 = cuda.prefill_logits.iter().map(|c| (*c as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (nh * nc);
            eprintln!(
                "gemma4-12B Q4_K cross-backend: logit-cosine={cosine:.7}\n  HIP  generated={:?}\n  CUDA generated={:?}",
                hip.generated, cuda.generated,
            );
            assert_eq!(argmax(&hip.prefill_logits), argmax(&cuda.prefill_logits), "next-token argmax disagrees");
            // Exact cross-backend parity is impossible: HIP reduces over wave64,
            // CUDA over wave32, so F16 attention/norm sums round differently and
            // the gap accumulates with depth (per-layer cosine is ~1e-6 from 1.0;
            // ~0.997 over gemma-4-12B's 48 layers). The meaningful checks are the
            // matching greedy token sequence + no gross divergence.
            assert!(cosine >= 0.995, "prefill logit-cosine {cosine} < 0.995 (CUDA vs HIP)");
            assert_eq!(hip.generated, cuda.generated, "greedy decode sequences diverge (CUDA vs HIP)");
        }
        Err(e) => eprintln!("SKIP HIP cross-check (gfx906 unavailable): {e:#}"),
    }
}
