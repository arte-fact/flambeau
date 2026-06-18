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

fn run<B: Backend>(
    file: &GgufFile,
    device: &B::Device,
    reg: &B::Registry,
    tokens: &[u32],
) -> anyhow::Result<Vec<f32>> {
    let mut model = Gemma4V2::load(file, device, ShardMode::Replicated, None, Some(CTX_CAP))?;
    // prefill_ubatch must cover the whole prompt — embed writes one residual row
    // per token and CUDA validates the DtoD copy stays inside the scratch slot.
    let cfg = Gemma4V2::scratch_config(
        &model,
        ShardMode::Replicated,
        tokens.len(),
        1,
        None,
        KvLayout::F16Contig,
    );
    let mut pool = ScratchPool::new(device, cfg)?;

    let positions: Vec<usize> = (0..tokens.len()).collect();
    let slot_ids = vec![0usize; tokens.len()];
    let stream = device.default_stream();
    let result = {
        let mut ctx = ForwardEngine::<B>::new_solo(device, stream, reg, &mut pool);
        Gemma4V2::forward(&model, &mut ctx, tokens, &positions, &slot_ids)
            .map(|()| ctx.logits().to_vec())
    };
    pool.dispose(device).ok();
    Gemma4V2::dispose(&mut model, device).ok();
    result
}

// KNOWN-FAILING reproducer. The HIP↔CUDA cross-check (gfx906 free) shows the
// CUDA gemma-4 *prefill* path reads uninitialized device memory: layer-0
// logit-cosine vs HIP is non-deterministic across runs (~0.98–0.99, not 1.0) and
// the error accumulates with depth (~0.68 at full depth); single-token output is
// degenerate. The synthetic decode tests (dense ~1e-6, MoE bit-identical) pass
// because they never exercise prefill or gemma's GELU/softcap/SWA/side-channel.
// Un-ignore once the prefill uninitialized-memory bug is fixed (see P5 doc).
#[test]
#[ignore = "reproduces a CUDA prefill uninitialized-memory bug; see P5-CUDA-DECODE-PARITY.md"]
fn gemma4_q4k_cuda_matches_hip() {
    let path = std::env::var("FLAMBEAU_TEST_GEMMA4_Q4K").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: model {path} not found (set FLAMBEAU_TEST_GEMMA4_Q4K)");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");

    // BOS (2) + a short deterministic prompt; one forward, compare next-token logits.
    // Length overridable to isolate prefill (n>1) from single-token decode.
    let n: usize = std::env::var("FLAMBEAU_TEST_NTOK").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let tokens: Vec<u32> = [2u32, 651, 1117, 576, 573, 4080, 603][..n.min(7)].to_vec();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };

    // CUDA is the primary subject — the RTX 3090 is dedicated to it. Real config
    // parse + full Q4_K weight upload + the gemma-4 op set + KV, end to end.
    let cuda_dev = CudaDevice::new(0).expect("CudaDevice 0");
    cuda_dev.bind().expect("cuda bind");
    let cuda_reg = flambeau_ops::CudaOpsRegistry::new(&cuda_dev).expect("CudaOpsRegistry");
    let cuda_logits = run::<CudaBackend>(&file, &cuda_dev, &cuda_reg, &tokens).expect("CUDA forward");

    assert!(cuda_logits.len() > 1000, "logits suspiciously small: {}", cuda_logits.len());
    assert!(cuda_logits.iter().all(|l| l.is_finite()), "CUDA logits not finite");
    let cmax = cuda_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let cmin = cuda_logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(cmax - cmin > 1.0, "CUDA logits degenerate (max-min={})", cmax - cmin);
    eprintln!(
        "gemma4-12B Q4_K CUDA: vocab={} argmax={} range=[{cmin:.3},{cmax:.3}]",
        cuda_logits.len(),
        argmax(&cuda_logits)
    );

    // HIP cross-check is best-effort: the gfx906 cards may be occupied (another
    // process holding VRAM). When they're free, assert exact-ish agreement.
    let hip_dev = HipDevice::new(0).expect("HipDevice 0");
    hip_dev.bind().expect("hip bind");
    let hip_reg = flambeau_ops::OpsRegistry::new(&hip_dev).expect("OpsRegistry");
    match run::<HipBackend>(&file, &hip_dev, &hip_reg, &tokens) {
        Ok(hip_logits) => {
            assert_eq!(hip_logits.len(), cuda_logits.len(), "vocab mismatch");
            let dot: f64 = hip_logits.iter().zip(&cuda_logits).map(|(h, c)| *h as f64 * *c as f64).sum();
            let nh: f64 = hip_logits.iter().map(|h| (*h as f64).powi(2)).sum::<f64>().sqrt();
            let nc: f64 = cuda_logits.iter().map(|c| (*c as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (nh * nc);
            eprintln!(
                "gemma4-12B Q4_K cross-backend: logit-cosine={cosine:.7} argmax hip={} cuda={}",
                argmax(&hip_logits),
                argmax(&cuda_logits)
            );
            assert_eq!(argmax(&hip_logits), argmax(&cuda_logits), "next-token argmax disagrees");
            assert!(cosine >= 0.9999, "logit-cosine {cosine} < 0.9999 (gemma4 Q4_K CUDA vs HIP)");
        }
        Err(e) => eprintln!("SKIP HIP cross-check (gfx906 unavailable): {e:#}"),
    }
}
