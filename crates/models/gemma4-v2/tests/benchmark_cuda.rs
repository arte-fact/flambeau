//! CUDA decode benchmark + generation for a gemma-4 model on the RTX 3090 —
//! load a real GGUF, prefill a chat prompt, greedily decode `n_gen` tokens with
//! the decode timed, print the generated token ids (compare to llama.cpp by hand)
//! and decode tok/s. Used to benchmark e.g. gemma-4-31B-qat against llama.cpp.
//! Skips when the model is absent. Run with `--release --ignored --nocapture`.

// `cuda` for the RTX run; `hip` because the shared forward/Arch machinery in
// gemma4-v2 is gated behind it (we only use CudaBackend here).
#![cfg(all(feature = "hip", feature = "cuda"))]

use std::time::Instant;

use flambeau_backend::CudaBackend;
use flambeau_backend_cuda::CudaDevice;
use flambeau_core::{Device, Stream};
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::{Arch, ForwardEngine, KvLayout, ScratchPool};
use flambeau_gemma4_v2::Gemma4V2;
use flambeau_quant::GgufFile;

const DEFAULT_MODEL: &str = "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf";

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

#[test]
#[ignore = "perf/bench — needs the GGUF + the RTX; run explicitly"]
fn gemma_cuda_decode_benchmark() {
    let Some(path) = flambeau_test_models::model_from_env("gemma_cuda_decode_benchmark", "FLAMBEAU_TEST_GEMMA4_Q4K", DEFAULT_MODEL) else {
        return;
    };
    let file = GgufFile::open(&path).expect("open gguf");

    // pp = n_tokens prefill, tg = n_tokens decode. Synthetic prompt — throughput
    // is content-independent (correctness is the parity test's job).
    let pp: usize = std::env::var("FLAMBEAU_PP").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let tg: usize = std::env::var("FLAMBEAU_TG").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let ctx_cap = pp + tg + 8;
    let prompt: Vec<u32> = (0..pp as u32).map(|i| (i % 30000) + 10).collect();

    let dev = CudaDevice::new(0).expect("CudaDevice 0");
    dev.bind().expect("cuda bind");
    let reg = flambeau_ops::CudaOpsRegistry::new(&dev).expect("CudaOpsRegistry");
    let mut model =
        Gemma4V2::load(&file, &dev, ShardMode::Replicated, None, Some(ctx_cap)).expect("load");
    let cfg =
        Gemma4V2::scratch_config(&model, ShardMode::Replicated, pp, 1, None, KvLayout::F16Contig, 0);
    let mut pool = ScratchPool::new(&dev, cfg).expect("pool");
    let stream = dev.default_stream();

    let (pp_tps, tg_tps) = {
        let mut ctx = ForwardEngine::<CudaBackend>::new_solo(&dev, stream, &reg, &mut pool);
        let positions: Vec<usize> = (0..pp).collect();
        let slots = vec![0usize; pp];

        // Prefill rate (one batched forward of `pp` tokens).
        Gemma4V2::forward(&model, &mut ctx, &prompt, &positions, &slots).expect("warmup pp");
        let _ = ctx.logits();
        stream.synchronize().ok();
        let t0 = Instant::now();
        Gemma4V2::forward(&model, &mut ctx, &prompt, &positions, &slots).expect("pp");
        let mut logits = ctx.logits().to_vec();
        let pp_tps = pp as f64 / t0.elapsed().as_secs_f64();

        // Decode rate (`tg` single-token steps).
        let mut pos = pp;
        for _ in 0..2 {
            let tok = argmax(&logits);
            Gemma4V2::forward(&model, &mut ctx, &[tok], &[pos], &[0]).expect("warmup tg");
            logits = ctx.logits().to_vec();
            pos += 1;
        }
        stream.synchronize().ok();
        let t1 = Instant::now();
        for _ in 0..tg {
            let tok = argmax(&logits);
            Gemma4V2::forward(&model, &mut ctx, &[tok], &[pos], &[0]).expect("decode");
            logits = ctx.logits().to_vec();
            pos += 1;
        }
        let tg_tps = tg as f64 / t1.elapsed().as_secs_f64();
        (pp_tps, tg_tps)
    };

    eprintln!(
        "gemma-4 CUDA (RTX) [{}]: PREFILL {pp_tps:7.1} tok/s (pp={pp})  |  DECODE {tg_tps:6.2} tok/s (tg={tg})",
        std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
    );

    pool.dispose(&dev).ok();
    Gemma4V2::dispose(&mut model, &dev).ok();
}
