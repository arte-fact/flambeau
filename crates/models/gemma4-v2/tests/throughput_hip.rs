//! End-to-end prefill + decode throughput on HIP gfx906 — wall-clock A/B harness
//! for dispatch decisions (e.g. the Q8_0 decode kernel alias). Loads a real GGUF,
//! times a `pp`-token prefill and `tg` single-token decode steps, reports tok/s.
//! Skips when the model is absent. Run with `--release`.
//!
//! KNOWN LIMITATIONS (this is a first probe, not yet a trustworthy `bench
//! matrix`):
//! - **Decode (tg) is the reliable metric.** `new_solo` is decode-mode.
//! - **Prefill (pp) is NOT yet trustworthy**: it measures ≈ `pp` × decode-step,
//!   i.e. it is not hitting the batched-MMQ prefill path through this
//!   `ForwardEngine::new_solo` entry. Treat the pp number as a placeholder until
//!   the batched-prefill engine is wired. Do not cite it.
//! - **Thermal sensitivity**: at the 200W cap, auto-clock throttles across
//!   consecutive runs (observed decode 31→27→21 tok/s over 3 back-to-back runs).
//!   A cross-process A/B can't resolve a <20% kernel delta — compare best-of-N on
//!   a cooled GPU, or time both kernels in one process (the `autotune` microbench
//!   already does the latter).
//!
//!   FLAMBEAU_Q8_MODEL=…/gemma-4-12b-it-Q8_0.gguf \
//!   cargo test -p flambeau-gemma4-v2 --features hip --release \
//!     --test throughput_hip -- --nocapture --ignored

#![cfg(feature = "hip")]

use std::time::Instant;

use flambeau_backend::HipBackend;
use flambeau_backend_hip::HipDevice;
use flambeau_core::{Device, Stream};
use flambeau_forward::ctx::ForwardCtx;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::{Arch, ForwardEngine, KvLayout, ScratchPool};
use flambeau_gemma4_v2::Gemma4V2;
use flambeau_quant::GgufFile;

const DEFAULT_MODEL: &str = "/home/artefact/models/gemma-4-12b-it-Q8_0.gguf";

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

#[test]
#[ignore = "perf harness — needs the GGUF + a quiet GPU; run explicitly"]
fn gemma_q8_0_prefill_decode_throughput() {
    let path = std::env::var("FLAMBEAU_Q8_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: model {path} not found (set FLAMBEAU_Q8_MODEL)");
        return;
    }
    let pp: usize = std::env::var("FLAMBEAU_PP").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let tg: usize = std::env::var("FLAMBEAU_TG").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let ctx_cap = pp + tg + 16;

    let file = GgufFile::open(&path).expect("open gguf");
    let dev = HipDevice::new(0).expect("HipDevice 0");
    dev.bind().expect("bind");
    let reg = flambeau_ops::OpsRegistry::new(&dev).expect("OpsRegistry");
    let mut model =
        Gemma4V2::load(&file, &dev, ShardMode::Replicated, None, Some(ctx_cap)).expect("load");
    let cfg =
        Gemma4V2::scratch_config(&model, ShardMode::Replicated, pp, 1, None, KvLayout::F16Contig);
    let mut pool = ScratchPool::new(&dev, cfg).expect("pool");
    let stream = dev.default_stream();

    // Token values don't affect throughput — the kernels run identically.
    let prompt: Vec<u32> = (0..pp as u32).map(|i| (i % 30000) + 10).collect();
    let positions: Vec<usize> = (0..pp).collect();
    let slots = vec![0usize; pp];

    let (pp_tps, tg_tps) = {
        let mut ctx = ForwardEngine::<HipBackend>::new_solo(&dev, stream, &reg, &mut pool);

        // Warmup prefill — first-touch allocs, instruction cache, clock ramp.
        Gemma4V2::forward(&model, &mut ctx, &prompt, &positions, &slots).expect("warmup pp");
        let _ = ctx.logits();

        // Prefill rate (MMQ path): re-run the prefill, timed to completion.
        stream.synchronize().ok();
        let t0 = Instant::now();
        Gemma4V2::forward(&model, &mut ctx, &prompt, &positions, &slots).expect("pp");
        let mut logits = ctx.logits().to_vec(); // host copy forces completion
        let pp_tps = pp as f64 / t0.elapsed().as_secs_f64();

        // Decode rate (MMVQ path): tg single-token steps from pos=pp.
        let mut pos = pp;
        for _ in 0..2 {
            // warmup decode steps
            let tok = argmax(&logits);
            Gemma4V2::forward(&model, &mut ctx, &[tok], &[pos], &[0]).expect("warmup tg");
            logits = ctx.logits().to_vec();
            pos += 1;
        }
        stream.synchronize().ok();
        let t1 = Instant::now();
        for _ in 0..tg {
            let tok = argmax(&logits);
            Gemma4V2::forward(&model, &mut ctx, &[tok], &[pos], &[0]).expect("tg");
            logits = ctx.logits().to_vec();
            pos += 1;
        }
        let tg_tps = tg as f64 / t1.elapsed().as_secs_f64();
        (pp_tps, tg_tps)
    };

    eprintln!(
        "gemma-Q8_0 gfx906: PREFILL {pp_tps:8.1} tok/s ({pp} tok)  |  DECODE {tg_tps:8.2} tok/s ({tg} tok)"
    );

    pool.dispose(&dev).ok();
    Gemma4V2::dispose(&mut model, &dev).ok();
}
