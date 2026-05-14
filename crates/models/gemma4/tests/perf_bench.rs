//! #9 — Gemma 4 perf bench (flambeau). Measures prefill + decode
//! throughput on real GGUFs across topologies the driver layer currently
//! supports end-to-end:
//!
//! - E4B-Q4_0 single-device (`hip:0`)
//! - 31B-Q4_0 PP2 (`hip:0,2`)
//! - 31B-Q4_0 PP4 (`hip:0,1,2,3`)
//!
//! TP + hybrid real-GGUF benches require the TP/hybrid upload paths
//! (#20 / #21) to land; tracked separately.
//!
//! Methodology (mirrors the "≥ 3 post-warmup runs" rule from memory
//! `qwen36_35B_A3B_vs_llamacpp_2026_04_29`): one upload + warmup, then
//! 3 prefill + 3 decode measurements with KV cleared between runs.
//! Report min wall-clock per phase → tok/s.
//!
//! NOTE on single-device gemma4: there is no batched-prefill kernel for
//! the single-device path (only PP has `forward_prefill_pp`). The
//! single-device "prefill" measurement loops `forward_one_token` over
//! the prompt, so it shows the same per-token cost as decode. That's
//! reported honestly. Adding a single-device batched-prefill path is a
//! future task.
//!
//! Output: tok/s per (model, topology) printed to stderr; the cert
//! markdown at `certs/perf/gemma4_v1_bench/cert.md` is composed by the
//! caller from the printed numbers + the corresponding llama-bench run.
//!
//! Skipped when GGUFs are absent or HIP device count is insufficient.

#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use flambeau_backend_hip::{device_count, HipCluster, HipDevice};
use flambeau_gemma4::{
    forward_one_token, partition_layers, Gemma4Config, Gemma4DeviceWeights, Gemma4PpDriver,
    Gemma4Session, ModelLayout,
};
use flambeau_quant::GgufFile;

const MODELS_DIR: &str = "/artefact/models";
const PROMPT_LEN: usize = 512;
const DECODE_LEN: usize = 128;
const N_RUNS: usize = 3;
const MAX_TOKENS: usize = PROMPT_LEN + DECODE_LEN + 16;

fn open_or_skip(name: &str) -> Option<GgufFile> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok()
}

fn cluster_or_skip(devices: &[i32]) -> Option<HipCluster> {
    let n = device_count().ok()?;
    if (n as usize) < devices.len() {
        eprintln!("skipping — need {} HIP devices, have {n}", devices.len());
        return None;
    }
    HipCluster::new(devices).ok()
}

/// Build a deterministic prompt of safe vocab ids.
fn synth_prompt(cfg: &Gemma4Config) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..PROMPT_LEN).map(|i| 256 + (i as u32)).collect();
    let vmax = (cfg.vocab_size as u32).saturating_sub(1);
    for t in &mut tokens {
        if *t > vmax {
            *t = vmax;
        }
    }
    tokens
}

fn min_of<F: FnMut() -> f64>(n: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..n {
        let ms = f();
        if ms < best {
            best = ms;
        }
    }
    best
}

fn report(model_id: &str, topology: &str, prefill_ms: f64, decode_ms: f64) {
    let prefill_tps = (PROMPT_LEN as f64) * 1000.0 / prefill_ms;
    let decode_tps = (DECODE_LEN as f64) * 1000.0 / decode_ms;
    eprintln!("\n=== BENCH | {model_id} | {topology} ===");
    eprintln!("  prefill ({} tok): {:.1} tok/s  ({:.2} ms)", PROMPT_LEN, prefill_tps, prefill_ms);
    eprintln!("  decode  ({} tok): {:.1} tok/s  ({:.2} ms)", DECODE_LEN, decode_tps, decode_ms);
}

#[test]
fn bench_e4b_q4_0_single() {
    let Some(file) = open_or_skip("gemma-4-E4B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 1).unwrap_or(true) {
        eprintln!("skipping — no HIP device");
        return;
    }
    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");

    let file_arc = Arc::new(file);
    let cfg = Gemma4Config::from_gguf(&file_arc).expect("cfg");
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let weights = Gemma4DeviceWeights::upload(&file_arc, &cfg, &layout, &device)
        .expect("upload E4B");
    let mut session = if cfg.per_layer_embed.is_some() {
        Gemma4Session::new_with_gguf(&device, weights, cfg.clone(), layout, MAX_TOKENS, file_arc.clone())
            .expect("session E4B (with gguf)")
    } else {
        Gemma4Session::new(&device, weights, cfg.clone(), layout, MAX_TOKENS)
            .expect("session E4B")
    };
    let prompt = synth_prompt(&cfg);

    // Warmup.
    let _ = forward_one_token(&mut session, &device, prompt[0], 0).expect("warmup");
    for _ in 0..16 {
        let _ = forward_one_token(&mut session, &device, prompt[0], 0).expect("warmup");
    }
    clear_session_kv(&mut session);

    // Prefill measurement: loop forward_one_token over `prompt` (single
    // device has no batched prefill kernel yet — see file header).
    let prefill_ms = min_of(N_RUNS, || {
        clear_session_kv(&mut session);
        let t0 = Instant::now();
        for (i, &t) in prompt.iter().enumerate() {
            let _ = forward_one_token(&mut session, &device, t, i).expect("prefill step");
        }
        t0.elapsed().as_secs_f64() * 1000.0
    });

    // Decode measurement: prefill first (untimed), then time DECODE_LEN steps.
    let decode_ms = min_of(N_RUNS, || {
        clear_session_kv(&mut session);
        let mut tok = prompt[0];
        for (i, &t) in prompt.iter().enumerate() {
            tok = forward_one_token(&mut session, &device, t, i).expect("prefill before decode");
        }
        let t0 = Instant::now();
        let mut pos = prompt.len();
        for _ in 0..DECODE_LEN {
            tok = forward_one_token(&mut session, &device, tok, pos).expect("decode");
            pos += 1;
        }
        t0.elapsed().as_secs_f64() * 1000.0
    });

    report("gemma4_E4B_Q4_0", "single (hip:0)", prefill_ms, decode_ms);
    session.dispose(&device).expect("dispose session");
}

fn clear_session_kv(session: &mut Gemma4Session) {
    for slot in session.kv_caches.iter_mut() {
        if let Some(kv) = slot {
            kv.clear();
        }
    }
}

#[test]
fn bench_31b_q4_0_pp2() {
    bench_31b_pp("gemma-4-31B-it-Q4_0.gguf", "PP2 (hip:0,2)", &[0, 2]);
}

#[test]
fn bench_31b_q4_0_pp4() {
    bench_31b_pp("gemma-4-31B-it-Q4_0.gguf", "PP4 (hip:0,1,2,3)", &[0, 1, 2, 3]);
}

fn bench_31b_pp(model: &str, topo_id: &str, devices: &[i32]) {
    let Some(file) = open_or_skip(model) else {
        return;
    };
    let Some(cluster) = cluster_or_skip(devices) else {
        return;
    };

    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let layer_to_rank = partition_layers(devices.len(), &layout).expect("partition");

    let mut driver = Gemma4PpDriver::upload(
        &file,
        cfg.clone(),
        layout,
        layer_to_rank,
        cluster,
        MAX_TOKENS,
    )
    .expect("upload 31B PP");

    let prompt = synth_prompt(&cfg);

    // Warmup.
    let warm = driver.forward_prefill(&prompt, 0).expect("warmup prefill");
    let mut tok = warm;
    let mut pos = prompt.len();
    for _ in 0..16 {
        tok = driver.forward_one_token(tok, pos).expect("warmup decode");
        pos += 1;
    }
    clear_driver_kv(&mut driver);

    let prefill_ms = min_of(N_RUNS, || {
        clear_driver_kv(&mut driver);
        let t0 = Instant::now();
        let _ = driver.forward_prefill(&prompt, 0).expect("prefill");
        t0.elapsed().as_secs_f64() * 1000.0
    });

    let decode_ms = min_of(N_RUNS, || {
        clear_driver_kv(&mut driver);
        let first = driver.forward_prefill(&prompt, 0).expect("prefill before decode");
        let t0 = Instant::now();
        let mut tok = first;
        let mut pos = prompt.len();
        for _ in 0..DECODE_LEN {
            tok = driver.forward_one_token(tok, pos).expect("decode");
            pos += 1;
        }
        t0.elapsed().as_secs_f64() * 1000.0
    });

    report("gemma4_31B_Q4_0", topo_id, prefill_ms, decode_ms);
    driver.dispose().expect("dispose");
}

fn clear_driver_kv(driver: &mut Gemma4PpDriver) {
    for stage in driver.stages.iter_mut() {
        for slot in stage.kv_caches.iter_mut() {
            if let Some(kv) = slot {
                kv.clear();
            }
        }
    }
}
