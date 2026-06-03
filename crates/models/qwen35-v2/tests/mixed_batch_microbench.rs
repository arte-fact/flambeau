//! Phase K3b — real-GGUF microbench of `forward_mixed` vs separate
//! `forward(prefill) + forward(decode)` calls on Qwen3.5-9B-Q4_1.
//!
//! Single-device path (hip:0). Set `FLAMBEAU_MIXED_MICROBENCH_GGUF`
//! to the model path; otherwise the test skips. Sweeps
//! {K=128/N=16, K=256/N=8, K=512/N=4} and reports per-call wall and
//! ratio per cell. Target per `doc/MIXED_BATCH_V2_PLAN.md`: match
//! v1's 1.06-1.17x ceiling.
//!
//! Run:
//!   FLAMBEAU_MIXED_MICROBENCH_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
//!     cargo test --release -p flambeau-qwen35-v2 --test mixed_batch_microbench -- --nocapture

#![cfg(feature = "hip")]

use std::time::Instant;

use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{Device, Stream};
use flambeau_forward::loader::ShardMode;
use flambeau_forward::runtime::Arch;
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::{forward, forward_mixed, Qwen35V2};

fn maybe_skip() -> Option<String> {
    let path = std::env::var("FLAMBEAU_MIXED_MICROBENCH_GGUF").ok()?;
    if device_count().ok()? < 1 {
        eprintln!("[skip] no HIP device");
        return None;
    }
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skip] GGUF not at {path}");
        return None;
    }
    Some(path)
}

struct Cell {
    k: usize,
    n_dec: usize,
}

fn bench_one(
    cell: &Cell,
    model: &<Qwen35V2 as Arch>::Model,
    device: &HipDevice,
    reg: &OpsRegistry,
    n_warmup: usize,
    n_iter: usize,
) -> (f64, f64) {
    let k = cell.k;
    let n_dec = cell.n_dec;
    let n_total = k + n_dec;
    let max_slots = n_dec + 1;
    let cfg = Qwen35V2::scratch_config(
        model,
        ShardMode::Replicated,
        n_total,
        max_slots,
        None,
        flambeau_forward::KvLayout::F16Contig,
    );
    let max_seq_len = cfg.max_seq_len;
    let mut pool = ScratchPool::new(device, cfg).expect("ScratchPool::new");
    let stream = device.default_stream();

    // Build deterministic tokens / positions / slot_ids.
    let tokens: Vec<u32> = (0..n_total as u32).map(|i| (i + 1) % 50).collect();
    let mut pref_positions: Vec<usize> = (0..k).collect();
    let mut pref_slots: Vec<usize> = vec![0usize; k];
    let mut dec_positions: Vec<usize> = vec![0usize; n_dec];
    let dec_slots: Vec<usize> = (1..=n_dec).collect();

    let mut mixed_positions = pref_positions.clone();
    mixed_positions.extend_from_slice(&dec_positions);
    let mut mixed_slots = pref_slots.clone();
    mixed_slots.extend_from_slice(&dec_slots);

    let mut ctx = SingleDeviceForwardCtx::new(device, stream, reg, &mut pool);

    // Warmup
    for _ in 0..n_warmup {
        forward(model, &mut ctx, &tokens[..k], &pref_positions, &pref_slots).expect("warmup pref");
        stream.synchronize().expect("sync");
        forward(
            model,
            &mut ctx,
            &tokens[k..],
            &dec_positions,
            &dec_slots,
        )
        .expect("warmup dec");
        stream.synchronize().expect("sync");
    }

    // Timed: SEPARATE
    let mut t_sep = 0.0f64;
    for it in 0..n_iter {
        // Advance positions per iteration (decode slots step by 1; prefill
        // continues from K * iter).
        for j in 0..k {
            pref_positions[j] = it * k + j;
        }
        for j in 0..n_dec {
            dec_positions[j] = (it * k + k) + j; // arbitrary distinct
        }
        if pref_positions[k - 1] >= max_seq_len {
            break;
        }
        let t0 = Instant::now();
        forward(model, &mut ctx, &tokens[..k], &pref_positions, &pref_slots).expect("sep pref");
        stream.synchronize().expect("sync");
        forward(
            model,
            &mut ctx,
            &tokens[k..],
            &dec_positions,
            &dec_slots,
        )
        .expect("sep dec");
        stream.synchronize().expect("sync");
        t_sep += t0.elapsed().as_secs_f64();
    }
    t_sep /= n_iter as f64;

    // Timed: MIXED
    let mut t_mix = 0.0f64;
    for it in 0..n_iter {
        for j in 0..k {
            mixed_positions[j] = it * k + j;
        }
        for j in 0..n_dec {
            mixed_positions[k + j] = (it * k + k) + j;
        }
        if mixed_positions[k - 1] >= max_seq_len {
            break;
        }
        let t0 = Instant::now();
        forward_mixed(
            model,
            &mut ctx,
            &tokens,
            &mixed_positions,
            &mixed_slots,
            k,
        )
        .expect("mixed");
        stream.synchronize().expect("sync");
        t_mix += t0.elapsed().as_secs_f64();
    }
    t_mix /= n_iter as f64;

    drop(ctx);
    pool.dispose(device).expect("dispose");
    let _ = pref_positions;
    let _ = dec_positions;
    let _ = mixed_positions;
    (t_sep * 1000.0, t_mix * 1000.0)
}

#[test]
fn microbench_mixed_vs_separate() {
    let Some(path) = maybe_skip() else {
        return;
    };
    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");

    let file = GgufFile::open(&path).expect("open gguf");
    // Cap ctx so the smallest per-rank KV doesn't OOM; we don't generate
    // long sequences in this bench anyway.
    let model = Qwen35V2::load(&file, &device, ShardMode::Replicated, None, Some(4096))
        .expect("Qwen35V2::load");

    // Per `doc/MIXED_BATCH_V2_PLAN.md` "Past results to beat (v1
    // reference)" — same shapes the v1 driver shipped.
    let cells = [
        Cell { k: 128, n_dec: 16 },
        Cell { k: 256, n_dec: 8 },
        Cell { k: 512, n_dec: 4 },
    ];

    println!();
    println!("Qwen3.5-9B-Q4_1 / hip:0 (single-device) / per-call wall");
    println!(
        "{:<6} {:<6} {:<12} {:<12} {:<10}",
        "K", "N", "seq (ms)", "mix (ms)", "speedup"
    );
    let mut min_speedup = f64::INFINITY;
    let mut max_speedup: f64 = 0.0;
    for cell in &cells {
        let (sep_ms, mix_ms) = bench_one(cell, &model, &device, &reg, 1, 5);
        let speedup = sep_ms / mix_ms;
        println!(
            "{:<6} {:<6} {:<12.2} {:<12.2} {:<10.3}",
            cell.k, cell.n_dec, sep_ms, mix_ms, speedup
        );
        min_speedup = min_speedup.min(speedup);
        max_speedup = max_speedup.max(speedup);
    }
    println!(
        "\nrange: {:.3}x .. {:.3}x  (v1 reference: 1.063x .. 1.174x)",
        min_speedup, max_speedup
    );

    // Don't assert a numeric bound — single-device shapes differ from v1's
    // pp2tp2 numbers. Print-only.
}
