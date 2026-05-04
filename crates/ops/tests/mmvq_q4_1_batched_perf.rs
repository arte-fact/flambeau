//! #288 microbench — wall-clock A/B between per-row `qmatmul(m=1)` ×N
//! and the new batched `mmvq_q4_1_batched` kernel via `qmatmul(m=N)`.
//!
//! Goal: confirm the weight-HBM amortization lever delivers a real
//! speedup at the GDN matmul shapes used by Qwen3.6-27B (k=4096,
//! n_rows in [3584, 14336]). Not the perf gate — that's #292 / live
//! cert. This bench just sanity-checks direction-of-win.
//!
//! Run with:
//!   cargo test --release -p flambeau-ops --features hip \
//!     --test mmvq_q4_1_batched_perf -- --nocapture --ignored

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "perf bench — every unsafe block is a memcpy / kernel launch over \
              locally-allocated bounded buffers."
)]

use std::time::Instant;

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::hip::norm::quantize_q8_1;
use flambeau_ops::hip::qmatmul::qmatmul;
use flambeau_ops::OpsRegistry;
use flambeau_quant::{BlockQ4_1, BlockQ8_1};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping perf bench");
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
}

fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn alloc_zeroed(dev: &HipDevice, bytes: usize) -> DevicePtr {
    let d = dev.alloc(bytes).unwrap();
    let host_zero = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(host_zero.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn seeded_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            ((u as f32 / u32::MAX as f32) - 0.5) * scale
        })
        .collect()
}

fn quantize_row_q4_1(row: &[f32]) -> Vec<BlockQ4_1> {
    assert_eq!(row.len() % 32, 0);
    let n_blocks = row.len() / 32;
    let mut blocks: Vec<BlockQ4_1> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let chunk = &row[b * 32..(b + 1) * 32];
        let mut min = chunk[0];
        let mut max = chunk[0];
        for &v in &chunk[1..] {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0u8; 16];
        for i in 0..16 {
            let lo = ((chunk[i] - min) * id).round().clamp(0.0, 15.0) as u8;
            let hi = ((chunk[i + 16] - min) * id).round().clamp(0.0, 15.0) as u8;
            qs[i] = (hi << 4) | (lo & 0x0F);
        }
        blocks.push(BlockQ4_1 {
            d: f16::from_f32(d),
            m: f16::from_f32(min),
            qs,
        });
    }
    blocks
}

fn time_us<F: FnMut() -> Result<()>>(stream: &flambeau_backend_hip::HipStream, iters: usize, mut f: F) -> Result<f64> {
    // Warm-up.
    for _ in 0..3 {
        f()?;
    }
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    stream.synchronize()?;
    let elapsed = t0.elapsed();
    Ok(elapsed.as_secs_f64() * 1e6 / iters as f64)
}

#[test]
#[ignore = "perf bench (run with --ignored)"]
fn mmvq_q4_1_batched_perf_sweep() -> Result<()> {
    let Some(dev) = dev_or_skip() else { return Ok(()); };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    // Qwen3.6-27B GDN-stage shapes (post TP=2 splits): hidden=3584,
    // d_inner=14336, etc. k=4096 is the typical inner dimension.
    let shapes: &[(usize, usize, &str)] = &[
        (3584, 4096, "qkv (≈hidden×k)"),
        (14336, 4096, "ssm_out width"),
    ];
    let slot_counts: &[usize] = &[1, 2, 4, 8];
    let iters = 50;

    eprintln!("# mmvq_q4_1_batched perf sweep — Qwen3.6-27B GDN matmul shapes");
    eprintln!("# Wall time per call (µs), 50 iters after 3 warm-up.");
    eprintln!("# n_rows × k          | N=1 (single)  | N=2 batched  | N=4 batched  | N=8 batched");

    for &(n_rows, k, label) in shapes {
        let n_blocks_per_row = k / 32;
        let max_n = *slot_counts.iter().max().unwrap();

        // Build weights once per shape.
        let w_f32 = seeded_f32(1, n_rows * k, 0.5);
        let mut w_q4_1: Vec<BlockQ4_1> = Vec::with_capacity(n_rows * n_blocks_per_row);
        for r in 0..n_rows {
            let row = &w_f32[r * k..(r + 1) * k];
            w_q4_1.extend_from_slice(&quantize_row_q4_1(row));
        }
        let d_w = upload(&dev, &w_q4_1);

        // Build activations sized for max N.
        let act_f32 = seeded_f32(2, max_n * k, 1.0);
        let d_act_f32 = upload(&dev, &act_f32);
        let act_q8_1_bytes = max_n * n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
        let d_act_q8_1 = alloc_zeroed(&dev, act_q8_1_bytes);
        quantize_q8_1(&reg, stream, d_act_f32, d_act_q8_1, max_n * k)?;
        stream.synchronize()?;

        let dst_bytes = max_n * n_rows * 4;
        let d_dst = alloc_zeroed(&dev, dst_bytes);

        // N=1 single-row baseline (single qmatmul call, m=1).
        let single_us = time_us(stream, iters, || {
            qmatmul(
                &reg, stream, d_w, d_act_q8_1, DevicePtr(0), d_dst,
                1, k, n_rows, QDtype::Q4_1,
            )
        })?;

        // Batched at each N. N=1 also exercises the batched path's
        // trivial case (won't actually go through the short-circuit
        // since the gate is m≥2; keep separate for completeness).
        let mut row_strs = vec![format!("{:>9.1}µs", single_us)];
        for &n in &slot_counts[1..] {
            let us = time_us(stream, iters, || {
                qmatmul(
                    &reg, stream, d_w, d_act_q8_1, DevicePtr(0), d_dst,
                    n, k, n_rows, QDtype::Q4_1,
                )
            })?;
            row_strs.push(format!("{:>9.1}µs ({:.2}×)", us, single_us * n as f64 / us));
        }
        eprintln!(
            "  {n_rows:>5} × {k:<5} {:<24} | {}",
            label,
            row_strs.join(" | ")
        );

        // Cleanup.
        unsafe {
            dev.dealloc(d_w, w_q4_1.len() * std::mem::size_of::<BlockQ4_1>())?;
            dev.dealloc(d_act_f32, max_n * k * 4)?;
            dev.dealloc(d_act_q8_1, act_q8_1_bytes)?;
            dev.dealloc(d_dst, dst_bytes)?;
        }
    }

    eprintln!();
    eprintln!("# Speedup factor printed = (N · single-row time) / batched time.");
    eprintln!("# > 1.0× means batched amortizes; ≈ 1.0× means no win;");
    eprintln!("# < 1.0× means batched costs more than per-row × N.");

    Ok(())
}
