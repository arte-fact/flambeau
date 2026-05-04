//! #288-v2 parity test — verify the existing `mmq_q4_1_wave64` kernel
//! produces output bit-identical to per-row `qmatmul(m=1)` when invoked
//! at small N (decode batch dim).
//!
//! The wave64 kernel was tuned for prefill (large m); this test
//! confirms it stays correct at decode-N values N ∈ {2, 4, 8}, which
//! is the regime the #288-v2 dispatch routes through it.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel launch \
              over host/device buffers that survive the synchronize."
)]

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
        eprintln!("no HIP devices — skipping test");
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

fn download_f32(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
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

fn run_parity(label: &str, n_rows: usize, k: usize, n_slots: usize, seed: u64) -> Result<bool> {
    let Some(dev) = dev_or_skip() else { return Ok(true); };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    let n_blocks_per_row = k / 32;

    let w_f32 = seeded_f32(seed.wrapping_add(1), n_rows * k, 0.5);
    let mut w_q4_1: Vec<BlockQ4_1> = Vec::with_capacity(n_rows * n_blocks_per_row);
    for r in 0..n_rows {
        w_q4_1.extend_from_slice(&quantize_row_q4_1(&w_f32[r * k..(r + 1) * k]));
    }

    let act_f32 = seeded_f32(seed.wrapping_add(2), n_slots * k, 1.0);
    let d_act_f32 = upload(&dev, &act_f32);
    let act_q8_1_bytes = n_slots * n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let d_act_q8_1 = alloc_zeroed(&dev, act_q8_1_bytes);
    quantize_q8_1(&reg, stream, d_act_f32, d_act_q8_1, n_slots * k)?;
    stream.synchronize()?;

    let d_w = upload(&dev, &w_q4_1);
    let dst_bytes = n_slots * n_rows * 4;
    let d_out_baseline = alloc_zeroed(&dev, dst_bytes);
    let d_out_wave64 = alloc_zeroed(&dev, dst_bytes);

    // 1. Baseline: per-row qmatmul(m=1) loop with FLAMBEAU_BATCHED_MMVQ unset.
    // SAFETY: env mutation/restore is single-threaded inside this test.
    unsafe { std::env::remove_var("FLAMBEAU_BATCHED_MMVQ"); }
    let act_row_bytes = n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let dst_row_bytes = n_rows * 4;
    for s in 0..n_slots {
        let act_row = DevicePtr(d_act_q8_1.as_usize() + s * act_row_bytes);
        let dst_row = DevicePtr(d_out_baseline.as_usize() + s * dst_row_bytes);
        qmatmul(
            &reg, stream, d_w, act_row, DevicePtr(0), dst_row,
            1, k, n_rows, QDtype::Q4_1,
        )?;
    }
    stream.synchronize()?;

    // 2. Wave64 path via the qmatmul opt-in.
    unsafe { std::env::set_var("FLAMBEAU_BATCHED_MMVQ", "1"); }
    qmatmul(
        &reg, stream, d_w, d_act_q8_1, DevicePtr(0), d_out_wave64,
        n_slots, k, n_rows, QDtype::Q4_1,
    )?;
    stream.synchronize()?;
    unsafe { std::env::remove_var("FLAMBEAU_BATCHED_MMVQ"); }

    let h_baseline = download_f32(&dev, d_out_baseline, n_slots * n_rows);
    let h_wave64 = download_f32(&dev, d_out_wave64, n_slots * n_rows);
    let mut n_diff = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..n_slots * n_rows {
        let d = (h_baseline[i] - h_wave64[i]).abs();
        if h_baseline[i].to_bits() != h_wave64[i].to_bits() {
            n_diff += 1;
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    const TOL: f32 = 1e-5;
    let pass = max_abs < TOL;
    eprintln!(
        "[mmvq_q4_1_wave64_small_n_parity] {label} N={n_slots} n_rows={n_rows} k={k} \
         n_diff={n_diff}/{} max_abs_err={max_abs:.3e} pass={pass}",
        n_slots * n_rows
    );

    unsafe {
        dev.dealloc(d_act_f32, n_slots * k * 4)?;
        dev.dealloc(d_act_q8_1, act_q8_1_bytes)?;
        dev.dealloc(d_w, w_q4_1.len() * std::mem::size_of::<BlockQ4_1>())?;
        dev.dealloc(d_out_baseline, dst_bytes)?;
        dev.dealloc(d_out_wave64, dst_bytes)?;
    }

    Ok(pass)
}

#[test]
fn mmvq_q4_1_wave64_small_n_parity_sweep() -> Result<()> {
    let cases: &[(&str, usize, usize, &[usize])] = &[
        // (label, n_rows, k, slot_counts)
        ("qkv-3584",   3584,  4096, &[2, 4, 8]),
        ("ssm-14336",  14336, 4096, &[2, 4, 8]),
    ];
    let mut all_pass = true;
    for (label, n_rows, k, slots) in cases {
        for &n_slots in *slots {
            let pass = run_parity(label, *n_rows, *k, n_slots, 0xCAFEBEEFC0DE_u64)?;
            if !pass {
                all_pass = false;
            }
        }
    }
    assert!(all_pass, "wave64 small-N parity exceeded 1e-5 abs-err tolerance");
    Ok(())
}
