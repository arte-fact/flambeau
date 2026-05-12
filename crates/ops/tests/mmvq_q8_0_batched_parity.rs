//! Parity test — `mmvq_q8_0_batched` (per-N compile-time, n2/n3/n4) vs
//! looping the single-row Q8_0 MMVQ once per slot. Q8_0 is symmetric
//! signed 8-bit, so each block contributes `dp4a(xi, yi, 0) · d_x · d_y`
//! with no min/scale correction.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a bounded memcpy or kernel \
              launch over host/device buffers that survive the synchronize."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::hip::norm::quantize_q8_1;
use flambeau_ops::hip::qmatmul::qmatmul;
use flambeau_ops::OpsRegistry;
use flambeau_quant::{BlockQ8_0, BlockQ8_1};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping mmvq_q8_0_batched_parity");
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

fn quantize_row_q8_0(row: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(row.len() % 32, 0);
    let n_blocks = row.len() / 32;
    let mut blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let chunk = &row[b * 32..(b + 1) * 32];
        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
        }
        blocks.push(BlockQ8_0 { d: f16::from_f32(d), qs });
    }
    blocks
}

#[derive(Clone, Copy)]
struct Shape {
    n_rows: usize,
    k: usize,
}

fn run_parity(label: &str, shape: Shape, n_slots: usize, seed: u64) -> Result<bool> {
    let Some(dev) = dev_or_skip() else { return Ok(true); };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    let Shape { n_rows, k } = shape;
    let n_blocks_per_row = k / 32;

    let w_f32 = seeded_f32(seed.wrapping_add(1), n_rows * k, 0.5);
    let mut w_q8_0: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * n_blocks_per_row);
    for r in 0..n_rows {
        let row = &w_f32[r * k..(r + 1) * k];
        w_q8_0.extend_from_slice(&quantize_row_q8_0(row));
    }

    let act_f32 = seeded_f32(seed.wrapping_add(2), n_slots * k, 1.0);
    let d_act_f32 = upload(&dev, &act_f32);
    let act_q8_1_bytes =
        n_slots * n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let d_act_q8_1 = alloc_zeroed(&dev, act_q8_1_bytes);
    quantize_q8_1(&reg, stream, d_act_f32, d_act_q8_1, n_slots * k)?;
    stream.synchronize()?;

    let d_w = upload(&dev, &w_q8_0);

    let dst_bytes = n_slots * n_rows * 4;
    let d_out_baseline = alloc_zeroed(&dev, dst_bytes);
    let d_out_batched = alloc_zeroed(&dev, dst_bytes);

    let act_row_bytes = n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let dst_row_bytes = n_rows * 4;
    for s in 0..n_slots {
        let act_row = DevicePtr(d_act_q8_1.as_usize() + s * act_row_bytes);
        let dst_row = DevicePtr(d_out_baseline.as_usize() + s * dst_row_bytes);
        qmatmul(
            &reg, stream, d_w, act_row, DevicePtr(0), dst_row,
            /* m = */ 1, k, n_rows, QDtype::Q8_0,
        )?;
    }
    stream.synchronize()?;

    qmatmul(
        &reg, stream, d_w, d_act_q8_1, DevicePtr(0), d_out_batched,
        n_slots, k, n_rows, QDtype::Q8_0,
    )?;
    stream.synchronize()?;

    let h_baseline = download_f32(&dev, d_out_baseline, n_slots * n_rows);
    let h_batched = download_f32(&dev, d_out_batched, n_slots * n_rows);
    let mut n_diff = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..n_slots * n_rows {
        let d = (h_baseline[i] - h_batched[i]).abs();
        if h_baseline[i].to_bits() != h_batched[i].to_bits() {
            n_diff += 1;
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    const TOL: f32 = 1e-5;
    let pass = max_abs < TOL;
    eprintln!(
        "[mmvq_q8_0_batched_parity] {label} N={n_slots} n_rows={n_rows} k={k} \
         n_diff={n_diff}/{} max_abs_err={max_abs:.3e} pass={pass}",
        n_slots * n_rows
    );

    unsafe {
        dev.dealloc(d_act_f32, n_slots * k * 4)?;
        dev.dealloc(d_act_q8_1, act_q8_1_bytes)?;
        dev.dealloc(d_w, w_q8_0.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_out_baseline, dst_bytes)?;
        dev.dealloc(d_out_batched, dst_bytes)?;
    }

    Ok(pass)
}

#[test]
fn mmvq_q8_0_batched_parity_sweep() -> Result<()> {
    let cases: &[(&str, Shape, &[usize])] = &[
        ("small",  Shape { n_rows: 64,   k: 1024 }, &[2, 3, 4]),
        ("k=4096", Shape { n_rows: 64,   k: 4096 }, &[2, 3, 4]),
        ("prod",   Shape { n_rows: 4096, k: 4096 }, &[2, 3, 4]),
    ];
    let mut all_pass = true;
    for (label, shape, slots) in cases {
        for &n_slots in *slots {
            let pass = run_parity(label, *shape, n_slots, 0xC0FFEEFEED_u64)?;
            if !pass {
                all_pass = false;
            }
        }
    }
    assert!(all_pass, "one or more parity cases exceeded the 1e-5 abs-err tolerance");
    Ok(())
}
