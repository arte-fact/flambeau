//! Parity test — `mmvq_q4_1_batched` (per-N compile-time, n2/n3/n4)
//! vs. looping the single-row `mmvq_q4_1_q8_1` kernel once per slot.
//! Math is algebraically identical (`sumi · (d_x · d_y) + (m_x · s_y) · 0.25`
//! per Q4_1 block), but the slot-loop changes hipcc's FMA-contraction
//! choices → ~1.5e-6 max abs drift at k=4096. Tolerance enforced at
//! `abs_err < 1e-5`, matching the batched-GDN cert tolerance.
//! Sweep: N ∈ {2, 3, 4} (the supported window); n_rows ∈ {64, 4096};
//! k ∈ {1024, 4096}.

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
use flambeau_quant::{BlockQ4_1, BlockQ8_1};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping mmvq_q4_1_batched_parity");
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

/// Deterministic LCG → small-range f32 fill.
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

/// CPU-side Q4_1 row quantizer. Per-block: scan 32 elements, find
/// (min, max), set d = (max-min)/15, m = min, qs[i] = round((x[i]-m)/d).
/// Pairs nibbles per ggml on-disk layout: byte i = (lo: x[i], hi: x[i+16]).
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

#[derive(Clone, Copy)]
struct Shape {
    n_rows: usize,
    k: usize,
}

fn run_parity(label: &str, shape: Shape, n_slots: usize, seed: u64) -> Result<bool> {
    let Some(dev) = dev_or_skip() else {
        return Ok(true);
    };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    let Shape { n_rows, k } = shape;
    let n_blocks_per_row = k / 32;

    // 1. Generate F32 weights [n_rows, k] and quantize to Q4_1.
    let w_f32 = seeded_f32(seed.wrapping_add(1), n_rows * k, 0.5);
    let mut w_q4_1: Vec<BlockQ4_1> = Vec::with_capacity(n_rows * n_blocks_per_row);
    for r in 0..n_rows {
        let row = &w_f32[r * k..(r + 1) * k];
        w_q4_1.extend_from_slice(&quantize_row_q4_1(row));
    }

    // 2. Generate F32 activations [n_slots, k] and quantize on device.
    let act_f32 = seeded_f32(seed.wrapping_add(2), n_slots * k, 1.0);
    let d_act_f32 = upload(&dev, &act_f32);
    let act_q8_1_bytes = n_slots * n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let d_act_q8_1 = alloc_zeroed(&dev, act_q8_1_bytes);
    quantize_q8_1(&reg, stream, d_act_f32, d_act_q8_1, n_slots * k)?;
    stream.synchronize()?;

    // 3. Upload Q4_1 weights.
    let d_w = upload(&dev, &w_q4_1);

    // 4. Output buffers.
    let dst_bytes = n_slots * n_rows * 4;
    let d_out_baseline = alloc_zeroed(&dev, dst_bytes);
    let d_out_batched = alloc_zeroed(&dev, dst_bytes);

    // 5. Baseline: N independent qmatmul(m=1) calls, one per slot.
    let act_row_bytes = n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let dst_row_bytes = n_rows * 4;
    for s in 0..n_slots {
        let act_row = DevicePtr(d_act_q8_1.as_usize() + s * act_row_bytes);
        let dst_row = DevicePtr(d_out_baseline.as_usize() + s * dst_row_bytes);
        qmatmul(
            &reg,
            stream,
            d_w,
            act_row,
            DevicePtr(0),
            dst_row,
            /* m = */ 1,
            k,
            n_rows,
            QDtype::Q4_1,
        )?;
    }
    stream.synchronize()?;

    // 6. Batched: single qmatmul(m=N) call. For N ∈ {2,3,4} this is
    // auto-dispatched to mmvq_q4_1_batched_n{N}; for N=1 it falls
    // through to the same per-row path as the baseline.
    qmatmul(
        &reg,
        stream,
        d_w,
        d_act_q8_1,
        DevicePtr(0),
        d_out_batched,
        n_slots,
        k,
        n_rows,
        QDtype::Q4_1,
    )?;
    stream.synchronize()?;

    // 7. Download both, compare bit-equal.
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
        "[mmvq_q4_1_batched_parity] {label} N={n_slots} n_rows={n_rows} k={k} \
         n_diff={n_diff}/{} max_abs_err={max_abs:.3e} pass={pass}",
        n_slots * n_rows
    );

    // Cleanup.
    unsafe {
        dev.dealloc(d_act_f32, n_slots * k * 4)?;
        dev.dealloc(d_act_q8_1, act_q8_1_bytes)?;
        dev.dealloc(d_w, w_q4_1.len() * std::mem::size_of::<BlockQ4_1>())?;
        dev.dealloc(d_out_baseline, dst_bytes)?;
        dev.dealloc(d_out_batched, dst_bytes)?;
    }

    Ok(pass)
}

#[test]
fn mmvq_q4_1_batched_parity_sweep() -> Result<()> {
    let cases: &[(&str, Shape, &[usize])] = &[
        (
            "small",
            Shape {
                n_rows: 64,
                k: 1024,
            },
            &[2, 3, 4],
        ),
        (
            "k=4096",
            Shape {
                n_rows: 64,
                k: 4096,
            },
            &[2, 3, 4],
        ),
        (
            "prod",
            Shape {
                n_rows: 4096,
                k: 4096,
            },
            &[2, 3, 4],
        ),
    ];
    let mut all_pass = true;
    for (label, shape, slots) in cases {
        for &n_slots in *slots {
            let pass = run_parity(label, *shape, n_slots, 0xDEADBEEFCAFE_u64)?;
            if !pass {
                all_pass = false;
            }
        }
    }
    assert!(
        all_pass,
        "one or more parity cases exceeded the 1e-5 abs-err tolerance"
    );
    Ok(())
}
