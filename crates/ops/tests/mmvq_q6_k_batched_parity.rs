//! Parity test — `mmvq_q6_k_batched` vs looping single-row Q6_K MMVQ.

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
use flambeau_quant::{BlockQ6K, BlockQ8_1, QK_K};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping mmvq_q6_k_batched_parity");
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

fn upload_bytes(dev: &HipDevice, data: &[u8]) -> DevicePtr {
    let d = dev.alloc(data.len()).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            data.len(),
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

fn seeded_u8(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 56) as u8
        })
        .collect()
}

fn tame_q6_k_scales(bytes: &mut [u8]) {
    let bs = std::mem::size_of::<BlockQ6K>();
    for chunk in bytes.chunks_exact_mut(bs) {
        for s in 0..16 {
            let raw = chunk[192 + s] as i8;
            chunk[192 + s] = ((raw as i32) >> 2) as u8;
        }
        let d = f16::from_f32((chunk[208] as f32 / 255.0) * 0.02 + 0.002);
        chunk[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
    }
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
    assert_eq!(k % QK_K, 0);
    let n_superblocks = n_rows * k / QK_K;
    let mut w_q6_k_bytes = seeded_u8(seed.wrapping_add(1), n_superblocks * std::mem::size_of::<BlockQ6K>());
    tame_q6_k_scales(&mut w_q6_k_bytes);

    let act_f32 = seeded_f32(seed.wrapping_add(2), n_slots * k, 1.0);
    let d_act_f32 = upload(&dev, &act_f32);
    let act_q8_1_bytes =
        n_slots * (k / 32) * std::mem::size_of::<BlockQ8_1>();
    let d_act_q8_1 = alloc_zeroed(&dev, act_q8_1_bytes);
    quantize_q8_1(&reg, stream, d_act_f32, d_act_q8_1, n_slots * k)?;
    stream.synchronize()?;

    let d_w = upload_bytes(&dev, &w_q6_k_bytes);

    let dst_bytes = n_slots * n_rows * 4;
    let d_out_baseline = alloc_zeroed(&dev, dst_bytes);
    let d_out_batched = alloc_zeroed(&dev, dst_bytes);

    let act_row_bytes = (k / 32) * std::mem::size_of::<BlockQ8_1>();
    let dst_row_bytes = n_rows * 4;
    for s in 0..n_slots {
        let act_row = DevicePtr(d_act_q8_1.as_usize() + s * act_row_bytes);
        let dst_row = DevicePtr(d_out_baseline.as_usize() + s * dst_row_bytes);
        qmatmul(
            &reg, stream, d_w, act_row, DevicePtr(0), dst_row,
            /* m = */ 1, k, n_rows, QDtype::Q6_K,
        )?;
    }
    stream.synchronize()?;

    qmatmul(
        &reg, stream, d_w, d_act_q8_1, DevicePtr(0), d_out_batched,
        n_slots, k, n_rows, QDtype::Q6_K,
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
    const TOL: f32 = 5e-4;
    let pass = max_abs < TOL;
    eprintln!(
        "[mmvq_q6_k_batched_parity] {label} N={n_slots} n_rows={n_rows} k={k} \
         n_diff={n_diff}/{} max_abs_err={max_abs:.3e} pass={pass}",
        n_slots * n_rows
    );

    unsafe {
        dev.dealloc(d_act_f32, n_slots * k * 4)?;
        dev.dealloc(d_act_q8_1, act_q8_1_bytes)?;
        dev.dealloc(d_w, w_q6_k_bytes.len())?;
        dev.dealloc(d_out_baseline, dst_bytes)?;
        dev.dealloc(d_out_batched, dst_bytes)?;
    }

    Ok(pass)
}

#[test]
fn mmvq_q6_k_batched_parity_sweep() -> Result<()> {
    let cases: &[(&str, Shape, &[usize])] = &[
        ("small", Shape { n_rows: 64,   k: 1024 }, &[2, 3, 4]),
        ("prod",  Shape { n_rows: 4096, k: 4096 }, &[2, 3, 4]),
    ];
    let mut all_pass = true;
    for (label, shape, slots) in cases {
        for &n_slots in *slots {
            let pass = run_parity(label, *shape, n_slots, 0x6BEEFCAFE_u64)?;
            if !pass {
                all_pass = false;
            }
        }
    }
    assert!(all_pass, "one or more parity cases exceeded the 5e-4 abs-err tolerance");
    Ok(())
}
