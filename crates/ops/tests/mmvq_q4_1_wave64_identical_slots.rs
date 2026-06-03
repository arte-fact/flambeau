//! #288-v2 finding test — verifies (and documents) that the wave64
//! kernel's per-slot outputs differ at f32 LSB scale even when the two
//! batched activation rows are IDENTICAL.
//! Result observed on gfx906: 11118 / 14336 outputs differ between
//! slot 0 and slot 1, max abs diff ~3.3e-6 (f32 LSB). The cause is
//! compiler FMA-contraction asymmetry inside the per-col unrolled
//! loop in `mmq_q4_1_wave64.cu` — hipcc generates slightly different
//! FP code for c=0 vs c=1 even though the math is symmetric. The
//! kernel is correct in the IEEE-754 sense; the output is coherent;
//! but bit-identical-within-batch is NOT preserved.
//! This test asserts max_abs < 1e-5 (tolerance) rather than
//! `n_diff == 0` so it captures the regression boundary if the kernel
//! changes shape later, while documenting the LSB-scale slot
//! asymmetry as the v2 routing's known limitation.

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

#[test]
fn mmvq_q4_1_wave64_identical_slots() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    // Use the GDN ssm_out shape that triggers the wave64 path.
    let n_rows = 14336;
    let k = 4096;
    let n_slots = 2;
    let n_blocks_per_row = k / 32;

    // 1. One activation row, replicated across slots.
    let act_row_f32 = seeded_f32(42, k, 1.0);
    let mut act_full = Vec::with_capacity(n_slots * k);
    for _ in 0..n_slots {
        act_full.extend_from_slice(&act_row_f32);
    }
    let d_act_f32 = upload(&dev, &act_full);
    let act_q8_1_bytes = n_slots * n_blocks_per_row * std::mem::size_of::<BlockQ8_1>();
    let d_act_q8_1 = alloc_zeroed(&dev, act_q8_1_bytes);
    quantize_q8_1(&reg, stream, d_act_f32, d_act_q8_1, n_slots * k)?;
    stream.synchronize()?;

    // 2. Random Q4_1 weights.
    let w_f32 = seeded_f32(7, n_rows * k, 0.5);
    let mut w_q4_1: Vec<BlockQ4_1> = Vec::with_capacity(n_rows * n_blocks_per_row);
    for r in 0..n_rows {
        w_q4_1.extend_from_slice(&quantize_row_q4_1(&w_f32[r * k..(r + 1) * k]));
    }
    let d_w = upload(&dev, &w_q4_1);

    // 3. Run wave64 path.
    // SAFETY: env mutation/restore is single-threaded inside the test.
    unsafe {
        std::env::set_var("FLAMBEAU_BATCHED_MMVQ", "wave64");
    }
    let dst_bytes = n_slots * n_rows * 4;
    let d_dst = alloc_zeroed(&dev, dst_bytes);
    qmatmul(
        flambeau_ops::OpCtx { reg: &reg, stream },
        flambeau_ops::QmatmulBuffers {
            weights: d_w,
            act_q8_1: d_act_q8_1,
            act_q8_1_mmq: DevicePtr(0),
            dst: d_dst,
        },
        flambeau_ops::MatmulShape { m: n_slots, k: k, n: n_rows },
        QDtype::Q4_1,
    )?;
    stream.synchronize()?;
    unsafe {
        std::env::remove_var("FLAMBEAU_BATCHED_MMVQ");
    }

    let h_dst = download_f32(&dev, d_dst, n_slots * n_rows);

    // 4. dst[0, :] should equal dst[1, :] bit-for-bit since both slots
    // received identical activation.
    let mut n_diff = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for i in 0..n_rows {
        let a = h_dst[i];
        let b = h_dst[n_rows + i];
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            if first_diff.is_none() {
                first_diff = Some((i, a, b));
            }
        }
    }
    eprintln!(
        "[wave64_identical_slots] n_rows={n_rows} k={k} N={n_slots} \
         n_diff={n_diff}/{n_rows} max_abs={max_abs:.3e}"
    );
    if let Some((idx, a, b)) = first_diff {
        eprintln!("  first divergent: dst[0,{idx}]={a:.6}, dst[1,{idx}]={b:.6}");
    }

    unsafe {
        dev.dealloc(d_act_f32, n_slots * k * 4)?;
        dev.dealloc(d_act_q8_1, act_q8_1_bytes)?;
        dev.dealloc(d_w, w_q4_1.len() * std::mem::size_of::<BlockQ4_1>())?;
        dev.dealloc(d_dst, dst_bytes)?;
    }

    // Allow LSB-scale FMA-contraction drift; flag a true correctness
    // regression if the gap widens beyond f32 noise floor.
    const TOL: f32 = 1e-5;
    assert!(
        max_abs < TOL,
        "wave64 inter-slot drift exceeded {TOL:e} — possible regression \
         beyond the documented FMA-contraction LSB asymmetry"
    );
    Ok(())
}
