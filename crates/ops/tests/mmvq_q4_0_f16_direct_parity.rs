//! Parity test for #120 prototype — `mmvq_q4_0_f16_direct` (F16-store
//! Q4_0 MMVQ) vs the legacy `mmvq + cast_f32_to_f16` two-launch path.
//!
//! Same `(weights, x)` fed through both paths; max abs F16 diff on the
//! output must be 0 (the F16 store is bit-identical to F32 → cast since
//! both apply the same final-store rounding mode — modulo the
//! `f16_direct` path's saturating clamp, which only differs from the
//! cast when |F32 acc| > 65504, a regime our small-magnitude random
//! deterministic Q4_0 weights don't reach).
//!
//! Skips when no HIP device is present.

#![cfg(feature = "hip")]
#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::cast::cast_f32_to_f16;
use flambeau_ops::hip::qmatmul::{mmvq, mmvq_q4_0_f16_direct};
use flambeau_ops::OpsRegistry;
use flambeau_core::QDtype;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
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

fn download<T: Copy + Default>(dev: &HipDevice, d: DevicePtr, n: usize) -> Vec<T> {
    let mut out: Vec<T> = vec![T::default(); n];
    let bytes = std::mem::size_of_val(&out[..]);
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            d,
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BlockQ4_0 {
    d: u16,
    qs: [u8; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BlockQ8_1 {
    d: u16,
    s: u16,
    qs: [i8; 32],
}

fn build_q4_0_weights(n_rows: usize, n_blocks: usize) -> Vec<BlockQ4_0> {
    // Deterministic mid-range nibbles (~[-4..6] post bias subtract).
    let mut weights = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                let lo = ((r * 11 + b * 7 + i) % 14) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 14) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            // d = 0.04 keeps row magnitudes well below F16 max.
            let d = f16::from_f32(0.04);
            weights.push(BlockQ4_0 {
                d: d.to_bits(),
                qs,
            });
        }
    }
    weights
}

fn build_q8_1_act(n_blocks: usize) -> Vec<BlockQ8_1> {
    let mut act = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let mut qs = [0i8; 32];
        let mut sum_i32: i32 = 0;
        for i in 0..32 {
            let v = (((b * 17 + i * 3) % 31) as i32) - 15;
            qs[i] = v as i8;
            sum_i32 += v;
        }
        let d = f16::from_f32(0.03);
        let s = f16::from_f32(d.to_f32() * sum_i32 as f32);
        act.push(BlockQ8_1 {
            d: d.to_bits(),
            s: s.to_bits(),
            qs,
        });
    }
    act
}

fn run_parity(dev: &HipDevice, n_rows: usize, k: usize) -> Result<()> {
    let n_blocks = k / 32;
    let weights = build_q4_0_weights(n_rows, n_blocks);
    let act = build_q8_1_act(n_blocks);

    let reg = OpsRegistry::new(dev)?;
    let stream = dev.default_stream();
    let w_dev = upload(dev, &weights);
    let act_dev = upload(dev, &act);

    let dst_f32 = dev.alloc(n_rows * 4).unwrap();
    let dst_f16_via_cast = dev.alloc(n_rows * 2).unwrap();
    let dst_f16_direct = dev.alloc(n_rows * 2).unwrap();

    mmvq(&reg, stream, w_dev, act_dev, dst_f32, n_rows, k, QDtype::Q4_0)?;
    cast_f32_to_f16(&reg, stream, dst_f32, dst_f16_via_cast, n_rows)?;
    mmvq_q4_0_f16_direct(&reg, stream, w_dev, act_dev, dst_f16_direct, n_rows, k)?;
    stream.synchronize()?;

    let via_cast: Vec<u16> = download(dev, dst_f16_via_cast, n_rows);
    let direct: Vec<u16> = download(dev, dst_f16_direct, n_rows);
    let f32_ref: Vec<f32> = download(dev, dst_f32, n_rows);

    let mut max_abs: f32 = 0.0;
    let mut first_diff: Option<usize> = None;
    for i in 0..n_rows {
        let a = f16::from_bits(via_cast[i]).to_f32();
        let b = f16::from_bits(direct[i]).to_f32();
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > 0.0 && first_diff.is_none() {
            first_diff = Some(i);
        }
    }
    eprintln!(
        "n_rows={n_rows} k={k}: max |F16(via cast) − F16(direct)| = {max_abs}; \
         f32_ref[0]={} via_cast[0]={} direct[0]={}",
        f32_ref[0],
        f16::from_bits(via_cast[0]).to_f32(),
        f16::from_bits(direct[0]).to_f32(),
    );
    if let Some(idx) = first_diff {
        let a = f16::from_bits(via_cast[idx]).to_f32();
        let b = f16::from_bits(direct[idx]).to_f32();
        eprintln!(
            "  first diff row={idx}: via_cast={a} direct={b} f32_ref={}",
            f32_ref[idx]
        );
    }
    assert!(
        max_abs == 0.0,
        "F16-direct diverges from F32+cast (max abs {max_abs})"
    );
    Ok(())
}

#[test]
fn mmvq_q4_0_f16_direct_matches_cast() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        eprintln!("no HIP device — skip");
        return Ok(());
    };
    for (n_rows, k) in [(64usize, 2048), (512, 2048), (4096, 5120)] {
        run_parity(&dev, n_rows, k)?;
    }
    Ok(())
}
