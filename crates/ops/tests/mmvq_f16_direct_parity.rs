//! Parity test for #120 — `mmvq_f16_direct` (F16-store MMVQ) vs the
//! legacy `mmvq + cast_f32_to_f16` two-launch path.
//!
//! Same `(weights, x)` fed through both paths; max abs F16 diff on the
//! output must be 0 (the F16 store is bit-identical to F32 → cast since
//! both apply the same final-store rounding mode — modulo the
//! `f16_direct` path's saturating clamp, which only differs from the
//! cast when |F32 acc| > 65504, a regime our small-magnitude
//! deterministic test weights don't reach).
//!
//! Covers Q4_0, Q4_1, Q8_0. Skips when no HIP device is present.

#![cfg(feature = "hip")]
#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::hip::cast::cast_f32_to_f16;
use flambeau_ops::hip::qmatmul::{mmvq, mmvq_f16_direct};
use flambeau_ops::OpsRegistry;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
}

fn upload_bytes(dev: &HipDevice, host: &[u8]) -> DevicePtr {
    let bytes = host.len();
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(host.as_ptr() as usize),
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
struct BlockQ4_1 {
    d: u16,
    m: u16,
    qs: [u8; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BlockQ5_0 {
    d: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BlockQ5_1 {
    d: u16,
    m: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BlockQ8_0 {
    d: u16,
    qs: [i8; 32],
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BlockQ8_1 {
    d: u16,
    s: u16,
    qs: [i8; 32],
}

fn build_q4_0(n_rows: usize, n_blocks: usize) -> Vec<BlockQ4_0> {
    let mut w = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                let lo = ((r * 11 + b * 7 + i) % 14) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 14) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            let d = f16::from_f32(0.04);
            w.push(BlockQ4_0 { d: d.to_bits(), qs });
        }
    }
    w
}

fn build_q4_1(n_rows: usize, n_blocks: usize) -> Vec<BlockQ4_1> {
    let mut w = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                let lo = ((r * 11 + b * 7 + i) % 15) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 15) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            let d = f16::from_f32(0.04);
            let m = f16::from_f32(-0.6);
            w.push(BlockQ4_1 {
                d: d.to_bits(),
                m: m.to_bits(),
                qs,
            });
        }
    }
    w
}

fn build_q5_0(n_rows: usize, n_blocks: usize) -> Vec<BlockQ5_0> {
    let mut w = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                let lo = ((r * 11 + b * 7 + i) % 14) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 14) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            let mut qh = [0u8; 4];
            for byte_idx in 0..4 {
                let mut v: u8 = 0;
                for bit_idx in 0..8 {
                    let global_bit = byte_idx * 8 + bit_idx;
                    if (r * 17 + b * 5 + global_bit) % 5 == 0 {
                        v |= 1 << bit_idx;
                    }
                }
                qh[byte_idx] = v;
            }
            let d = f16::from_f32(0.04);
            w.push(BlockQ5_0 {
                d: d.to_bits(),
                qh,
                qs,
            });
        }
    }
    w
}

fn build_q5_1(n_rows: usize, n_blocks: usize) -> Vec<BlockQ5_1> {
    let mut w = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                let lo = ((r * 11 + b * 7 + i) % 15) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 15) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            let mut qh = [0u8; 4];
            for byte_idx in 0..4 {
                let mut v: u8 = 0;
                for bit_idx in 0..8 {
                    let global_bit = byte_idx * 8 + bit_idx;
                    if (r * 19 + b * 7 + global_bit) % 5 == 0 {
                        v |= 1 << bit_idx;
                    }
                }
                qh[byte_idx] = v;
            }
            let d = f16::from_f32(0.04);
            let m = f16::from_f32(-0.6);
            w.push(BlockQ5_1 {
                d: d.to_bits(),
                m: m.to_bits(),
                qh,
                qs,
            });
        }
    }
    w
}

fn build_q8_0(n_rows: usize, n_blocks: usize) -> Vec<BlockQ8_0> {
    let mut w = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0i8; 32];
            for i in 0..32 {
                let v = (((r * 17 + b * 7 + i * 3) % 31) as i32) - 15;
                qs[i] = v as i8;
            }
            let d = f16::from_f32(0.03);
            w.push(BlockQ8_0 { d: d.to_bits(), qs });
        }
    }
    w
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

fn run_parity(dev: &HipDevice, dtype: QDtype, n_rows: usize, k: usize) -> Result<()> {
    let n_blocks = k / 32;
    let act = build_q8_1_act(n_blocks);

    let w_bytes: Vec<u8> = match dtype {
        QDtype::Q4_0 => {
            let w = build_q4_0(n_rows, n_blocks);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q4_1 => {
            let w = build_q4_1(n_rows, n_blocks);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q5_0 => {
            let w = build_q5_0(n_rows, n_blocks);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q5_1 => {
            let w = build_q5_1(n_rows, n_blocks);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q8_0 => {
            let w = build_q8_0(n_rows, n_blocks);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        _ => unreachable!(),
    };
    let act_bytes_len = std::mem::size_of_val(act.as_slice());
    let act_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(act.as_ptr() as *const u8, act_bytes_len) };

    let reg = OpsRegistry::new(dev)?;
    let stream = dev.default_stream();
    let w_dev = upload_bytes(dev, &w_bytes);
    let act_dev = upload_bytes(dev, act_bytes);

    let dst_f32 = dev.alloc(n_rows * 4).unwrap();
    let dst_f16_via_cast = dev.alloc(n_rows * 2).unwrap();
    let dst_f16_direct = dev.alloc(n_rows * 2).unwrap();

    mmvq(&reg, stream, w_dev, act_dev, dst_f32, n_rows, k, dtype)?;
    cast_f32_to_f16(&reg, stream, dst_f32, dst_f16_via_cast, n_rows)?;
    mmvq_f16_direct(&reg, stream, w_dev, act_dev, dst_f16_direct, n_rows, k, dtype)?;
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
        "{dtype:?} n_rows={n_rows} k={k}: max |F16(via cast) − F16(direct)| = {max_abs}; \
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
        "{dtype:?}: F16-direct diverges from F32+cast (max abs {max_abs})"
    );
    Ok(())
}

#[test]
fn mmvq_f16_direct_matches_cast() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        eprintln!("no HIP device — skip");
        return Ok(());
    };
    let shapes = [(64usize, 2048), (512, 2048), (4096, 5120)];
    for dtype in [QDtype::Q4_0, QDtype::Q4_1, QDtype::Q5_0, QDtype::Q5_1, QDtype::Q8_0] {
        for (n_rows, k) in shapes {
            run_parity(&dev, dtype, n_rows, k)?;
        }
    }
    Ok(())
}
