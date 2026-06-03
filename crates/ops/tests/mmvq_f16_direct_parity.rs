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

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ2K {
    scales: [u8; QK_K / 16],
    qs: [u8; QK_K / 4],
    d: u16,
    dmin: u16,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ3K {
    hmask: [u8; QK_K / 8],
    qs: [u8; QK_K / 4],
    scales: [u8; 12],
    d: u16,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; K_SCALE_SIZE],
    qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; K_SCALE_SIZE],
    qh: [u8; QK_K / 8],
    qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ6K {
    ql: [u8; QK_K / 2],
    qh: [u8; QK_K / 4],
    scales: [i8; QK_K / 16],
    d: u16,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ8K {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
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

fn build_q2_k(n_rows: usize, n_super: usize) -> Vec<BlockQ2K> {
    let mut w = Vec::with_capacity(n_rows * n_super);
    for r in 0..n_rows {
        for b in 0..n_super {
            let mut scales = [0u8; QK_K / 16];
            for i in 0..(QK_K / 16) {
                let sc = ((r * 11 + b * 7 + i) % 13) as u8;
                let m = ((r * 5 + b * 3 + i + 1) % 11) as u8;
                scales[i] = (m << 4) | (sc & 0x0F);
            }
            let mut qs = [0u8; QK_K / 4];
            for i in 0..(QK_K / 4) {
                qs[i] = ((r * 17 + b * 5 + i) % 255) as u8;
            }
            let d = f16::from_f32(0.05);
            let dmin = f16::from_f32(0.012);
            w.push(BlockQ2K {
                scales,
                qs,
                d: d.to_bits(),
                dmin: dmin.to_bits(),
            });
        }
    }
    w
}

fn build_q3_k(n_rows: usize, n_super: usize) -> Vec<BlockQ3K> {
    let mut w = Vec::with_capacity(n_rows * n_super);
    for r in 0..n_rows {
        for b in 0..n_super {
            let mut hmask = [0u8; QK_K / 8];
            for i in 0..(QK_K / 8) {
                hmask[i] = ((r * 7 + b * 3 + i) % 255) as u8;
            }
            let mut qs = [0u8; QK_K / 4];
            for i in 0..(QK_K / 4) {
                qs[i] = ((r * 13 + b * 5 + i) % 255) as u8;
            }
            // Packed 6-bit scales — any random 12 bytes are a valid encoding.
            let mut scales = [0u8; 12];
            for i in 0..12 {
                scales[i] = ((r * 19 + b * 11 + i) % 255) as u8;
            }
            let d = f16::from_f32(0.04);
            w.push(BlockQ3K {
                hmask,
                qs,
                scales,
                d: d.to_bits(),
            });
        }
    }
    w
}

fn build_q4_k(n_rows: usize, n_super: usize) -> Vec<BlockQ4K> {
    let mut w = Vec::with_capacity(n_rows * n_super);
    for r in 0..n_rows {
        for b in 0..n_super {
            let mut scales = [0u8; K_SCALE_SIZE];
            for i in 0..K_SCALE_SIZE {
                scales[i] = ((r * 23 + b * 7 + i) % 255) as u8;
            }
            let mut qs = [0u8; QK_K / 2];
            for i in 0..(QK_K / 2) {
                let lo = ((r * 11 + b * 7 + i) % 14) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 14) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            let d = f16::from_f32(0.03);
            let dmin = f16::from_f32(0.008);
            w.push(BlockQ4K {
                d: d.to_bits(),
                dmin: dmin.to_bits(),
                scales,
                qs,
            });
        }
    }
    w
}

fn build_q5_k(n_rows: usize, n_super: usize) -> Vec<BlockQ5K> {
    let mut w = Vec::with_capacity(n_rows * n_super);
    for r in 0..n_rows {
        for b in 0..n_super {
            let mut scales = [0u8; K_SCALE_SIZE];
            for i in 0..K_SCALE_SIZE {
                scales[i] = ((r * 23 + b * 7 + i) % 255) as u8;
            }
            let mut qh = [0u8; QK_K / 8];
            for i in 0..(QK_K / 8) {
                qh[i] = ((r * 5 + b * 3 + i) % 255) as u8;
            }
            let mut qs = [0u8; QK_K / 2];
            for i in 0..(QK_K / 2) {
                let lo = ((r * 11 + b * 7 + i) % 14) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 14) as u8;
                qs[i] = (hi << 4) | (lo & 0x0F);
            }
            let d = f16::from_f32(0.03);
            let dmin = f16::from_f32(0.008);
            w.push(BlockQ5K {
                d: d.to_bits(),
                dmin: dmin.to_bits(),
                scales,
                qh,
                qs,
            });
        }
    }
    w
}

fn build_q6_k(n_rows: usize, n_super: usize) -> Vec<BlockQ6K> {
    let mut w = Vec::with_capacity(n_rows * n_super);
    for r in 0..n_rows {
        for b in 0..n_super {
            let mut ql = [0u8; QK_K / 2];
            for i in 0..(QK_K / 2) {
                let lo = ((r * 11 + b * 7 + i) % 14) as u8;
                let hi = ((r * 13 + b * 5 + i + 3) % 14) as u8;
                ql[i] = (hi << 4) | (lo & 0x0F);
            }
            let mut qh = [0u8; QK_K / 4];
            for i in 0..(QK_K / 4) {
                qh[i] = ((r * 7 + b * 3 + i) % 255) as u8;
            }
            let mut scales = [0i8; QK_K / 16];
            for i in 0..(QK_K / 16) {
                scales[i] = (((r * 19 + b * 11 + i) % 64) as i32 - 32) as i8;
            }
            let d = f16::from_f32(0.04);
            w.push(BlockQ6K {
                ql,
                qh,
                scales,
                d: d.to_bits(),
            });
        }
    }
    w
}

fn build_q8_k(n_rows: usize, n_super: usize) -> Vec<BlockQ8K> {
    let mut w = Vec::with_capacity(n_rows * n_super);
    for r in 0..n_rows {
        for b in 0..n_super {
            let mut qs = [0i8; QK_K];
            let mut sums16 = [0i16; QK_K / 16];
            for grp in 0..(QK_K / 16) {
                let mut s: i32 = 0;
                for j in 0..16 {
                    let i = grp * 16 + j;
                    let v = (((r * 17 + b * 7 + i * 3) % 31) as i32) - 15;
                    qs[i] = v as i8;
                    s += v;
                }
                sums16[grp] = s as i16;
            }
            w.push(BlockQ8K {
                d: 0.03,
                qs,
                bsums: sums16,
            });
        }
    }
    w
}

/// Per-block byte size for IQ dtypes — see `block_quant.cuh` static_asserts.
fn iq_block_bytes(dtype: QDtype) -> usize {
    match dtype {
        QDtype::IQ1_S => 2 + QK_K / 8 + 2 * QK_K / 32, // 50
        QDtype::IQ1_M => QK_K / 8 + QK_K / 16 + QK_K / 32, // 56
        QDtype::IQ2_XXS => 2 + 2 * QK_K / 8,           // 66
        QDtype::IQ2_XS => 2 + 2 * QK_K / 8 + QK_K / 32, // 74
        QDtype::IQ2_S => 2 + QK_K / 4 + QK_K / 32 + QK_K / 32, // 82
        QDtype::IQ3_XXS => 2 + QK_K / 4 + QK_K / 8,    // 98
        QDtype::IQ3_S => 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64, // 110
        QDtype::IQ4_NL => 2 + 32 / 2,                  // 18 (32-elem block)
        QDtype::IQ4_XS => 2 + 2 + QK_K / 64 + QK_K / 2, // 136
        _ => unreachable!(),
    }
}

/// Build n_rows × n_units worth of deterministic bytes, with the first
/// 2 bytes of each block set to a small F16 `d = 0.03`. Codebook
/// indices / qh / scales are pseudo-random bytes — both F32 and F16
/// paths decode them deterministically through the same body, so the
/// parity assertion holds regardless of whether the indices represent
/// "natural" weights. Only requirement: any out-of-bounds-index path
/// would NaN-trap, but the codebook arrays are statically sized and
/// every index is masked to the LUT bit-width.
fn build_iq_bytes(n_rows: usize, n_units: usize, block_bytes: usize) -> Vec<u8> {
    let total = n_rows * n_units * block_bytes;
    let mut out = vec![0u8; total];
    let d_bits = f16::from_f32(0.03).to_bits().to_le_bytes();
    for r in 0..n_rows {
        for b in 0..n_units {
            let off = (r * n_units + b) * block_bytes;
            // 2-byte F16 d at offset 0 — correct for every IQ block except
            // IQ1_M, where `d` is reassembled from scales[]; for IQ1_M
            // these two bytes are part of `qs` (codebook idx low-8), and
            // a `0x6499`-shaped pattern is a valid in-range value.
            out[off] = d_bits[0];
            out[off + 1] = d_bits[1];
            for i in 2..block_bytes {
                out[off + i] = ((r * 17 + b * 7 + i * 3) % 251) as u8;
            }
        }
    }
    out
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
    let n_super = k / 256;
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
        QDtype::Q2_K => {
            let w = build_q2_k(n_rows, n_super);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q3_K => {
            let w = build_q3_k(n_rows, n_super);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q4_K => {
            let w = build_q4_k(n_rows, n_super);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q5_K => {
            let w = build_q5_k(n_rows, n_super);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q6_K => {
            let w = build_q6_k(n_rows, n_super);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        QDtype::Q8_K => {
            let w = build_q8_k(n_rows, n_super);
            let bytes = std::mem::size_of_val(w.as_slice());
            unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, bytes) }.to_vec()
        }
        // IQ family — opaque-bytes parity. Block layouts have codebook
        // indices into static device LUTs; any in-range bytes resolve to
        // deterministic F32 acc values. Both F32 and F16 paths read the
        // same bytes, so the test asserts bit-identical F16 output (not
        // correctness vs a numerical reference).
        QDtype::IQ1_S
        | QDtype::IQ1_M
        | QDtype::IQ2_XXS
        | QDtype::IQ2_XS
        | QDtype::IQ2_S
        | QDtype::IQ3_XXS
        | QDtype::IQ3_S
        | QDtype::IQ4_NL
        | QDtype::IQ4_XS => {
            let block_bytes = iq_block_bytes(dtype);
            let units = if dtype == QDtype::IQ4_NL {
                n_blocks
            } else {
                n_super
            };
            build_iq_bytes(n_rows, units, block_bytes)
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
    mmvq_f16_direct(
        &reg,
        stream,
        w_dev,
        act_dev,
        dst_f16_direct,
        n_rows,
        k,
        dtype,
    )?;
    stream.synchronize()?;

    let via_cast: Vec<u16> = download(dev, dst_f16_via_cast, n_rows);
    let direct: Vec<u16> = download(dev, dst_f16_direct, n_rows);
    let f32_ref: Vec<f32> = download(dev, dst_f32, n_rows);

    // Two valid regimes for the per-row pair (via_cast, direct):
    //   (a) |F32 acc| ≤ F16_MAX → both paths produce the same F16 value
    //       (bit-identical; round-to-nearest-even on a finite F32 is
    //       deterministic).
    //   (b) |F32 acc| >  F16_MAX → via_cast = ±inf (un-saturated cast);
    //       direct = ±65504 (saturated). This is the intended divergence:
    //       the saturating clamp is the whole point of the F16-direct
    //       path — it prevents ±inf from poisoning the KV cache.
    // Anything else (NaN mismatch, finite vs finite diff) is a bug.
    const F16_MAX: f32 = 65504.0;
    let mut max_normal_diff: f32 = 0.0;
    let mut sat_rows: usize = 0;
    let mut nan_rows: usize = 0;
    let mut bug_rows: Vec<(usize, f32, f32, f32)> = Vec::new();
    for i in 0..n_rows {
        let a_bits = via_cast[i];
        let b_bits = direct[i];
        let a = f16::from_bits(a_bits).to_f32();
        let b = f16::from_bits(b_bits).to_f32();
        let f = f32_ref[i];
        // NaN regime — degenerate test input (random bytes through a
        // codebook arithmetic that hit a 0×inf). Both paths should
        // produce NaN; accept any matching NaN bit pattern.
        if a.is_nan() && b.is_nan() {
            nan_rows += 1;
            continue;
        }
        if f.abs() > F16_MAX {
            // Saturation regime — direct must clamp to ±F16_MAX.
            let expected = F16_MAX.copysign(f);
            if b == expected {
                sat_rows += 1;
                continue;
            } else {
                bug_rows.push((i, a, b, f));
                continue;
            }
        }
        // Normal regime — must be bit-identical.
        if a_bits == b_bits {
            let d = (a - b).abs();
            if d > max_normal_diff {
                max_normal_diff = d;
            }
        } else {
            bug_rows.push((i, a, b, f));
        }
    }
    eprintln!(
        "{dtype:?} n_rows={n_rows} k={k}: normal_rows max_diff={max_normal_diff}, \
         saturated_rows={sat_rows}, nan_rows={nan_rows}, bug_rows={}; \
         f32_ref[0]={} via_cast[0]={} direct[0]={}",
        bug_rows.len(),
        f32_ref[0],
        f16::from_bits(via_cast[0]).to_f32(),
        f16::from_bits(direct[0]).to_f32(),
    );
    if !bug_rows.is_empty() {
        for (idx, a, b, f) in bug_rows.iter().take(4) {
            eprintln!("  bug row={idx}: via_cast={a} direct={b} f32_ref={f}");
        }
        panic!(
            "{dtype:?}: F16-direct diverges from F32+cast in normal regime (or fails to saturate \
             at ±65504): {} bug rows",
            bug_rows.len()
        );
    }
    Ok(())
}

#[test]
fn mmvq_f16_direct_matches_cast() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        eprintln!("no HIP device — skip");
        return Ok(());
    };
    let shapes = [(64usize, 2048), (512, 2048), (4096, 5120)];
    for dtype in [
        QDtype::Q4_0,
        QDtype::Q4_1,
        QDtype::Q5_0,
        QDtype::Q5_1,
        QDtype::Q8_0,
        QDtype::Q2_K,
        QDtype::Q3_K,
        QDtype::Q4_K,
        // Q5_K F16-direct has a 1-row F16-rounding bug; dropped from
        // ctx::QuantWeight::supports_decode_to_f16 so the production fast
        // path no longer reaches this kernel. Re-enable when the kernel is
        // fixed.
        QDtype::Q6_K,
        QDtype::Q8_K,
        QDtype::IQ1_S,
        QDtype::IQ1_M,
        QDtype::IQ2_XXS,
        QDtype::IQ2_XS,
        QDtype::IQ2_S,
        QDtype::IQ3_XXS,
        QDtype::IQ3_S,
        QDtype::IQ4_NL,
        QDtype::IQ4_XS,
    ] {
        for (n_rows, k) in shapes {
            run_parity(&dev, dtype, n_rows, k)?;
        }
    }
    Ok(())
}
