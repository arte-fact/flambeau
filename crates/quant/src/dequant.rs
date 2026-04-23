//! CPU dequantize reference (`dequantize_row_*`) for every V1 GGUF block dtype.
//!
//! These are ports of candle's `GgmlType::to_float` impls (crates/candle-core/
//! src/quantized/k_quants.rs), which in turn mirror llama.cpp's
//! `ggml-cpu-quants.c::dequantize_row_*`. The arithmetic is written to be
//! bit-identical to llama.cpp on IEEE754-friendly hardware — round-off only
//! differs if reordering / fused FMA is applied, which we explicitly avoid.
//!
//! These functions are the correctness oracle against which every GPU MMVQ /
//! MMQ kernel is certed. Performance is not a concern here.

// Dequantize code routinely casts packed unsigned nibbles / bytes to signed
// quant codes (the GGUF format uses u8 storage for [-N, N-1] signed quants
// that the decoder recovers by a bias subtract). These casts are intentional
// and bit-preserving — clippy::cast_possible_wrap doesn't apply.
//
// Test asserts compare against exact-representable f32 constants (powers of
// 2, zeros, small integers) that round-trip through quant → dequant with
// zero error — strict `==` is correct there, not an epsilon comparison.
#![expect(
    clippy::cast_possible_wrap,
    reason = "bit-preserving u8/usize → i8/i32 casts on quant codes"
)]

use byteorder::{ByteOrder, LittleEndian};
use half::{bf16, f16};

use crate::blocks::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1,
};
use crate::dtype::{GgmlDType, QK4_0, QK4_1, QK5_0, QK5_1, QK8_0, QK_K};
use crate::error::QuantError;

/// Dispatch on `dtype` and dequantize `raw` (one tensor's packed bytes) into a
/// freshly-allocated `Vec<f32>` of length `elem_count`.
///
/// # Errors
/// Returns an error if `raw` is the wrong byte length for `elem_count` of
/// `dtype`, or if `elem_count` is not a multiple of the dtype's block size.
pub fn dequantize_to_vec(
    dtype: GgmlDType,
    raw: &[u8],
    elem_count: usize,
) -> Result<Vec<f32>, QuantError> {
    let mut out = vec![0.0f32; elem_count];
    dequantize_into(dtype, raw, &mut out)?;
    Ok(out)
}

/// Dequantize `raw` into a caller-supplied `out` buffer of exactly `elem_count`
/// elements. This is the allocation-free form used by the sweep harness.
///
/// # Errors
/// Returns an error if `raw` is the wrong byte length for `out.len()` of
/// `dtype`, or if `out.len()` is not a multiple of the dtype's block size.
pub fn dequantize_into(
    dtype: GgmlDType,
    raw: &[u8],
    out: &mut [f32],
) -> Result<(), QuantError> {
    let elem_count = out.len();
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();

    if elem_count % block_size != 0 {
        return Err(QuantError::ElemCountNotDivisible {
            elem_count,
            block_size,
            dtype: dtype.name(),
        });
    }
    let n_blocks = elem_count / block_size;
    let expected_bytes = n_blocks * type_size;
    if raw.len() != expected_bytes {
        return Err(QuantError::ByteLenMismatch {
            got: raw.len(),
            expected: expected_bytes,
            dtype: dtype.name(),
        });
    }

    match dtype {
        GgmlDType::F32 => {
            let src: &[f32] = bytemuck::cast_slice(raw);
            out.copy_from_slice(src);
        }
        GgmlDType::F16 => {
            let src: &[f16] = bytemuck::cast_slice(raw);
            for (o, x) in out.iter_mut().zip(src) {
                *o = x.to_f32();
            }
        }
        GgmlDType::BF16 => {
            let src: &[bf16] = bytemuck::cast_slice(raw);
            for (o, x) in out.iter_mut().zip(src) {
                *o = x.to_f32();
            }
        }
        GgmlDType::Q4_0 => dequant_q4_0(bytemuck::cast_slice(raw), out),
        GgmlDType::Q4_1 => dequant_q4_1(bytemuck::cast_slice(raw), out),
        GgmlDType::Q5_0 => dequant_q5_0(bytemuck::cast_slice(raw), out),
        GgmlDType::Q5_1 => dequant_q5_1(bytemuck::cast_slice(raw), out),
        GgmlDType::Q8_0 => dequant_q8_0(bytemuck::cast_slice(raw), out),
        GgmlDType::Q8_1 => dequant_q8_1(bytemuck::cast_slice(raw), out),
        GgmlDType::Q2K => dequant_q2_k(bytemuck::cast_slice(raw), out),
        GgmlDType::Q3K => dequant_q3_k(bytemuck::cast_slice(raw), out),
        GgmlDType::Q4K => dequant_q4_k(bytemuck::cast_slice(raw), out),
        GgmlDType::Q5K => dequant_q5_k(bytemuck::cast_slice(raw), out),
        GgmlDType::Q6K => dequant_q6_k(bytemuck::cast_slice(raw), out),
        GgmlDType::Q8K => dequant_q8_k(bytemuck::cast_slice(raw), out),
    }
    Ok(())
}

// ---- legacy Q*_0 / Q*_1 ----------------------------------------------------

fn dequant_q4_0(xs: &[BlockQ4_0], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        for j in 0..(QK4_0 / 2) {
            let x0 = (x.qs[j] & 0x0F) as i16 - 8;
            let x1 = (x.qs[j] >> 4) as i16 - 8;
            ys[i * QK4_0 + j] = x0 as f32 * d;
            ys[i * QK4_0 + j + QK4_0 / 2] = x1 as f32 * d;
        }
    }
}

fn dequant_q4_1(xs: &[BlockQ4_1], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let m = x.m.to_f32();
        for j in 0..(QK4_1 / 2) {
            let x0 = x.qs[j] & 0x0F;
            let x1 = x.qs[j] >> 4;
            ys[i * QK4_1 + j] = x0 as f32 * d + m;
            ys[i * QK4_1 + j + QK4_1 / 2] = x1 as f32 * d + m;
        }
    }
}

fn dequant_q5_0(xs: &[BlockQ5_0], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let qh = LittleEndian::read_u32(&x.qh);
        for j in 0..(QK5_0 / 2) {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((x.qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((x.qs[j] >> 4) | xh_1) as i32 - 16;
            ys[i * QK5_0 + j] = x0 as f32 * d;
            ys[i * QK5_0 + j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
}

fn dequant_q5_1(xs: &[BlockQ5_1], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let m = x.m.to_f32();
        let qh = LittleEndian::read_u32(&x.qh);
        for j in 0..(QK5_1 / 2) {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = (x.qs[j] & 0x0F) | xh_0;
            let x1 = (x.qs[j] >> 4) | xh_1;
            ys[i * QK5_1 + j] = x0 as f32 * d + m;
            ys[i * QK5_1 + j + QK5_1 / 2] = x1 as f32 * d + m;
        }
    }
}

fn dequant_q8_0(xs: &[BlockQ8_0], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        for j in 0..QK8_0 {
            ys[i * QK8_0 + j] = x.qs[j] as f32 * d;
        }
    }
}

fn dequant_q8_1(xs: &[BlockQ8_1], ys: &mut [f32]) {
    // Q8_1 is normally an activation-side quant (scale + sum), never a weight,
    // but GGUFs may carry it. The sum component is not part of the dequantised
    // value — only `d` scales `qs`.
    const QK: usize = 32;
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        for j in 0..QK {
            ys[i * QK + j] = x.qs[j] as f32 * d;
        }
    }
}

// ---- K-quants --------------------------------------------------------------

/// Reconstruct the 6-bit (scale, min) pair indexed by `j` from the packed
/// 12-byte Q4_K / Q5_K scale array. Matches candle's
/// `utils::get_scale_min_k4`.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q2_k(xs: &[BlockQ2K], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let min = x.dmin.to_f32();
        let y = &mut ys[i * QK_K..(i + 1) * QK_K];
        let mut is = 0usize;
        let mut y_idx = 0usize;
        for qs_chunk in x.qs.chunks_exact(32) {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = x.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &qs_chunk[..16] {
                    y[y_idx] = dl * ((q >> shift) & 3) as f32 - ml;
                    y_idx += 1;
                }
                let sc = x.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &qs_chunk[16..] {
                    y[y_idx] = dl * ((q >> shift) & 3) as f32 - ml;
                    y_idx += 1;
                }
                shift += 2;
            }
        }
    }
}

fn dequant_q3_k(xs: &[BlockQ3K], ys: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    for (i, x) in xs.iter().enumerate() {
        let y = &mut ys[i * QK_K..(i + 1) * QK_K];

        // Unpack the packed 6-bit scales into 16 signed bytes.
        let mut aux = [0u32; 4];
        LittleEndian::read_u32_into(&x.scales[..12], &mut aux[..3]);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scales = [0i8; 16];
        for (byte, out) in aux
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .zip(scales.iter_mut())
        {
            *out = byte as i8;
        }

        let d_all = x.d.to_f32();
        let mut m: u8 = 1;
        let mut is = 0usize;

        // QK_K = 256 = 2 × 128; each 128-block consumes 32 qs bytes (4 values per byte via
        // `>> shift`) but shares the same 32-byte `hmask` with the other 128-block — the
        // walking `m` bit picks which hmask bit contributes to each element.
        for blk128 in 0..(QK_K / 128) {
            let qs_chunk = &x.qs[blk128 * 32..blk128 * 32 + 32];
            let mut shift: u32 = 0;
            for shift_iter in 0..4 {
                for scale_idx in 0..2 {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    for l in 0..16 {
                        let qi = l + 16 * scale_idx;
                        let q = qs_chunk[qi];
                        let sub: i8 = if (x.hmask[qi] & m) == 0 { 4 } else { 0 };
                        let q_val = ((q >> shift) & 3) as i8 - sub;
                        let y_idx =
                            blk128 * 128 + shift_iter * 32 + scale_idx * 16 + l;
                        y[y_idx] = dl * q_val as f32;
                    }
                    is += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
}

fn dequant_q4_k(xs: &[BlockQ4K], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let min = x.dmin.to_f32();
        let y = &mut ys[i * QK_K..(i + 1) * QK_K];
        let q = &x.qs;
        let mut is = 0usize;
        let mut y_idx = 0usize;
        for j in (0..QK_K).step_by(64) {
            let qslice = &q[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, &x.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &x.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for qb in qslice {
                y[y_idx] = d1 * (qb & 0xF) as f32 - m1;
                y_idx += 1;
            }
            for qb in qslice {
                y[y_idx] = d2 * (qb >> 4) as f32 - m2;
                y_idx += 1;
            }
            is += 2;
        }
    }
}

fn dequant_q5_k(xs: &[BlockQ5K], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let min = x.dmin.to_f32();
        let y = &mut ys[i * QK_K..(i + 1) * QK_K];
        let ql = &x.qs;
        let qh = &x.qh;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut y_idx = 0usize;
        for j in (0..QK_K).step_by(64) {
            let ql_slice = &ql[j / 2..j / 2 + 32];
            let (sc, m) = get_scale_min_k4(is, &x.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &x.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for (qlb, qhb) in ql_slice.iter().zip(qh) {
                let to_add = if qhb & u1 != 0 { 16.0 } else { 0.0 };
                y[y_idx] = d1 * ((qlb & 0xF) as f32 + to_add) - m1;
                y_idx += 1;
            }
            for (qlb, qhb) in ql_slice.iter().zip(qh) {
                let to_add = if qhb & u2 != 0 { 16.0 } else { 0.0 };
                y[y_idx] = d2 * ((qlb >> 4) as f32 + to_add) - m2;
                y_idx += 1;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn dequant_q6_k(xs: &[BlockQ6K], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d.to_f32();
        let y = &mut ys[i * QK_K..(i + 1) * QK_K];
        for n in (0..QK_K).step_by(128) {
            let idx = n / 128;
            let sc = &x.scales[8 * idx..];
            let ql = &x.ql[64 * idx..];
            let qh = &x.qh[32 * idx..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                y[n + l] = d * sc[is] as f32 * q1 as f32;
                y[n + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                y[n + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                y[n + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
}

fn dequant_q8_k(xs: &[BlockQ8K], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        let d = x.d;
        for j in 0..QK_K {
            ys[i * QK_K + j] = d * x.qs[j] as f32;
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "test asserts use exact-representable f32 sentinel values"
)]
mod tests {
    use super::*;

    // ---- legacy ----

    #[test]
    fn q4_0_zero_block_dequants_to_zero() {
        let block = BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs: [0x88u8; QK4_0 / 2], // nibbles 8 and 8 → values 0 and 0
        };
        let mut out = vec![42.0f32; QK4_0];
        dequant_q4_0(&[block], &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn q4_0_known_values() {
        // d = 0.5, qs[0] = 0x42 → low nibble 2 → (2 - 8) * 0.5 = -3.0
        //                       high nibble 4 → (4 - 8) * 0.5 = -2.0
        let mut block = BlockQ4_0 {
            d: f16::from_f32(0.5),
            qs: [0x88u8; QK4_0 / 2],
        };
        block.qs[0] = 0x42;
        let mut out = vec![0.0f32; QK4_0];
        dequant_q4_0(&[block], &mut out);
        assert_eq!(out[0], -3.0);
        assert_eq!(out[QK4_0 / 2], -2.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn q8_0_identity_scale() {
        let mut block = BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs: [0; QK8_0],
        };
        for i in 0..QK8_0 {
            block.qs[i] = (i as i8) - 16;
        }
        let mut out = vec![0.0f32; QK8_0];
        dequant_q8_0(&[block], &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i as i8 - 16) as f32);
        }
    }

    #[test]
    fn q5_0_msb_is_restored() {
        // d = 1.0, all low nibbles = 0, qh high-bit set for lane 0 low-half only:
        //   lane 0 low: (0 | 0x10) - 16 = 0, lane 0 high: (0 | 0) - 16 = -16.
        let mut block = BlockQ5_0 {
            d: f16::from_f32(1.0),
            qh: [0; 4],
            qs: [0; QK5_0 / 2],
        };
        // Set bit 0 of qh → high bit of lane 0's low-half = 1, high-half = 0.
        block.qh[0] = 0x01;
        let mut out = vec![0.0f32; QK5_0];
        dequant_q5_0(&[block], &mut out);
        assert_eq!(out[0], 0.0); // (0|16)-16
        assert_eq!(out[QK5_0 / 2], -16.0); // (0|0)-16
        // Lane 1 low-half is unaffected.
        assert_eq!(out[1], -16.0);
    }

    // ---- K-quants ----

    #[test]
    fn q8_k_direct_scale() {
        let mut block = BlockQ8K {
            d: 0.25,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        for i in 0..QK_K {
            block.qs[i] = (i as i32 - 128) as i8;
        }
        let mut out = vec![0.0f32; QK_K];
        dequant_q8_k(&[block], &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.25 * (i as i32 - 128) as f32);
        }
    }

    #[test]
    fn q4_k_zero_block_dequants_to_zero() {
        let block = BlockQ4K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0; 12],
            qs: [0x77; QK_K / 2],
        };
        let mut out = vec![1.0f32; QK_K];
        dequant_q4_k(&[block], &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn q4_k_constant_block_gives_m1() {
        // With qs all zero, d irrelevant, dmin = 0.5, scales packed so that
        // both halves of the first 64 elements use (sc=0, m=1) → value = -0.5.
        // Easiest: pick scales = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        let mut scales = [0u8; 12];
        scales[4] = 1; // m for j=0
        scales[5] = 1; // m for j=1
        let block = BlockQ4K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.5),
            scales,
            qs: [0u8; QK_K / 2],
        };
        let mut out = vec![0.0f32; QK_K];
        dequant_q4_k(&[block], &mut out);
        // First 64 elements: d1/d2 = 0, m1/m2 = 0.5 → value = -0.5.
        for v in &out[..64] {
            assert_eq!(*v, -0.5);
        }
        // Remaining elements: m = 0 → value = 0.
        for v in &out[64..] {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn q6_k_zero_block_dequants_to_zero() {
        let block = BlockQ6K {
            ql: [0x88; QK_K / 2], // low nibbles: (8|0)-32 = -24; but scales=0 → 0
            qh: [0; QK_K / 4],
            scales: [0; QK_K / 16],
            d: f16::from_f32(1.0),
        };
        let mut out = vec![7.0f32; QK_K];
        dequant_q6_k(&[block], &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dispatch_f16_round_trip() {
        use half::f16;
        let vals: Vec<f16> = (0..64).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
        let raw: &[u8] = bytemuck::cast_slice(&vals);
        let out = dequantize_to_vec(GgmlDType::F16, raw, 64).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32 * 0.5);
        }
    }

    #[test]
    fn q2_k_zero_block_dequants_to_zero() {
        let block = BlockQ2K {
            scales: [0; QK_K / 16],
            qs: [0; QK_K / 4],
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
        };
        let mut out = vec![99.0f32; QK_K];
        dequant_q2_k(&[block], &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn q3_k_zero_block_dequants_to_nonzero_offset() {
        // hmask=0 and scales all zero → every element is dl * (q_bits - 4).
        // With d = 1 and scales → (0 - 32) = -32 → dl = -32. q_bits = 0 → y = -32*(-4)=128.
        // Wait — scales[is] is packed; if we zero scales array, the unpacked scales
        // are all 0, so dl = (0 - 32) = -32, and y = dl*(0 - 4) = 128.
        let block = BlockQ3K {
            hmask: [0; QK_K / 8],
            qs: [0; QK_K / 4],
            scales: [0; 12],
            d: f16::from_f32(1.0),
        };
        let mut out = vec![0.0f32; QK_K];
        dequant_q3_k(&[block], &mut out);
        for v in out {
            assert_eq!(v, 128.0);
        }
    }

    #[test]
    fn q5_k_zero_block_dequants_to_zero() {
        let block = BlockQ5K {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0; 12],
            qh: [0; QK_K / 8],
            qs: [0; QK_K / 2],
        };
        let mut out = vec![5.0f32; QK_K];
        dequant_q5_k(&[block], &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn q5_k_msb_adds_16() {
        // d = 1.0, first scale packed so sc=1,m=0; ql=0; qh bit 0 of byte 0 set.
        // Expected: first element = 1.0 * (0 + 16) - 0 = 16.0
        let mut scales = [0u8; 12];
        scales[0] = 1; // sc for j=0 low (6-bit)
        let mut qh = [0u8; QK_K / 8];
        qh[0] = 0x01;
        let block = BlockQ5K {
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
            scales,
            qh,
            qs: [0u8; QK_K / 2],
        };
        let mut out = vec![0.0f32; QK_K];
        dequant_q5_k(&[block], &mut out);
        assert_eq!(out[0], 16.0);
    }

    #[test]
    fn round_trip_raw_bytes_via_dispatch_for_k_quants() {
        // Zeroed buffer of QK_K blocks for each K-dtype → exercise the
        // cast_slice path via the public dispatch entry point.
        let q4k_raw = vec![0u8; std::mem::size_of::<BlockQ4K>()];
        let out = dequantize_to_vec(GgmlDType::Q4K, &q4k_raw, QK_K).unwrap();
        assert!(out.iter().all(|v| *v == 0.0));

        let q6k_raw = vec![0u8; std::mem::size_of::<BlockQ6K>()];
        let out = dequantize_to_vec(GgmlDType::Q6K, &q6k_raw, QK_K).unwrap();
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn byte_len_mismatch_reported() {
        let q4k_raw = vec![0u8; 7];
        let err = dequantize_to_vec(GgmlDType::Q4K, &q4k_raw, QK_K).unwrap_err();
        assert!(format!("{err}").contains("Q4_K"));
    }
}
