//! CPU dequantize reference (`dequantize_row_*`) for every V1 GGUF block dtype.
//! These are ports of candle's `GgmlType::to_float` impls (crates/candle-core/
//! src/quantized/k_quants.rs), which in turn mirror llama.cpp's
//! `ggml-cpu-quants.c::dequantize_row_*`. The arithmetic is written to be
//! bit-identical to llama.cpp on IEEE754-friendly hardware — round-off only
//! differs if reordering / fused FMA is applied, which we explicitly avoid.
//! These functions are the correctness oracle against which every GPU MMVQ /
//! MMQ kernel is certed. Performance is not a concern here.

// Dequantize code routinely casts packed unsigned nibbles / bytes to signed
// quant codes (the GGUF format uses u8 storage for [-N, N-1] signed quants
// that the decoder recovers by a bias subtract). These casts are intentional
// and bit-preserving — clippy::cast_possible_wrap doesn't apply.
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
        GgmlDType::Mxfp4 => dequant_mxfp4(raw, out),
        GgmlDType::Iq4Xs => dequant_iq4_xs(raw, out),
        GgmlDType::Iq3Xxs => dequant_iq3_xxs(raw, out),
        GgmlDType::Iq4Nl => dequant_iq4_nl(raw, out),
        GgmlDType::Iq3S => dequant_iq3_s(raw, out),
        GgmlDType::Iq2Xxs => dequant_iq2_xxs(raw, out),
    }
    Ok(())
}

fn dequant_iq4_nl(raw: &[u8], out: &mut [f32]) {
    // Block layout (18 B): f16 d (2) + u8 qs[16]. 32 elements per block.
    // y_i = d * KVALUES_IQ4NL[nibble_i]; low nibbles at byte j → elem j,
    // high nibbles at byte j → elem j+16 (Q4_0-family layout).
    const QK: usize = QK4_0;
    const BLOCK: usize = 2 + QK / 2;
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs = &raw[off + 2..off + BLOCK];
        let dst = &mut out[b * QK..(b + 1) * QK];
        for j in 0..QK / 2 {
            dst[j]          = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
            dst[j + QK / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize]   as f32;
        }
    }
}

// IQ4_NL nonlinear lookup (16 signed int8 values). Shared by IQ4_XS,
// IQ4_NL, and a couple of other IQ variants in llama.cpp. Values from
// ggml-quants.c::kvalues_iq4nl.
static KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn dequant_iq4_xs(raw: &[u8], out: &mut [f32]) {
    // Block layout (136 B): f16 d (2) + u16 scales_h (2) + u8 scales_l[4] (4)
    // + u8 qs[128]. 8 sub-blocks of 32 elements; each sub-block has a signed
    // 6-bit scale split low/high across scales_l / scales_h, biased by -32.
    const QK: usize = QK_K;             // 256
    const SUB: usize = 32;
    const N_SUBS: usize = QK / SUB;     // 8
    const BLOCK: usize = 2 + 2 + (QK / 64) + (QK / 2);  // 136
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let scales_h = u16::from_le_bytes([raw[off + 2], raw[off + 3]]);
        let scales_l = &raw[off + 4..off + 4 + QK / 64];
        let qs = &raw[off + 4 + QK / 64..off + BLOCK];
        for ib in 0..N_SUBS {
            let l_nib = (scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            let h_bits = ((scales_h >> (2 * ib)) & 0x03) as u8;
            let ls = (l_nib | (h_bits << 4)) as i32 - 32;
            let dl = d * ls as f32;
            let qs_sub = &qs[ib * 16..(ib + 1) * 16];
            let dst = &mut out[b * QK + ib * SUB..b * QK + (ib + 1) * SUB];
            for j in 0..16 {
                dst[j]      = dl * KVALUES_IQ4NL[(qs_sub[j] & 0x0F) as usize] as f32;
                dst[j + 16] = dl * KVALUES_IQ4NL[(qs_sub[j] >> 4) as usize]   as f32;
            }
        }
    }
}

// IQ3_XXS tables: 256-entry codebook (each u32 packs 4 × 8-bit quant values
// 0..127) + 128-entry sign-bit lookup. From llama.cpp's ggml-common.h.
static IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];
static KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

fn dequant_iq3_xxs(raw: &[u8], out: &mut [f32]) {
    // Block layout (98 B): f16 d (2) + u8 qs[96]. Within qs, the first
    // QK_K/4 = 64 bytes index the codebook (2 indices per ib32 chunk),
    // and the last 32 bytes hold per-ib32 (scale, sign-bits) as packed
    // uint32_t (4 bits scale + 7 bits sign × 4 = 28 bits).
    const QK: usize = QK_K;            // 256
    const N_IB32: usize = QK / 32;     // 8 chunks of 32 elements
    const BLOCK: usize = 2 + 3 * QK / 8;  // 2 + 96 = 98
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs = &raw[off + 2..off + 2 + QK / 4];                // 64 bytes
        let scs = &raw[off + 2 + QK / 4..off + BLOCK];           // 32 bytes
        let mut q_off = 0usize;
        let mut y_off = b * QK;
        for ib32 in 0..N_IB32 {
            let aux32 = u32::from_le_bytes([
                scs[4 * ib32], scs[4 * ib32 + 1],
                scs[4 * ib32 + 2], scs[4 * ib32 + 3],
            ]);
            let db = d * (0.5f32 + ((aux32 >> 28) as f32)) * 0.5f32;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[q_off + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[q_off + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let s1 = if signs & (1 << j) != 0 { -1.0f32 } else { 1.0f32 };
                    let s2 = if signs & (1 << (j + 4)) != 0 { -1.0f32 } else { 1.0f32 };
                    out[y_off + j]     = db * g1[j] as f32 * s1;
                    out[y_off + j + 4] = db * g2[j] as f32 * s2;
                }
                y_off += 8;
            }
            q_off += 8;
        }
    }
}

// IQ3_S codebook: 512-entry × u32 (each u32 packs 4 unsigned i8 quants 0..15)
// + the 1-bit kmask_iq2xs (1, 2, 4, 8, 16, 32, 64, 128) for sign extraction.
static IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

fn dequant_iq3_s(raw: &[u8], out: &mut [f32]) {
    // Block layout (110 B): f16 d (2) + u8 qs[64] + u8 qh[8] + u8 signs[32]
    // + u8 scales[4]. 8 sub-blocks of 32 elements; 9-bit codebook index =
    // qs[byte] | ((qh_bit) << 8); per-sub-block scale = 1 + 2 * 4bit-nibble
    // from scales[ib32/2]; per-byte sign in signs[l].
    const QK: usize = QK_K;
    const BLOCK: usize = 2 + QK / 4 + QK / 32 + QK / 8 + QK / 64;  // 110
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs_base = off + 2;
        let qh_base = qs_base + QK / 4;
        let sgn_base = qh_base + QK / 32;
        let sc_base = sgn_base + QK / 8;
        let mut y_off = b * QK;
        let mut qs_p = 0usize;     // offset into qs[]
        let mut signs_p = 0usize;  // offset into signs[]
        let mut ib32 = 0usize;
        while ib32 < QK / 32 {
            let sc_byte = raw[sc_base + ib32 / 2];
            let db1 = d * (1.0f32 + 2.0 * (sc_byte & 0x0F) as f32);
            let db2 = d * (1.0f32 + 2.0 * (sc_byte >> 4) as f32);
            let qh1 = raw[qh_base + ib32];
            let qh2 = raw[qh_base + ib32 + 1];
            for l in 0..4 {
                let g1_idx = (raw[qs_base + qs_p + 2 * l] as u32)
                    | (((qh1 as u32) << (8 - 2 * l)) & 256);
                let g2_idx = (raw[qs_base + qs_p + 2 * l + 1] as u32)
                    | (((qh1 as u32) << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[g1_idx as usize].to_le_bytes();
                let g2 = IQ3S_GRID[g2_idx as usize].to_le_bytes();
                let signs = raw[sgn_base + signs_p + l];
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                    out[y_off + j]     = db1 * g1[j] as f32 * s1;
                    out[y_off + j + 4] = db1 * g2[j] as f32 * s2;
                }
                y_off += 8;
            }
            qs_p += 8;
            signs_p += 4;
            for l in 0..4 {
                let g1_idx = (raw[qs_base + qs_p + 2 * l] as u32)
                    | (((qh2 as u32) << (8 - 2 * l)) & 256);
                let g2_idx = (raw[qs_base + qs_p + 2 * l + 1] as u32)
                    | (((qh2 as u32) << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[g1_idx as usize].to_le_bytes();
                let g2 = IQ3S_GRID[g2_idx as usize].to_le_bytes();
                let signs = raw[sgn_base + signs_p + l];
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                    out[y_off + j]     = db2 * g1[j] as f32 * s1;
                    out[y_off + j + 4] = db2 * g2[j] as f32 * s2;
                }
                y_off += 8;
            }
            qs_p += 8;
            signs_p += 4;
            ib32 += 2;
        }
    }
}

// IQ2_XXS codebook: 256-entry × u64 (each u64 packs 8 unsigned i8 quants).
static IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

fn dequant_iq2_xxs(raw: &[u8], out: &mut [f32]) {
    // Block layout (66 B): f16 d (2) + u16 qs[32] (64). 8 sub-blocks of 32
    // elements. Each sub-block reads 8 qs bytes as 2 × u32 (aux32[0..2]):
    //   aux32[0] → 4 codebook indices (1 byte each, low-byte-first)
    //   aux32[1] → 4 × 7-bit sign-LUT indices (bits 0..27) + 4-bit scale (top)
    // Codebook lookup yields 8 i8 quants; sign-LUT (KSIGNS_IQ2XS, 128 entries
    // shared with IQ3_XXS) inflates a 7-bit index to an 8-bit per-byte sign
    // mask.
    const QK: usize = QK_K;
    const N_IB32: usize = QK / 32;  // 8
    const BLOCK: usize = 2 + 2 * (QK / 8);  // 66
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs = &raw[off + 2..off + BLOCK];
        let mut y_off = b * QK;
        for ib32 in 0..N_IB32 {
            let q0 = ib32 * 8;
            let aux0 = u32::from_le_bytes([qs[q0], qs[q0 + 1], qs[q0 + 2], qs[q0 + 3]]);
            let aux1 = u32::from_le_bytes([qs[q0 + 4], qs[q0 + 5], qs[q0 + 6], qs[q0 + 7]]);
            let db = d * (0.5f32 + ((aux1 >> 28) as f32)) * 0.25f32;
            let aux8 = aux0.to_le_bytes();
            for l in 0..4 {
                let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let s = if signs & (1 << j) != 0 { -1.0f32 } else { 1.0f32 };
                    out[y_off + j] = db * grid[j] as f32 * s;
                }
                y_off += 8;
            }
        }
    }
}

// MXFP4 lookup (sign << 3 | exp << 1 | mantissa).
static MXFP4_LUT: [f32; 16] = [
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn dequant_mxfp4(raw: &[u8], out: &mut [f32]) {
    const QK: usize = 32;
    const BLOCK: usize = 1 + QK / 2; // e + 16 nibbles = 17 B
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let e = raw[off];
        // E8M0 → FP32: scale = 2^(e - 127). Pairs with our half-magnitude
        // `MXFP4_LUT` (true E2M1 values: 0/0.5/1/.../6); llama.cpp uses
        // a doubled LUT × half-scale, but the product is identical.
        let scale = if e == 0 { 0.0f32 } else { f32::from_bits((e as u32) << 23) };
        let nibbles = &raw[off + 1..off + 1 + QK / 2];
        let dst = &mut out[b * QK..(b + 1) * QK];
        // ggml Q4_0-family layout: lo nibble at byte j → elem j,
        // hi nibble at byte j → elem j + QK/2. Earlier impl placed them
        // adjacent (elem 2j / 2j+1) which is the post-shuffle layout and
        // does not match what GGUFs store.
        for (j, &byte) in nibbles.iter().enumerate() {
            dst[j]          = MXFP4_LUT[(byte & 0x0F) as usize] * scale;
            dst[j + QK / 2] = MXFP4_LUT[((byte >> 4) & 0x0F) as usize] * scale;
        }
    }
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
        // high nibble 4 → (4 - 8) * 0.5 = -2.0
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
        // lane 0 low: (0 | 0x10) - 16 = 0, lane 0 high: (0 | 0) - 16 = -16.
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
