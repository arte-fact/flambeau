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
use crate::iq_tables::{
    IQ1S_GRID, IQ1_DELTA, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

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
pub fn dequantize_into(dtype: GgmlDType, raw: &[u8], out: &mut [f32]) -> Result<(), QuantError> {
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
        GgmlDType::Iq2Xs => dequant_iq2_xs(raw, out),
        GgmlDType::Iq2S => dequant_iq2_s(raw, out),
        GgmlDType::Iq1S => dequant_iq1_s(raw, out),
        GgmlDType::Iq1M => dequant_iq1_m(raw, out),
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
            dst[j] = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
            dst[j + QK / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

fn dequant_iq4_xs(raw: &[u8], out: &mut [f32]) {
    // Block layout (136 B): f16 d (2) + u16 scales_h (2) + u8 scales_l[4] (4)
    // + u8 qs[128]. 8 sub-blocks of 32 elements; each sub-block has a signed
    // 6-bit scale split low/high across scales_l / scales_h, biased by -32.
    const QK: usize = QK_K; // 256
    const SUB: usize = 32;
    const N_SUBS: usize = QK / SUB; // 8
    const BLOCK: usize = 2 + 2 + (QK / 64) + (QK / 2); // 136
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
                dst[j] = dl * KVALUES_IQ4NL[(qs_sub[j] & 0x0F) as usize] as f32;
                dst[j + 16] = dl * KVALUES_IQ4NL[(qs_sub[j] >> 4) as usize] as f32;
            }
        }
    }
}

fn dequant_iq3_xxs(raw: &[u8], out: &mut [f32]) {
    // Block layout (98 B): f16 d (2) + u8 qs[96]. Within qs, the first
    // QK_K/4 = 64 bytes index the codebook (2 indices per ib32 chunk),
    // and the last 32 bytes hold per-ib32 (scale, sign-bits) as packed
    // uint32_t (4 bits scale + 7 bits sign × 4 = 28 bits).
    const QK: usize = QK_K; // 256
    const N_IB32: usize = QK / 32; // 8 chunks of 32 elements
    const BLOCK: usize = 2 + 3 * QK / 8; // 2 + 96 = 98
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs = &raw[off + 2..off + 2 + QK / 4]; // 64 bytes
        let scs = &raw[off + 2 + QK / 4..off + BLOCK]; // 32 bytes
        let mut q_off = 0usize;
        let mut y_off = b * QK;
        for ib32 in 0..N_IB32 {
            let aux32 = u32::from_le_bytes([
                scs[4 * ib32],
                scs[4 * ib32 + 1],
                scs[4 * ib32 + 2],
                scs[4 * ib32 + 3],
            ]);
            let db = d * (0.5f32 + ((aux32 >> 28) as f32)) * 0.5f32;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[q_off + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[q_off + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let s1 = if signs & (1 << j) != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let s2 = if signs & (1 << (j + 4)) != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db * g1[j] as f32 * s1;
                    out[y_off + j + 4] = db * g2[j] as f32 * s2;
                }
                y_off += 8;
            }
            q_off += 8;
        }
    }
}

fn dequant_iq3_s(raw: &[u8], out: &mut [f32]) {
    // Block layout (110 B): f16 d (2) + u8 qs[64] + u8 qh[8] + u8 signs[32]
    // + u8 scales[4]. 8 sub-blocks of 32 elements; 9-bit codebook index =
    // qs[byte] | ((qh_bit) << 8); per-sub-block scale = 1 + 2 * 4bit-nibble
    // from scales[ib32/2]; per-byte sign in signs[l].
    const QK: usize = QK_K;
    const BLOCK: usize = 2 + QK / 4 + QK / 32 + QK / 8 + QK / 64; // 110
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs_base = off + 2;
        let qh_base = qs_base + QK / 4;
        let sgn_base = qh_base + QK / 32;
        let sc_base = sgn_base + QK / 8;
        let mut y_off = b * QK;
        let mut qs_p = 0usize; // offset into qs[]
        let mut signs_p = 0usize; // offset into signs[]
        let mut ib32 = 0usize;
        while ib32 < QK / 32 {
            let sc_byte = raw[sc_base + ib32 / 2];
            let db1 = d * (1.0f32 + 2.0 * (sc_byte & 0x0F) as f32);
            let db2 = d * (1.0f32 + 2.0 * (sc_byte >> 4) as f32);
            let qh1 = raw[qh_base + ib32];
            let qh2 = raw[qh_base + ib32 + 1];
            for l in 0..4 {
                let g1_idx =
                    (raw[qs_base + qs_p + 2 * l] as u32) | (((qh1 as u32) << (8 - 2 * l)) & 256);
                let g2_idx = (raw[qs_base + qs_p + 2 * l + 1] as u32)
                    | (((qh1 as u32) << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[g1_idx as usize].to_le_bytes();
                let g2 = IQ3S_GRID[g2_idx as usize].to_le_bytes();
                let signs = raw[sgn_base + signs_p + l];
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db1 * g1[j] as f32 * s1;
                    out[y_off + j + 4] = db1 * g2[j] as f32 * s2;
                }
                y_off += 8;
            }
            qs_p += 8;
            signs_p += 4;
            for l in 0..4 {
                let g1_idx =
                    (raw[qs_base + qs_p + 2 * l] as u32) | (((qh2 as u32) << (8 - 2 * l)) & 256);
                let g2_idx = (raw[qs_base + qs_p + 2 * l + 1] as u32)
                    | (((qh2 as u32) << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[g1_idx as usize].to_le_bytes();
                let g2 = IQ3S_GRID[g2_idx as usize].to_le_bytes();
                let signs = raw[sgn_base + signs_p + l];
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db2 * g1[j] as f32 * s1;
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

fn dequant_iq2_xxs(raw: &[u8], out: &mut [f32]) {
    // Block layout (66 B): f16 d (2) + u16 qs[32] (64). 8 sub-blocks of 32
    // elements. Each sub-block reads 8 qs bytes as 2 × u32 (aux32[0..2]):
    //   aux32[0] → 4 codebook indices (1 byte each, low-byte-first)
    //   aux32[1] → 4 × 7-bit sign-LUT indices (bits 0..27) + 4-bit scale (top)
    // Codebook lookup yields 8 i8 quants; sign-LUT (KSIGNS_IQ2XS, 128 entries
    // shared with IQ3_XXS) inflates a 7-bit index to an 8-bit per-byte sign
    // mask.
    const QK: usize = QK_K;
    const N_IB32: usize = QK / 32; // 8
    const BLOCK: usize = 2 + 2 * (QK / 8); // 66
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
                    let s = if signs & (1 << j) != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db * grid[j] as f32 * s;
                }
                y_off += 8;
            }
        }
    }
}

fn dequant_iq2_xs(raw: &[u8], out: &mut [f32]) {
    // Block layout (74 B): f16 d (2) + u16 qs[32] (64) + u8 scales[8].
    // Each qs u16 packs 9-bit codebook index (low 9) + 7-bit sign-LUT
    // index (high 7). Scales array holds 8 × 4-bit-pair sub-block scales,
    // each contributing db = d * (0.5 + nib) * 0.25.
    const QK: usize = QK_K;
    const N_IB32: usize = QK / 32; // 8
    const BLOCK: usize = 2 + 2 * (QK / 8) + QK / 32; // 74
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs_base = off + 2;
        let sc_base = qs_base + 2 * (QK / 8);
        let mut y_off = b * QK;
        let mut ib32 = 0usize;
        while ib32 < N_IB32 {
            let sc_byte = raw[sc_base + ib32 / 2];
            let db_lo = d * (0.5f32 + (sc_byte & 0x0F) as f32) * 0.25f32;
            let db_hi = d * (0.5f32 + (sc_byte >> 4) as f32) * 0.25f32;
            // First sub-block (32 elems): 4 × u16 starting at qs[8*(ib32/2)].
            let qbase = qs_base + 16 * (ib32 / 2);
            for l in 0..4 {
                let q = u16::from_le_bytes([raw[qbase + 2 * l], raw[qbase + 2 * l + 1]]);
                let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((q >> 9) & 127) as usize];
                for j in 0..8 {
                    let s = if signs & (1 << j) != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db_lo * grid[j] as f32 * s;
                }
                y_off += 8;
            }
            // Second sub-block (32 elems): next 4 × u16.
            let qbase2 = qbase + 8;
            for l in 0..4 {
                let q = u16::from_le_bytes([raw[qbase2 + 2 * l], raw[qbase2 + 2 * l + 1]]);
                let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((q >> 9) & 127) as usize];
                for j in 0..8 {
                    let s = if signs & (1 << j) != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db_hi * grid[j] as f32 * s;
                }
                y_off += 8;
            }
            ib32 += 2;
        }
    }
}

fn dequant_iq2_s(raw: &[u8], out: &mut [f32]) {
    // Block layout (82 B): f16 d (2) + u8 qs[64] + u8 qh[8] + u8 scales[8].
    // qs is split: qs[0..32] = 10-bit codebook index low byte, qs[32..64] =
    // sign bytes (one per 8-elem chunk). qh[ib32] holds 2 bits × 4 = 8 bits
    // shared across 4 indices of one half-sub-block (high bits 8-9 of the
    // 10-bit index). scales[ib32/2] = two 4-bit sub-block scales (lo/hi nib).
    const QK: usize = QK_K;
    const N_IB32: usize = QK / 32;
    const BLOCK: usize = 2 + QK / 4 + QK / 32 + QK / 32; // 82
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs_base = off + 2; // [..32] idx low, [32..64] signs
        let qh_base = qs_base + QK / 4;
        let sc_base = qh_base + QK / 32;
        let mut y_off = b * QK;
        let mut qs_p = 0usize; // walks qs[0..32]
        let mut sgn_p = QK / 8; // walks qs[32..64] (initial offset 32)
        let mut ib32 = 0usize;
        while ib32 < N_IB32 {
            let sc_byte = raw[sc_base + ib32 / 2];
            let db_lo = d * (0.5f32 + (sc_byte & 0x0F) as f32) * 0.25f32;
            let db_hi = d * (0.5f32 + (sc_byte >> 4) as f32) * 0.25f32;
            // First half (sub-block ib32, 32 elems): qh[ib32] supplies high bits.
            let qh1 = raw[qh_base + ib32];
            for l in 0..4 {
                let idx =
                    (raw[qs_base + qs_p + l] as usize) | (((qh1 as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx].to_le_bytes();
                let signs = raw[qs_base + sgn_p + l];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db_lo * grid[j] as f32 * s;
                }
                y_off += 8;
            }
            qs_p += 4;
            sgn_p += 4;
            // Second half (sub-block ib32+1): qh[ib32+1].
            let qh2 = raw[qh_base + ib32 + 1];
            for l in 0..4 {
                let idx =
                    (raw[qs_base + qs_p + l] as usize) | (((qh2 as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx].to_le_bytes();
                let signs = raw[qs_base + sgn_p + l];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    out[y_off + j] = db_hi * grid[j] as f32 * s;
                }
                y_off += 8;
            }
            qs_p += 4;
            sgn_p += 4;
            ib32 += 2;
        }
    }
}

fn dequant_iq1_s(raw: &[u8], out: &mut [f32]) {
    // Block layout (50 B): f16 d (2) + u8 qs[32] + u16 qh[8] (16).
    // Per sub-block (32 elems): dl = d * (2*scale_3b + 1) where scale_3b
    // = (qh[ib] >> 12) & 7; delta sign = qh[ib] bit 15. 11-bit codebook
    // index = qs[4*ib+l] | (((qh[ib] >> (3*l)) & 7) << 8).
    const QK: usize = QK_K;
    const N_IB: usize = QK / 32;
    const BLOCK: usize = 2 + QK / 8 + 2 * (QK / 32); // 50
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let d = half::f16::from_le_bytes([raw[off], raw[off + 1]]).to_f32();
        let qs_base = off + 2;
        let qh_base = qs_base + QK / 8;
        let mut y_off = b * QK;
        for ib in 0..N_IB {
            let qh = u16::from_le_bytes([raw[qh_base + 2 * ib], raw[qh_base + 2 * ib + 1]]);
            let dl = d * (2.0f32 * (((qh >> 12) & 7) as f32) + 1.0f32);
            let delta = if qh & 0x8000 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            for l in 0..4 {
                let lo = raw[qs_base + 4 * ib + l] as usize;
                let hi = ((qh as usize) >> (3 * l)) & 7;
                let idx = lo | (hi << 8);
                let grid_u = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    let g = grid_u[j] as i8 as f32;
                    out[y_off + j] = dl * (g + delta);
                }
                y_off += 8;
            }
        }
    }
}

fn dequant_iq1_m(raw: &[u8], out: &mut [f32]) {
    // Block layout (56 B): u8 qs[32] + u8 qh[16] + u8 scales[8]. No
    // per-block d in the source bytes — d is reassembled from 4 nibbles
    // spread across the 4 u16 scale-words, then interpreted as fp16.
    const QK: usize = QK_K;
    const N_IB: usize = QK / 32;
    const BLOCK: usize = QK / 8 + QK / 16 + QK / 32; // 56
    let n_blocks = out.len() / QK;
    for b in 0..n_blocks {
        let off = b * BLOCK;
        let qs_base = off;
        let qh_base = qs_base + QK / 8;
        let sc_base = qh_base + QK / 16;
        // Re-assemble d (f16) from 4 nibbles, one from each u16 scale-word.
        let sc: [u16; 4] = [
            u16::from_le_bytes([raw[sc_base], raw[sc_base + 1]]),
            u16::from_le_bytes([raw[sc_base + 2], raw[sc_base + 3]]),
            u16::from_le_bytes([raw[sc_base + 4], raw[sc_base + 5]]),
            u16::from_le_bytes([raw[sc_base + 6], raw[sc_base + 7]]),
        ];
        let d_bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = half::f16::from_bits(d_bits).to_f32();
        let mut y_off = b * QK;
        for ib in 0..N_IB {
            let sc_ib2 = sc[ib / 2];
            let shift1 = 6 * (ib % 2);
            let shift2 = shift1 + 3;
            let dl1 = d * (2.0f32 * (((sc_ib2 >> shift1) & 7) as f32) + 1.0f32);
            let dl2 = d * (2.0f32 * (((sc_ib2 >> shift2) & 7) as f32) + 1.0f32);
            let qh0 = raw[qh_base + 2 * ib];
            let qh1 = raw[qh_base + 2 * ib + 1];
            let qs0 = raw[qs_base + 4 * ib];
            let qs1 = raw[qs_base + 4 * ib + 1];
            let qs2 = raw[qs_base + 4 * ib + 2];
            let qs3 = raw[qs_base + 4 * ib + 3];
            let idx0 = qs0 as usize | (((qh0 as usize) << 8) & 0x700);
            let idx1 = qs1 as usize | (((qh0 as usize) << 4) & 0x700);
            let idx2 = qs2 as usize | (((qh1 as usize) << 8) & 0x700);
            let idx3 = qs3 as usize | (((qh1 as usize) << 4) & 0x700);
            let delta0 = if qh0 & 0x08 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            let delta1 = if qh0 & 0x80 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            let delta2 = if qh1 & 0x08 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            let delta3 = if qh1 & 0x80 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            // First half uses dl1
            for &(idx, delta) in &[(idx0, delta0), (idx1, delta1)] {
                let g = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    out[y_off + j] = dl1 * ((g[j] as i8 as f32) + delta);
                }
                y_off += 8;
            }
            // Second half uses dl2
            for &(idx, delta) in &[(idx2, delta2), (idx3, delta3)] {
                let g = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    out[y_off + j] = dl2 * ((g[j] as i8 as f32) + delta);
                }
                y_off += 8;
            }
        }
    }
}

// MXFP4 lookup (sign << 3 | exp << 1 | mantissa).
static MXFP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
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
        let scale = if e == 0 {
            0.0f32
        } else {
            f32::from_bits((e as u32) << 23)
        };
        let nibbles = &raw[off + 1..off + 1 + QK / 2];
        let dst = &mut out[b * QK..(b + 1) * QK];
        // ggml Q4_0-family layout: lo nibble at byte j → elem j,
        // hi nibble at byte j → elem j + QK/2. Earlier impl placed them
        // adjacent (elem 2j / 2j+1) which is the post-shuffle layout and
        // does not match what GGUFs store.
        for (j, &byte) in nibbles.iter().enumerate() {
            dst[j] = MXFP4_LUT[(byte & 0x0F) as usize] * scale;
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
                        let y_idx = blk128 * 128 + shift_iter * 32 + scale_idx * 16 + l;
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
