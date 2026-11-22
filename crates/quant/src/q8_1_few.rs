//! Q8_1 activation in the int8 MMA B-fragment layout of the few-column MMQ
//! (m16n8k32 tiles, [`FEW_NCOLS`] columns per group). Host reference of the
//! `quantize_f16_q8_1_few` kernels.
//!
//! Per unit of 256 values (8 sub-blocks of 32) and group of 8 columns,
//! [`FEW_UNIT_GROUP_BYTES`] bytes; the groups of a unit are adjacent and the
//! units follow in k order. Inside, 144 int4:
//! - `qs` int4\[4\]\[32\]: \[j\]\[lane\] for lane = 4·g + t holds values
//!   \[4t, 4t + 4) and \[16 + 4t, 16 + 4t + 4) of sub-block 2j, then of
//!   sub-block 2j + 1, of column g;
//! - `ds` int4\[4\]\[4\]: \[t\]\[j\] holds the scales of columns 2t and 2t + 1
//!   for sub-block 2j, then 2j + 1 — `d` as f32 ([`FewScale::D4`]) or
//!   `(d, d · Σq)` as two f16 ([`FewScale::Ds4`]).
//!
//! Columns past the batch are zero.

use half::f16;

use crate::dtype::{QK8_1, QK_K};

/// Columns of one group (one m16n8k32 B tile).
pub const FEW_NCOLS: usize = 8;
/// Bytes per unit of 256 values and column group: 144 int4.
pub const FEW_UNIT_GROUP_BYTES: usize = (4 * 32 + 4 * 4) * 16;

/// Scale record of a sub-block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FewScale {
    /// `d` only (f32).
    D4,
    /// `d` and `d · Σq` (f16 pair), for weight types with mins.
    Ds4,
}

/// Bytes of the layout for `total_b` columns of `ncols` values.
pub fn few_bytes(ncols: usize, total_b: usize) -> usize {
    (ncols / QK_K) * total_b.div_ceil(FEW_NCOLS) * FEW_UNIT_GROUP_BYTES
}

/// Byte offset of the int that holds values `[k0, k0 + 4)` of sub-block `s`
/// (`k0 % 4 == 0`), column `g` of a unit-group record.
pub fn few_qs_offset(s: usize, k0: usize, g: usize) -> usize {
    let int4 = (s / 2) * 32 + 4 * g + (k0 % 16) / 4;
    let int = 2 * (s % 2) + k0 / 16;
    int4 * 16 + int * 4
}

/// Byte offset of the scale record of sub-block `s`, column `g` of a
/// unit-group record (4 bytes in either scale format).
pub fn few_ds_offset(s: usize, g: usize) -> usize {
    let int4 = 4 * 32 + 4 * (g / 2) + s / 2;
    let slot = 2 * (s % 2) + g % 2;
    int4 * 16 + slot * 4
}

/// Quantises `x` (`total_b` rows of `ncols` F16 values, `ncols % 256 == 0`)
/// into the few-column layout, with the kernel's arithmetic: per sub-block
/// `d_inv = 127 / amax`, `q = round(x · d_inv)` (half away from zero),
/// `d = 1 / d_inv`, both 0 when `amax == 0`.
pub fn quantize_f16_q8_1_few(x: &[f16], ncols: usize, total_b: usize, scale: FewScale) -> Vec<u8> {
    let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    quantize_f32_q8_1_few(&xf, ncols, total_b, scale)
}

/// [`quantize_f16_q8_1_few`] for an F32 source.
pub fn quantize_f32_q8_1_few(x: &[f32], ncols: usize, total_b: usize, scale: FewScale) -> Vec<u8> {
    assert_eq!(ncols % QK_K, 0, "few layout needs ncols % 256 == 0");
    assert!(x.len() >= ncols * total_b);
    let n_groups = total_b.div_ceil(FEW_NCOLS);
    let mut out = vec![0u8; few_bytes(ncols, total_b)];
    for col in 0..total_b {
        let (gi, g) = (col / FEW_NCOLS, col % FEW_NCOLS);
        for unit in 0..ncols / QK_K {
            let rec = (unit * n_groups + gi) * FEW_UNIT_GROUP_BYTES;
            for s in 0..QK_K / QK8_1 {
                let base = col * ncols + unit * QK_K + s * QK8_1;
                let xs = &x[base..base + QK8_1];
                let amax = xs.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let d_inv = if amax == 0.0 { 0.0 } else { 127.0f32 / amax };
                let d = if amax == 0.0 { 0.0 } else { 1.0f32 / d_inv };
                let qs: Vec<i8> = xs.iter().map(|v| (v * d_inv).round() as i8).collect();
                for k0 in (0..QK8_1).step_by(4) {
                    let at = rec + few_qs_offset(s, k0, g);
                    for (i, q) in qs[k0..k0 + 4].iter().enumerate() {
                        out[at + i] = *q as u8;
                    }
                }
                let at = rec + few_ds_offset(s, g);
                match scale {
                    FewScale::D4 => out[at..at + 4].copy_from_slice(&d.to_le_bytes()),
                    FewScale::Ds4 => {
                        let qsum: i32 = qs.iter().map(|&q| i32::from(q)).sum();
                        let sum = d * qsum as f32;
                        out[at..at + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                        out[at + 2..at + 4].copy_from_slice(&f16::from_f32(sum).to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lane (g, t) of the m16n8k32 B fragment reads, for sub-block 2j, ints
    /// 0 and 1 of int4 [j][4g + t] = values [4t, 4t + 4) and [16 + 4t, ...).
    #[test]
    fn qs_offsets_follow_the_fragment_lanes() {
        assert_eq!(few_qs_offset(0, 0, 0), 0);
        assert_eq!(few_qs_offset(0, 16, 0), 4);
        assert_eq!(few_qs_offset(1, 0, 0), 8);
        assert_eq!(few_qs_offset(1, 16, 0), 12);
        assert_eq!(few_qs_offset(0, 4, 0), 16);
        assert_eq!(few_qs_offset(0, 0, 1), 4 * 16);
        assert_eq!(few_qs_offset(2, 0, 0), 32 * 16);
        assert_eq!(few_qs_offset(7, 28, 7), (3 * 32 + 4 * 7 + 3) * 16 + 3 * 4);
    }

    #[test]
    fn ds_offsets_pair_columns_by_sub_block() {
        assert_eq!(few_ds_offset(0, 0), 128 * 16);
        assert_eq!(few_ds_offset(0, 1), 128 * 16 + 4);
        assert_eq!(few_ds_offset(1, 0), 128 * 16 + 8);
        assert_eq!(few_ds_offset(2, 0), 129 * 16);
        assert_eq!(few_ds_offset(0, 2), 132 * 16);
        assert_eq!(few_ds_offset(7, 7), (128 + 12 + 3) * 16 + 12);
    }

    /// Every byte of a unit-group record is written exactly once by the 8
    /// columns × 8 sub-blocks × 8 value quads plus the 64 scale records.
    #[test]
    fn offsets_tile_the_record() {
        let mut seen = vec![0u8; FEW_UNIT_GROUP_BYTES];
        for g in 0..FEW_NCOLS {
            for s in 0..8 {
                for k0 in (0..32).step_by(4) {
                    let o = few_qs_offset(s, k0, g);
                    seen[o..o + 4].iter_mut().for_each(|b| *b += 1);
                }
                let o = few_ds_offset(s, g);
                seen[o..o + 4].iter_mut().for_each(|b| *b += 1);
            }
        }
        assert!(seen.iter().all(|&b| b == 1));
    }

    #[test]
    fn values_round_trip_through_the_layout() {
        let (ncols, total_b) = (512, 3);
        let x: Vec<f16> = (0..ncols * total_b)
            .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 64.0))
            .collect();
        for scale in [FewScale::D4, FewScale::Ds4] {
            let y = quantize_f16_q8_1_few(&x, ncols, total_b, scale);
            assert_eq!(y.len(), 2 * FEW_UNIT_GROUP_BYTES);
            for col in 0..total_b {
                for i in 0..ncols {
                    let (unit, s, k) = (i / QK_K, (i % QK_K) / QK8_1, i % QK8_1);
                    let rec = unit * FEW_UNIT_GROUP_BYTES;
                    let q = y[rec + few_qs_offset(s, k - k % 4, col) + k % 4] as i8;
                    let at = rec + few_ds_offset(s, col);
                    let d = match scale {
                        FewScale::D4 => f32::from_le_bytes(y[at..at + 4].try_into().unwrap()),
                        FewScale::Ds4 => f16::from_le_bytes([y[at], y[at + 1]]).to_f32(),
                    };
                    let v = x[col * ncols + i].to_f32();
                    assert!((f32::from(q) * d - v).abs() <= d * 0.5 + 1e-3 * v.abs(), "col {col} i {i}");
                }
            }
            // Columns 3..8 of the group are zero, scales included.
            for g in total_b..FEW_NCOLS {
                for s in 0..8 {
                    let o = few_ds_offset(s, g);
                    assert_eq!(&y[o..o + 4], &[0, 0, 0, 0]);
                    assert_eq!(&y[few_qs_offset(s, 0, g)..few_qs_offset(s, 0, g) + 4], &[0, 0, 0, 0]);
                }
            }
        }
    }

    #[test]
    fn ds4_sum_is_d_times_the_quantized_sum() {
        let x: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32 / 256.0)).collect();
        let y = quantize_f16_q8_1_few(&x, 256, 1, FewScale::Ds4);
        let at = few_ds_offset(0, 0);
        let d = f16::from_le_bytes([y[at], y[at + 1]]).to_f32();
        let sum = f16::from_le_bytes([y[at + 2], y[at + 3]]).to_f32();
        let qsum: i32 = (0..32).map(|k| i32::from(y[few_qs_offset(0, k - k % 4, 0) + k % 4] as i8)).sum();
        assert!((sum - d * qsum as f32).abs() <= 1e-3 * sum.abs());
    }
}
