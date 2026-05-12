//! GGUF block-quant dtype discriminant.
//! Wire-format codes come from ggml (`enum ggml_type` in `ggml.h`) — the numeric
//! ids are stable and we must match them byte-for-byte against GGUFs written by
//! llama.cpp / llamacpp-turbo.

use thiserror::Error;

use crate::blocks::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1,
};

/// Block size for K-quants (`QK_K`). Candle, llama.cpp, llamacpp-turbo all use 256.
pub const QK_K: usize = 256;

/// Block size for legacy Q*_0 / Q*_1 / Q8_0 / Q8_1 blocks.
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

/// Packed superblock-scale size used by Q4_K / Q5_K (6+6 bit packed scales/mins).
pub const K_SCALE_SIZE: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    /// Microscaling FP4 (OCP MX). 32 elements per block at E2M1 (1 sign + 2
    /// exp + 1 mantissa) plus a shared E8M0 (1-byte exponent) microscale.
    /// Block layout: `[uint8 e, uint8 qs[16]]` = 17 B/block. Used by Unsloth
    /// Dynamic Quants on sensitivity-tagged layers (e.g. qwen3next shared
    /// experts in Coder-Next-80B-Q4_0). Not natively supported by V1 kernels;
    /// the loader transparently dequant→Q8_0 at load (mirrors 2 BF16→Q8_0).
    Mxfp4,
    /// IQ4_XS — 4-bit non-linear quant. 256-element super-block layout:
    /// `[half d, u16 scales_h, u8 scales_l[4], u8 qs[128]]` = 136 B/block.
    /// 8 sub-blocks of 32 elements, each with a 6-bit signed scale split
    /// across `scales_l` (low 4 bits) + `scales_h` (high 2 bits). Quant
    /// values dequant through the `kvalues_iq4nl` LUT (16 entries spanning
    /// `[-127, 113]`). Used by Unsloth Dynamic Quants on UD-*-XL builds.
    /// Like MXFP4, the loader dequant→Q8_0 at upload time; no native kernel.
    Iq4Xs,
    /// IQ3_XXS — extreme 3-bit non-linear quant. 256-element super-block,
    /// 98 B/block: `[half d, u8 qs[96]]`. The `qs` array encodes 256 quant
    /// values in 96 bytes via the 1024-byte `iq3xxs_grid` codebook plus
    /// per-32-elem scale + sign bits. Used by UD-Q3_K_XL builds (≈ 50%
    /// of layers). At-load dequant→Q8_0; no native kernel.
    Iq3Xxs,
}

#[derive(Debug, Error)]
pub enum DTypeError {
    #[error("unknown ggml dtype id {0}")]
    UnknownWireId(u32),
    #[error("ggml dtype id {0} is not supported in flambeau V1 (IQ/MXFP4 family)")]
    UnsupportedWireId(u32),
}

impl GgmlDType {
    /// Read the wire-format dtype code (`enum ggml_type`).
    pub fn from_wire(u: u32) -> Result<Self, DTypeError> {
        Ok(match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            // IQ3_XXS / IQ4_XS — accepted at parse-time; loader dequant→Q8_0
            // at upload. Both appear in UD-Q3_K_XL builds; no native kernel.
            18 => Self::Iq3Xxs,
            23 => Self::Iq4Xs,
            // MXFP4 (Microscaling FP4) — accepted at parse-time so the loader
            // can transparently dequant→Q8_0 at upload time. Used by Unsloth
            // Dynamic Quants in qwen3next shared experts.
            39 => Self::Mxfp4,
            // Other IQ quants exist in GGUFs in the wild but are not part of
            // V1's dtype set. Distinct error so loaders reject cleanly.
            16..=17 | 19..=22 | 24..=29 | 31..=38 => return Err(DTypeError::UnsupportedWireId(u)),
            _ => return Err(DTypeError::UnknownWireId(u)),
        })
    }

    pub fn to_wire(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::BF16 => 30,
            Self::Iq3Xxs => 18,
            Self::Iq4Xs => 23,
            Self::Mxfp4 => 39,
        }
    }

    /// Human-readable name (matches `ggml_type_name`).
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq4Xs => "IQ4_XS",
            Self::Mxfp4 => "MXFP4",
        }
    }

    /// Elements packed in one block. `1` for F32/F16/BF16.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => QK4_0,
            Self::Q4_1 => QK4_1,
            Self::Q5_0 => QK5_0,
            Self::Q5_1 => QK5_1,
            Self::Q8_0 => QK8_0,
            Self::Q8_1 => QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => QK_K,
            // IQ4_XS uses a 256-element super-block (8 sub-blocks of 32).
            Self::Iq4Xs | Self::Iq3Xxs => QK_K,
            // MXFP4 uses Q4_0's 32-element block.
            Self::Mxfp4 => QK4_0,
        }
    }

    /// Bytes occupied by one block on disk and in memory.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
            // IQ4_XS block: f16 d (2) + u16 scales_h (2) + u8 scales_l[4] (4)
            // + u8 qs[128] = 136 B.
            Self::Iq4Xs => 2 + 2 + (QK_K / 64) + (QK_K / 2),
            // IQ3_XXS block: f16 d (2) + u8 qs[96] = 98 B.
            Self::Iq3Xxs => 2 + 3 * QK_K / 8,
            // MXFP4 block: 1 byte E8M0 microscale + 16 bytes nibbles = 17 B.
            Self::Mxfp4 => 1 + QK4_0 / 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_round_trip() {
        for d in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q8_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
            GgmlDType::Mxfp4,
        ] {
            assert_eq!(GgmlDType::from_wire(d.to_wire()).unwrap(), d, "{d:?}");
        }
    }

    #[test]
    fn mxfp4_block_size() {
        assert_eq!(GgmlDType::Mxfp4.block_size(), 32);
        assert_eq!(GgmlDType::Mxfp4.type_size(), 17);
    }

    #[test]
    fn mxfp4_dequant_round_trip() {
        // match llama.cpp `dequantize_row_mxfp4` final values:
        // layout: lo nibble at byte j → element j; hi → element j + QK/2
        // scale: half-LUT × 2^(e-127) ≡ doubled-LUT × 2^(e-128)
        // One block, e=128 → scale=2^1=2.0. byte0=0x21 (lo=1=0.5, hi=2=1.0),
        // byte1=0x43 (lo=3=1.5, hi=4=2.0). Scaled: out[0]=1.0, out[1]=3.0,
        // out[16]=2.0, out[17]=4.0.
        let mut raw = vec![0u8; 17];
        raw[0] = 128; // scale = 2^(128-127) = 2.0
        raw[1] = 0x21;
        raw[2] = 0x43;
        let mut out = [0.0f32; 32];
        crate::dequantize_into(GgmlDType::Mxfp4, &raw, &mut out).unwrap();
        assert!((out[0]  - 1.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1]  - 3.0).abs() < 1e-6, "out[1]={}", out[1]);
        assert!((out[16] - 2.0).abs() < 1e-6, "out[16]={}", out[16]);
        assert!((out[17] - 4.0).abs() < 1e-6, "out[17]={}", out[17]);
        // Other positions should be zero.
        assert_eq!(out[2], 0.0);
        assert_eq!(out[15], 0.0);
        assert_eq!(out[18], 0.0);
        assert_eq!(out[31], 0.0);
    }

    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!(GgmlDType::Q4_0.type_size(), 18);
        assert_eq!(GgmlDType::Q4_1.type_size(), 20);
        assert_eq!(GgmlDType::Q5_0.type_size(), 22);
        assert_eq!(GgmlDType::Q5_1.type_size(), 24);
        assert_eq!(GgmlDType::Q8_0.type_size(), 34);
        assert_eq!(GgmlDType::Q8_1.type_size(), 36);
        assert_eq!(GgmlDType::Q2K.type_size(), QK_K / 16 + QK_K / 4 + 4);
        assert_eq!(GgmlDType::Q3K.type_size(), QK_K / 8 + QK_K / 4 + 12 + 2);
        assert_eq!(GgmlDType::Q4K.type_size(), 2 + 2 + K_SCALE_SIZE + QK_K / 2);
        assert_eq!(
            GgmlDType::Q5K.type_size(),
            2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2
        );
        assert_eq!(GgmlDType::Q6K.type_size(), QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);
        assert_eq!(GgmlDType::Q8K.type_size(), 4 + QK_K + QK_K / 16 * 2);
    }
}
