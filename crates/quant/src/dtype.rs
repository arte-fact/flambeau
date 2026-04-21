//! GGUF block-quant dtype discriminant.
//!
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
            // IQ quants + MXFP4 exist in GGUFs in the wild (e.g. Unsloth UD, qwen3next shared
            // experts, gpt-oss) but are not part of V1 Qwen3.6 Q4_K_M's dtype set. Callers see
            // a distinct error so loaders can reject cleanly instead of misreading bytes.
            16..=29 | 31..=39 => return Err(DTypeError::UnsupportedWireId(u)),
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
        ] {
            assert_eq!(GgmlDType::from_wire(d.to_wire()).unwrap(), d, "{d:?}");
        }
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
