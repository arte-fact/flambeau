//! flambeau-quant — GGUF v2 / v3 reader + CPU dequantize reference for every
//! V1 block dtype (F32 / F16 / BF16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 /
//! Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K).
//! The CPU dequantiser is the correctness oracle for every GPU MMVQ / MMQ
//! kernel. The GGUF reader is mmap-backed and exposes per-rank tensor-range
//! readers so TP / EP worker loads don't peak at 2× VRAM (candle X5 pattern).
//! Scope: ships the loader surface. GPU uploads, K-transposed repacking,
//! and quality certs for Q8/turbo-quant KV layouts land in later V1 steps.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod blocks;
pub mod chat_template;
pub mod dequant;
pub mod dtype;
pub mod error;
pub mod gguf;
mod iq_tables;
pub mod quantize_k;
pub mod tokenizer;

pub use blocks::{
    BlockIq3S, BlockIq3Xxs, BlockIq4Nl, BlockIq4Xs, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0,
    BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1,
    BlockQ8_1Mmq, QK8_1_MMQ,
};
pub use dequant::{dequantize_into, dequantize_to_vec};
pub use dtype::{
    GgmlDType, K_SCALE_SIZE, QK4_0, QK4_1, QK5_0, QK5_1, QK8_0, QK8_1, QK_K,
};
pub use error::{QuantError, Result};
pub use gguf::{GgufFile, GgufVersion, TensorInfo, Value, ValueType, DEFAULT_ALIGNMENT};
pub use tokenizer::{load_from_gguf, FimTokens, GgufTokenizer};
pub use chat_template::{ChatMessage, ChatTemplate};
