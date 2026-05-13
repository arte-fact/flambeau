use std::io;

use thiserror::Error;

use crate::dtype::DTypeError;

#[derive(Debug, Error)]
pub enum QuantError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    DType(#[from] DTypeError),

    #[error("not a GGUF file: magic {magic:#010x}")]
    BadMagic { magic: u32 },

    #[error("unsupported GGUF version {version}")]
    UnsupportedVersion { version: u32 },

    #[error("invalid GGUF metadata value-type tag {tag}")]
    InvalidValueType { tag: u32 },

    #[error("invalid UTF-8 in GGUF string field")]
    InvalidUtf8,

    #[error("tensor {name:?} has no matching entry in the file")]
    UnknownTensor { name: String },

    #[error("tensor {name:?} declared in two different split files")]
    DuplicateTensor { name: String },

    #[error(
        "byte range [{start}..+{len}) on tensor {name:?} exceeds its {total} bytes"
    )]
    RangeOutOfBounds {
        name: String,
        start: u64,
        len: u64,
        total: u64,
    },

    #[error(
        "byte range on tensor {name:?} is unaligned to dtype {dtype} (type_size={type_size}, start={start}, len={len})"
    )]
    RangeUnaligned {
        name: String,
        dtype: &'static str,
        type_size: usize,
        start: u64,
        len: u64,
    },

    #[error(
        "tensor elem_count {elem_count} is not a multiple of block_size {block_size} ({dtype})"
    )]
    ElemCountNotDivisible {
        elem_count: usize,
        block_size: usize,
        dtype: &'static str,
    },

    #[error("dequantise {dtype}: got {got} bytes, expected {expected}")]
    ByteLenMismatch {
        got: usize,
        expected: usize,
        dtype: &'static str,
    },
}

pub type Result<T> = std::result::Result<T, QuantError>;
