//! `#[repr(C)]` GGUF block-quant layouts, byte-for-byte compatible with
//! `ggml-common.h` in llama.cpp. All structs are `bytemuck::Pod` so raw
//! byte slices from mmap'd GGUFs can be reinterpreted with zero copy.

use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::dtype::{K_SCALE_SIZE, QK4_0, QK4_1, QK5_0, QK5_1, QK8_0, QK8_1, QK_K};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub d: f16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub d: f16,
    pub m: f16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub d: f16,
    pub s: f16,
    pub qs: [i8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

/// Turbo / candle 4-warp MMQ activation block — 128 elements (4 sub-blocks of 32).
/// Header = 4 × half2 ds pairs (one `(d, d*sum)` per 32-element sub-block, 16 B).
/// Body = 4 × QK8_1 = 128 signed int8 quants.
/// Total 144 B.
/// Storage order on device: `(k_big_block, col)` row-major (see
/// `mmq_turbo.cu:159-167` for the source layout). The MMQ kernel's
/// inner loop fetches a 144-B block at stride `ncols_y * 144` along K;
/// all threads in a tile's col-group share the block via LDS broadcast.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8_1Mmq {
    pub ds: [f16; 8],        // 4 × (d, d*sum) packed as half2[4]
    pub qs: [i8; 4 * QK8_1], // 128 quants
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1Mmq>() == 16 + 4 * QK8_1);

/// Number of F32 elements covered by one `BlockQ8_1Mmq`.
pub const QK8_1_MMQ: usize = 4 * QK8_1;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
    pub d: f16,
    pub dmin: f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ2K>() == QK_K / 16 + QK_K / 4 + 4);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12],
    pub d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ3K>() == QK_K / 8 + QK_K / 4 + 12 + 2);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ4K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 2 + 2 + K_SCALE_SIZE + QK_K / 2);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}
const _: () =
    assert!(std::mem::size_of::<BlockQ5K>() == 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockQ6K>() == QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 4 + QK_K + QK_K / 16 * 2);

/// IQ4_NL — 4-bit non-linear quant, 32-element block. Per-block f16 scale
/// + 16 bytes of nibble-packed LUT indices. Byte-identical to
///   `ggml-common.h`'s `block_iq4_nl`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIq4Nl>() == 2 + QK4_0 / 2);

/// IQ3_XXS — ~3.06-bpw K-quant with 256-entry u32 codebook. Super-block of
/// 256 elements: f16 scale + 96 bytes split into 64 codebook indices + 32
/// bytes of packed (4-bit scale + 4 × 7-bit sign-LUT idx) per ib32.
/// Byte-identical to `ggml-common.h`'s `block_iq3_xxs`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub d: f16,
    pub qs: [u8; QK_K / 4 + QK_K / 8], // 64 + 32 = 96
}
const _: () = assert!(std::mem::size_of::<BlockIq3Xxs>() == 2 + QK_K / 4 + QK_K / 8);

/// IQ3_S — ~3.44-bpw K-quant with 512-entry u32 codebook. Super-block of
/// 256 elements: f16 scale + 64-byte codebook-low + 8-byte codebook-high-bit
/// + 32-byte per-element sign masks + 4-byte 4-bit scales (8 sub-block
///   scales total). Byte-identical to `ggml-common.h`'s `block_iq3_s`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq3S {
    pub d: f16,
    pub qs: [u8; QK_K / 4],      // 64
    pub qh: [u8; QK_K / 32],     //  8
    pub signs: [u8; QK_K / 8],   // 32
    pub scales: [u8; QK_K / 64], //  4
}
const _: () =
    assert!(std::mem::size_of::<BlockIq3S>() == 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64);

/// IQ2_XXS — ~2.0625-bpw K-quant. Super-block of 256 elements: f16 scale +
/// 64 bytes of `u16 qs[32]` (viewed byte-wise). Byte-identical to
/// `ggml-common.h`'s `block_iq2_xxs`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq2Xxs {
    pub d: f16,
    pub qs: [u8; 2 * QK_K / 8], // 64
}
const _: () = assert!(std::mem::size_of::<BlockIq2Xxs>() == 2 + 2 * QK_K / 8);

/// IQ2_XS — ~2.3125-bpw K-quant. Super-block of 256 elements: f16 scale +
/// 64-byte `u16 qs[32]` (packed 9-bit grid + 7-bit sign idx) + 8-byte
/// scales (4-bit nibble pairs). Byte-identical to `block_iq2_xs`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq2Xs {
    pub d: f16,
    pub qs: [u8; 2 * QK_K / 8],  // 64
    pub scales: [u8; QK_K / 32], //  8
}
const _: () = assert!(std::mem::size_of::<BlockIq2Xs>() == 2 + 2 * QK_K / 8 + QK_K / 32);

/// IQ2_S — 2.5-bpw K-quant. Super-block of 256 elements: f16 scale +
/// 64-byte qs ([0..32] = 10-bit idx low, [32..64] = sign bytes) +
/// 8-byte qh (2 high bits per qs byte) + 8-byte scales (4-bit nibble
/// pairs). Byte-identical to `block_iq2_s`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq2S {
    pub d: f16,
    pub qs: [u8; QK_K / 4],      // 64
    pub qh: [u8; QK_K / 32],     //  8
    pub scales: [u8; QK_K / 32], //  8
}
const _: () = assert!(std::mem::size_of::<BlockIq2S>() == 2 + QK_K / 4 + QK_K / 32 + QK_K / 32);

/// IQ1_S — 1.5625-bpw K-quant. Super-block of 256 elements: f16 scale +
/// 32-byte qs (low 8 of 11-bit codebook idx) + 16-byte qh (u16 × 8 with
/// 3-bit scale + 1-bit delta sign + 4 × 3 high-bits per sub-block).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq1S {
    pub d: f16,
    pub qs: [u8; QK_K / 8],      // 32
    pub qh: [u8; 2 * QK_K / 32], // 16
}
const _: () = assert!(std::mem::size_of::<BlockIq1S>() == 2 + QK_K / 8 + 2 * QK_K / 32);

/// IQ1_M — 1.75-bpw K-quant. NO per-block `d` field; `d` is reassembled
/// from 4 nibbles spread across the 4 u16 `scales` words. Super-block of
/// 256 elements: 32-byte qs + 16-byte qh (per-elem (3-bit hi-idx +
/// 1-bit delta-sign) × 2 packed per byte) + 8-byte scales.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq1M {
    pub qs: [u8; QK_K / 8],      // 32
    pub qh: [u8; QK_K / 16],     // 16
    pub scales: [u8; QK_K / 32], //  8
}
const _: () = assert!(std::mem::size_of::<BlockIq1M>() == QK_K / 8 + QK_K / 16 + QK_K / 32);

/// IQ4_XS — 4-bit non-linear K-quant, 256-element super-block with 8
/// sub-blocks of 32. Per-sub-block signed 6-bit scale split into
/// `scales_l` (low 4 bits × 8 → 4 bytes) and `scales_h` (high 2 bits × 8
/// → u16). Byte-identical to `ggml-common.h`'s `block_iq4_xs`.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub d: f16,
    pub scales_h: u16,
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIq4Xs>() == 2 + 2 + QK_K / 64 + QK_K / 2);
