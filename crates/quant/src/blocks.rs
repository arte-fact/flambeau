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
    pub ds: [f16; 8],               // 4 × (d, d*sum) packed as half2[4]
    pub qs: [i8; 4 * QK8_1],        // 128 quants
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
