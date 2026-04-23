#pragma once
// Backend-neutral block-quant layouts.
// Byte-identical to ggml-common.h + candle k_quants.rs so pointer-casts
// from mmap'd GGUF data work on GPU without repacking.

#include <hip/hip_runtime.h>

#ifndef QK8_0
#define QK8_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif
#ifndef QK4_1
#define QK4_1 32
#endif
#ifndef QK_K
#define QK_K 256
#endif
#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

// fp16 — `_Float16` is the HIP/clang built-in half type; ggml calls this
// `ggml_fp16_t` (a typedef of `__half`). Using the built-in keeps us away
// from hip/amd_hip_fp16.h's heavier wrapper class for pointer casts.
typedef _Float16 fb_fp16_t;

typedef struct {
    fb_fp16_t d;             // scale
    int8_t    qs[QK8_0];     // signed quants
} flambeau_block_q8_0;
static_assert(sizeof(flambeau_block_q8_0) == 2 + QK8_0, "block_q8_0 size");

typedef struct {
    fb_fp16_t d;             // scale
    fb_fp16_t s;             // sum(x) * d — used by Q4_1/Q5_1 vec_dot
    int8_t    qs[QK8_1];     // signed quants
} flambeau_block_q8_1;
static_assert(sizeof(flambeau_block_q8_1) == 4 + QK8_1, "block_q8_1 size");

// Q8_1 MMQ layout — turbo / candle 4-warp LDS-tiled prefill path's activation
// block. One MMQ block holds 4 × 32 = 128 elements (vs standard Q8_1's 32).
// Header: 4 × half2 ds (one (d, d*sum) per 32-element sub-block) = 16 B.
// Body:   4 × QK8_1 = 128 int8 quants = 128 B.
// Total: 144 B.
//
// Storage: (k_big_block, col) row-major in the device buffer — the MMQ
// K-loop fetches 144-B strides along the col axis, sharing each block
// across all threads in the tile's col-group.
//
// Source: /artefact/candle/candle-hip-kernels/src/mmq_turbo.cu:159-167.
typedef struct {
    fb_fp16_t ds[8];                  // 4 × half2 = 8 fp16 ((d, d*sum) × 4)
    int8_t    qs[4 * QK8_1];          // 128 signed quants
} flambeau_block_q8_1_mmq;
static_assert(sizeof(flambeau_block_q8_1_mmq) == 16 + 4 * QK8_1,
              "block_q8_1_mmq size");
#ifndef QK8_1_MMQ
#define QK8_1_MMQ (4 * QK8_1)         // 128 elements per MMQ block
#endif

// Q4_1 — 4-bit legacy quant with min offset. Block of 32 elements, 16 bytes
// of nibble-packed unsigned quants. Reconstruction: y = d * q - m (where
// q in [0, 15], so effectively y = d * (q - m/d) but kept separate).
typedef struct {
    fb_fp16_t d;                  // delta (scale)
    fb_fp16_t m;                  // min
    uint8_t   qs[QK4_1 / 2];      // 16 bytes, 4-bit nibbles (low | high)
} flambeau_block_q4_1;
static_assert(sizeof(flambeau_block_q4_1) == 2 + 2 + QK4_1 / 2, "block_q4_1 size");

#ifndef QK4_0
#define QK4_0 32
#endif
#ifndef QK5_0
#define QK5_0 32
#endif

// Q4_0 — 4-bit legacy quant with zero-point -8 (no min). Block of 32
// elements, 16 bytes nibble-packed unsigned quants. Reconstruction:
// y = d * (q - 8) where q ∈ [0, 15].
typedef struct {
    fb_fp16_t d;                  // delta (scale)
    uint8_t   qs[QK4_0 / 2];      // 16 bytes, 4-bit nibbles (low | high)
} flambeau_block_q4_0;
static_assert(sizeof(flambeau_block_q4_0) == 2 + QK4_0 / 2, "block_q4_0 size");

// Q5_0 — 5-bit legacy quant with zero-point -16 (no min). Block of 32
// elements: 16 bytes of low-4-bit nibbles, 4 bytes packing the 5th bit
// for each of 32 elements (bit i of the uint32 → 5th bit of element i).
// Reconstruction: q5 = (qh_bit_i << 4) | nibble_i, then y = d * (q5 - 16).
typedef struct {
    fb_fp16_t d;                  // delta (scale)
    uint8_t   qh[4];              // 32 "5th bits", one per element
    uint8_t   qs[QK5_0 / 2];      // 16 bytes, 4-bit nibbles (low | high)
} flambeau_block_q5_0;
static_assert(sizeof(flambeau_block_q5_0) == 2 + 4 + QK5_0 / 2, "block_q5_0 size");

#ifndef QK5_1
#define QK5_1 32
#endif

// Q5_1 — 5-bit legacy quant with min offset. Block of 32 elements: 16 bytes
// of low-4-bit nibbles + 4 bytes 5th-bit pack, like Q5_0, but adds a per-
// block `m` (min) so reconstruction is `y = d · q5 + m` where q5 ∈ [0, 31].
typedef struct {
    fb_fp16_t d;                  // delta (scale)
    fb_fp16_t m;                  // min
    uint8_t   qh[4];              // 32 "5th bits"
    uint8_t   qs[QK5_1 / 2];      // 16 bytes, 4-bit nibbles (low | high)
} flambeau_block_q5_1;
static_assert(sizeof(flambeau_block_q5_1) == 2 + 2 + 4 + QK5_1 / 2, "block_q5_1 size");

// Q4_K — 4-bit K-quant, super-block of 256 elements split into 8 sub-blocks
// of 32. Byte-identical to ggml-common.h block_q4_K and to flambeau-quant's
// BlockQ4K.
typedef struct {
    fb_fp16_t d;                           // super-block scale
    fb_fp16_t dmin;                        // super-block min scale
    uint8_t   scales[K_SCALE_SIZE];        // packed 6-bit (scale, min) pairs × 8
    uint8_t   qs[QK_K / 2];                // 256 × 4-bit quants (low/high nibble)
} flambeau_block_q4_K;
static_assert(sizeof(flambeau_block_q4_K) == 2 + 2 + K_SCALE_SIZE + QK_K / 2,
              "block_q4_K size");

// Reconstruct the 6-bit (scale, min) pair for sub-block `j` (0..7) from the
// packed 12-byte scales array. Mirrors candle's `get_scale_min_k4` exactly.
__device__ __forceinline__ void flambeau_q4k_scale_min(
    int j, const uint8_t* __restrict__ q, uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = q[j]     & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
    }
}

// Q5_K — 5-bit K-quant: 4-bit low nibble in `qs` + 1 high bit in `qh`.
// Byte layout identical to flambeau-quant's BlockQ5K.
typedef struct {
    fb_fp16_t d;
    fb_fp16_t dmin;
    uint8_t   scales[K_SCALE_SIZE];        // same 6-bit (sc, m) × 8 layout as Q4_K
    uint8_t   qh[QK_K / 8];                // one high bit per element, 32 bytes
    uint8_t   qs[QK_K / 2];                // low 4 bits per element, 128 bytes
} flambeau_block_q5_K;
static_assert(sizeof(flambeau_block_q5_K) ==
                  2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
              "block_q5_K size");

// Q6_K — 6-bit K-quant: 4-bit low nibble in `ql` + 2 high bits in `qh` +
// per-16-element i8 scales + super-block `d`. Byte layout identical to
// flambeau-quant's BlockQ6K.
typedef struct {
    uint8_t   ql[QK_K / 2];                // 128 bytes, 4 bits per element
    uint8_t   qh[QK_K / 4];                // 64 bytes, 2 bits per element
    int8_t    scales[QK_K / 16];           // 16 signed byte scales
    fb_fp16_t d;                           // super-block scale
} flambeau_block_q6_K;
static_assert(sizeof(flambeau_block_q6_K) ==
                  QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
              "block_q6_K size");
