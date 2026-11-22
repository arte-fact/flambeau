#pragma once
// Backend-neutral block-quant layouts.
// Byte-identical to ggml-common.h + candle k_quants.rs so pointer-casts
// from mmap'd GGUF data work on GPU without repacking.

#include "backend_compat.cuh"

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

// fb_fp16_t is defined in backend_compat.cuh (_Float16 on HIP, __half on CUDA).

// bf16 — Brain Float 16: 1 sign + 8 exponent + 7 mantissa bits, byte-
// compatible with the upper half of an IEEE-754 F32. gfx906 has no
// native BF16 arithmetic (CDNA2/MI200+ only), so storage is uint16_t
// and arithmetic routes through F32 via bit-shift conversion.
typedef unsigned short fb_bf16_t;

__device__ static inline fb_bf16_t fb_f32_to_bf16(float x) {
    // Round-to-nearest-even truncation of the lower 16 bits.
    // NaN inputs are preserved as NaN: the test forces a quiet-NaN
    // mantissa bit so the truncated result still satisfies isnan.
    unsigned int u = __builtin_bit_cast(unsigned int, x);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
        return (fb_bf16_t) ((u >> 16) | 0x0040u);
    }
    unsigned int lsb  = (u >> 16) & 1u;
    unsigned int bias = 0x7FFFu + lsb;
    return (fb_bf16_t) ((u + bias) >> 16);
}

__device__ static inline float fb_bf16_to_f32(fb_bf16_t b) {
    unsigned int u = ((unsigned int) b) << 16;
    return __builtin_bit_cast(float, u);
}

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
// Body: 4 × QK8_1 = 128 int8 quants = 128 B.
// Total: 144 B.
// Storage: (k_big_block, col) row-major in the device buffer — the MMQ
// K-loop fetches 144-B strides along the col axis, sharing each block
// across all threads in the tile's col-group.
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

// Q2_K — 2-bit K-quant, super-block of 256 elements. Byte-identical to
// ggml-common.h block_q2_K + flambeau-quant's BlockQ2K.
typedef struct {
    uint8_t   scales[QK_K / 16];           // 16 × packed 4-bit (scale, min)
    uint8_t   qs[QK_K / 4];                // 64 bytes, 4 elements / byte
    fb_fp16_t d;                           // super-block scale
    fb_fp16_t dmin;                        // super-block min scale
} flambeau_block_q2_K;
static_assert(sizeof(flambeau_block_q2_K) == QK_K / 16 + QK_K / 4 + 4,
              "block_q2_K size");

// Q3_K — 3-bit K-quant, super-block of 256 elements. Byte-identical to
// ggml-common.h block_q3_K + flambeau-quant's BlockQ3K. High bit lives in
// `hmask`; low 2 bits in `qs`; signed 6-bit scales packed in 12 bytes.
typedef struct {
    uint8_t   hmask[QK_K / 8];             // 32 bytes — 1 bit per element
    uint8_t   qs[QK_K / 4];                // 64 bytes — 2 bits per element
    uint8_t   scales[12];                  // 16 × signed 6-bit scales
    fb_fp16_t d;                           // super-block scale
} flambeau_block_q3_K;
static_assert(sizeof(flambeau_block_q3_K) == QK_K / 8 + QK_K / 4 + 12 + 2,
              "block_q3_K size");

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

// quantize_q8_1_to_shared — block-level helper that takes a length-D
// FP16 vector in registers/LDS and emits a Q8_1-style packed
// representation in shared memory: int8 quants + per-32-element
// (d, s) header pair. Used by the INT8 attention score path so the
// inner KQ matmul can stay in integer (dp4a) instead of dequanting K
// to FP16 per element.
// Layout in `out_qs`/`out_ds`:
// out_qs[D] : int8 quantized values
// out_ds[(D/32) * 2] : interleaved (d, d*sum) FP16 per 32-elem block
// (matches `flambeau_block_q8_1.ds` semantics)
// Caller invariants: `D` is a multiple of 32, and `D <= blockDim.x`.
// One thread reads `q_in[tid]` (or zero-pads if `tid >= D`). Reduction
// is wave-wide (`__shfl_xor` with WARP=64 on gfx906); for blocks
// larger than one warp, caller must run this WITH `n_warps = D/64`
// per group and keep block-amax per 32-element subgroup separately.
// Reference: `quantize_q8_1_to_shared` in
// `/artefact/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:292`.
__device__ __forceinline__ void flambeau_quantize_q8_1_to_shared(
    const fb_fp16_t* __restrict__ q_in,   // [D] fp16
    int               D,                   // length, multiple of 32
    int               tid,                 // 0..blockDim.x-1
    int8_t*  __restrict__ out_qs,          // [D]
    fb_fp16_t* __restrict__ out_ds          // [(D/32) * 2] interleaved (d, d*sum)
) {
    const int block_idx    = tid >> 5;     // tid / 32 → which Q8_1 block
    const int block_offset = tid & 31;     // tid % 32

    if (tid >= D) return;

    float v = (float) q_in[tid];

    // amax over each 32-lane group via warp shfl-xor. With WARP=64 on
    // gfx906, lanes 0..31 form group 0, 32..63 form group 1, so we
    // reduce within the half-warp (shfl_xor up to 16). For blocks
    // wider than one warp, each warp's lane 0..31 / 32..63 contribute
    // distinct (d, s) pairs to consecutive Q8_1 blocks — caller is
    // expected to launch with blockDim.x = D so each tid covers one
    // element. score_parts-style cross-warp aggregation is NOT needed
    // because each Q8_1 block is owned by exactly one half-warp.
    float amax = fabsf(v);
    float sum  = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 32));
        sum +=         __shfl_xor(sum,  off, 32);
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const int   qi = min(127, max(-127, (int) rintf(v * id)));
    out_qs[tid] = (int8_t) qi;

    // Lane 0 of each 32-element group writes the (d, d*sum) header.
    if (block_offset == 0) {
        out_ds[block_idx * 2 + 0] = (fb_fp16_t) d;
        out_ds[block_idx * 2 + 1] = (fb_fp16_t) (d * sum);
    }
}

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

// Byte-wise u32 load. Q3_K block size is 110 B (non-multiple-of-4), so
// scales[], qs[], and hmask[] alternate 4-byte / 2-byte alignment across
// consecutive blocks; a `(uint32_t*)` cast there is unaligned UB on the
// 2-byte-aligned half (empirically 18-137% rel-err in Q3_K MMVQ before fix).
__device__ __forceinline__ uint32_t flambeau_load_u32_unaligned(const uint8_t* p) {
    return (uint32_t) p[0]
         | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16)
         | ((uint32_t) p[3] << 24);
}

// Unpack the packed 6-bit signed scales (12 bytes) into 16 raw bytes in
// [0, 63]. Caller applies the -32 bias at use time. Uses byte-wise loads
// to handle Q3_K's misaligned scales[] across blocks.
__device__ __forceinline__ void flambeau_q3k_unpack_scales(
    const uint8_t* __restrict__ scales, int8_t out[16]) {
    const uint32_t k1 = 0x0303'0303u;
    const uint32_t k2 = 0x0f0f'0f0fu;
    uint32_t aux[4];
    aux[0] = flambeau_load_u32_unaligned(scales);
    aux[1] = flambeau_load_u32_unaligned(scales + 4);
    const uint32_t tmp = flambeau_load_u32_unaligned(scales + 8);
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    const uint8_t* bytes = (const uint8_t*) aux;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = (int8_t) bytes[i];
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

// Q8_K — 8-bit K-quant, super-block of 256 elements. Activation-side
// quant (ggml uses it for K · Q dot products). Byte-identical to
// ggml-common.h block_q8_K + flambeau-quant's BlockQ8K.
typedef struct {
    float   d;                             // super-block scale (F32)
    int8_t  qs[QK_K];                      // 256 signed quants
    int16_t bsums[QK_K / 16];              // per-16-elem sum of qs (precomputed)
} flambeau_block_q8_K;
static_assert(sizeof(flambeau_block_q8_K) == 4 + QK_K + QK_K / 16 * 2,
              "block_q8_K size");

// IQ4_NL — 4-bit non-linear quant, 32-element block. f16 scale + 16 bytes
// of nibble-packed unsigned 4-bit codes. Reconstruction:
//   y_i = d * KVALUES_IQ4NL[code_i]
// where code_i indexes a 16-entry signed i8 LUT (ported from llama.cpp's
// `kvalues_iq4nl`). Low nibble at byte j → elem j; high nibble at byte j →
// elem j + 16. Byte-identical to ggml-common.h block_iq4_nl.
typedef struct {
    fb_fp16_t d;                  // per-block scale
    uint8_t   qs[QK4_0 / 2];      // 16 bytes, 4-bit LUT indices (low | high)
} flambeau_block_iq4_nl;
static_assert(sizeof(flambeau_block_iq4_nl) == 2 + QK4_0 / 2,
              "block_iq4_nl size");

// IQ4_XS — 4-bit non-linear K-quant, super-block of 256 elements with 8
// sub-blocks of 32. Per-sub-block signed 6-bit scale split into
// `scales_l` (low 4 bits × 8 in 4 bytes) and `scales_h` (high 2 bits × 8
// in u16). Reconstruction (per sub-block ib):
//   ls = (scales_l[ib/2] >> (4*(ib&1)) & 0xF) | ((scales_h >> (2*ib)) & 3) << 4
//   ls_signed = (i32) ls - 32                    (range [-32, 31])
//   y = d * ls_signed * KVALUES_IQ4NL[code]      (same LUT as IQ4_NL)
// qs layout: sub-block ib owns bytes qs[ib*16 .. ib*16+16]; low nibble →
// elem 0..15 of sub-block, high nibble → elem 16..31. Byte-identical to
// ggml-common.h block_iq4_xs.
typedef struct {
    fb_fp16_t d;                           // super-block scale
    uint16_t  scales_h;                    // high 2 bits of each of 8 sub-block scales
    uint8_t   scales_l[QK_K / 64];         // low 4 bits × 8 → 4 bytes
    uint8_t   qs[QK_K / 2];                // 128 bytes, 4-bit LUT indices
} flambeau_block_iq4_xs;
static_assert(sizeof(flambeau_block_iq4_xs) == 2 + 2 + QK_K / 64 + QK_K / 2,
              "block_iq4_xs size");

// IQ3_XXS — 3-bit-ish (3.06 bpw) K-quant with codebook lookup. Super-block
// of 256 elements; per-block layout:
//   f16 d (2 bytes)
//   u8 qs[96]:
//     qs[0..64]   — 64 codebook indices (2 indices per ib32 × 4 groups × 8 ib32)
//     qs[64..96]  — 32 packed (4-bit scale + 4 × 7-bit sign-LUT index) per ib32,
//                   loaded as 8 × u32 with the scale at bits [28..32]
// Byte-identical to ggml-common.h `block_iq3_xxs`.
typedef struct {
    fb_fp16_t d;
    uint8_t   qs[QK_K / 4 + QK_K / 8];     // 64 + 32 = 96
} flambeau_block_iq3_xxs;
static_assert(sizeof(flambeau_block_iq3_xxs) == 2 + QK_K / 4 + QK_K / 8,
              "block_iq3_xxs size");

// IQ3_S — refined 3.44-bpw K-quant. Super-block of 256 elements with explicit
// 9th-bit and sign arrays alongside the 8-bit codebook indices:
//   f16 d (2 bytes)
//   u8 qs[64]      — codebook low 8 bits (one byte per 4-element group)
//   u8 qh[8]       — codebook 9th bit, one bit per qs byte (8 bytes pack 64 bits)
//   u8 signs[32]   — per-byte 8-bit sign masks
//   u8 scales[4]   — 4-bit nibble scales, two per byte, 8 sub-block scales total
// Byte-identical to ggml-common.h `block_iq3_s`.
typedef struct {
    fb_fp16_t d;
    uint8_t   qs[QK_K / 4];                // 64
    uint8_t   qh[QK_K / 32];               //  8
    uint8_t   signs[QK_K / 8];             // 32
    uint8_t   scales[QK_K / 64];           //  4
} flambeau_block_iq3_s;
static_assert(sizeof(flambeau_block_iq3_s)
              == 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64,
              "block_iq3_s size");

// IQ2_XXS — 2.0625 bpw super-block (256 elems / 66 bytes). f16 d + 64 bytes
// of u16 qs[32]. Per ib32: 8 qs bytes read as 2 × u32, top u32 holds the
// 4-bit scale at [28..32] + 4 × 7-bit sign-LUT indices, bottom u32 holds
// 4 × 8-bit codebook indices into IQ2XXS_GRID (256 × u64).
typedef struct {
    fb_fp16_t d;
    uint8_t   qs[2 * QK_K / 8];           // 64 bytes (u16 × 32 viewed byte-wise)
} flambeau_block_iq2_xxs;
static_assert(sizeof(flambeau_block_iq2_xxs) == 2 + 2 * QK_K / 8,
              "block_iq2_xxs size");

// IQ2_XS — 2.3125 bpw. 256 elems / 74 bytes. f16 d + 64-byte u16 qs[32] +
// 8-byte sub-block scales (8 × 4-bit nibbles, two per byte). Each qs u16
// packs (9-bit grid index | 7-bit sign-LUT index).
typedef struct {
    fb_fp16_t d;
    uint8_t   qs[2 * QK_K / 8];           // 64 bytes
    uint8_t   scales[QK_K / 32];          //  8 bytes
} flambeau_block_iq2_xs;
static_assert(sizeof(flambeau_block_iq2_xs)
              == 2 + 2 * QK_K / 8 + QK_K / 32,
              "block_iq2_xs size");

// IQ2_S — 2.5 bpw. 256 elems / 82 bytes. f16 d + qs[32] (low-8 of 10-bit
// codebook index) + qs[32..64] (per-byte sign masks) + qh[8] (2 high bits
// of the codebook index per qs byte, four bytes packed into one qh byte)
// + scales[8] (8 × 4-bit nibbles, two per byte).
typedef struct {
    fb_fp16_t d;
    uint8_t   qs[QK_K / 4];                // 64 bytes: [0..32] idx-low, [32..64] signs
    uint8_t   qh[QK_K / 32];               //  8 bytes: 2 high bits per qs byte
    uint8_t   scales[QK_K / 32];           //  8 bytes: 4-bit nibble pairs
} flambeau_block_iq2_s;
static_assert(sizeof(flambeau_block_iq2_s)
              == 2 + QK_K / 4 + QK_K / 32 + QK_K / 32,
              "block_iq2_s size");

// IQ1_S — 1.5625 bpw. 256 elems / 50 bytes. f16 d + qs[32] (low-8 of
// 11-bit codebook index) + qh[8] u16 (per-sub-block 3-bit scale + 1-bit
// delta-sign + 4 × 3 high-bits of codebook indices in one u16).
typedef struct {
    fb_fp16_t d;
    uint8_t   qs[QK_K / 8];                // 32 bytes
    uint8_t   qh[2 * QK_K / 32];           // 16 bytes (u16 × 8)
} flambeau_block_iq1_s;
static_assert(sizeof(flambeau_block_iq1_s)
              == 2 + QK_K / 8 + 2 * QK_K / 32,
              "block_iq1_s size");

// IQ1_M — 1.75 bpw. 256 elems / 56 bytes. NO per-block `d` in the bytes —
// `d` is reassembled from 4 nibbles spread across the 4 u16 scale-words
// (scales[]). qs[32] = low-8 codebook idx, qh[16] = pairs of (3-bit
// high-idx + 1-bit delta-sign) × 2 per byte, scales[8] = 4 × u16 packing
// (d-nibble + 2 × 3-bit sub-block scales).
typedef struct {
    uint8_t qs[QK_K / 8];                  // 32 bytes
    uint8_t qh[QK_K / 16];                 // 16 bytes
    uint8_t scales[QK_K / 32];             //  8 bytes
} flambeau_block_iq1_m;
static_assert(sizeof(flambeau_block_iq1_m)
              == QK_K / 8 + QK_K / 16 + QK_K / 32,
              "block_iq1_m size");

// Shared signed-i8 LUT for IQ4_NL and IQ4_XS. Byte-identical port of
// llama.cpp `kvalues_iq4nl` (ggml-common.h). Inline so each kernel TU
// gets a register-resident copy without ODR conflicts; the compiler is
// free to keep it in constant memory if launch occupancy benefits.
__device__ __forceinline__ int8_t flambeau_iq4nl_lut(int idx) {
    constexpr int8_t lut[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return lut[idx & 0xF];
}

// Reconstruct the signed 6-bit sub-block scale for sub-block `ib` (0..7)
// in an IQ4_XS super-block. Returns the bias-corrected i32 in [-32, 31].
__device__ __forceinline__ int flambeau_iq4_xs_scale(
    int ib, uint16_t scales_h, const uint8_t* __restrict__ scales_l
) {
    const int l_nib = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int h_bits = (scales_h >> (2 * ib)) & 0x03;
    return (l_nib | (h_bits << 4)) - 32;
}
