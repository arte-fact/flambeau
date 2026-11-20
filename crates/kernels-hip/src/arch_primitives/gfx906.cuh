#pragma once
// gfx906 arch primitives — DPP-fused warp reductions on MI50 / MI60 (GCN 5.1).
// Ported from candle-hip-kernels/src/gfx906_primitives.cuh (the subset // MMVQ actually needs). We keep the full-warp and half-warp reduce paths
// behind the same names candle uses so future ports stay byte-searchable.
// Architectural rule 5: this header contains HIP/gfx906 intrinsics and
// therefore cannot live in kernels-shared. Shared algorithmic-core headers
// (block-quant unpack math, softmax math) stay in kernels-shared/include.

#include <hip/hip_runtime.h>
#include "mmq_prefetch.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// ---------------------------------------------------------------------------
// DPP fused add/max primitives.
// Vega ISA Table 8 (§4.5) wait-state rules:
// VALU→VGPR then VALU-DPP reads that VGPR → 2 wait states (s_nop 1)
// VALU writes EXEC then VALU-DPP → 5 wait states (s_nop 4)
// The first DPP of a reduction chain follows the op that wrote EXEC (the
// compare/branch above the call site), so it uses `s_nop 4`. Subsequent
// DPPs follow DPP ops and only need `s_nop 1`.
// Single-arg inline asm ("%1, %1") forces the compiler to keep src0 and
// src1 in the same VGPR, avoiding a MOV that would break the wait-state
// bookkeeping. "memory" clobber pins instruction order across the DPP.
// ---------------------------------------------------------------------------

#ifdef __HIP_PLATFORM_AMD__

#define FLAMBEAU_FUSED_DPP_F32(name, barrier, dpp_ctrl, vop)                  \
    static __device__ __forceinline__ float name(float x) {                   \
        float r;                                                              \
        asm volatile(                                                         \
            barrier                                                           \
            vop " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf"         \
            : "=v"(r) : "v"(x) : "memory"                                     \
        );                                                                    \
        return r;                                                             \
    }

FLAMBEAU_FUSED_DPP_F32(gfx906_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
FLAMBEAU_FUSED_DPP_F32(gfx906_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
FLAMBEAU_FUSED_DPP_F32(gfx906_dpp_add_ror8, "s_nop 1\n", "row_ror:8",           "v_add_f32_dpp")

#undef FLAMBEAU_FUSED_DPP_F32

// xor-4 via split-bank v_mov_b32_dpp (row_shl:4 + row_shr:4).
static __device__ __forceinline__ float gfx906_shuffle_xor4(float x) {
    int src = __float_as_int(x);
    int dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(dst) : "v"(src) : "memory"
    );
    return __int_as_float(dst);
}

// xor-16 via the LDS crossbar swizzle (no actual LDS traffic).
static __device__ __forceinline__ float gfx906_swizzle_xor16(float x) {
    int src = __float_as_int(x);
    int dst;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(dst) : "v"(src) : "memory"
    );
    return __int_as_float(dst);
}

// Full 64-wide sum reduction. Every lane ends with the sum across the warp.
static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    x = gfx906_dpp_add_xor1(x);   // xor 1
    x = gfx906_dpp_add_xor2(x);   // xor 2
    x  += gfx906_shuffle_xor4(x); // xor 4 — split-bank returns other-lane value
    x = gfx906_dpp_add_ror8(x);   // xor 8 — row_ror:8 is a full-row swap
    x  += gfx906_swizzle_xor16(x);// xor 16
    x  += __shfl_xor(x, 32, 64);  // xor 32 — cross-half, no DPP pattern for this
    return x;
}

// Half-warp (32 lanes) — stops at xor-16, skips the cross-half swap. Used by
// multi-row MMVQ kernels with `nw1_r2` layout where each 32-lane group owns a
// distinct output row.
static __device__ __forceinline__ float gfx906_half_warp_reduce_sum(float x) {
    x = gfx906_dpp_add_xor1(x);
    x = gfx906_dpp_add_xor2(x);
    x += gfx906_shuffle_xor4(x);
    x = gfx906_dpp_add_ror8(x);
    x += gfx906_swizzle_xor16(x);
    return x;
}

// Quarter-warp (16 lanes) — stops at xor-8. Used by `nw1_r4` layout
// (Q6_K's default per candle P29).
static __device__ __forceinline__ float gfx906_quarter_warp_reduce_sum(float x) {
    x = gfx906_dpp_add_xor1(x);
    x = gfx906_dpp_add_xor2(x);
    x += gfx906_shuffle_xor4(x);
    x = gfx906_dpp_add_ror8(x);
    return x;
}

// Eighth-warp (8 lanes) — stops at xor-4. Used by r8 MoE layouts.
// All 3 steps are DPP within the same 16-lane bank so no cross-bank shuffle
// is needed.
static __device__ __forceinline__ float gfx906_eighth_warp_reduce_sum(float x) {
    x = gfx906_dpp_add_xor1(x);
    x = gfx906_dpp_add_xor2(x);
    x += gfx906_shuffle_xor4(x);
    return x;
}

// ---------------------------------------------------------------------------
// Fast-math primitives — ported from candle's gfx906_primitives.cuh
// (flash-attn port). Each maps to a single gfx906 SFU
// instruction (2-4 cycles) vs software math's ~10-20 instructions.
// ---------------------------------------------------------------------------

#ifndef GFX906_LOG2E
#define GFX906_LOG2E 1.4426950408889634f
#endif

static __device__ __forceinline__ float gfx906_rcp(float x) {
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

static __device__ __forceinline__ float gfx906_exp2(float x) {
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

/// exp(x) = exp2(x * log2(e)) — 1 FMA + 1 SFU cycle.
static __device__ __forceinline__ float gfx906_fast_exp(float x) {
    return gfx906_exp2(x * GFX906_LOG2E);
}

// ---------------------------------------------------------------------------
// dp4a — 4×int8 × 4×int8 → int32 fused multiply-add. Single-cycle on gfx906
// via `v_dot4_i32_i8`. Used by INT8 score paths (Q8_0 KV attention,
// MMVQ Q4_K/Q4_0/Q5_1, MMQ turbo). Inputs are int32-packed int8 quads.
// Saturate flag false → plain integer MAC.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int gfx906_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Neutral primitive names for the backend-neutral kernels-shared templates
// (mmvq.cuh / quant_traits.cuh), mirroring what kernels-cuda/sm_80.cuh exposes.
// The arch-specific bodies stay above; only the names cross into shared code.
static __device__ __forceinline__ float warp_reduce_sum(float x)         { return gfx906_warp_reduce_sum(x); }
static __device__ __forceinline__ float half_warp_reduce_sum(float x)    { return gfx906_half_warp_reduce_sum(x); }
static __device__ __forceinline__ float quarter_warp_reduce_sum(float x) { return gfx906_quarter_warp_reduce_sum(x); }
static __device__ __forceinline__ float eighth_warp_reduce_sum(float x)  { return gfx906_eighth_warp_reduce_sum(x); }
static __device__ __forceinline__ int   dp4a(int a, int b, int c)        { return gfx906_dp4a(a, b, c); }
static __device__ __forceinline__ float fast_exp(float x)                { return gfx906_fast_exp(x); }
static __device__ __forceinline__ float fast_rcp(float x)                { return gfx906_rcp(x); }

// Byte-select permute for codebook LUTs (IQ4_NL/XS). out.byte[i] =
// src8[sel.byte[i]] where src8 = {b.0..3 (low), a.0..3 (high)}; selector bytes
// are 0..7 (the LUTs never use the 8..15 special selectors). sm_80.cuh mirrors
// this via __byte_perm with the selector repacked to nibbles.
static __device__ __forceinline__ unsigned int byte_perm(unsigned int a, unsigned int b, unsigned int sel) {
    return __builtin_amdgcn_perm(a, b, sel);
}

// Map the 8 packed 4-bit indices of `q4` through a 16-entry int8 table packed
// little-endian in t0..t3: `lo` gets the low nibble of each byte, `hi` the high
// nibble. Per 4 values: two permutes gather the low / high octet of the table,
// a third selects between them on bit 3 of each index.
static __device__ __forceinline__ unsigned int lut16_s8_x4(
    unsigned int q, unsigned int t0, unsigned int t1, unsigned int t2, unsigned int t3) {
    const unsigned int v_low  = byte_perm(t1, t0, q & 0x07070707u);
    const unsigned int v_high = byte_perm(t3, t2, q & 0x07070707u);
    const unsigned int mask   = 0x03020100u | ((q & 0x08080808u) >> 1);
    return byte_perm(v_high, v_low, mask);
}
static __device__ __forceinline__ void lut16_s8_nibbles(
    unsigned int q4, unsigned int t0, unsigned int t1, unsigned int t2, unsigned int t3,
    int& lo, int& hi) {
    lo = (int) lut16_s8_x4(q4 & 0x0F0F0F0Fu, t0, t1, t2, t3);
    hi = (int) lut16_s8_x4((q4 >> 4) & 0x0F0F0F0Fu, t0, t1, t2, t3);
}

// Neutral L2-prefetch seam for the LDS-tiled MMQ kernels. gfx906 issues real
// global_load_dword hints (mmq_prefetch.cuh); sm_80.cuh provides no-ops. The
// shared kernel calls these names so the prefetch stays a backend detail.
static __device__ __forceinline__ int  mmq_prefetch_y_next(const int* p)                { return gfx906_prefetch_y_next(p); }
static __device__ __forceinline__ int  mmq_prefetch_x_next(const char* b, int o, int s) { return gfx906_prefetch_x_next(b, o, s); }
static __device__ __forceinline__ void mmq_prefetch_consume(int v)                      { gfx906_prefetch_consume(v); }

#else  // non-HIP fallback, should never be reached in V1

static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, WARP_SIZE);
    }
    return x;
}

#endif  // __HIP_PLATFORM_AMD__
