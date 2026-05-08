#pragma once
// gfx906 arch primitives — DPP-fused warp reductions on MI50 / MI60 (GCN 5.1).
// Ported from candle-hip-kernels/src/gfx906_primitives.cuh (the subset // MMVQ actually needs). We keep the full-warp and half-warp reduce paths
// behind the same names candle uses so future ports stay byte-searchable.
// Architectural rule 5: this header contains HIP/gfx906 intrinsics and
// therefore cannot live in kernels-shared. Shared algorithmic-core headers
// (block-quant unpack math, softmax math) stay in kernels-shared/include.

#include <hip/hip_runtime.h>

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

#else  // non-HIP fallback, should never be reached in V1

static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, WARP_SIZE);
    }
    return x;
}

#endif  // __HIP_PLATFORM_AMD__
