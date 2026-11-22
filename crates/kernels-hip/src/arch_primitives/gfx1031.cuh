#pragma once
// gfx1031 arch primitives — RDNA2 (Navi 22), **wave32**. The portability canary
// (CLAUDE.md): wave32 on the HIP side, so it mirrors `sm_80.cuh`'s warp-32
// reductions but with HIP's `__shfl_xor` (no `_sync`/mask). The neutral names
// (`warp_reduce_sum`, `dp4a`, …) are what the backend-neutral kernels-shared
// templates call; the arch-specific bodies stay here (rule 5). gfx906's DPP
// chains are wave64-only and live in gfx906.cuh; RDNA2's cross-lane path is a
// plain `__shfl_xor` butterfly at the explicit group width.
//
// Compile-portability scaffold: correctness-first wave32 lowering, no perf
// tuning. Runtime correctness/perf certs need a physical gfx1031 device.

#include <hip/hip_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef FLAMBEAU_LOG2E
#define FLAMBEAU_LOG2E 1.4426950408889634f
#endif

// Sum across a contiguous group of `width` lanes; every lane returns the group
// sum. Ascending offsets match the order the shared reduce helpers
// (mmvq_reduce.cuh) assume, so results agree lane-for-lane with the wave32 CUDA
// path.
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor(x, off, 32);
    return x;
}
static __device__ __forceinline__ float half_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) x += __shfl_xor(x, off, 16);
    return x;
}
static __device__ __forceinline__ float quarter_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) x += __shfl_xor(x, off, 8);
    return x;
}
static __device__ __forceinline__ float eighth_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1) x += __shfl_xor(x, off, 4);
    return x;
}

// SFU math — RDNA2 has v_rcp_f32 / v_exp_f32, same as GCN.
static __device__ __forceinline__ float fast_rcp(float x) {
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}
static __device__ __forceinline__ float fast_exp2(float x) {
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}
static __device__ __forceinline__ float fast_exp(float x) { return fast_exp2(x * FLAMBEAU_LOG2E); }

// dp4a — RDNA2 has v_dot4_i32_i8 (same builtin as GCN). Saturate flag false.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Byte-select permute for codebook LUTs (IQ4_NL/XS) — RDNA2 has v_perm_b32,
// same builtin + semantics as gfx906. out.byte[i] = src8[sel.byte[i]],
// src8 = {b.0..3 (low), a.0..3 (high)}.
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

// Neutral L2-prefetch seam — no-ops (gfx906's global_load_dword hints are a
// wave64 perf optimization; sm_80.cuh no-ops these too). Signatures mirror
// gfx906.cuh exactly.
static __device__ __forceinline__ int  mmq_prefetch_y_next(const int*)            { return 0; }
static __device__ __forceinline__ int  mmq_prefetch_x_next(const char*, int, int) { return 0; }
static __device__ __forceinline__ void mmq_prefetch_consume(int)                   {}
