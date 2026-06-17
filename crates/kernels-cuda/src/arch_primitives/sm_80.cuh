#pragma once
// Warp-32 cross-lane reductions + SFU/dp4a primitives for sm_80+.
// Architectural rule 5: carries CUDA intrinsics, so it stays in kernels-cuda.

#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef FLAMBEAU_LOG2E
#define FLAMBEAU_LOG2E 1.4426950408889634f
#endif

// Full mask: kernels launch warp-aligned, so every lane participates.
#define FLAMBEAU_WARP_MASK 0xffffffffu

// Sum across a contiguous group of `width` lanes; every lane in the group
// returns the group sum.
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(FLAMBEAU_WARP_MASK, x, off, 32);
    return x;
}
static __device__ __forceinline__ float half_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) x += __shfl_xor_sync(FLAMBEAU_WARP_MASK, x, off, 16);
    return x;
}
static __device__ __forceinline__ float quarter_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) x += __shfl_xor_sync(FLAMBEAU_WARP_MASK, x, off, 8);
    return x;
}
static __device__ __forceinline__ float eighth_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1) x += __shfl_xor_sync(FLAMBEAU_WARP_MASK, x, off, 4);
    return x;
}

static __device__ __forceinline__ float fast_rcp(float x) { return __frcp_rn(x); }
// CUDA has no __exp2f intrinsic; exp2f lowers to the MUFU.EX2 SFU op.
static __device__ __forceinline__ float fast_exp2(float x) { return exp2f(x); }
static __device__ __forceinline__ float fast_exp(float x) { return exp2f(x * FLAMBEAU_LOG2E); }

static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

// Byte-select permute for codebook LUTs (IQ4_NL/XS), matching gfx906.cuh's
// `byte_perm`: out.byte[i] = src8[sel.byte[i]], src8 = {b.0..3 (low), a.0..3
// (high)}. amdgcn_perm takes a byte-per-output selector; __byte_perm takes a
// nibble-per-output selector with source {x.0..3, y.0..3}. So x=b, y=a and pack
// each selector byte's low 3 bits into nibble i. Valid for selectors 0..7 (the
// LUTs never use the 8..15 special bytes).
static __device__ __forceinline__ unsigned int byte_perm(unsigned int a, unsigned int b, unsigned int sel) {
    const unsigned int packed = (sel & 0x7u)
                              | (((sel >> 8) & 0x7u) << 4)
                              | (((sel >> 16) & 0x7u) << 8)
                              | (((sel >> 24) & 0x7u) << 12);
    return __byte_perm(b, a, packed);
}

// Neutral L2-prefetch seam for the LDS-tiled MMQ kernels — no-ops on CUDA
// (correctness-first; cp.async staging is a perf follow-up). gfx906.cuh issues
// real global_load_dword hints; signatures mirror it exactly.
static __device__ __forceinline__ int  mmq_prefetch_y_next(const int*)            { return 0; }
static __device__ __forceinline__ int  mmq_prefetch_x_next(const char*, int, int) { return 0; }
static __device__ __forceinline__ void mmq_prefetch_consume(int)                   {}
