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
