#pragma once
// Per-dtype quantization policy for the templated matmul kernels: the block
// type plus the per-int32 partial dot against a Q8_1 activation block. This is
// the only per-dtype code the mmvq/mmq skeletons need. Backend-neutral: the
// `dp4a` it calls comes from the backend's arch_primitives header, which must
// be included before this one.
#include "block_quant.cuh"

// Load int32 `i` from a byte-packed quant array that is only 2-byte aligned
// (e.g. Q8_0's `qs` sits at offset 2). Reads two aligned uint16 and combines,
// reconstructing the little-endian int32 without a misaligned 4-byte load —
// NVIDIA faults on those (CUDA_ERROR_MISALIGNED_ADDRESS); AMD tolerates them.
static __device__ __forceinline__ int load_int_b2(const int8_t* p, int i) {
    const unsigned short* u = (const unsigned short*) p;
    return (int) u[2 * i] | ((int) u[2 * i + 1] << 16);
}

// Weight-dtype tag types.
struct Q8_0 {};

template<class Q> struct quant_traits;

template<> struct quant_traits<Q8_0> {
    using block_t = flambeau_block_q8_0;
    // int32s of packed int8 quants per block (QK8_0 / 4).
    static constexpr int INT32_PER_BLOCK = QK8_0 / 4;

    // Float contribution of int32 lane `j` of one weight block against the
    // matching Q8_1 activation block. Q8_0 is symmetric (no min/zero-point),
    // so the block scale is a per-block constant and applying it per lane then
    // summing equals applying it to the block sum.
    static __device__ __forceinline__ float
    partial(const block_t& bx, const flambeau_block_q8_1& by, int j) {
        const int xi = load_int_b2(bx.qs, j);
        const int yi = load_int_b2(by.qs, j);
        return (float) bx.d * (float) by.d * (float) dp4a(xi, yi, 0);
    }
};
