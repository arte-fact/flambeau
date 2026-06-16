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
struct Q4_0 {};

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

template<> struct quant_traits<Q4_0> {
    using block_t = flambeau_block_q4_0;
    // int32s of nibble-packed quants per block: 16 bytes of qs => QK4_0 / 8.
    static constexpr int INT32_PER_BLOCK = QK4_0 / 8;

    // Float contribution of int32 lane `j` of one Q4_0 weight block. Q4_0 has
    // zero-point -8 (q in [0,15], y = d*(q-8)); DP4A can't bias packed bytes,
    // so (q-8)·y = q·y - 8·sum(y), with sum(y)·d_y baked into the Q8_1 header
    // as `s`. Lane `j` unpacks the 8 quants of one qs int32 into low / high
    // nibble planes (matching activation int32s `j` and `j+4`) and folds a
    // quarter of the per-block -8·d_x·s_y correction so the warp-reduce sums to
    // exactly one correction per block.
    static __device__ __forceinline__ float
    partial(const block_t& bx, const flambeau_block_q8_1& by, int j) {
        const int v     = load_int_b2((const int8_t*) bx.qs, j);
        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;
        const int u_lo  = load_int_b2(by.qs, j);
        const int u_hi  = load_int_b2(by.qs, j + 4);
        int sumi = dp4a(vi_lo, u_lo, 0);
        sumi     = dp4a(vi_hi, u_hi, sumi);
        const float d_x = (float) bx.d;
        return sumi * (d_x * (float) by.d) - 8.0f * d_x * (float) by.s * 0.25f;
    }
};
