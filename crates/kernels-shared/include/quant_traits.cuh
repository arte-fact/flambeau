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

// Expand 4 consecutive bits of `qh` starting at `start` into 4 bytes (each 0
// or 1) packed into an int32 — the 5th-bit plane for Q5_0/Q5_1, fed to `dp4a`.
static __device__ __forceinline__ int expand_bits4(unsigned int qh, int start) {
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

// Weight-dtype tag types.
struct Q8_0 {};
struct Q4_0 {};
struct Q4_1 {};
struct Q5_0 {};
struct Q5_1 {};

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

    // VDR variant: one thread owns `grp`'s VDR consecutive int32 lanes, chains
    // VDR dp4a into one int32 sum, then a single block-scale multiply (vs VDR
    // float multiplies in `partial`). For VDR=1 this is identical to `partial`.
    template<int VDR>
    static __device__ __forceinline__ float
    partial_vdr(const block_t& bx, const flambeau_block_q8_1& by, int grp) {
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < VDR; ++i) {
            const int xi = load_int_b2(bx.qs, grp * VDR + i);
            const int yi = load_int_b2(by.qs, grp * VDR + i);
            sumi = dp4a(xi, yi, sumi);
        }
        return (float) bx.d * (float) by.d * (float) sumi;
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

template<> struct quant_traits<Q4_1> {
    using block_t = flambeau_block_q4_1;
    static constexpr int INT32_PER_BLOCK = QK4_1 / 8;

    // Q4_1 carries a per-block min: y = d*q + m, q in [0,15]. The dot folds
    // m*Σy via the Q8_1 header `s` (= d_y·Σy_qs): + m_x·s_y per block, split
    // across the 4 lanes by 0.25.
    static __device__ __forceinline__ float
    partial(const block_t& bx, const flambeau_block_q8_1& by, int j) {
        const int v     = load_int_b2((const int8_t*) bx.qs, j);
        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;
        const int u_lo  = load_int_b2(by.qs, j);
        const int u_hi  = load_int_b2(by.qs, j + 4);
        int sumi = dp4a(vi_lo, u_lo, 0);
        sumi     = dp4a(vi_hi, u_hi, sumi);
        return sumi * ((float) bx.d * (float) by.d) + (float) bx.m * (float) by.s * 0.25f;
    }
};

template<> struct quant_traits<Q5_0> {
    using block_t = flambeau_block_q5_0;
    static constexpr int INT32_PER_BLOCK = QK5_0 / 8;

    // Q5_0: 5-bit, zero-point -16. The 5th bit of each element lives in the
    // 32-bit `qh` plane; expand_bits4 unpacks it into a dp4a-able byte plane.
    // (q5-16)·y = nibble·y + 16·bit·y - 16·Σy → -16·d_x·s_y correction, /4.
    static __device__ __forceinline__ float
    partial(const block_t& bx, const flambeau_block_q8_1& by, int j) {
        const int v     = load_int_b2((const int8_t*) bx.qs, j);
        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;
        const int u_lo  = load_int_b2(by.qs, j);
        const int u_hi  = load_int_b2(by.qs, j + 4);
        const unsigned int qh = (unsigned int) load_int_b2((const int8_t*) bx.qh, 0);
        const int bit_lo = expand_bits4(qh, j * 4);
        const int bit_hi = expand_bits4(qh, j * 4 + 16);
        int sumi_nib = dp4a(vi_lo, u_lo, 0);
        sumi_nib     = dp4a(vi_hi, u_hi, sumi_nib);
        int sumi_bit = dp4a(bit_lo, u_lo, 0);
        sumi_bit     = dp4a(bit_hi, u_hi, sumi_bit);
        const float d_x = (float) bx.d;
        return (sumi_nib + 16 * sumi_bit) * (d_x * (float) by.d)
             - 16.0f * d_x * (float) by.s * 0.25f;
    }
};

template<> struct quant_traits<Q5_1> {
    using block_t = flambeau_block_q5_1;
    static constexpr int INT32_PER_BLOCK = QK5_1 / 8;

    // Q5_1: 5-bit with min. y = d*q5 + m, q5 in [0,31]. nibble + 5th-bit plane
    // (expand_bits4) + the m_x·s_y min fold (/4 across lanes).
    static __device__ __forceinline__ float
    partial(const block_t& bx, const flambeau_block_q8_1& by, int j) {
        const int v     = load_int_b2((const int8_t*) bx.qs, j);
        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;
        const int u_lo  = load_int_b2(by.qs, j);
        const int u_hi  = load_int_b2(by.qs, j + 4);
        const unsigned int qh = (unsigned int) load_int_b2((const int8_t*) bx.qh, 0);
        const int bit_lo = expand_bits4(qh, j * 4);
        const int bit_hi = expand_bits4(qh, j * 4 + 16);
        int sumi_nib = dp4a(vi_lo, u_lo, 0);
        sumi_nib     = dp4a(vi_hi, u_hi, sumi_nib);
        int sumi_bit = dp4a(bit_lo, u_lo, 0);
        sumi_bit     = dp4a(bit_hi, u_hi, sumi_bit);
        return (sumi_nib + 16 * sumi_bit) * ((float) bx.d * (float) by.d)
             + (float) bx.m * (float) by.s * 0.25f;
    }
};
