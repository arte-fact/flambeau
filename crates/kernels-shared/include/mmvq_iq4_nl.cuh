#pragma once
// Templated single-row IQ4_NL MMVQ (decode GEMV). IQ4_NL is a 32-element block
// like Q4_0 (fp16 `d` + 16 nibble bytes), but each 4-bit index addresses the
// 16-entry `kvalues_iq4nl` non-linear codebook for a signed int8 value. Ports
// llama.cpp's `vec_dot_iq4_nl_q8_1`: 256 threads, 4 lanes per block; each lane
// LUT-decodes 8 nibbles (low+high) via the neutral `byte_perm` then two dp4a.
// No bias term (codebook is signed). Backend-neutral: `byte_perm` / `dp4a` /
// the reduce come from the arch header + shared headers; int loads go through
// `load_int_b2`. `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

// The 16 `kvalues_iq4nl` codebook entries as four little-endian int32 words.
static constexpr unsigned int IQ4NL_KV0 = 0xBFAD9881u; // {-127, -104, -83, -65}
static constexpr unsigned int IQ4NL_KV1 = 0xF6EADDCFu; // {-49, -35, -22, -10}
static constexpr unsigned int IQ4NL_KV2 = 0x26190D01u; // {1, 13, 25, 38}
static constexpr unsigned int IQ4NL_KV3 = 0x71594535u; // {53, 69, 89, 113}

// Map 4 packed 4-bit indices to 4 codebook int8 values. Two byte_perm gather
// the low / high octet of the codebook, a third selects between them on bit 3
// of each index.
static __device__ __forceinline__ unsigned int iq4_nl_table_4(unsigned int q4) {
    const unsigned int v_low  = byte_perm(IQ4NL_KV1, IQ4NL_KV0, q4 & 0x07070707u);
    const unsigned int v_high = byte_perm(IQ4NL_KV3, IQ4NL_KV2, q4 & 0x07070707u);
    const unsigned int mask   = 0x03020100u | ((q4 & 0x08080808u) >> 1);
    return byte_perm(v_high, v_low, mask);
}

// Map the 8 packed 4-bit indices of `q4` to codebook values: `lo` gets the low
// nibble of each byte, `hi` the high nibble (arch primitive `lut16_s8_nibbles`).
static __device__ __forceinline__ void iq4_nl_table_8(unsigned int q4, int& lo, int& hi) {
    lut16_s8_nibbles(q4, IQ4NL_KV0, IQ4NL_KV1, IQ4NL_KV2,
                     IQ4NL_KV3, lo, hi);
}

template<class OutT>
static __device__ __forceinline__ void mmvq_iq4_nl_dp4a_row(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    constexpr int THREADS = 256;
    constexpr int IPB     = 4;               // int32s of qs per block (16 bytes)
    constexpr int STEP    = THREADS / IPB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid   = threadIdx.x;
    const int lane4 = tid & 3;
    const int b0    = tid >> 2;

    const flambeau_block_iq4_nl* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = b0; b < n_blocks_per_row; b += STEP) {
        const flambeau_block_iq4_nl* bx = xrow + b;
        const flambeau_block_q8_1*   by = y + b;

        const int aux_q4 = load_int_b2((const int8_t*) bx->qs, lane4);
        const unsigned int q_low  = (unsigned int) aux_q4 & 0x0F0F0F0Fu;
        const unsigned int q_high = ((unsigned int) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const unsigned int v_lo = iq4_nl_table_4(q_low);
        const unsigned int v_hi = iq4_nl_table_4(q_high);

        const int u_lo = load_int_b2(by->qs, lane4);
        const int u_hi = load_int_b2(by->qs, lane4 + 4);

        int sumi = dp4a((int) v_lo, u_lo, 0);
        sumi     = dp4a((int) v_hi, u_hi, sumi);

        acc += (float) bx->d * (float) by->d * (float) sumi;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
