#pragma once
// Row-tiled batched dp4a MMVQ for the 256-element super-block dtypes (Q4_K,
// Q5_K, IQ4_XS), N activation columns. 256 threads = 8 output rows per block,
// each row owned by a 32-lane group. Per weight super-block the block stages
// the N columns' Q8_1 strip (8 sub-blocks each) into LDS once, then every lane
// decodes its weight ints in registers once and dots them against the N
// staged columns. Small-batch counterpart of the LDS-tiled MMQ: no weight
// tile, the activation is fetched from global memory once per block instead
// of once per (row, column). Standard slot-major Q8_1 `y`, slot-major `dst`.
// `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"
#include "mmvq_iq4_nl.cuh"

namespace mmvq_rt {
constexpr int THREADS   = 256;
constexpr int ROW_GROUP = 32;
constexpr int ROWS      = THREADS / ROW_GROUP;
constexpr int SUBS      = 8;
constexpr int BLK_INTS  = 9;  // sizeof(flambeau_block_q8_1) / 4

// Copy the N columns' 8-sub-block strip of super-block `b` into `s_y`.
template<int N>
static __device__ __forceinline__ void stage_strip(
    int* s_y, const flambeau_block_q8_1* __restrict__ y,
    size_t act_blocks_per_row, int b)
{
    constexpr int COL_INTS = SUBS * BLK_INTS;
    #pragma unroll
    for (int i0 = 0; i0 < N * COL_INTS; i0 += THREADS) {
        const int i = i0 + threadIdx.x;
        if (i < N * COL_INTS) {
            const int c = i / COL_INTS;
            const int o = i - c * COL_INTS;
            const int* src = (const int*) (y + c * act_blocks_per_row + (size_t) b * SUBS);
            s_y[i] = src[o];
        }
    }
}

struct Q4K {
    typedef flambeau_block_q4_K block_t;
    // lane = (pair 0..3) × (int 0..7): one int of both sub-blocks of a pair.
    template<int N>
    static __device__ __forceinline__ void step(
        const block_t* bk, const flambeau_block_q8_1* s_blk, int lane_lo, float (&acc)[N])
    {
        const int pair_idx = lane_lo >> 3;
        const int iqs      = lane_lo & 7;
        const int sub_lo   = pair_idx * 2;

        const int qs_int = load_int_b2((const int8_t*) bk->qs, pair_idx * 8 + iqs);
        const int q_lo = qs_int & 0x0F0F0F0F;
        const int q_hi = (qs_int >> 4) & 0x0F0F0F0F;

        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_lo + 1, bk->scales, &sc_hi, &m_hi);
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;
        const float ds_lo = d * (float) sc_lo;
        const float ds_hi = d * (float) sc_hi;
        const float dm_lo = dmin * (float) m_lo;
        const float dm_hi = dmin * (float) m_hi;

        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya_lo = s_blk + c * SUBS + sub_lo;
            const flambeau_block_q8_1* ya_hi = ya_lo + 1;
            const int u_lo = load_int_b2(ya_lo->qs, iqs);
            const int u_hi = load_int_b2(ya_hi->qs, iqs);
            acc[c] += (float) ya_lo->d
                    * (ds_lo * (float) dp4a(q_lo, u_lo, 0) - dm_lo * (float) dp4a(0x01010101, u_lo, 0))
                    + (float) ya_hi->d
                    * (ds_hi * (float) dp4a(q_hi, u_hi, 0) - dm_hi * (float) dp4a(0x01010101, u_hi, 0));
        }
    }
};

struct Q5K {
    typedef flambeau_block_q5_K block_t;
    // lane = (sub 0..7) × (int pair 0..3): ints `lane4` and `lane4 + 4` of one sub-block.
    template<int N>
    static __device__ __forceinline__ void step(
        const block_t* bk, const flambeau_block_q8_1* s_blk, int lane_lo, float (&acc)[N])
    {
        const int sub   = lane_lo >> 2;
        const int lane4 = lane_lo & 3;

        uint8_t sc_u8 = 0, m_u8 = 0;
        flambeau_q4k_scale_min(sub, bk->scales, &sc_u8, &m_u8);
        const float ds = (float) bk->d * (float) sc_u8;
        const float dm = (float) bk->dmin * (float) m_u8 * 0.25f;

        const int8_t* qs_pair = (const int8_t*) &bk->qs[(sub >> 1) * 32];
        const int vl_lo = load_int_b2(qs_pair, lane4);
        const int vl_hi = load_int_b2(qs_pair, lane4 + 4);
        const int vh_lo = load_int_b2((const int8_t*) bk->qh, lane4);
        const int vh_hi = load_int_b2((const int8_t*) bk->qh, lane4 + 4);

        const int vlq_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
        const int vlq_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);
        const int v_lo = vlq_lo | (((vh_lo >> sub) << 4) & 0x10101010);
        const int v_hi = vlq_hi | (((vh_hi >> sub) << 4) & 0x10101010);

        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = s_blk + c * SUBS + sub;
            const int u_lo = load_int_b2(ya->qs, lane4);
            const int u_hi = load_int_b2(ya->qs, lane4 + 4);
            const int sumi = dp4a(v_hi, u_hi, dp4a(v_lo, u_lo, 0));
            acc[c] += ds * (float) ya->d * (float) sumi - dm * (float) ya->s;
        }
    }
};

struct IQ4XS {
    typedef flambeau_block_iq4_xs block_t;
    template<int N>
    static __device__ __forceinline__ void step(
        const block_t* bk, const flambeau_block_q8_1* s_blk, int lane_lo, float (&acc)[N])
    {
        const int sub   = lane_lo >> 2;
        const int lane4 = lane_lo & 3;

        const float dl = (float) bk->d * (float) flambeau_iq4_xs_scale(sub, bk->scales_h, bk->scales_l);

        const int aux_q4 = load_int_b2((const int8_t*) bk->qs, sub * 4 + lane4);
        const int v_lo = (int) iq4_nl_table_4((unsigned int) aux_q4 & 0x0F0F0F0Fu);
        const int v_hi = (int) iq4_nl_table_4(((unsigned int) aux_q4 >> 4) & 0x0F0F0F0Fu);

        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = s_blk + c * SUBS + sub;
            const int u_lo = load_int_b2(ya->qs, lane4);
            const int u_hi = load_int_b2(ya->qs, lane4 + 4);
            acc[c] += dl * (float) ya->d * (float) dp4a(v_hi, u_hi, dp4a(v_lo, u_lo, 0));
        }
    }
};
}  // namespace mmvq_rt

template<class Tr, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_dp4a_rowtile_batched_row(
    const typename Tr::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    using namespace mmvq_rt;
    const int tid     = threadIdx.x;
    const int lane_lo = tid & (ROW_GROUP - 1);
    const int row     = blockIdx.x * ROWS + tid / ROW_GROUP;
    const bool row_ok = row < n_rows;
    const typename Tr::block_t* xrow = row_ok ? (x + (size_t) row * n_superblocks_per_row) : x;
    const size_t act_blocks_per_row = (size_t) n_superblocks_per_row * SUBS;

    __shared__ int s_y[N * SUBS * BLK_INTS];
    const flambeau_block_q8_1* s_blk = (const flambeau_block_q8_1*) s_y;

    float acc[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) acc[c] = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        stage_strip<N>(s_y, y, act_blocks_per_row, b);
        __syncthreads();
        if (row_ok) Tr::template step<N>(xrow + b, s_blk, lane_lo, acc);
        __syncthreads();
    }

    if (!row_ok) return;
    mmvq_reduce_store_2row_batched<N, OutT>(acc, dst, row, n_rows, lane_lo);
}
