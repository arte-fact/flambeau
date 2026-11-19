#pragma once
// Row-tiled batched dp4a MMVQ for the 256-element super-block dtypes (Q4_K,
// Q5_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S), N activation columns. 256 threads = 8 output rows per block,
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
#include "mmvq_iq3_s.cuh"
#include "mmvq_iq3_xxs.cuh"
#include "mmvq_iq2_s.cuh"

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
// Expand the low / high nibble of a sign byte into per-byte 0xFF masks.
static __device__ __forceinline__ void sign_packs(
    unsigned int signs_byte, unsigned int& lo, unsigned int& hi)
{
    lo = 0; hi = 0;
    lo |= (signs_byte & 0x01) ? 0x000000FFu : 0;
    lo |= (signs_byte & 0x02) ? 0x0000FF00u : 0;
    lo |= (signs_byte & 0x04) ? 0x00FF0000u : 0;
    lo |= (signs_byte & 0x08) ? 0xFF000000u : 0;
    hi |= (signs_byte & 0x10) ? 0x000000FFu : 0;
    hi |= (signs_byte & 0x20) ? 0x0000FF00u : 0;
    hi |= (signs_byte & 0x40) ? 0x00FF0000u : 0;
    hi |= (signs_byte & 0x80) ? 0xFF000000u : 0;
}

// The grid i-quants share one lane layout: lane = (sub 0..7) × (pair 0..3),
// covering ints `2·pair` and `2·pair + 1` of the sub-block. `sumi` of column
// `c` over those two ints, for signed grid words `g0` / `g1`.
static __device__ __forceinline__ int grid_sumi(
    const flambeau_block_q8_1* ya, int l0, int g0, int g1)
{
    const int u0 = load_int_b2(ya->qs, l0);
    const int u1 = load_int_b2(ya->qs, l0 + 1);
    return dp4a(g1, u1, dp4a(g0, u0, 0));
}

struct IQ3S {
    typedef flambeau_block_iq3_s block_t;
    template<int N>
    static __device__ __forceinline__ void step(
        const block_t* bk, const flambeau_block_q8_1* s_blk, int lane_lo, float (&acc)[N])
    {
        const int sub   = lane_lo >> 2;
        const int l_grp = lane_lo & 3;
        const int iqs   = 2 * sub;
        const int l0    = 2 * l_grp;

        const uint8_t* qs_base = bk->qs + 4 * iqs;
        const unsigned int qh = bk->qh[sub];
        const int g_idx0 = (int) qs_base[l0] | (((int) qh << (8 - l0)) & 0x100);
        const int g_idx1 = (int) qs_base[l0 + 1] | (((int) qh << (7 - l0)) & 0x100);

        unsigned int s0, s1;
        sign_packs(bk->signs[4 * sub + l_grp], s0, s1);
        const int g0 = iq2xxs_apply_signs(IQ3S_GRID[g_idx0], s0);
        const int g1 = iq2xxs_apply_signs(IQ3S_GRID[g_idx1], s1);

        const int sc_byte = bk->scales[iqs >> 2];
        const int scale   = 1 + 2 * ((sc_byte >> ((iqs << 1) & 0x04)) & 0x0F);
        const float d_sb  = (float) bk->d;

        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = s_blk + c * SUBS + sub;
            acc[c] += d_sb * (float) ya->d * (float) (grid_sumi(ya, l0, g0, g1) * scale);
        }
    }
};

struct IQ3XXS {
    typedef flambeau_block_iq3_xxs block_t;
    template<int N>
    static __device__ __forceinline__ void step(
        const block_t* bk, const flambeau_block_q8_1* s_blk, int lane_lo, float (&acc)[N])
    {
        const int sub   = lane_lo >> 2;
        const int l_grp = lane_lo & 3;
        const int iqs   = 2 * sub;
        const int l0    = 2 * l_grp;

        const uint8_t* q3_base = bk->qs + 4 * iqs;
        const unsigned int aux32 =
            (unsigned int) load_int_b2((const int8_t*) (bk->qs + 64 + 2 * iqs), 0);
        const int ls = (int) (aux32 >> 28);
        const unsigned int signs_full = iq2xxs_unpack_ksigns(aux32 >> (7 * l_grp));

        const unsigned int sel_lo = signs_full & 0x08040201u;
        const unsigned int sel_hi = signs_full & 0x80402010u;
        unsigned int s0 = 0, s1 = 0;
        s0 |= ((sel_lo >> 0) & 0xFFu) ? 0x000000FFu : 0;
        s0 |= ((sel_lo >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        s0 |= ((sel_lo >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        s0 |= ((sel_lo >> 24) & 0xFFu) ? 0xFF000000u : 0;
        s1 |= ((sel_hi >> 0) & 0xFFu) ? 0x000000FFu : 0;
        s1 |= ((sel_hi >> 8) & 0xFFu) ? 0x0000FF00u : 0;
        s1 |= ((sel_hi >> 16) & 0xFFu) ? 0x00FF0000u : 0;
        s1 |= ((sel_hi >> 24) & 0xFFu) ? 0xFF000000u : 0;

        const int g0 = iq2xxs_apply_signs(IQ3XXS_GRID[q3_base[l0]], s0);
        const int g1 = iq2xxs_apply_signs(IQ3XXS_GRID[q3_base[l0 + 1]], s1);
        const float d_sb = (float) bk->d;

        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = s_blk + c * SUBS + sub;
            const int sumi = grid_sumi(ya, l0, g0, g1);
            acc[c] += d_sb * (float) ya->d * (float) ((ls * sumi + sumi / 2) / 2);
        }
    }
};

struct IQ2S {
    typedef flambeau_block_iq2_s block_t;
    template<int N>
    static __device__ __forceinline__ void step(
        const block_t* bk, const flambeau_block_q8_1* s_blk, int lane_lo, float (&acc)[N])
    {
        const int sub   = lane_lo >> 2;
        const int l_grp = lane_lo & 3;
        const int iqs   = 2 * sub;
        const int l0    = 2 * l_grp;

        const int qh = (int) bk->qh[sub];
        const int grid_idx = (int) bk->qs[2 * iqs + l_grp] | ((qh << (8 - l0)) & 0x300);

        unsigned int s0, s1;
        sign_packs(bk->qs[32 + 2 * iqs + l_grp], s0, s1);
        const uint64_t grid_u64 = IQ2S_GRID[grid_idx];
        const int g0 = iq2xxs_apply_signs((unsigned int) grid_u64, s0);
        const int g1 = iq2xxs_apply_signs((unsigned int) (grid_u64 >> 32), s1);

        const int sc_byte = (int) bk->scales[sub >> 1];
        const int ls      = (sub & 1) ? (sc_byte >> 4) : (sc_byte & 0x0F);
        const float dls   = (float) bk->d * ((float) ls + 0.5f) * 0.25f;

        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = s_blk + c * SUBS + sub;
            acc[c] += dls * (float) ya->d * (float) grid_sumi(ya, l0, g0, g1);
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
