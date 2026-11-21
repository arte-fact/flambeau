#pragma once
// Batched (N = 1..8 activation columns) single-row MMVQ in llama.cpp's CUDA
// layout (`mul_mat_vec_q`, ncols_dst = N): one output row per block, the
// weight-block split per thread of `vec_dot_<type>_q8_1` (VDR), each thread
// decoding its weight ints once and dotting them against the N columns, N
// accumulators reduced together at the end. Columns are slot-major: column c
// of `y` starts at `y + c * blocks_per_col` Q8_1 blocks, `dst[c * n_rows + row]`.
// A trait gives the weight block type, the threads per weight block, the Q8_1
// blocks per weight block, and `step<N>` = one weight block's contribution.
// Backend-neutral: dp4a / byte_perm / the LUT and reduce come from the arch
// header and shared headers.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"
#include "mmvq_iq4_nl.cuh"

namespace mmvq_vdr {

// IQ4_XS, VDR 4: 8 threads per super-block, one 32-weight sub-block each.
struct Iq4Xs {
    typedef flambeau_block_iq4_xs block_t;
    static constexpr int TPB = 8;
    static constexpr int YPB = 8;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        int lo[R][4], hi[R][4];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                iq4_nl_table_8((unsigned int) load_int_b4(bk[r]->qs, 4 * lane + j), lo[r][j], hi[r][j]);
            }
            d[r] = (float) bk[r]->d * (float) flambeau_iq4_xs_scale(lane, bk[r]->scales_h, bk[r]->scales_l);
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = y + c * ys + lane;
            int u[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) u[j] = load_int_b4(ya->qs, j);
            const float dy = (float) ya->d;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi = dp4a(lo[r][j], u[j], sumi);
                    sumi = dp4a(hi[r][j], u[j + 4], sumi);
                }
                acc[r][c] += d[r] * dy * (float) sumi;
            }
        }
    }
};

template<bool Q5> struct QkBlock { typedef flambeau_block_q4_K type; };
template<> struct QkBlock<true> { typedef flambeau_block_q5_K type; };

// Q4_K / Q5_K, VDR 2: 16 threads per super-block; a thread loads 2 ints of
// one sub-block pair's `qs` (and `qh` for Q5_K) and covers both nibbles.
template<bool Q5>
struct QkVdr2 {
    typedef typename QkBlock<Q5>::type block_t;
    static constexpr int TPB = 16;
    static constexpr int YPB = 8;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        const int pair   = lane >> 2;
        const int k      = lane & 3;
        const int sub_lo = 2 * pair;
        int v[R][2][2];
        uint8_t sc[R][2], mn[R][2];
        float d[R], dmin[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const int vl0 = load_int_b4(bk[r]->qs + 32 * pair, k);
            const int vl1 = load_int_b4(bk[r]->qs + 32 * pair, k + 4);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                v[r][i][0] = (vl0 >> (4 * i)) & 0x0F0F0F0F;
                v[r][i][1] = (vl1 >> (4 * i)) & 0x0F0F0F0F;
            }
            if constexpr (Q5) {
                const int vh0 = load_int_b4(bk[r]->qh, k) >> sub_lo;
                const int vh1 = load_int_b4(bk[r]->qh, k + 4) >> sub_lo;
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    v[r][i][0] |= ((vh0 >> i) << 4) & 0x10101010;
                    v[r][i][1] |= ((vh1 >> i) << 4) & 0x10101010;
                }
            }
            flambeau_q4k_scale_min(sub_lo,     bk[r]->scales, &sc[r][0], &mn[r][0]);
            flambeau_q4k_scale_min(sub_lo + 1, bk[r]->scales, &sc[r][1], &mn[r][1]);
            d[r]    = (float) bk[r]->d;
            dmin[r] = (float) bk[r]->dmin;
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            int u[2][2];
            float d8[2];
            int dot_1[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const flambeau_block_q8_1* ya = y + c * ys + sub_lo + i;
                u[i][0] = load_int_b4(ya->qs, k);
                u[i][1] = load_int_b4(ya->qs, k + 4);
                d8[i] = (float) ya->d;
                dot_1[i] = dp4a(0x01010101, u[i][1], dp4a(0x01010101, u[i][0], 0));
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float sumf_d = 0.0f;
                float sumf_m = 0.0f;
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int dot_q = dp4a(v[r][i][1], u[i][1], dp4a(v[r][i][0], u[i][0], 0));
                    sumf_d += d8[i] * (float) (dot_q * sc[r][i]);
                    sumf_m += d8[i] * (float) (dot_1[i] * mn[r][i]);
                }
                acc[r][c] += d[r] * sumf_d - dmin[r] * sumf_m;
            }
        }
    }
};

// Q6_K, VDR 2 (`vec_dot_q6_K_q8_1`): 16 threads per super-block, 2 ints of
// `ql` / `qh` each over 2 planes; the -32 bias is a dp4a against 0xE0 bytes.
struct Q6K {
    typedef flambeau_block_q6_K block_t;
    static constexpr int TPB = 16;
    static constexpr int YPB = 8;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        constexpr int QI6_K = 32;
        constexpr int QR6_K = 2;
        constexpr int QI8_1 = 8;
        const int iqs          = 2 * lane;
        const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        const int qh_idx       = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
        int v[R][QR6_K][2];
        int sc[R][QR6_K];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int vl[2], vh[2];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                vl[j] = load_int_b2((const int8_t*) bk[r]->ql, iqs + j);
                vh[j] = load_int_b2((const int8_t*) bk[r]->qh, qh_idx + j) >> vh_shift;
            }
            #pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                sc[r][i] = bk[r]->scales[scale_offset + 4 * i];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    v[r][i][j] = ((vl[j] >> (4 * i)) & 0x0F0F0F0F) | (((vh[j] >> (4 * i)) << 4) & 0x30303030);
                }
            }
            d[r] = (float) bk[r]->d;
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            int u[QR6_K][2];
            float d8[QR6_K];
            #pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                const flambeau_block_q8_1* ya = y + c * ys + bq8_offset + 2 * i;
                #pragma unroll
                for (int j = 0; j < 2; ++j) u[i][j] = load_int_b4(ya->qs, iqs % QI8_1 + j);
                d8[i] = (float) ya->d;
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float sumf = 0.0f;
                #pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    int sumi = 0;
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        sumi = dp4a(v[r][i][j], u[i][j], sumi);
                        sumi = dp4a((int) 0xE0E0E0E0, u[i][j], sumi);
                    }
                    sumf += d8[i] * (float) (sumi * sc[r][i]);
                }
                acc[r][c] += d[r] * sumf;
            }
        }
    }
};

// Q3_K, VDR 2 (`vec_dot_q3_K_q8_1`): 8 threads per super-block, 2 ints of
// `qs` / `hmask` over 4 two-bit planes; the -4 bias is a dp4a against 0xFC bytes.
struct Q3K {
    typedef flambeau_block_q3_K block_t;
    static constexpr int TPB = 8;
    static constexpr int YPB = 8;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        constexpr int QI3_K = 16;
        constexpr int QR3_K = 4;
        constexpr int QI8_1 = 8;
        const int iqs        = 2 * lane;
        const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
        const int iqs8       = iqs % QI8_1;
        const int sc_shift   = 8 * (iqs8 / (QI8_1 / 2)) + 4 * (iqs / (QI3_K / 2));
        int v[R][QR3_K][2];
        int sc[R][QR3_K];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int vl[2], vh[2];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                vl[j] = load_int_b2((const int8_t*) bk[r]->qs, iqs + j);
                vh[j] = load_int_b2((const int8_t*) bk[r]->hmask, iqs8 + j) >> bq8_offset;
            }
            int scales[3];
            #pragma unroll
            for (int l = 0; l < 3; ++l) {
                scales[l] = load_int_b2((const int8_t*) bk[r]->scales, l) >> sc_shift;
            }
            #pragma unroll
            for (int i = 0; i < QR3_K; ++i) {
                const int sc_low  = (scales[i / 2] >> (16 * (i % 2))) & 0x0F;
                const int sc_high = (scales[2] >> (16 * (i % 2) + 2 * (i / 2))) & 0x03;
                sc[r][i] = (sc_low | (sc_high << 4)) - 32;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    v[r][i][j] = ((vl[j] >> (2 * i)) & 0x03030303) | (((vh[j] >> i) << 2) & 0x04040404);
                }
            }
            d[r] = (float) bk[r]->d;
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            int u[QR3_K][2];
            float d8[QR3_K];
            #pragma unroll
            for (int i = 0; i < QR3_K; ++i) {
                const flambeau_block_q8_1* ya = y + c * ys + bq8_offset + i;
                #pragma unroll
                for (int j = 0; j < 2; ++j) u[i][j] = load_int_b4(ya->qs, iqs8 + j);
                d8[i] = (float) ya->d;
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float sumf = 0.0f;
                #pragma unroll
                for (int i = 0; i < QR3_K; ++i) {
                    int sumi = 0;
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        sumi = dp4a(v[r][i][j], u[i][j], sumi);
                        sumi = dp4a((int) 0xFCFCFCFC, u[i][j], sumi);
                    }
                    sumf += d8[i] * (float) (sumi * sc[r][i]);
                }
                acc[r][c] += d[r] * sumf;
            }
        }
    }
};

// IQ4_NL, VDR 2: 2 threads per 32-element block, 2 ints of `qs` each.
struct Iq4Nl {
    typedef flambeau_block_iq4_nl block_t;
    static constexpr int TPB = 2;
    static constexpr int YPB = 1;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        const int iqs = 2 * lane;
        int lo[R][2], hi[R][2];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                iq4_nl_table_8((unsigned int) load_int_b2((const int8_t*) bk[r]->qs, iqs + l), lo[r][l], hi[r][l]);
            }
            d[r] = (float) bk[r]->d;
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = y + c * ys;
            int u[4];
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                u[l]     = load_int_b4(ya->qs, iqs + l);
                u[l + 2] = load_int_b4(ya->qs, iqs + l + 4);
            }
            const float dy = (float) ya->d;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int sumi = 0;
                #pragma unroll
                for (int l = 0; l < 2; ++l) {
                    sumi = dp4a(lo[r][l], u[l], sumi);
                    sumi = dp4a(hi[r][l], u[l + 2], sumi);
                }
                acc[r][c] += d[r] * dy * (float) sumi;
            }
        }
    }
};

// Q8_0, VDR 2 (`vec_dot_q8_0_q8_1`): 4 threads per 32-element block.
struct Q8_0 {
    typedef flambeau_block_q8_0 block_t;
    static constexpr int TPB = 4;
    static constexpr int YPB = 1;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        const int iqs = 2 * lane;
        int v[R][2];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            v[r][0] = load_int_b2(bk[r]->qs, iqs);
            v[r][1] = load_int_b2(bk[r]->qs, iqs + 1);
            d[r] = (float) bk[r]->d;
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = y + c * ys;
            const int u0 = load_int_b4(ya->qs, iqs);
            const int u1 = load_int_b4(ya->qs, iqs + 1);
            const float dy = (float) ya->d;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r][c] += d[r] * dy * (float) dp4a(v[r][1], u1, dp4a(v[r][0], u0, 0));
            }
        }
    }
};

}  // namespace mmvq_vdr

// One block of G threads (G <= the hardware warp) owns R output rows: each
// weight block of the R rows is decoded once and each activation column read
// once for the R rows (the llama.cpp `rows_per_cuda_block` amortisation); the
// reduce is a G-wide butterfly, no LDS or barrier. `n_blocks_per_row` is in
// weight blocks; a column holds n_blocks_per_row * Tr::YPB Q8_1 blocks.
template<class Tr, int N, int G, int R, class OutT>
static __device__ __forceinline__ void mmvq_vdr_batched_row(
    const typename Tr::block_t* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_blocks_per_row)
{
    constexpr int BLK_PER_ITER = G / Tr::TPB;

    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x * R;
    const int lane = tid % Tr::TPB;
    const size_t ys = (size_t) n_blocks_per_row * Tr::YPB;

    const typename Tr::block_t* xrow[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r < n_rows ? row0 + r : n_rows - 1;
        xrow[r] = x + (size_t) row * n_blocks_per_row;
    }

    float acc[R][N];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < N; ++c) acc[r][c] = 0.0f;
    }

    for (int b = tid / Tr::TPB; b < n_blocks_per_row; b += BLK_PER_ITER) {
        const typename Tr::block_t* bk[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) bk[r] = xrow[r] + b;
        Tr::template step<N, R>(bk, y + (size_t) b * Tr::YPB, ys, lane, acc);
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            #pragma unroll
            for (int off = G / 2; off > 0; off >>= 1) {
                acc[r][c] += __shfl_xor(acc[r][c], off, G);
            }
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r < n_rows) {
                #pragma unroll
                for (int c = 0; c < N; ++c) mmvq_store<OutT>(dst, c * n_rows + row0 + r, acc[r][c]);
            }
        }
    }
}
