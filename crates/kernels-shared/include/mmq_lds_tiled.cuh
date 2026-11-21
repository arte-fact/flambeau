#pragma once
// Generic 4-warp LDS-tiled MMQ (the prefill GEMM) over the DS4 144-B
// activation. One block computes MMQ_Y weight rows × MMQ_X activation columns:
// per 256-element super-block the weight tile is decoded once into LDS as
// dp4a-ready int8 lanes, then dotted against two 128-element activation tiles.
// Lanes are logical (LDS + __syncthreads only, no cross-lane shuffles), so block
// (MMQ_LDS_LANES, NWARPS) is correct on wave64 and wave32 alike.
//
// Per-dtype policy `Tr` supplies:
//   x_ints(MMQ_Y)          LDS int32 count of the decoded weight tile
//   K_STEP                 activation int32 lanes consumed per `dot` call
//   load_tiles<Y, W>(...)  decode one super-block of MMQ_Y rows into the tile
//   load_row<Y>(...)       pull K_STEP lanes + scales of row `i` into registers
//   dot(row, y_col, k01)   partial dot of that row against one activation column
// `load_row` runs once per row and is reused across all of the thread's columns,
// so the weight lanes are not re-read from LDS per column.
// Only the per-sub-block activation scale `d` (and for Q4_K/Q5_K the raw sum) of
// the DS4 header is read, so dtypes that need no bias term share the same layout.
#include "quant_traits.cuh"
#include "iq_grid.cuh"
#include "mmq_iq4_nl_wave64.cuh"
#include "mmq_q4_k_lds_tiled.cuh"
#include "mmvq_store.cuh"

namespace mmq_lds {

constexpr int LANES     = MMQ_LDS_LANES;
constexpr int NE_K      = 32;              // int32 lanes per 128-element activation tile
constexpr int Y_STRIDE  = NE_K + NE_K / 8; // 36 int32 = one 144-B DS4 block
constexpr int QS_STRIDE = 2 * NE_K + 1;    // 8-bit weight lanes per row, +1 de-aliases LDS banks

// Per-byte `a - b` for bytes of `a` and `b` both below 0x80 (no cross-byte borrow).
static __device__ __forceinline__ int sub_bytes(int a, unsigned b) {
    return (int) ((((unsigned) a | 0x80808080u) - b) ^ 0x80808080u);
}

struct Q4K {
    static constexpr int K_STEP = 16;
    static constexpr int QS4_STRIDE = NE_K + 1;   // packed 4-bit lanes per row
    static constexpr int x_ints(int mmq_y) {
        return mmq_y * QS4_STRIDE + mmq_y + mmq_y * (NE_K / 8) + mmq_y / 8;
    }

    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*       x_qs = x_tile;
        fb_fp16_t* x_dm = (fb_fp16_t*) (x_qs + MMQ_Y * QS4_STRIDE);
        int*       x_sc = (int*) (x_dm + MMQ_Y * 2);
        const flambeau_block_q4_K* x = (const flambeau_block_q4_K*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 32;
        constexpr int nrows = LANES / threads_per_row;
        const int txi = tx % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_q4_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_qs[i * QS4_STRIDE + txi] = load_int_b2((const int8_t*) bxi->qs, txi);
            }
        }
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * LANES) {
            const int i = (i0 + ty * LANES + tx) % MMQ_Y;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q4_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_dm[i * 2 + 0] = bxi->d;
                x_dm[i * 2 + 1] = bxi->dmin;
            }
        }
        constexpr int rows_per_warp_sc = LANES / 4;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i = (i0 + ty * rows_per_warp_sc + tx / 4) % MMQ_Y;
            const int ksc = tx % 4;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q4_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_sc[i * 4 + i / 8 + ksc] =
                    flambeau_unpack_scales_q45_K((const int*) bxi->scales, ksc);
            }
        }
    }

    struct Row {
        int   v[16];     // 64 weights as 4-bit lanes, low then high nibbles per 32
        float fsc[2];    // d * scale per 32-element sub-block
        float fm[2];     // dmin * min per 32-element sub-block
    };

    template<int MMQ_Y>
    static __device__ __forceinline__ Row load_row(
        const int* __restrict__ x_tile, int i, int k00, int k01)
    {
        const int*       x_qs = x_tile;
        const fb_fp16_t* x_dm = (const fb_fp16_t*) (x_qs + MMQ_Y * QS4_STRIDE);
        const int*       x_sc = (const int*) (x_dm + MMQ_Y * 2);
        const int*       q    = &x_qs[i * QS4_STRIDE + (k00 + k01) / 2];
        const uint8_t*   sc   = (const uint8_t*) &x_sc[i * 4 + i / 8 + k00 / 32] + 2 * (k01 / 16);
        const float      d    = (float) x_dm[i * 2 + 0];
        const float      dmin = (float) x_dm[i * 2 + 1];
        Row r;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            r.v[j]     = q[j]        & 0x0F0F0F0F;
            r.v[8 + j] = (q[j] >> 4) & 0x0F0F0F0F;
        }
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            r.fsc[s] = d * (float) sc[s];
            r.fm[s]  = dmin * (float) sc[8 + s];
        }
        return r;
    }

    static __device__ __forceinline__ float dot(
        const Row& r, const int* __restrict__ y_col, int k01)
    {
        const int*       u   = y_col + 4 + k01;
        const fb_fp16_t* ds8 = (const fb_fp16_t*) y_col + (k01 / 8) * 2;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) sumi = dp4a(r.v[s * 8 + j], u[s * 8 + j], sumi);
            acc += (float) ds8[s * 2 + 0] * (r.fsc[s] * (float) sumi) - (float) ds8[s * 2 + 1] * r.fm[s];
        }
        return acc;
    }
};

struct Q5K {
    static constexpr int K_STEP = 16;
    static constexpr int x_ints(int mmq_y) {
        return mmq_y * QS_STRIDE + mmq_y + mmq_y * (NE_K / 8) + mmq_y / 8;
    }

    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*       x_qs = x_tile;
        fb_fp16_t* x_dm = (fb_fp16_t*) (x_qs + MMQ_Y * QS_STRIDE);
        int*       x_sc = (int*) (x_dm + MMQ_Y * 2);
        const flambeau_block_q5_K* x = (const flambeau_block_q5_K*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 32;
        constexpr int nrows = LANES / threads_per_row;
        const int txi = tx % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_q5_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                const int ql  = load_int_b2((const int8_t*) bxi->qs, txi);
                const int ql0 = (ql >> 0) & 0x0F0F0F0F;
                const int ql1 = (ql >> 4) & 0x0F0F0F0F;
                const int qh  = load_int_b2((const int8_t*) bxi->qh, txi % 8);
                const int qh0 = ((qh >> (2 * (txi / 8) + 0)) << 4) & 0x10101010;
                const int qh1 = ((qh >> (2 * (txi / 8) + 1)) << 4) & 0x10101010;
                const int kq0 = 16 * (txi / 8) + txi % 8;
                x_qs[i * QS_STRIDE + kq0]     = ql0 | qh0;
                x_qs[i * QS_STRIDE + kq0 + 8] = ql1 | qh1;
            }
        }
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * LANES) {
            const int i = (i0 + ty * LANES + tx) % MMQ_Y;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q5_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_dm[i * 2 + 0] = bxi->d;
                x_dm[i * 2 + 1] = bxi->dmin;
            }
        }
        constexpr int rows_per_warp_sc = LANES / 4;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i = (i0 + ty * rows_per_warp_sc + tx / 4) % MMQ_Y;
            const int ksc = tx % 4;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q5_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_sc[i * 4 + i / 8 + ksc] =
                    flambeau_unpack_scales_q45_K((const int*) bxi->scales, ksc);
            }
        }
    }

    struct Row {
        int   v[16];
        float fsc[2];    // d * scale per 32-element sub-block
        float fm[2];     // dmin * min per 32-element sub-block
    };

    template<int MMQ_Y>
    static __device__ __forceinline__ Row load_row(
        const int* __restrict__ x_tile, int i, int k00, int k01)
    {
        const int*       x_qs = x_tile;
        const fb_fp16_t* x_dm = (const fb_fp16_t*) (x_qs + MMQ_Y * QS_STRIDE);
        const int*       x_sc = (const int*) (x_dm + MMQ_Y * 2);
        const int*       q    = &x_qs[i * QS_STRIDE + k00 + k01];
        const uint8_t*   sc   = (const uint8_t*) &x_sc[i * 4 + i / 8 + k00 / 32] + 2 * (k01 / 16);
        const float      d    = (float) x_dm[i * 2 + 0];
        const float      dmin = (float) x_dm[i * 2 + 1];
        Row r;
        #pragma unroll
        for (int j = 0; j < 16; ++j) r.v[j] = q[j];
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            r.fsc[s] = d * (float) sc[s];
            r.fm[s]  = dmin * (float) sc[8 + s];
        }
        return r;
    }

    static __device__ __forceinline__ float dot(
        const Row& r, const int* __restrict__ y_col, int k01)
    {
        const int*       u   = y_col + 4 + k01;
        const fb_fp16_t* ds8 = (const fb_fp16_t*) y_col + (k01 / 8) * 2;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) sumi = dp4a(r.v[s * 8 + j], u[s * 8 + j], sumi);
            acc += (float) ds8[s * 2 + 0] * (r.fsc[s] * (float) sumi) - (float) ds8[s * 2 + 1] * r.fm[s];
        }
        return acc;
    }
};

struct Q6K {
    static constexpr int K_STEP = 16;
    static constexpr int x_ints(int mmq_y) {
        return mmq_y * QS_STRIDE + mmq_y + mmq_y / 32 + mmq_y * (NE_K / 8) + mmq_y / 8;
    }

    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*   x_qs = x_tile;
        float* x_df = (float*) (x_qs + MMQ_Y * QS_STRIDE);
        int*   x_sc = (int*) (x_df + MMQ_Y + MMQ_Y / 32);
        const flambeau_block_q6_K* x = (const flambeau_block_q6_K*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 32;
        constexpr int nrows = LANES / threads_per_row;
        const int txi = tx % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_q6_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                const int ql  = load_int_b2((const int8_t*) bxi->ql, txi);
                const int ql0 = (ql >> 0) & 0x0F0F0F0F;
                const int ql1 = (ql >> 4) & 0x0F0F0F0F;
                const int qh  = load_int_b2((const int8_t*) bxi->qh, 8 * (txi / 16) + txi % 8);
                const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
                const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;
                const int kq0 = 2 * txi - txi % 16;
                x_qs[i * QS_STRIDE + kq0]      = sub_bytes(ql0 | qh0, 0x20202020u);
                x_qs[i * QS_STRIDE + kq0 + 16] = sub_bytes(ql1 | qh1, 0x20202020u);
            }
        }
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * LANES) {
            const int i = (i0 + ty * LANES + tx) % MMQ_Y;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q6_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_df[i + i / 32] = (float) bxi->d;
            }
        }
        constexpr int rows_per_warp_sc = LANES / 4;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i = (i0 + ty * rows_per_warp_sc + tx / 4) % MMQ_Y;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q6_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_sc[i * 4 + i / 8 + tx % 4] = load_int_b2(bxi->scales, tx % 4);
            }
        }
    }

    struct Row {
        int   v[16];
        int   sc;        // 4 signed scales, one per 16 elements
        float d;
    };

    template<int MMQ_Y>
    static __device__ __forceinline__ Row load_row(
        const int* __restrict__ x_tile, int i, int k00, int k01)
    {
        const int*   x_qs = x_tile;
        const float* x_df = (const float*) (x_qs + MMQ_Y * QS_STRIDE);
        const int*   x_sc = (const int*) (x_df + MMQ_Y + MMQ_Y / 32);
        const int    k0   = k00 + k01;
        const int*   q    = &x_qs[i * QS_STRIDE + k0];
        Row r;
        #pragma unroll
        for (int j = 0; j < 16; ++j) r.v[j] = q[j];
        r.sc = x_sc[i * 4 + i / 8 + k0 / 16];
        r.d  = x_df[i + i / 32];
        return r;
    }

    static __device__ __forceinline__ float dot(
        const Row& r, const int* __restrict__ y_col, int k01)
    {
        const int*       u  = y_col + 4 + k01;
        const fb_fp16_t* d8 = (const fb_fp16_t*) y_col + (k01 / 8) * 2;
        float sumf_d = 0.0f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int sumi_lo = 0;
            int sumi_hi = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                sumi_lo = dp4a(r.v[s * 8 + j],     u[s * 8 + j],     sumi_lo);
                sumi_hi = dp4a(r.v[s * 8 + 4 + j], u[s * 8 + 4 + j], sumi_hi);
            }
            const int sc_lo = (int) (int8_t) (r.sc >> (16 * s));
            const int sc_hi = (int) (int8_t) (r.sc >> (16 * s + 8));
            sumf_d += (float) d8[s * 2] * (float) (sc_lo * sumi_lo + sc_hi * sumi_hi);
        }
        return r.d * sumf_d;
    }
};

struct Q3K {
    static constexpr int K_STEP = 8;
    static constexpr int x_ints(int mmq_y) {
        return mmq_y * QS_STRIDE + mmq_y + mmq_y * (NE_K / 8) + mmq_y / 8;
    }

    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*   x_qs = x_tile;
        float* x_df = (float*) (x_qs + MMQ_Y * QS_STRIDE);
        int*   x_sc = (int*) (x_df + MMQ_Y);
        const flambeau_block_q3_K* x = (const flambeau_block_q3_K*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 16;
        constexpr int nrows = LANES / threads_per_row;
        const int kqsx = tx % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_q3_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                const int ql = load_int_b2((const int8_t*) bxi->qs, kqsx);
                const int qh = load_int_b2((const int8_t*) bxi->hmask, kqsx % 8) >> (4 * (kqsx / 8));
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
                    const int ql_k =  (ql >> (2 * l))       & 0x03030303;
                    const int qh_k = ((qh >>      l)  << 2) & 0x04040404;
                    x_qs[i * QS_STRIDE + k] = sub_bytes(ql_k | qh_k, 0x04040404u);
                }
            }
        }
        constexpr int rows_per_warp_sc = LANES / 4;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i = (i0 + ty * rows_per_warp_sc + tx / 4) % MMQ_Y;
            const int ksc = tx % 4;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q3_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                const int8_t* scales = (const int8_t*) bxi->scales;
                const int sc_low  = (load_int_b2(scales, ksc % 2) >> (4 * (ksc / 2))) & 0x0F0F0F0F;
                const int sc_high = ((load_int_b2(scales, 2) >> (2 * ksc)) << 4) & 0x30303030;
                x_sc[i * 4 + i / 8 + ksc] = sub_bytes(sc_low | sc_high, 0x20202020u);
            }
        }
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * LANES) {
            const int i = (i0 + ty * LANES + tx) % MMQ_Y;
            const int row = tile_m + i;
            if (row < nrows_x) {
                const flambeau_block_q3_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                x_df[i] = (float) bxi->d;
            }
        }
    }

    struct Row {
        int   v[8];
        int   sc[2];     // signed scale per 16 elements
        float d;
    };

    template<int MMQ_Y>
    static __device__ __forceinline__ Row load_row(
        const int* __restrict__ x_tile, int i, int k00, int k01)
    {
        const int*    x_qs = x_tile;
        const float*  x_df = (const float*) (x_qs + MMQ_Y * QS_STRIDE);
        const int*    x_sc = (const int*) (x_df + MMQ_Y);
        const int     k0   = k00 + k01;
        const int*    q    = &x_qs[i * QS_STRIDE + k0];
        const int8_t* sc   = (const int8_t*) (x_sc + i * 4 + i / 8) + k0 / 4;
        Row r;
        #pragma unroll
        for (int j = 0; j < 8; ++j) r.v[j] = q[j];
        r.sc[0] = sc[0];
        r.sc[1] = sc[1];
        r.d = x_df[i];
        return r;
    }

    static __device__ __forceinline__ float dot(
        const Row& r, const int* __restrict__ y_col, int k01)
    {
        const int*  u  = y_col + 4 + k01;
        const float d8 = (float) ((const fb_fp16_t*) y_col)[(k01 / 8) * 2];
        int sumi = 0;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int sumi_sc = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) sumi_sc = dp4a(r.v[s * 4 + j], u[s * 4 + j], sumi_sc);
            sumi += sumi_sc * r.sc[s];
        }
        return r.d * d8 * (float) sumi;
    }
};

// Shared tile layout + dot for dtypes decoded to plain int8 lanes with one
// float scale per 32 elements (8 per row per super-block iteration).
struct Q8Style {
    static constexpr int K_STEP = 8;
    static constexpr int x_ints(int mmq_y) {
        return mmq_y * QS_STRIDE + mmq_y * 8 + mmq_y / 4;
    }

    struct Row {
        int   v[8];
        float d;
    };

    template<int MMQ_Y>
    static __device__ __forceinline__ Row load_row(
        const int* __restrict__ x_tile, int i, int k00, int k01)
    {
        const int*   x_qs = x_tile;
        const float* x_df = (const float*) (x_qs + MMQ_Y * QS_STRIDE);
        const int    k0   = k00 + k01;
        const int*   q    = &x_qs[i * QS_STRIDE + k0];
        Row r;
        #pragma unroll
        for (int j = 0; j < 8; ++j) r.v[j] = q[j];
        r.d = x_df[i * 8 + i / 4 + k0 / 8];
        return r;
    }

    static __device__ __forceinline__ float dot(
        const Row& r, const int* __restrict__ y_col, int k01)
    {
        const int*  u  = y_col + 4 + k01;
        const float d8 = (float) ((const fb_fp16_t*) y_col)[(k01 / 8) * 2];
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sumi = dp4a(r.v[j], u[j], sumi);
        return r.d * d8 * (float) sumi;
    }
};

struct IQ4XS : Q8Style {
    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*   x_qs = x_tile;
        float* x_df = (float*) (x_qs + MMQ_Y * QS_STRIDE);
        const flambeau_block_iq4_xs* x = (const flambeau_block_iq4_xs*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 32;
        constexpr int nrows = LANES / threads_per_row;
        const int kqsx = tx % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_iq4_xs* bxi = x + kb0 + (size_t) row * stride_row_x;
                const int word = load_int_b2((const int8_t*) bxi->qs, kqsx);
                const int k0 = 8 * (kqsx / 4) + kqsx % 4;
                x_qs[i * QS_STRIDE + k0]     = flambeau_pack_iq4_lut(word        & 0x0F0F0F0F);
                x_qs[i * QS_STRIDE + k0 + 4] = flambeau_pack_iq4_lut((word >> 4) & 0x0F0F0F0F);
            }
        }
        constexpr int rows_per_warp_df = LANES / 8;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_df) {
            const int i = i0 + ty * rows_per_warp_df + tx / 8;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_iq4_xs* bxi = x + kb0 + (size_t) row * stride_row_x;
                const int sub = tx % 8;
                const int ls = flambeau_iq4_xs_scale(sub, bxi->scales_h, bxi->scales_l);
                x_df[i * 8 + i / 4 + sub] = (float) bxi->d * (float) ls;
            }
        }
    }
};

// 32-element weight blocks: `kb0` still counts 256-element iterations, so the
// block index is `kb0 * 8 + kbx` and `stride_row_x` is in 32-element blocks.
struct IQ4NL : Q8Style {
    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*   x_qs = x_tile;
        float* x_df = (float*) (x_qs + MMQ_Y * QS_STRIDE);
        const flambeau_block_iq4_nl* x = (const flambeau_block_iq4_nl*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 32;
        constexpr int nrows = LANES / threads_per_row;
        const int txi  = tx % threads_per_row;
        const int kbx  = txi / 4;
        const int kqsx = txi % 4;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_iq4_nl* bxi = x + kb0 * 8 + (size_t) row * stride_row_x + kbx;
                const int word = load_int_b2((const int8_t*) bxi->qs, kqsx);
                const int k0 = kbx * 8 + kqsx;
                x_qs[i * QS_STRIDE + k0]     = flambeau_pack_iq4_lut(word        & 0x0F0F0F0F);
                x_qs[i * QS_STRIDE + k0 + 4] = flambeau_pack_iq4_lut((word >> 4) & 0x0F0F0F0F);
            }
        }
        constexpr int rows_per_warp_df = LANES / 8;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_df) {
            const int i = i0 + ty * rows_per_warp_df + tx / 8;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const int kbxd = tx % 8;
                const flambeau_block_iq4_nl* bxi = x + kb0 * 8 + (size_t) row * stride_row_x + kbxd;
                x_df[i * 8 + i / 4 + kbxd] = (float) bxi->d;
            }
        }
    }
};

struct Q8_0 : Q8Style {
    template<int MMQ_Y, int NWARPS>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ vx, int* __restrict__ x_tile,
        int kb0, int tile_m, int nrows_x, int stride_row_x)
    {
        int*   x_qs = x_tile;
        float* x_df = (float*) (x_qs + MMQ_Y * QS_STRIDE);
        const flambeau_block_q8_0* x = (const flambeau_block_q8_0*) vx;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        constexpr int threads_per_row = 32;
        constexpr int nrows = LANES / threads_per_row;
        const int txi  = tx % threads_per_row;
        const int kbx  = txi / 8;
        const int kqsx = txi % 8;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + ty * nrows + tx / threads_per_row;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const flambeau_block_q8_0* bxi = x + kb0 * 8 + (size_t) row * stride_row_x + kbx;
                x_qs[i * QS_STRIDE + txi]        = load_int_b2(bxi[0].qs, kqsx);
                x_qs[i * QS_STRIDE + NE_K + txi] = load_int_b2(bxi[4].qs, kqsx);
            }
        }
        constexpr int rows_per_warp_df = LANES / 8;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_df) {
            const int i = i0 + ty * rows_per_warp_df + tx / 8;
            const int row = tile_m + i;
            if (i < MMQ_Y && row < nrows_x) {
                const int kbxd = tx % 8;
                const flambeau_block_q8_0* bxi = x + kb0 * 8 + (size_t) row * stride_row_x + kbxd;
                x_df[i * 8 + i / 4 + kbxd] = (float) bxi->d;
            }
        }
    }
};

}  // namespace mmq_lds

// Dynamic LDS bytes = 4 * (MMQ_X * mmq_lds::Y_STRIDE + Tr::x_ints(MMQ_Y)).
template<class Tr, int MMQ_X, int MMQ_Y, int NWARPS, class OutT = float>
static __device__ __forceinline__ void mmq_lds_tiled_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y,
    int stride_row_x, int nrows_dst)
{
    constexpr int LANES    = mmq_lds::LANES;
    constexpr int NE_K     = mmq_lds::NE_K;
    constexpr int Y_STRIDE = mmq_lds::Y_STRIDE;
    constexpr int ROW_GRPS = (MMQ_Y + LANES - 1) / LANES;

    extern __shared__ int lds[];
    int* tile_y = lds;
    int* tile_x = tile_y + MMQ_X * Y_STRIDE;

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * MMQ_X;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    constexpr int n_sum = (MMQ_X / NWARPS) * ROW_GRPS;
    float sum[n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) sum[s] = 0.0f;

    const int* y_int_tile = (const int*) vy + tile_n * Y_STRIDE;
    const int kb0_stop = ncols_x / 256;

    for (int kb0 = 0; kb0 < kb0_stop; ++kb0) {
        Tr::template load_tiles<MMQ_Y, NWARPS>(vx, tile_x, kb0, tile_m, nrows_x, stride_row_x);

        #pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int k00 = phase * NE_K;
            const int* by0 = y_int_tile + (size_t) ncols_y * (kb0 * 2 + phase) * Y_STRIDE;
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * Y_STRIDE; l0 += NWARPS * LANES) {
                const int l = l0 + ty * LANES + tx;
                if (l < MMQ_X * Y_STRIDE) tile_y[l] = by0[l];
            }
            __syncthreads();

            #pragma nounroll
            for (int k01 = 0; k01 < NE_K; k01 += Tr::K_STEP) {
                #pragma unroll
                for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
                    const int i = i0 + tx;
                    if (i < MMQ_Y) {
                        const typename Tr::Row row = Tr::template load_row<MMQ_Y>(tile_x, i, k00, k01);
                        #pragma unroll
                        for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                            sum[j0 / NWARPS * ROW_GRPS + i0 / LANES] +=
                                Tr::dot(row, tile_y + (j0 + ty) * Y_STRIDE, k01);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int col = tile_n + j0 + ty;
        if (col >= ncols_y) continue;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
            const int row = tile_m + i0 + tx;
            if (i0 + tx < MMQ_Y && row < nrows_x && row < nrows_dst) {
                mmvq_store<OutT>(dst, (int) ((size_t) col * nrows_dst + row),
                    sum[j0 / NWARPS * ROW_GRPS + i0 / LANES]);
            }
        }
    }
}
