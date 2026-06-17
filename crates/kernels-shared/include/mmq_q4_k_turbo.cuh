#pragma once
// Templated Q4_K 4-warp LDS-tiled MMQ (turbo prefill kernel), DS4 144-B
// activation. MMQ_Y=128 weight rows × MMQ_X=16 activation cols, double-buffered
// Y per half-super-block. Port of llamacpp-turbo's load_tiles_q4_K /
// vec_dot_q4_K_q8_1_impl_mmq. Same logical-lanes structure as the Q4_0/Q4_1
// 4warp_lds (the "wave64" is a logical 64-wide tile — LDS + __syncthreads only,
// NO __shfl — so block (LANES, NWARPS)=(64,4) is correct on wave64 AND wave32).
// Backend-neutral: `dp4a` from the shared headers; the half2 (d,dmin)/(d_y,Σx)
// pairs are stored/read as two `fb_fp16_t`; int loads via `load_int_b2`. Q4_K
// 6-bit scale unpack (`unpack_scales_q45_K`) is pure int ops. `OutT` is the dest.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

#ifndef MMQ_LDS_LANES
#define MMQ_LDS_LANES 64
#endif

// Turbo's unpack_scales_q45_K — 8 packed 6-bit (scale, min) → int32 lane.
static __device__ __forceinline__ int flambeau_unpack_scales_q45_K(const int* scales, int ksc) {
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F)
         | ((scales[ksc / 2]               >> (2 * (ksc % 2)))          & 0x30303030);
}

// Q4_K × DS4-Q8_1 sub-block dot. d4/dmin4 = super-block (d, dmin); ds8 = 2×(d_y,
// Σx) fp16 pairs for the two Q8_1 sub-blocks covered.
static __device__ __forceinline__ float flambeau_vec_dot_q4_k_mmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
    float d4, float dmin4, const fb_fp16_t* __restrict__ ds8)
{
    constexpr int QR4_K = 2;
    constexpr int VDR = 8;
    constexpr int QI8_1 = 8;
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR4_K * VDR / QI8_1; ++i) {
        int sumi_d = 0;
        #pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * QI8_1 + j], sumi_d);
        }
        const float ds_x = (float) ds8[i * 2 + 0];
        const float ds_y = (float) ds8[i * 2 + 1];
        sumf_d += ds_x * (sc[i] * sumi_d);
        sumf_m += ds_y * m[i];
    }
    return d4 * sumf_d - dmin4 * sumf_m;
}

template<int MMQ_X, int MMQ_Y, int NWARPS, class OutT = float>
static __device__ __forceinline__ void mmq_q4_k_turbo_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y,
    int stride_col_y, int stride_row_x, int nrows_dst)
{
    constexpr int LANES = MMQ_LDS_LANES;
    constexpr int ITER_K = 256;
    constexpr int TILE_NE_K = 32;
    constexpr int QR4_K = 2;
    constexpr int QI4_K = 32;
    constexpr int VDR = 8;
    constexpr int QI8_1L = 8;
    constexpr int TILE_Y_K_LDS = TILE_NE_K + TILE_NE_K / QI8_1L;   // 36
    constexpr int TXS_QS = MMQ_Y * TILE_NE_K + MMQ_Y;
    constexpr int TXS_DM = MMQ_Y * TILE_NE_K / QI4_K;             // half2 count
    constexpr int sz_mmq_int = 144 / 4;                          // 36

    extern __shared__ int lds[];
    int*       tile_y = lds;
    int*       x_qs   = tile_y + MMQ_X * TILE_Y_K_LDS;
    fb_fp16_t* x_dm   = (fb_fp16_t*) (x_qs + TXS_QS);   // 2 fp16 per (d, dmin)
    int*       x_sc   = (int*) (x_dm + TXS_DM * 2);

    const flambeau_block_q4_K* x = (const flambeau_block_q4_K*) vx;

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * MMQ_X;
    const int tx = threadIdx.x;   // 0..LANES-1
    const int ty = threadIdx.y;   // 0..NWARPS-1

    const int kb0_stop = ncols_x / 256;

    constexpr int n_sum = (MMQ_X / NWARPS) * ((MMQ_Y + LANES - 1) / LANES);
    float sum[n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) sum[s] = 0.0f;

    for (int kb0 = 0; kb0 < kb0_stop; ++kb0) {
        // -- Load tile_x (Q4_K super-block, MMQ_Y rows).
        {
            constexpr int threads_per_row = ITER_K / (4 * QR4_K);   // 32
            constexpr int nrows           = LANES / threads_per_row; // 2
            const int txi = tx % threads_per_row;
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
                const int i = i0 + ty * nrows + tx / threads_per_row;
                if (i < MMQ_Y) {
                    const int row = tile_m + i;
                    if (row < nrows_x) {
                        const flambeau_block_q4_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                        x_qs[i * (TILE_NE_K + 1) + txi] = load_int_b2((const int8_t*) bxi->qs, txi);
                    }
                }
            }
            // dm pass: (d, dmin) per row.
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
            // Scales pass: 8 int32/row (turbo layout).
            constexpr int rows_per_warp_sc = LANES / 4;   // 16
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
                const int i = (i0 + ty * rows_per_warp_sc + tx / (TILE_NE_K / 8)) % MMQ_Y;
                const int ksc = tx % (TILE_NE_K / 8);   // 0..3
                const int row = tile_m + i;
                if (row < nrows_x) {
                    const flambeau_block_q4_K* bxi = x + kb0 + (size_t) row * stride_row_x;
                    const int* scales = (const int*) bxi->scales;
                    const int scales8 = flambeau_unpack_scales_q45_K(scales, ksc);
                    x_sc[i * (TILE_NE_K / 8) + i / 8 + ksc] = scales8;
                }
            }
        }

        const int* y_int_tile = (const int*) vy + tile_n * sz_mmq_int;

        #pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int k00 = phase * TILE_NE_K;
            const int* by0 = y_int_tile
                + (size_t) ncols_y * ((kb0 * (256 / (4 * QK8_1))) * sz_mmq_int + phase * sz_mmq_int);
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * TILE_Y_K_LDS; l0 += NWARPS * LANES) {
                const int l = l0 + ty * LANES + tx;
                if (l < MMQ_X * TILE_Y_K_LDS) tile_y[l] = by0[l];
            }
            __syncthreads();

            const int*       y_qs = tile_y + 4;
            const fb_fp16_t* y_ds = (const fb_fp16_t*) tile_y;

            for (int k01 = 0; k01 < TILE_NE_K; k01 += QR4_K * VDR) {
                const int k0 = k00 + k01;
                #pragma unroll
                for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                    const int j = j0 + ty;
                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
                        const int i = i0 + tx;
                        if (i < MMQ_Y) {
                            const uint8_t* sc = (const uint8_t*)
                                &x_sc[i * (TILE_NE_K / 8) + i / 8 + k0 / 32] + 2 * (k01 / 16);
                            sum[j0 / NWARPS * ((MMQ_Y + LANES - 1) / LANES) + i0 / LANES] +=
                                flambeau_vec_dot_q4_k_mmq(
                                    &x_qs[i * (TILE_NE_K + 1) + k0 / 2],
                                    &y_qs[j * TILE_Y_K_LDS + k01],
                                    sc, sc + 8,
                                    (float) x_dm[i * 2 + 0], (float) x_dm[i * 2 + 1],
                                    &y_ds[(j * TILE_Y_K_LDS + k01 / QI8_1L) * 2]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    (void) stride_col_y;
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j = j0 + ty;
        const int col = tile_n + j;
        if (col >= ncols_y) continue;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
            const int i = i0 + tx;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < nrows_x && row < nrows_dst) {
                    mmvq_store<OutT>(dst, (int) ((size_t) col * nrows_dst + row),
                        sum[j0 / NWARPS * ((MMQ_Y + LANES - 1) / LANES) + i0 / LANES]);
                }
            }
        }
    }
}
