#pragma once
// Templated Q4_0 4-warp LDS-tiled MMQ (the dense-prefill perf kernel), DS4 144-B
// activation layout. MMQ_Y=128 weight rows × MMQ_X=64 activation cols per block,
// staged through dynamic LDS; dp4a inner with the Q4_0 bias identity
// d·(d_y·Σq·q8 − 8·y_s). The hand kernel's "wave64" is a LOGICAL 64-wide tile
// (LANES), not a hardware warp: all cross-thread comms are LDS + __syncthreads,
// there are NO __shfl/warp-reduce ops, so launching block (LANES, NWARPS)=(64,4)
// is correct on both wave64 and wave32. Backend-neutral: `dp4a` + the neutral
// `mmq_prefetch_*` seam from the arch headers; weight int loads via `load_int_b2`
// (Q4_0 qs at block-offset 2); the (d, m) / (d_y, y_s) pairs are read as two
// `fb_fp16_t` (the byte layout the DS4 quantizer writes). `OutT` is the dest dtype.
#include "quant_traits.cuh"
#include "mmvq_store.cuh"

#ifndef MMQ_LDS_LANES
#define MMQ_LDS_LANES 64
#endif
#ifndef QR4_0
#define QR4_0 2
#endif

template<int MMQ_X, int MMQ_Y, int NWARPS, class OutT = float>
static __device__ __forceinline__ void mmq_q4_0_4warp_lds_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    OutT* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y,
    int stride_col_y, int stride_row_x, int nrows_dst)
{
    constexpr int LANES = MMQ_LDS_LANES;
    constexpr int QI4_0v = QK4_0 / (4 * QR4_0);   // 4 ints / Q4_0 block
    constexpr int QI8_1v = QK8_1 / 4;             // 8 ints / Q8_1 block
    constexpr int TILE_NE_K = 32;
    constexpr int TILE_Y_K  = TILE_NE_K + TILE_NE_K / QI8_1v;   // 36
    constexpr int ITER_K    = 256;
    constexpr int VDR       = 4;
    constexpr int MMQ_BYTES = 144;
    constexpr int MMQ_INTS  = MMQ_BYTES / 4;       // 36
    constexpr int X_QS_INTS = MMQ_Y * (TILE_NE_K + 1);
    constexpr int X_DM_PAIRS = MMQ_Y * (TILE_NE_K / QI4_0v) + MMQ_Y / QI4_0v; // half2 count
    constexpr int TILE_Y_INTS = ((MMQ_X * TILE_Y_K) + (NWARPS * LANES) - 1)
                              / (NWARPS * LANES) * (NWARPS * LANES);          // GGML_PAD

    extern __shared__ int shared_buf[];
    int*       tile_y = shared_buf;
    int*       x_qs   = tile_y + TILE_Y_INTS;
    fb_fp16_t* x_dm   = (fb_fp16_t*) (x_qs + X_QS_INTS);   // 2 fp16 per (d, m) pair

    const int it = blockIdx.x;
    const int jt = blockIdx.y;
    const int tx = threadIdx.x;   // 0..LANES-1 (logical lane)
    const int ty = threadIdx.y;   // 0..NWARPS-1 (logical warp)

    constexpr int sum_slots = MMQ_X * MMQ_Y / (NWARPS * LANES);
    float sum[sum_slots];
    #pragma unroll
    for (int s = 0; s < sum_slots; ++s) sum[s] = 0.0f;

    const int* y_base = (const int*) vy + jt * MMQ_X * MMQ_INTS;
    constexpr int blocks_per_iter = ITER_K / QK4_0;   // 8
    const int n_blocks_x = ncols_x / QK4_0;
    const int kb0_stop = (n_blocks_x + blocks_per_iter - 1) / blocks_per_iter * blocks_per_iter;
    const int i_max = nrows_x - it * MMQ_Y - 1;
    constexpr int q4_0_block_bytes = 18;

    const bool need_check = (it + 1) * MMQ_Y > nrows_x;

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int kb_remaining =
            (n_blocks_x - kb0 < blocks_per_iter) ? (n_blocks_x - kb0) : blocks_per_iter;

        // -- L2 prefetch of next iter's X/Y tiles (gfx906 real, CUDA no-op).
        const int kb0_next = kb0 + blocks_per_iter;
        if (kb0_next < kb0_stop) {
            const int* by_next0 = y_base + (size_t) stride_col_y * (kb0_next / 4) * MMQ_INTS;
            const int yp = mmq_prefetch_y_next(by_next0);
            const long long off_x_next =
                ((long long) it * MMQ_Y * stride_row_x + kb0_next) * q4_0_block_bytes;
            const int xp = mmq_prefetch_x_next((const char*) vx, (int) off_x_next,
                                               stride_row_x * q4_0_block_bytes);
            mmq_prefetch_consume(yp);
            mmq_prefetch_consume(xp);
        }

        // -- Load X tile (quants + (d, m=0) pairs) into LDS.
        {
            constexpr int threads_per_row = ITER_K / (4 * QR4_0);   // 32
            constexpr int nrows           = LANES / threads_per_row; // 2
            const int txi  = tx % threads_per_row;
            const int kbx  = txi / QI4_0v;
            const int kqsx = txi % QI4_0v;
            const int kbx_safe = (kbx < kb_remaining) ? kbx : (kb_remaining - 1);
            const int kbx0 = it * MMQ_Y * stride_row_x + kb0;

            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
                int i = i0 + ty * nrows + tx / threads_per_row;
                if (need_check) i = (i < i_max) ? i : i_max;
                const flambeau_block_q4_0* bxi =
                    (const flambeau_block_q4_0*) vx + kbx0 + i * stride_row_x + kbx_safe;
                x_qs[i * (TILE_NE_K + 1) + txi] = load_int_b2((const int8_t*) bxi->qs, kqsx);
            }

            constexpr int blocks_per_tile_x_row = TILE_NE_K / QI4_0v;     // 8
            constexpr int rows_per_warp         = LANES / blocks_per_tile_x_row; // 8
            const int kbxd = tx % blocks_per_tile_x_row;
            const bool kbxd_valid = (kbxd < kb_remaining);
            const int kbxd_safe = kbxd_valid ? kbxd : 0;

            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp) {
                int i = i0 + ty * rows_per_warp + tx / blocks_per_tile_x_row;
                if (need_check) i = (i < i_max) ? i : i_max;
                const flambeau_block_q4_0* bxi =
                    (const flambeau_block_q4_0*) vx + kbx0 + i * stride_row_x + kbxd_safe;
                const int pair = i * (TILE_NE_K / QI4_0v) + i / QI4_0v + kbxd;
                x_dm[pair * 2 + 0] = kbxd_valid ? bxi->d : (fb_fp16_t) 0.0f;
                x_dm[pair * 2 + 1] = (fb_fp16_t) 0.0f;   // Q4_0 has no min
            }
        }

        // -- Two-phase Y load + dot per half super-block (k00 = 0, then TILE_NE_K).
        #pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int k00 = phase * TILE_NE_K;
            const int* by =
                y_base + (size_t) stride_col_y * ((kb0 / 4) * MMQ_INTS + phase * MMQ_INTS);
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * TILE_Y_K; l0 += NWARPS * LANES) {
                const int l = l0 + ty * LANES + tx;
                tile_y[l] = by[l];
            }
            __syncthreads();

            const int*       y_qs = tile_y + 4;
            const fb_fp16_t* y_ds = (const fb_fp16_t*) tile_y;

            for (int k01 = 0; k01 < TILE_NE_K; k01 += QR4_0 * VDR) {
                const int k0 = k00 + k01;
                #pragma unroll
                for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                    const int j = j0 + ty;
                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
                        const int i = i0 + tx;

                        const int kyqs = QI8_1v * ((k01 / 2) / (QI8_1v / 2))
                                       + (k01 / 2) % (QI8_1v / 2);
                        const int4 vec0 = *((const int4*) &y_qs[j * TILE_Y_K + kyqs]);
                        const int4 vec1 = *((const int4*) &y_qs[j * TILE_Y_K + kyqs + QI4_0v]);
                        int u[2 * VDR];
                        u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
                        u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;

                        const int dm_pair = i * (TILE_NE_K / QI4_0v) + i / QI4_0v + k0 / (QR4_0 * QI4_0v);
                        const float d_x = (float) x_dm[dm_pair * 2 + 0];
                        const int ds_idx = j * TILE_Y_K + k01 / QI8_1v;
                        const float d_y = (float) y_ds[ds_idx * 2 + 0];
                        const float y_s = (float) y_ds[ds_idx * 2 + 1];

                        const int* v = &x_qs[i * (TILE_NE_K + 1) + k0 / QR4_0];
                        int sumi = 0;
                        #pragma unroll
                        for (int t = 0; t < VDR; ++t) {
                            const int vi0 = (v[t] >> 0) & 0x0F0F0F0F;
                            const int vi1 = (v[t] >> 4) & 0x0F0F0F0F;
                            sumi = dp4a(vi0, u[2 * t + 0], sumi);
                            sumi = dp4a(vi1, u[2 * t + 1], sumi);
                        }
                        sum[(j0 / NWARPS) * (MMQ_Y / LANES) + i0 / LANES] +=
                            d_x * (d_y * (float) sumi - 8.0f * y_s);
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j = j0 + ty;
        const int col_g = jt * MMQ_X + j;
        if (col_g >= ncols_y) return;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
            const int i = i0 + tx;
            const int row_g = it * MMQ_Y + i;
            if (need_check && row_g >= nrows_x) continue;
            mmvq_store<OutT>(dst, (int) ((size_t) col_g * nrows_dst + row_g),
                             sum[(j0 / NWARPS) * (MMQ_Y / LANES) + i0 / LANES]);
        }
    }
}
