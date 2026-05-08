// mmq_q4_K_turbo — 4.b port of llamacpp-turbo's 4-warp LDS-tiled Q4_K MMQ.
// Source: /artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/ggml/src/ggml-cuda/mmq.cuh
// - load_tiles_q4_K (DP4A branch)
// - vec_dot_q4_K_q8_1_dp4a
// - vec_dot_q4_K_q8_1_impl_mmq (from vecdotq.cuh)
// - mul_mat_q_process_tile (outer K loop with double-buffered Y LDS)
// This is the DENSE standalone port — no indexed-MoE wrapping. That lands in
// once this passes sweep_mmq correctness.
// Fixed template parameters for 4.b:
// mmq_x = 16 (activation cols per block)
// mmq_y = 16 (weight rows per block)
// nwarps = 4 × warp_size 64 = 256 threads/block
// DP4A path only (no MFMA/MMA)
// Launch:
// block = (64, 4, 1)
// grid = (⌈n/mmq_y⌉, ⌈m/mmq_x⌉, 1)
// shared = (mmq_x * MMQ_TILE_Y_K_LDS + tile_x_qs + tile_x_dm + tile_x_sc) * 4 B
// Args:
// vx — device ptr to block_q4_K array, shape [N, K / QK_K]
// vy_mmq — device ptr to block_q8_1_mmq array (DS4 layout), shape [M_big, K / QK8_1_MMQ]
// dst — device ptr to f32 output, shape [M, N] col-major (dst[col*nrows_dst + row])
// ncols_x (=K), nrows_x (=N), ncols_y (=M), stride_col_y, stride_row_x, nrows_dst (=N)

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Turbo constants (verbatim from mmq.cuh).
#define MMQ_ITER_K       256
#define MMQ_TILE_NE_K    32
#define QR4_K            2
#define QI4_K            32             // QK_K / (4 * QR4_K) = 256 / 8
#define VDR_Q4_K_Q8_1_MMQ 8
#define QK_K             256
#define QK8_1_FLAMBEAU   32             // standard Q8_1, but the MMQ activation
                                        // uses DS4 layout (4 sub-blocks per MMQ block).
#define QI8_1            8              // QK8_1 / 4
#define MMQ_TILE_Y_K_LDS (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)   // 36 ints per col

// Fixed block geometry for this instantiation.
// mmq_y = 128 is turbo's gfx906 default (get_mmq_y_device returns 128 for non-RDNA1 AMD).
// mmq_x = 16 is a mid-sized batch tile; turbo's mmq_x_best picks 8-64 shape-dependent.
#define MMQ_X 16
#define MMQ_Y 128
#define NWARPS 4
#define BLOCK_THREADS (WARP_SIZE * NWARPS)  // 256

// DP4A tile sizes for Q4_K (from MMQ_DP4A_TXS_Q4_K in turbo mmq.cuh:185):
// qs: mmq_y * MMQ_TILE_NE_K + mmq_y
// dm: mmq_y * MMQ_TILE_NE_K / QI4_K (half2 count; half2 = 1 int slot each)
// sc: mmq_y * MMQ_TILE_NE_K / 8 + mmq_y / 8
#define TXS_QS (MMQ_Y * MMQ_TILE_NE_K + MMQ_Y)         // 128*32 + 128 = 4224 ints
#define TXS_DM (MMQ_Y * MMQ_TILE_NE_K / QI4_K)          // 128 half2 = 128 int slots
#define TXS_SC (MMQ_Y * MMQ_TILE_NE_K / 8 + MMQ_Y / 8)  // 128*4 + 16 = 528 ints

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Turbo's unpack_scales_q45_K (verbatim from mmq.cuh:2127-2135).
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

// Turbo's vec_dot_q4_K_q8_1_impl_mmq (verbatim from vecdotq.cuh:558-583).
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m,
    const __half2 & dm4, const __half2 * __restrict__ ds8
) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    #pragma unroll
    for (int i = 0; i < QR4_K * VDR_Q4_K_Q8_1_MMQ / QI8_1; ++i) {
        int sumi_d = 0;
        #pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * QI8_1 + j], sumi_d);
        }
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i];
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

extern "C" __global__ __launch_bounds__(BLOCK_THREADS, 1)
void flambeau_mmq_q4_K_turbo_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float*      __restrict__ dst,
    const int ncols_x,       // K
    const int nrows_x,       // N (output features)
    const int ncols_y,       // M (batch rows)
    const int stride_col_y,  // bytes per MMQ activation col (= sizeof(block_q8_1_mmq) * (K/QK8_1_MMQ))
    const int stride_row_x,  // blocks per X row (= K / QK_K) for Q4_K
    const int nrows_dst      // (= N for our layout)
) {
    extern __shared__ int lds[];
    // LDS layout:
    // tile_y[MMQ_X * MMQ_TILE_Y_K_LDS] ints (double-buffered per-half-super-block)
    // tile_x_qs[TXS_QS] ints
    // tile_x_dm[TXS_DM * 2] ints (stored as half2 pairs, 2 ints per half2)
    // tile_x_sc[TXS_SC] ints
    int   * tile_y  = lds;
    int   * tile_x  = tile_y  + MMQ_X * MMQ_TILE_Y_K_LDS;
    int   * x_qs    = tile_x;
    __half2 * x_dm  = (__half2 *) (x_qs + TXS_QS);
    int   * x_sc    = (int *)    (x_dm + TXS_DM);

    const flambeau_block_q4_K*      x = (const flambeau_block_q4_K*)      vx;
    const flambeau_block_q8_1_mmq*  y = (const flambeau_block_q8_1_mmq*)  vy;

    const int tile_m = blockIdx.x * MMQ_Y;    // base weight-row (out channel)
    const int tile_n = blockIdx.y * MMQ_X;    // base activation-col (batch row)

    const int kb0_start = 0;
    const int kb0_stop  = ncols_x / QK_K;     // super-blocks per row
    const int blocks_per_iter = 1;

    // Per-thread accumulators: one fp32 per (j, i) pair this thread owns.
    // With nwarps=4, warp_size=64, mmq_x=16, mmq_y=16:
    // j slot count = mmq_x / nwarps = 4 (each warp owns 4 cols)
    // i slot count = mmq_y / warp_size = 16 / 64 = 0 → each thread owns partial row via modulo
    // Actually turbo's setup uses j0 += nwarps → j = j0 + threadIdx.y (4 iters × 4 warps = 16).
    // And i0 += warp_size → i = i0 + threadIdx.x. With mmq_y=16 < warp_size=64, only threads
    // with threadIdx.x < mmq_y participate on the dot side. Accumulator size = mmq_x/nwarps * 1 = 4.
    constexpr int n_sum = (MMQ_X / NWARPS) * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE);  // 4 * 1 = 4
    float sum[n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) sum[s] = 0.0f;

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        // =========================================================
        // Load tile_x (Q4_K super-block for this kb0, mmq_y rows).
        // Follows turbo's load_tiles_q4_K DP4A branch (mmq.cuh:2137+).
        // =========================================================
        {
            constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);   // 32
            constexpr int nrows           = WARP_SIZE / threads_per_row; // 2
            const int txi = threadIdx.x % threads_per_row;

            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
                const int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
                if (i < MMQ_Y) {
                    const int row = tile_m + i;
                    if (row < nrows_x) {
                        const flambeau_block_q4_K* bxi =
                            (const flambeau_block_q4_K*) x + kb0 + row * stride_row_x;
                        const int qs0 = ((const int*) bxi->qs)[txi];  // get_int_b4
                        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
                    }
                }
            }

            // dm pass: one half2 per row.
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * WARP_SIZE) {
                const int i = (i0 + threadIdx.y * WARP_SIZE + threadIdx.x) % MMQ_Y;
                const int row = tile_m + i;
                if (row < nrows_x) {
                    const flambeau_block_q4_K* bxi =
                        (const flambeau_block_q4_K*) x + kb0 + row * stride_row_x;
                    // block stores d, dmin as fp16 at offset 0, 2.
                    const __half* dm_ptr = (const __half*) bxi;  // d at [0], dmin at [1]
                    x_dm[i] = __halves2half2(dm_ptr[0], dm_ptr[1]);
                }
            }

            // Scales pass: unpack scales/mins into 8 int32 per row (turbo layout).
            constexpr int rows_per_warp_sc = WARP_SIZE / 4;   // 16
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
                const int i_raw = i0 + threadIdx.y * rows_per_warp_sc + threadIdx.x / (MMQ_TILE_NE_K / 8);
                const int i = i_raw % MMQ_Y;
                const int ksc = threadIdx.x % (MMQ_TILE_NE_K / 8);   // 0..3
                const int row = tile_m + i;
                if (row < nrows_x) {
                    const flambeau_block_q4_K* bxi =
                        (const flambeau_block_q4_K*) x + kb0 + row * stride_row_x
                        + (threadIdx.x % (MMQ_TILE_NE_K / 8)) / (QI4_K / 8);
                    const int* scales = (const int*) bxi->scales;
                    const int scales8 = unpack_scales_q45_K(scales, ksc);
                    x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = scales8;
                }
            }
        }

        // =========================================================
        // Y LDS load pass 1 (first half-super-block).
        // =========================================================
        constexpr int sz_mmq_int = (int) (sizeof(flambeau_block_q8_1_mmq) / sizeof(int));  // 144 / 4 = 36
        // Turbo applies `offset_y += (col_low + jt*mmq_x) * sz_mmq_int` to the Y
        // pointer BEFORE entering mul_mat_q_process_tile (mmq.cuh:3718). That
        // baked-in offset is what makes by0 address THIS tile's mmq_x cols
        // instead of always cols [0..mmq_x). We fold it back in here.
        const int* y_int_tile = (const int*) y + tile_n * sz_mmq_int;
        {
            // by0 = y_tile + ncols_y * (kb0 * qk / ne_block) * sz
            // qk = QK_K = 256, ne_block = 4*QK8_1 = 128
            // (kb0 * 256 / 128) = kb0 * 2 — each Q4_K super-block covers 2 Q8_1_MMQ blocks.
            const int* by0 = y_int_tile + ncols_y * (kb0 * (QK_K / (4 * QK8_1_FLAMBEAU))) * sz_mmq_int;
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_LDS; l0 += NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_LDS) {
                    tile_y[l] = by0[l];
                }
            }
        }

        __syncthreads();

        // =========================================================
        // vec_dot pass 1 (k00 = 0, processes first half-super-block).
        // Follows turbo's vec_dot_q4_K_q8_1_dp4a (mmq.cuh:2247+).
        // =========================================================
        {
            const int   * y_qs = tile_y + 4;                   // skip 4 ds ints header
            const __half2 * y_ds = (const __half2 *) tile_y;

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
                const int k0 = /* k00 */ 0 + k01;

                #pragma unroll
                for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                    const int j = j0 + threadIdx.y;

                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                        const int i = i0 + threadIdx.x;
                        if (i < MMQ_Y) {
                            const uint8_t* sc = (const uint8_t*) &x_sc[
                                i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32
                            ] + 2 * (k01 / 16);

                            sum[j0 / NWARPS * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE) + i0 / WARP_SIZE]
                              += vec_dot_q4_K_q8_1_impl_mmq(
                                &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2],
                                &y_qs[j * MMQ_TILE_Y_K_LDS + k01],
                                sc, sc + 8,
                                x_dm[i],
                                &y_ds[j * MMQ_TILE_Y_K_LDS + k01 / QI8_1]);
                        }
                    }
                }
            }
        }

        __syncthreads();

        // =========================================================
        // Y LDS load pass 2 (second half-super-block).
        // by0 += sz_mmq_int (turbo: `(kb0 * qk / ne_block) * sz + sz`).
        // =========================================================
        {
            const int* by0 = y_int_tile
                + ncols_y * ((kb0 * (QK_K / (4 * QK8_1_FLAMBEAU))) * sz_mmq_int + sz_mmq_int);
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_LDS; l0 += NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_LDS) {
                    tile_y[l] = by0[l];
                }
            }
        }

        __syncthreads();

        // =========================================================
        // vec_dot pass 2 (k00 = MMQ_TILE_NE_K, second half-super-block).
        // =========================================================
        {
            const int   * y_qs = tile_y + 4;
            const __half2 * y_ds = (const __half2 *) tile_y;

            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
                const int k0 = MMQ_TILE_NE_K + k01;

                #pragma unroll
                for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
                    const int j = j0 + threadIdx.y;

                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                        const int i = i0 + threadIdx.x;
                        if (i < MMQ_Y) {
                            const uint8_t* sc = (const uint8_t*) &x_sc[
                                i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32
                            ] + 2 * (k01 / 16);

                            sum[j0 / NWARPS * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE) + i0 / WARP_SIZE]
                              += vec_dot_q4_K_q8_1_impl_mmq(
                                &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2],
                                &y_qs[j * MMQ_TILE_Y_K_LDS + k01],
                                sc, sc + 8,
                                x_dm[i],
                                &y_ds[j * MMQ_TILE_Y_K_LDS + k01 / QI8_1]);
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // =========================================================
    // Write back. Output layout matches other flambeau MMQ kernels:
    // dst[col * nrows_dst + row].
    // =========================================================
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col = tile_n + j;
        if (col >= ncols_y) continue;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < nrows_x && row < nrows_dst) {
                    dst[(size_t) col * nrows_dst + row] =
                        sum[j0 / NWARPS * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE) + i0 / WARP_SIZE];
                }
            }
        }
    }
    (void) stride_col_y;  // unused for row-major MMQ layout
}
