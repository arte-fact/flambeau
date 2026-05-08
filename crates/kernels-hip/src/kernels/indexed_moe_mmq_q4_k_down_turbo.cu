// indexed_moe_mmq_q4_k_down_turbo — 4.c sibling of gate_up_turbo.
// Down-projection has a single weight matrix and uses the SwiGLU'd per-pair
// activation (each pair has its own activation vector), so both the
// activation index AND the output index use slot_pair directly.
// Structurally identical to gate_up_turbo with half the weight LDS and one
// set of accumulators.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define MMQ_ITER_K       256
#define MMQ_TILE_NE_K    32
#define QR4_K            2
#define QI4_K            32
#define VDR_Q4_K_Q8_1_MMQ 8
#define QK_K             256
#define QK8_1_FLAMBEAU   32
#define QI8_1            8
#define MMQ_TILE_Y_K_LDS (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)

// MMQ_X=8 matches padded-sort alignment (see gate_up_turbo note).
#define MMQ_X 8
#define MMQ_Y 128
#define NWARPS 4
#define BLOCK_THREADS (WARP_SIZE * NWARPS)

#define TXS_QS (MMQ_Y * MMQ_TILE_NE_K + MMQ_Y)
#define TXS_DM (MMQ_Y * MMQ_TILE_NE_K / QI4_K)
#define TXS_SC (MMQ_Y * MMQ_TILE_NE_K / 8 + MMQ_Y / 8)
#define TILE_X_TOTAL (TXS_QS + TXS_DM + TXS_SC)

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

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

static __device__ __forceinline__ void load_x_q4_K_super_block(
    const flambeau_block_q4_K * __restrict__ w,
    const int kb0,
    const int tile_m,
    const int n_rows,
    const int n_sb_per_row,
    int * __restrict__ x_qs,
    __half2 * __restrict__ x_dm,
    int * __restrict__ x_sc
) {
    {
        constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
        constexpr int nrows           = WARP_SIZE / threads_per_row;
        const int txi = threadIdx.x % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const flambeau_block_q4_K* bxi = w + row * n_sb_per_row + kb0;
                    const int qs0 = ((const int*) bxi->qs)[txi];
                    x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
                }
            }
        }
    }
    {
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * WARP_SIZE) {
            const int i = (i0 + threadIdx.y * WARP_SIZE + threadIdx.x) % MMQ_Y;
            const int row = tile_m + i;
            if (row < n_rows) {
                const flambeau_block_q4_K* bxi = w + row * n_sb_per_row + kb0;
                const __half* dm_ptr = (const __half*) bxi;
                x_dm[i] = __halves2half2(dm_ptr[0], dm_ptr[1]);
            }
        }
    }
    {
        constexpr int rows_per_warp_sc = WARP_SIZE / 4;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i_raw = i0 + threadIdx.y * rows_per_warp_sc + threadIdx.x / (MMQ_TILE_NE_K / 8);
            const int i = i_raw % MMQ_Y;
            const int ksc = threadIdx.x % (MMQ_TILE_NE_K / 8);
            const int row = tile_m + i;
            if (row < n_rows) {
                const flambeau_block_q4_K* bxi =
                    w + row * n_sb_per_row + kb0
                    + (threadIdx.x % (MMQ_TILE_NE_K / 8)) / (QI4_K / 8);
                const int* scales = (const int*) bxi->scales;
                const int scales8 = unpack_scales_q45_K(scales, ksc);
                x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = scales8;
            }
        }
    }
}

static __device__ __forceinline__ void vec_dot_pass(
    const int * __restrict__ x_qs,
    const __half2 * __restrict__ x_dm,
    const int * __restrict__ x_sc,
    const int * __restrict__ tile_y,
    float * __restrict__ sum,
    const int k00
) {
    const int   * y_qs = tile_y + 4;
    const __half2 * y_ds = (const __half2 *) tile_y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;
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

extern "C" __global__ __launch_bounds__(BLOCK_THREADS, 1)
void flambeau_indexed_moe_mmq_q4_k_down_turbo_q8_1(
    const flambeau_block_q4_K * __restrict__ down_w,
    const flambeau_block_q8_1_mmq * __restrict__ y_mmq,
    const int * __restrict__ expert_ids,
    const int * __restrict__ sorted_pair_idx_padded,
    const int * __restrict__ padded_offsets,
    float * __restrict__ dst,
    const int n_rows,          // hidden
    const int n_pairs,
    const int n_sb_per_row,    // = inter / QK_K for down_w
    const int n_experts
) {
    extern __shared__ int lds[];
    int * tile_y    = lds;
    int * tile_x    = tile_y + MMQ_X * MMQ_TILE_Y_K_LDS;

    int   * x_qs = tile_x;
    __half2 * x_dm = (__half2 *) (x_qs + TXS_QS);
    int   * x_sc = (int *) (x_dm + TXS_DM);

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * MMQ_X;

    __shared__ int padded_total_shared;
    if (threadIdx.x == 0 && threadIdx.y == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert     = expert_ids[first_pair];

    __shared__ int slot_pair[MMQ_X];
    if (threadIdx.y == 0 && threadIdx.x < MMQ_X) {
        slot_pair[threadIdx.x] = sorted_pair_idx_padded[tile_n + threadIdx.x];
    }
    __syncthreads();

    const flambeau_block_q4_K * down_w_expert = down_w + (size_t) expert * n_rows * n_sb_per_row;

    constexpr int sz_mmq_int = (int) (sizeof(flambeau_block_q8_1_mmq) / sizeof(int));
    const int * y_int = (const int *) y_mmq;

    constexpr int n_sum = (MMQ_X / NWARPS) * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE);
    float sum[n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) sum[s] = 0.0f;

    for (int kb0 = 0; kb0 < n_sb_per_row; ++kb0) {
        load_x_q4_K_super_block(down_w_expert, kb0, tile_m, n_rows, n_sb_per_row, x_qs, x_dm, x_sc);

        {
            const int big_block = kb0 * (QK_K / (4 * QK8_1_FLAMBEAU));
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_LDS; l0 += NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_LDS) {
                    const int c_in_tile = l / MMQ_TILE_Y_K_LDS;
                    const int intra     = l % MMQ_TILE_Y_K_LDS;
                    const int pair      = slot_pair[c_in_tile];
                    tile_y[l] = y_int[((size_t) big_block * n_pairs + pair) * sz_mmq_int + intra];
                }
            }
        }
        __syncthreads();
        vec_dot_pass(x_qs, x_dm, x_sc, tile_y, sum, 0);
        __syncthreads();

        {
            const int big_block = kb0 * (QK_K / (4 * QK8_1_FLAMBEAU)) + 1;
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_LDS; l0 += NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_LDS) {
                    const int c_in_tile = l / MMQ_TILE_Y_K_LDS;
                    const int intra     = l % MMQ_TILE_Y_K_LDS;
                    const int pair      = slot_pair[c_in_tile];
                    tile_y[l] = y_int[((size_t) big_block * n_pairs + pair) * sz_mmq_int + intra];
                }
            }
        }
        __syncthreads();
        vec_dot_pass(x_qs, x_dm, x_sc, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j_local = j0 + threadIdx.y;
        const int pair    = slot_pair[j_local];
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const size_t out_idx = (size_t) pair * n_rows + row;
                    const int s = j0 / NWARPS * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE) + i0 / WARP_SIZE;
                    dst[out_idx] = sum[s];
                }
            }
        }
    }
}
