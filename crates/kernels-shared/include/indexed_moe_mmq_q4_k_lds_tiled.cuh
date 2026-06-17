#pragma once
// Templated Q4_K 4-warp LDS-tiled indexed-MoE MMQ (prefill), DS4 144-B
// activation. Same logical-lanes structure as the dense `mmq_q4_k_lds_tiled.cuh`
// (the "wave64" is a logical LANES-wide tile — LDS + __syncthreads only, NO
// __shfl — so block (LANES, NWARPS) is correct on wave64 AND wave32). MoE
// addressing: weight base is offset by `expert_ids[first_pair]`; the DS4
// activation is gathered per output column (gate_up: by token, shared across a
// token's top_k slots; down: by pair) from the padded expert sort. gate_up fuses
// two weight tiles (gate + up) sharing one per-token activation; down is the
// single-weight per-pair sibling. Backend-neutral: reuses the dense header's
// `flambeau_unpack_scales_q45_K` / `flambeau_vec_dot_q4_k_mmq` (fb_fp16_t pairs),
// `dp4a`, `load_int_b2`, `mmvq_store`.
#include "mmq_q4_k_lds_tiled.cuh"

// Load one Q4_K super-block (MMQ_Y rows of one expert) into the x_qs/x_dm/x_sc
// LDS tile. `w` is already offset to the expert base.
template<int MMQ_Y, int NWARPS>
static __device__ __forceinline__ void flambeau_moe_load_x_q4_k_lds(
    const flambeau_block_q4_K* __restrict__ w,
    int kb0, int tile_m, int n_rows, int n_sb_per_row,
    int* __restrict__ x_qs, fb_fp16_t* __restrict__ x_dm, int* __restrict__ x_sc)
{
    constexpr int LANES = MMQ_LDS_LANES;
    constexpr int ITER_K = 256;
    constexpr int TILE_NE_K = 32;
    constexpr int QR4_K = 2;
    constexpr int QI4_K = 32;

    {
        constexpr int threads_per_row = ITER_K / (4 * QR4_K);   // 32
        constexpr int nrows           = LANES / threads_per_row; // 2
        const int txi = threadIdx.x % threads_per_row;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const flambeau_block_q4_K* bxi = w + (size_t) row * n_sb_per_row + kb0;
                    x_qs[i * (TILE_NE_K + 1) + txi] = load_int_b2((const int8_t*) bxi->qs, txi);
                }
            }
        }
    }
    {
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * LANES) {
            const int i = (i0 + threadIdx.y * LANES + threadIdx.x) % MMQ_Y;
            const int row = tile_m + i;
            if (row < n_rows) {
                const flambeau_block_q4_K* bxi = w + (size_t) row * n_sb_per_row + kb0;
                x_dm[i * 2 + 0] = bxi->d;
                x_dm[i * 2 + 1] = bxi->dmin;
            }
        }
    }
    {
        constexpr int rows_per_warp_sc = LANES / 4;   // 16
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i = (i0 + threadIdx.y * rows_per_warp_sc + threadIdx.x / (TILE_NE_K / 8)) % MMQ_Y;
            const int ksc = threadIdx.x % (TILE_NE_K / 8);   // 0..3
            const int row = tile_m + i;
            if (row < n_rows) {
                const flambeau_block_q4_K* bxi = w + (size_t) row * n_sb_per_row + kb0;
                const int* scales = (const int*) bxi->scales;
                const int scales8 = flambeau_unpack_scales_q45_K(scales, ksc);
                x_sc[i * (TILE_NE_K / 8) + i / 8 + ksc] = scales8;
            }
        }
    }
    (void) QI4_K;
}

// One DP4A vec_dot pass against (x_qs, x_dm, x_sc, tile_y) at k00, accumulating
// into sum[].
template<int MMQ_X, int MMQ_Y, int NWARPS>
static __device__ __forceinline__ void flambeau_moe_vec_dot_pass_q4_k(
    const int* __restrict__ x_qs, const fb_fp16_t* __restrict__ x_dm,
    const int* __restrict__ x_sc, const int* __restrict__ tile_y,
    float* __restrict__ sum, int k00)
{
    constexpr int LANES = MMQ_LDS_LANES;
    constexpr int TILE_NE_K = 32;
    constexpr int QR4_K = 2;
    constexpr int VDR = 8;
    constexpr int QI8_1L = 8;
    constexpr int TILE_Y_K_LDS = TILE_NE_K + TILE_NE_K / QI8_1L;   // 36

    const int* y_qs = tile_y + 4;
    const fb_fp16_t* y_ds = (const fb_fp16_t*) tile_y;

    for (int k01 = 0; k01 < TILE_NE_K; k01 += QR4_K * VDR) {
        const int k0 = k00 + k01;
        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
            const int j = j0 + threadIdx.y;
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
                const int i = i0 + threadIdx.x;
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
}

template<int MMQ_X, int MMQ_Y, int NWARPS, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q4_k_gate_up_lds_tiled_row(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1_mmq* __restrict__ y_mmq,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_sb_per_row, int n_experts)
{
    constexpr int LANES = MMQ_LDS_LANES;
    constexpr int TILE_NE_K = 32;
    constexpr int QI4_K = 32;
    constexpr int QI8_1L = 8;
    constexpr int QK_K_L = 256;
    constexpr int QK8_1L = 32;
    constexpr int TILE_Y_K_LDS = TILE_NE_K + TILE_NE_K / QI8_1L;   // 36
    constexpr int TXS_QS = MMQ_Y * TILE_NE_K + MMQ_Y;
    constexpr int TXS_DM = MMQ_Y * TILE_NE_K / QI4_K;             // half2 count
    constexpr int TXS_SC = MMQ_Y * TILE_NE_K / 8 + MMQ_Y / 8;
    constexpr int TILE_X_TOTAL = TXS_QS + TXS_DM + TXS_SC;        // ints (dm: TXS_DM half2 = TXS_DM ints)
    constexpr int sz_mmq_int = 144 / 4;                          // 36

    extern __shared__ int lds[];
    int* tile_y      = lds;
    int* tile_x_gate = tile_y + MMQ_X * TILE_Y_K_LDS;
    int* tile_x_up   = tile_x_gate + TILE_X_TOTAL;

    int*       xg_qs = tile_x_gate;
    fb_fp16_t* xg_dm = (fb_fp16_t*) (xg_qs + TXS_QS);
    int*       xg_sc = (int*) (xg_dm + TXS_DM * 2);

    int*       xu_qs = tile_x_up;
    fb_fp16_t* xu_dm = (fb_fp16_t*) (xu_qs + TXS_QS);
    int*       xu_sc = (int*) (xu_dm + TXS_DM * 2);

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * MMQ_X;

    __shared__ int padded_total_shared;
    if (threadIdx.x == 0 && threadIdx.y == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert     = expert_ids[first_pair];

    __shared__ int slot_pair[MMQ_X];
    __shared__ int slot_token[MMQ_X];
    if (threadIdx.y == 0 && threadIdx.x < MMQ_X) {
        const int pair = sorted_pair_idx_padded[tile_n + threadIdx.x];
        slot_pair[threadIdx.x]  = pair;
        slot_token[threadIdx.x] = pair / top_k;
    }
    __syncthreads();

    const flambeau_block_q4_K* gate_w_expert = gate_w + (size_t) expert * n_rows * n_sb_per_row;
    const flambeau_block_q4_K* up_w_expert   = up_w   + (size_t) expert * n_rows * n_sb_per_row;

    const int* y_int = (const int*) y_mmq;

    constexpr int n_sum = (MMQ_X / NWARPS) * ((MMQ_Y + LANES - 1) / LANES);
    float sum_gate[n_sum];
    float sum_up[n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) { sum_gate[s] = 0.0f; sum_up[s] = 0.0f; }

    for (int kb0 = 0; kb0 < n_sb_per_row; ++kb0) {
        flambeau_moe_load_x_q4_k_lds<MMQ_Y, NWARPS>(gate_w_expert, kb0, tile_m, n_rows, n_sb_per_row, xg_qs, xg_dm, xg_sc);
        flambeau_moe_load_x_q4_k_lds<MMQ_Y, NWARPS>(up_w_expert,   kb0, tile_m, n_rows, n_sb_per_row, xu_qs, xu_dm, xu_sc);

        #pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int big_block = kb0 * (QK_K_L / (4 * QK8_1L)) + phase;
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * TILE_Y_K_LDS; l0 += NWARPS * LANES) {
                const int l = l0 + threadIdx.y * LANES + threadIdx.x;
                if (l < MMQ_X * TILE_Y_K_LDS) {
                    const int c_in_tile = l / TILE_Y_K_LDS;
                    const int intra     = l % TILE_Y_K_LDS;
                    const int token     = slot_token[c_in_tile];
                    tile_y[l] = y_int[((size_t) big_block * n_tokens + token) * sz_mmq_int + intra];
                }
            }
            __syncthreads();
            flambeau_moe_vec_dot_pass_q4_k<MMQ_X, MMQ_Y, NWARPS>(xg_qs, xg_dm, xg_sc, tile_y, sum_gate, phase * TILE_NE_K);
            flambeau_moe_vec_dot_pass_q4_k<MMQ_X, MMQ_Y, NWARPS>(xu_qs, xu_dm, xu_sc, tile_y, sum_up,   phase * TILE_NE_K);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j_local = j0 + threadIdx.y;
        const int pair    = slot_pair[j_local];
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
            const int i = i0 + threadIdx.x;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const int s = j0 / NWARPS * ((MMQ_Y + LANES - 1) / LANES) + i0 / LANES;
                    const int idx = (int) ((size_t) pair * n_rows + row);
                    mmvq_store<OutT>(gate_out, idx, sum_gate[s]);
                    mmvq_store<OutT>(up_out,   idx, sum_up[s]);
                }
            }
        }
    }
}

template<int MMQ_X, int MMQ_Y, int NWARPS, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q4_k_down_lds_tiled_row(
    const flambeau_block_q4_K* __restrict__ down_w,
    const flambeau_block_q8_1_mmq* __restrict__ y_mmq,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ dst,
    int n_rows, int n_pairs, int n_sb_per_row, int n_experts)
{
    constexpr int LANES = MMQ_LDS_LANES;
    constexpr int TILE_NE_K = 32;
    constexpr int QI4_K = 32;
    constexpr int QI8_1L = 8;
    constexpr int QK_K_L = 256;
    constexpr int QK8_1L = 32;
    constexpr int TILE_Y_K_LDS = TILE_NE_K + TILE_NE_K / QI8_1L;   // 36
    constexpr int TXS_QS = MMQ_Y * TILE_NE_K + MMQ_Y;
    constexpr int TXS_DM = MMQ_Y * TILE_NE_K / QI4_K;
    constexpr int sz_mmq_int = 144 / 4;

    extern __shared__ int lds[];
    int* tile_y = lds;
    int* tile_x = tile_y + MMQ_X * TILE_Y_K_LDS;

    int*       x_qs = tile_x;
    fb_fp16_t* x_dm = (fb_fp16_t*) (x_qs + TXS_QS);
    int*       x_sc = (int*) (x_dm + TXS_DM * 2);

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

    const flambeau_block_q4_K* down_w_expert = down_w + (size_t) expert * n_rows * n_sb_per_row;
    const int* y_int = (const int*) y_mmq;

    constexpr int n_sum = (MMQ_X / NWARPS) * ((MMQ_Y + LANES - 1) / LANES);
    float sum[n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) sum[s] = 0.0f;

    for (int kb0 = 0; kb0 < n_sb_per_row; ++kb0) {
        flambeau_moe_load_x_q4_k_lds<MMQ_Y, NWARPS>(down_w_expert, kb0, tile_m, n_rows, n_sb_per_row, x_qs, x_dm, x_sc);

        #pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int big_block = kb0 * (QK_K_L / (4 * QK8_1L)) + phase;
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * TILE_Y_K_LDS; l0 += NWARPS * LANES) {
                const int l = l0 + threadIdx.y * LANES + threadIdx.x;
                if (l < MMQ_X * TILE_Y_K_LDS) {
                    const int c_in_tile = l / TILE_Y_K_LDS;
                    const int intra     = l % TILE_Y_K_LDS;
                    const int pair      = slot_pair[c_in_tile];
                    tile_y[l] = y_int[((size_t) big_block * n_pairs + pair) * sz_mmq_int + intra];
                }
            }
            __syncthreads();
            flambeau_moe_vec_dot_pass_q4_k<MMQ_X, MMQ_Y, NWARPS>(x_qs, x_dm, x_sc, tile_y, sum, phase * TILE_NE_K);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j_local = j0 + threadIdx.y;
        const int pair    = slot_pair[j_local];
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += LANES) {
            const int i = i0 + threadIdx.x;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const int s = j0 / NWARPS * ((MMQ_Y + LANES - 1) / LANES) + i0 / LANES;
                    mmvq_store<OutT>(dst, (int) ((size_t) pair * n_rows + row), sum[s]);
                }
            }
        }
    }
}
