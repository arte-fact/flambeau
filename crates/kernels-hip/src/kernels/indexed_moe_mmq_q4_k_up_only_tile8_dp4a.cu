// Up-only half of the fused MoE Q4_K MMQ tile8 kernel. Mirror of the
// gate-only variant — see that file's header for the structural
// rationale. Same VGPR-pressure relief: only `u_v[8]` per thread.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK_K
#define QK_K 256
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 2)
void flambeau_indexed_moe_mmq_q4_k_up_only_tile8_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row,
    const int n_experts
) {
    const int tile_m = blockIdx.x * WARP_SIZE;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    __shared__ int padded_total_shared;
    if (tid == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert = expert_ids[first_pair];

    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        const int t = pair / top_k;
        const int s = pair - t * top_k;
        slot_token[c] = t;
        slot_out_idx[c] = t * top_k + s;
    }

    const int blocks_per_row_x = n_sb_per_row;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums_up[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float u_d = 0.0f, u_dmin = 0.0f;
        uint8_t u_sub_sc[8] = {0}, u_sub_m[8] = {0};

        const flambeau_block_q4_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            ubx = &up_w[w_row_off];
            u_d    = (float) ubx->d;
            u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, ubx->scales, &u_sub_sc[j], &u_sub_m[j]);
            }
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int u_v[8] = {0};
            if (row_ok) {
                const int* u_ql_ptr = (const int*) (ubx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int uw = u_ql_ptr[j];
                    u_v[j] = (half == 0) ? (uw & 0x0F0F0F0F) : ((uw >> 4) & 0x0F0F0F0F);
                }
            }

            const float ud_sc = u_d    * (float) u_sub_sc[sub];
            const float ud_m  = u_dmin * (float) u_sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_ud = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
                    sumi_ud = dp4a(u_v[j], y_j, sumi_ud);
                    sumi_y  = dp4a(0x01010101, y_j, sumi_y);
                }
                const float sumi_y_f = (float) sumi_y;
                sums_up[c] += d8 * ((float) sumi_ud * ud_sc - sumi_y_f * ud_m);
            }
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_out_idx[c] * n_rows + row;
        up_out[out_idx] = sums_up[c];
    }
}
