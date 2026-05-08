// indexed_moe_mmq_q4_k_down_tile8_dp4a — down-projection MoE MMQ.
// Structural sibling of `indexed_moe_mmq_q4_k_gate_up_tile8_dp4a.cu`. Same
// per-block layout (64 rows × 8 slots, wave64, all slots share expert
// via padded sort), different activation indexing:
// Unlike gate+up (activation indexed by `token`, weight by `expert`),
// the down projection consumes the per-pair SwiGLU'd activation:
// `activated_q8_1[pair_idx, n_sb_per_row_inter]`, where pair_idx =
// token * top_k + slot. Output is also per-pair.
// Padding slots repeat the last real pair_idx → redundant compute but
// no branches.

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

// see gate_up_tile8 sibling — same 1-wave/SIMD occupancy floor +
// same (WARP_SIZE, 2) fix.
extern "C" __global__ __launch_bounds__(WARP_SIZE, 2)
void flambeau_indexed_moe_mmq_q4_k_down_tile8_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ down_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,          // [n_experts + 1]
    float*      __restrict__ dst,
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

    // Cache per-col pair_idx (also doubles as activation row AND output row).
    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        slot_pair[c] = sorted_pair_idx_padded[tile_n + c];
    }

    const int blocks_per_row_x = n_sb_per_row;
    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    // Suppress unused-variable warning when we don't need n_tokens / top_k.
    (void) n_tokens; (void) top_k;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float d = 0.0f, dmin = 0.0f;
        uint8_t sub_sc[8] = {0};
        uint8_t sub_m [8] = {0};
        const flambeau_block_q4_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            d    = (float) bx->d;
            dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, bx->scales, &sub_sc[j], &sub_m[j]);
            }
        }

        float sumf_d[TILE_N];
        float sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int v[8] = {0};
            if (row_ok) {
                const int* ql_ptr = (const int*) (bx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_ptr[j];
                    v[j] = (half == 0)
                        ? (ql_word & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);
                }
            }

            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_d = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
                    sumi_d = dp4a(v[j], y_j, sumi_d);
                    sumi_y = dp4a(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * ((float) sumi_d) * sc_f;
                sumf_m[c] += d8 * ((float) sumi_y) * m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += d * sumf_d[c] - dmin * sumf_m[c];
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
}
