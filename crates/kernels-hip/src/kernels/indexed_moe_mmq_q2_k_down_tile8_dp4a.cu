// Q2_K MoE MMQ tile8 (down). Affine quant — same shape as Q4_K/Q5_K down
// tile8 except the 2-bit weight decode and 4-bit packed (scale, min) per
// sub-block. Q2_K is 84 B / super-block (mod-4), no alignment workaround.

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
void flambeau_indexed_moe_mmq_q2_k_down_tile8_dp4a_q8_1(
    const flambeau_block_q2_K* __restrict__ down_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
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

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        slot_pair[c] = sorted_pair_idx_padded[tile_n + c];
    }

    constexpr int q8_per_super = QK_K / QK8_1;

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    (void) n_tokens; (void) top_k;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sc_buf[16] = {0};
        const flambeau_block_q2_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            super_d    = (float) bx->d;
            super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                sc_buf[j] = bx->scales[j];
            }
        }

        float sumf_d[TILE_N];
        float sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int chunk_idx  = sub >> 2;
            const int shift_iter = sub & 3;
            const int shift      = 2 * shift_iter;

            int v[8] = {0};
            if (row_ok) {
                const int* qs_words = (const int*) (bx->qs + chunk_idx * 32);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = qs_words[j];
                    v[j] = (ql_word >> shift) & 0x03030303;
                }
            }

            const int sc_a = (int) (sc_buf[2 * sub + 0] & 0xF);
            const int sc_b = (int) (sc_buf[2 * sub + 1] & 0xF);
            const int m_a  = (int) (sc_buf[2 * sub + 0] >> 4);
            const int m_b  = (int) (sc_buf[2 * sub + 1] >> 4);

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_a = 0, sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi_a = dp4a(v[j], y_packed[j], sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_packed[j], sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    sumi_b = dp4a(v[j], y_packed[j], sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_packed[j], sumi_y_b);
                }

                sumf_d[c] += d8 * (float) (sc_a * sumi_a + sc_b * sumi_b);
                sumf_m[c] += d8 * (float) (m_a  * sumi_y_a + m_b  * sumi_y_b);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
}
