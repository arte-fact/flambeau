#pragma once
// Templated down-projection Q8_0 indexed-MoE MMQ, tile8 (register-tiled prefill).
// The single-weight sibling of the gate+up tile8: one weight tensor
// (ffn_down_exps), activation indexed per-PAIR (`y[pair,:]`, top_k_inner=1),
// output `dst[pair,row]`. MMQ_Y=64 rows × TILE_N=8 slot-columns, register-tiled,
// no LDS/no cross-thread comms (block height is the logical tile MMQ_Y, not a
// hardware warp). Q8_0 dot collapses to `w_d·d_y·sumi`. Backend-neutral: `dp4a`
// from the arch header; weight qs (block offset 2) loads through `load_int_b2`.
#include "quant_traits.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q8_0_down_tile8_row(
    const flambeau_block_q8_0* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row, int n_experts)
{
    constexpr int MMQ_Y  = 64;
    constexpr int TILE_N = 8;

    const int tile_m = blockIdx.x * MMQ_Y;
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

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        float w_d = 0.0f;
        int w_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_q8_0* wbx = &w[w_row_off];
            w_d = (float) wbx->d;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                w_v[j] = load_int_b2((const int8_t*) wbx->qs, j);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_pair[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sumi = dp4a(w_v[j], load_int_b2(by->qs, j), sumi);
            }
            sums[c] += w_d * d8 * (float) sumi;
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
    (void) top_k; (void) n_tokens;
}
