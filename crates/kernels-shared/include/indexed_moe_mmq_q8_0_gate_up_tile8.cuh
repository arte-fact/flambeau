#pragma once
// Templated fused gate+up Q8_0 indexed-MoE MMQ, tile8 (register-tiled prefill).
// MMQ_Y=64 weight rows (one per thread) × TILE_N=8 slot-columns per block: each
// thread holds its row's Q8_0 weight block (8 int32) in registers and runs the
// dp4a against 8 sorted slot activations, writing 8 gate + 8 up outputs — no LDS,
// no cross-thread comms, so the block height is a logical tile (MMQ_Y), NOT a
// hardware warp (correct on wave64 AND wave32). Experts are pre-sorted +
// padded-to-8 (`sorted_pair_idx_padded` / `padded_offsets`) so all 8 slots in a
// block share one expert. Q8_0 is signed/centred → the dot collapses to
// `x_d·d_y·sumi`. Backend-neutral: `dp4a` from the arch header; weight qs (block
// offset 2) loads through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"

template<class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_q8_0_gate_up_tile8_row(
    const flambeau_block_q8_0* __restrict__ gate_w,
    const flambeau_block_q8_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
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

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        float g_d = 0.0f, u_d = 0.0f;
        int g_v[8] = {0};
        int u_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_q8_0* gbx = &gate_w[w_row_off];
            const flambeau_block_q8_0* ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                g_v[j] = load_int_b2((const int8_t*) gbx->qs, j);
                u_v[j] = load_int_b2((const int8_t*) ubx->qs, j);
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;

            int sumi_g = 0, sumi_u = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int y_j = load_int_b2(by->qs, j);
                sumi_g = dp4a(g_v[j], y_j, sumi_g);
                sumi_u = dp4a(u_v[j], y_j, sumi_u);
            }
            sums_gate[c] += g_d * d8 * (float) sumi_g;
            sums_up[c]   += u_d * d8 * (float) sumi_u;
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_out_idx[c] * n_rows + row;
        gate_out[out_idx] = sums_gate[c];
        up_out[out_idx]   = sums_up[c];
    }

    (void) n_tokens;
}
