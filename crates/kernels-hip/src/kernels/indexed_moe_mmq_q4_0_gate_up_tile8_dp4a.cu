// indexed_moe_mmq_q4_0_gate_up_tile8_dp4a — fused gate+up MoE MMQ for
// Q4_0 expert weights.
// Mirrors `indexed_moe_mmq_q4_k_gate_up_tile8_dp4a.cu`: 64 threads /
// wave64, MMQ_Y=64 rows/block, TILE_N=8 slot-cols/block, padded-sort
// expert routing — but substitutes Q4_K's super-block decode for
// Q4_0's flat 32-elem block dequant. Same bias-correction identity as
// dense Q4_0 MMQ:
// (q - 8) · y = dp4a(q, y) - 8 · y_s, y_s = d_y · Σ q8
// and no per-sub-block min → folded into `sums[c] += x_d · (y_d·sumi − 8·y_s)`.
// Used by Qwen3.6-35B-A3B-Q4_0 MoE prefill (Q4_0 ffn_gate/up/down_exps
// are the dominant prefill fraction).
// Launch:
// grid = (⌈n_rows / 64⌉, padded_total / 8, 1)
// block = (64, 1, 1)

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK4_0
#define QK4_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,         // [n_experts + 1]
    float*      __restrict__ gate_out,
    float*      __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_blocks_per_row,
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

    // All 8 slots in this block share the same expert (pad-to-8).
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

    const int blocks_per_row_x = n_blocks_per_row;

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        // Decode Q4_0 weight block(s) for this thread's row — 4 int32 of qs,
        // split into low (elements 0..15) and high (elements 16..31) nibbles.
        float g_d = 0.0f, u_d = 0.0f;
        int g_v[8] = {0};
        int u_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * blocks_per_row_x + ib;
            const flambeau_block_q4_0* gbx = &gate_w[w_row_off];
            const flambeau_block_q4_0* ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
            const int* g_ql = (const int*) gbx->qs;
            const int* u_ql = (const int*) ubx->qs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gw = g_ql[j];
                const int uw = u_ql[j];
                g_v[j]     = (gw >> 0) & 0x0F0F0F0F;
                g_v[j + 4] = (gw >> 4) & 0x0F0F0F0F;
                u_v[j]     = (uw >> 0) & 0x0F0F0F0F;
                u_v[j + 4] = (uw >> 4) & 0x0F0F0F0F;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const float s8 = (float) by->s;  // d_y · Σ q8
            const int* y_packed = (const int*) by->qs;

            int sumi_g = 0, sumi_u = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int y_j = y_packed[j];
                sumi_g = dp4a(g_v[j], y_j, sumi_g);
                sumi_u = dp4a(u_v[j], y_j, sumi_u);
            }
            // (q-8)·y per-block: x_d · (d_y · sumi − 8 · y_s)
            sums_gate[c] += g_d * (d8 * (float) sumi_g - 8.0f * s8);
            sums_up[c]   += u_d * (d8 * (float) sumi_u - 8.0f * s8);
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_out_idx[c] * n_rows + row;
        gate_out[out_idx] = sums_gate[c];
        up_out[out_idx]   = sums_up[c];
    }
}
