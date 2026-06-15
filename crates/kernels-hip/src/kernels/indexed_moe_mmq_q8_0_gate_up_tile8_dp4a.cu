// indexed_moe_mmq_q8_0_gate_up_tile8_dp4a — fused gate+up MoE MMQ
// for Q8_0 expert weights.
// Direct sibling of `indexed_moe_mmq_q4_0_gate_up_tile8_dp4a.cu`
// with the nibble-unpack and (q-8) bias correction removed. Q8_0 is already
// signed/centred so the inner formula collapses to
// sums[c] += x_d · d_y · sumi.
// Q8_0 block: {fp16 d, int8 qs[32]} = 8 × int32 of signed quants per block.
// n_blocks_per_row = hidden / 32 (QK8_0 == QK8_1 == 32), same stride as
// the activation's Q8_1 blocks → one weight block ↔ one activation block
// in the k-loop.
// shipped Q8_0 indexed-MoE MMVQ (256 threads, 1 output/block).
// On Qwen3.6-35B-A3B-UD-Q8_K_XL prefill L=512 that kernel held 77 % of
// GPU time — the sole dominant hotspot. Tile8 delivers 64 rows
// × 8 slot-cols = 512 outputs per block with one weight-tile decode per
// thread per block, matching Q4_K structural win (~3× per-kernel).
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
#ifndef QK8_0
#define QK8_0 32
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
void flambeau_indexed_moe_mmq_q8_0_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q8_0* __restrict__ gate_w,
    const flambeau_block_q8_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,          // [n_experts + 1]
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

    // All 8 slots in this block share the same expert (pad-to-8
    // invariant, enforced by moe_sort_by_expert_padded).
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
        // Load Q8_0 weight blocks for this thread's row: 8 × int32 each
        // direct from qs[], no nibble decode, no sub-block scales.
        float g_d = 0.0f, u_d = 0.0f;
        int g_v[8] = {0};
        int u_v[8] = {0};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            const flambeau_block_q8_0* gbx = &gate_w[w_row_off];
            const flambeau_block_q8_0* ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
            const int* g_ql = (const int*) gbx->qs;
            const int* u_ql = (const int*) ubx->qs;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                g_v[j] = g_ql[j];
                u_v[j] = u_ql[j];
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            const float d8 = (float) by->d;
            const int* y_packed = (const int*) by->qs;

            int sumi_g = 0, sumi_u = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int y_j = y_packed[j];
                sumi_g = dp4a(g_v[j], y_j, sumi_g);
                sumi_u = dp4a(u_v[j], y_j, sumi_u);
            }
            // Q8_0 is signed/centred → no (q - 8) bias, no by->s term.
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
