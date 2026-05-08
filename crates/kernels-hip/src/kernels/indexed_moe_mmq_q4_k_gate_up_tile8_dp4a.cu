// indexed_moe_mmq_q4_k_gate_up_tile8_dp4a — fused gate+up MoE MMQ.
// Structural lever closing the 2× gap to Qwen3.5-9B dense prefill:
// - 64 threads per block, 1 wave64
// - MMQ_Y = 64 output rows per block (1 row per thread)
// - TILE_N = 8 slot-cols per block (iterated per-thread)
// - Total 64 × 8 = 512 outputs per block
// All 8 slots in a block are GUARANTEED to map to the same expert
// (padded sort enforces this by padding each expert's range
// to a multiple of 8, with tail slots repeating the last real
// pair_idx). Weight slab is thus loaded per-thread exactly once per
// sub-block and reused across all 8 activation columns via the
// 8-dp4a inner loop — cache-efficient in both L1 and registers.
// Padding slots compute redundantly (same expert, same activation as
// the last real slot → same output); their output write is a duplicate
// that either overwrites or is overwritten. No per-slot validity check
// in the inner loop → branch-free hot path.
// Launch:
// grid = (ceil(n_rows / 64), padded_total / 8, 1)
// block = (64, 1, 1)
// Compared to sorted r4 gate_up (4 rows × 1 col = 4 outputs/block):
// 32× more outputs per block AND weight tile shared across 8 cols
// → expected 2-3× per-kernel speedup on prefill.

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

// `__launch_bounds__(WARP_SIZE, 1)` is intentional. (WARP_SIZE, 2) was
// 2.0× slower (Scratch 156 → 684 B). An inline-accumulator refactor
// (eliminating the per-super-block sumf_* FP32 transients) likewise
// failed to recover (_, 2) — the real spill sources are the
// `g_v[8]`/`u_v[8]` int weight packs + scale buffers across the
// unrolled super-block loop. Past 1 wave/SIMD needs a structural change
// (split gate/up, persistent-thread, or LDS-staged designs).
extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_q4_k_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,          // [n_experts + 1] — reads [n_experts] for total
    float*      __restrict__ gate_out,
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

    // Early-exit blocks past the actual padded total. Caller sets grid.y to
    // an upper bound; the kernel self-limits via the on-device padded_total.
    __shared__ int padded_total_shared;
    if (tid == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    // All 8 slots in this block share the same expert (padding
    // guarantee). Read once, use for the whole block.
    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert = expert_ids[first_pair];

    // Per-slot (token, slot_idx) decode. Cache to avoid re-reading padded idx
    // eight times in the inner loop.
    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        const int t = pair / top_k;
        const int s = pair - t * top_k;
        slot_token[c] = t;
        slot_out_idx[c] = t * top_k + s;  // flat index into gate_out / up_out
    }

    const int blocks_per_row_x = n_sb_per_row;       // super-blocks per row
    constexpr int q8_per_super = QK_K / QK8_1;       // = 8

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float g_d = 0.0f, g_dmin = 0.0f;
        float u_d = 0.0f, u_dmin = 0.0f;
        uint8_t g_sub_sc[8] = {0}, g_sub_m[8] = {0};
        uint8_t u_sub_sc[8] = {0}, u_sub_m[8] = {0};

        const flambeau_block_q4_K* gbx = nullptr;
        const flambeau_block_q4_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d    = (float) gbx->d;
            g_dmin = (float) gbx->dmin;
            u_d    = (float) ubx->d;
            u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, gbx->scales, &g_sub_sc[j], &g_sub_m[j]);
                flambeau_q4k_scale_min(j, ubx->scales, &u_sub_sc[j], &u_sub_m[j]);
            }
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                const int* g_ql_ptr = (const int*) (gbx->qs + 32 * il);
                const int* u_ql_ptr = (const int*) (ubx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gw = g_ql_ptr[j];
                    const int uw = u_ql_ptr[j];
                    g_v[j] = (half == 0) ? (gw & 0x0F0F0F0F) : ((gw >> 4) & 0x0F0F0F0F);
                    u_v[j] = (half == 0) ? (uw & 0x0F0F0F0F) : ((uw >> 4) & 0x0F0F0F0F);
                }
            }

            // Pre-fold block-level d/dmin with sub-block scales. Single
            // multiply per (sub, kernel) column done once, reused across all
            // TILE_N slots.
            const float gd_sc = g_d    * (float) g_sub_sc[sub];
            const float gd_m  = g_dmin * (float) g_sub_m [sub];
            const float ud_sc = u_d    * (float) u_sub_sc[sub];
            const float ud_m  = u_dmin * (float) u_sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_gd = 0, sumi_ud = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
                    sumi_gd = dp4a(g_v[j], y_j, sumi_gd);
                    sumi_ud = dp4a(u_v[j], y_j, sumi_ud);
                    sumi_y  = dp4a(0x01010101, y_j, sumi_y);
                }
                // Fold into persistent sums directly — no per-super-block
                // sumf_* transient arrays. Reorders FP32 adds within this
                // super-block vs the original (accumulator-then-scale).
                const float sumi_y_f = (float) sumi_y;
                sums_gate[c] += d8 * ((float) sumi_gd * gd_sc - sumi_y_f * gd_m);
                sums_up[c]   += d8 * ((float) sumi_ud * ud_sc - sumi_y_f * ud_m);
            }
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
