// indexed_moe_mmq_q4_k_gate_up_tile16_dp4a — V2.31.b NULL. Kept under
// arch-rule-10. Measured Qwen3.6-35B-A3B-UD-Q4_K_S Mesh<4> prefill:
//
//   L=128: tile8 493.8 → tile16 297.9 tok/s  (−39.7 %)
//   L=512: tile8 699.6 → tile16 469.9 tok/s  (−32.8 %)
//   L=1024: tile8 710.0 → tile16 507.2 tok/s (−28.6 %)
//   decode tg=64 unchanged; last_id=447 bit-exact (parity preserved)
//
// Ship gate was ≥ 750 tok/s at L=512 (+9.8 %). Measured −33 % → KILL.
//
// why: V2.31.a floor projection (+11 %) assumed V2.6.b tile8 was
// partially weight-HBM-bound at ~65 % MemBusy per V2.9.a notes. The
// measurement disproves that — 2× activation reuse barely moved
// per-row time, meaning tile8 is actually compute-bound on the Q4_K
// DP4A decode ladder (8 sub-blocks × 8 DP4As × TILE_N). Doubling
// TILE_N adds more compute per block without reducing per-output
// compute, so the 48 % padding-redundant-fraction at pad-to-16
// (vs 30 % at pad-to-8) has nothing to offset it. Plus the extra 16
// FP32 accumulators (sums_{gate,up}[16]) likely push VGPR past the
// 10-waves/SIMD threshold → occupancy loss compounds the compute
// regression.
//
// Lesson: V2.6.b tile8 register-resident is at or near the per-row
// optimum on gfx906 Q4_K MoE prefill. Further wins need a different
// structural lever (e.g., cross-layer pipelining, CUDA-Graph batching,
// dense-tensor-core — none available on gfx906), NOT wider tiles.
//
// Ruled out for re-entry: TILE_N ≥ 16 alone. Possible re-entry: a
// different decode ladder (skip some DP4As via sub-block layout
// reorganisation) that reduces per-output compute AND allows wider
// tile. Unlikely given this null plus V2.14.d's turbo-@-8 null.
//
// Not registered in KERNEL_STEMS; build.rs's non-recursive enumerate
// skips _unverified/. The ops wrapper `indexed_moe_mmq_q4_k_gate_up_tile16`
// in crates/ops/src/hip/moe.rs is kept but annotated unreachable.
// pad-to-16 sort infrastructure kept — generic, may serve future
// tile16-class attempts with different kernel internals.
//
// indexed_moe_mmq_q4_k_gate_up_tile16_dp4a — V2.31.b tile16 (MMQ_X=16)
// register-resident fused gate+up MoE MMQ for Q4_K.
//
// Direct extension of V2.6.b's `indexed_moe_mmq_q4_k_gate_up_tile8_dp4a.cu`:
//   MMQ_Y  = 64 (unchanged, 1 wave64, 1 row/thread)
//   TILE_N = 16 (doubled from 8 — each block covers 16 slot-cols)
//   sums_gate/up[16] instead of [8] (+16 FP32 accumulators/thread)
//   Everything else is byte-identical: weights in registers, Y from L1,
//   0 LDS, 0 syncthreads.
//
// V2.31.a scope rationale — V2.6.b at MMQ_X=8 has ~65 % MemBusy on gfx906
// per V2.9.a PMC notes; doubling activation reuse per weight-register-
// resident read gets close to 1.5× per-row speedup. Paired with pad-to-16
// sort overhead (+35 % GPU work vs pad-to-8), net projected band is
// +11 % to +48 % on 35B-A3B-UD-Q4_K_S prefill.
//
// Requires pad-to-16 sort (V2.31.b `moe_sort_by_expert_padded_16`). Uses
// the same V2.6.a padding invariant — all 16 slot-cols in a block share
// the same expert, padded slots repeat the last real pair_idx.
//
// Launch:
//   grid  = (⌈n_rows / 64⌉, padded_total_16 / 16, 1)
//   block = (64, 1, 1)
//
// VGPR budget: V2.6.b tile8 had ~48 VGPR (sums × 2, scales × 8, gqld × 2,
// etc.). Doubling sums to sums × 16 adds 16 VGPR → ~64 VGPR total. Still
// under the 128 VGPR / 2 waves-per-SIMD threshold on gfx906 per V2.3.b
// wave64 Q4_K precedent (which ships at 64 VGPR).

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
#define TILE_N 16

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_q4_k_gate_up_tile16_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,          // [n_experts + 1]
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

    __shared__ int padded_total_shared;
    if (tid == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    // All 16 slot-cols share the same expert (V2.31.b pad-to-16 invariant).
    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert = expert_ids[first_pair];

    // Per-slot (token, slot_idx) decode.
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
    constexpr int q8_per_super = QK_K / QK8_1;            // = 8

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
