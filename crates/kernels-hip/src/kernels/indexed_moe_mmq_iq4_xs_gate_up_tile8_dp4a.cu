// indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a — fused gate+up MoE MMQ for
// IQ4_XS expert weights.
// Mirrors `indexed_moe_mmq_q4_k_gate_up_tile8_dp4a` shape:
//   - 64 threads = one wave64 = 64 output rows per block (MMQ_Y).
//   - TILE_N = 8 slot-pairs per block (gate+up share the slot tile).
//   - sorted_pair_idx_padded for per-expert bucketing.
//   - inner = pack-LUT-then-dp4a over 8 i32 weight words × 8 i32 Q8_1 act.
// IQ4_XS decode mirrors `mmq_iq4_xs_wave64.cu`: pre-apply the IQ4_NL LUT
// to each 4-bit code → packed signed-i8 → dp4a vs Q8_1. No min /
// bias-correction term (IQ4_XS reconstruction is symmetric, just
// `d * ls * LUT[code]`).
// Phase 4 Slice C.

#include "block_quant.cuh"
#include "iq_grid.cuh"
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

static __device__ __forceinline__ int pack_iq4_lut(int nibbles) {
    const int b0 = (int) flambeau_iq4nl_lut(nibbles & 0xFF);
    const int b1 = (int) flambeau_iq4nl_lut((nibbles >>  8) & 0xFF);
    const int b2 = (int) flambeau_iq4nl_lut((nibbles >> 16) & 0xFF);
    const int b3 = (int) flambeau_iq4nl_lut((nibbles >> 24) & 0xFF);
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_iq4_xs_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_iq4_xs* __restrict__ gate_w,
    const flambeau_block_iq4_xs* __restrict__ up_w,
    const flambeau_block_q8_1*   __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
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

    // All 8 slots in this block share the same expert (pad-to-8 bucketing).
    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert     = expert_ids[first_pair];

    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        const int t = pair / top_k;
        const int s = pair - t * top_k;
        slot_token[c]   = t;
        slot_out_idx[c] = t * top_k + s;
    }

    constexpr int q8_per_super = QK_K / QK8_1;  // 8

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    (void) n_tokens;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float g_d = 0.0f, u_d = 0.0f;
        uint16_t g_sh = 0, u_sh = 0;
        const flambeau_block_iq4_xs* gbx = nullptr;
        const flambeau_block_iq4_xs* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d  = (float) gbx->d;
            u_d  = (float) ubx->d;
            g_sh = gbx->scales_h;
            u_sh = ubx->scales_h;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int g_v[8] = {0};
            int u_v[8] = {0};
            int g_ls = 0, u_ls = 0;
            if (row_ok) {
                g_ls = flambeau_iq4_xs_scale(sub, g_sh, gbx->scales_l);
                u_ls = flambeau_iq4_xs_scale(sub, u_sh, ubx->scales_l);
                const int* g_ql = (const int*) (gbx->qs + 16 * sub);
                const int* u_ql = (const int*) (ubx->qs + 16 * sub);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int gw = g_ql[j];
                    const int uw = u_ql[j];
                    g_v[j]     = pack_iq4_lut(gw        & 0x0F0F0F0F);
                    g_v[j + 4] = pack_iq4_lut((gw >> 4) & 0x0F0F0F0F);
                    u_v[j]     = pack_iq4_lut(uw        & 0x0F0F0F0F);
                    u_v[j + 4] = pack_iq4_lut((uw >> 4) & 0x0F0F0F0F);
                }
            }

            const float g_lsf = (float) g_ls;
            const float u_lsf = (float) u_ls;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int g_sumi = 0, u_sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    g_sumi = dp4a(g_v[j], y_packed[j], g_sumi);
                    u_sumi = dp4a(u_v[j], y_packed[j], u_sumi);
                }
                sums_gate[c] += g_d * d8 * (float) g_sumi * g_lsf;
                sums_up[c]   += u_d * d8 * (float) u_sumi * u_lsf;
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
