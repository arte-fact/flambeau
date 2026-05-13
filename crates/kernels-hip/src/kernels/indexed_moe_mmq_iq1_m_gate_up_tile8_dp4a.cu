// indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a — IQ1_M MoE MMQ tile8 gate+up.
// IQ1_M has no per-block `d` — reassembled from spread nibbles in `scales`.
// Per-l (dl, delta) vary; the per-l sum_qi is computed via dp4a-with-ones.
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
static __device__ __forceinline__ float iq1m_reassemble_d(const uint8_t* __restrict__ scales) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_iq1_m_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_iq1_m* __restrict__ gate_w,
    const flambeau_block_iq1_m* __restrict__ up_w,
    const flambeau_block_q8_1*  __restrict__ y,
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

    constexpr int q8_per_super = QK_K / QK8_1;
    float sums_gate[TILE_N], sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    (void) n_tokens;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float g_d = 0.0f, u_d = 0.0f;
        const flambeau_block_iq1_m* gbx = nullptr;
        const flambeau_block_iq1_m* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = iq1m_reassemble_d(gbx->scales);
            u_d = iq1m_reassemble_d(ubx->scales);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int g_v[8] = {0}, u_v[8] = {0};
            float g_dl_l[4] = {0}, u_dl_l[4] = {0};
            float g_delta_l[4] = {0}, u_delta_l[4] = {0};
            if (row_ok) {
                const int g_sw = (int) gbx->scales[2 * (sub >> 1)]
                               | ((int) gbx->scales[2 * (sub >> 1) + 1] << 8);
                const int u_sw = (int) ubx->scales[2 * (sub >> 1)]
                               | ((int) ubx->scales[2 * (sub >> 1) + 1] << 8);
                const int sh1 = 6 * (sub & 1);
                const int sh2 = sh1 + 3;
                const float g_dl1 = g_d * (2.0f * (float)((g_sw >> sh1) & 7) + 1.0f);
                const float g_dl2 = g_d * (2.0f * (float)((g_sw >> sh2) & 7) + 1.0f);
                const float u_dl1 = u_d * (2.0f * (float)((u_sw >> sh1) & 7) + 1.0f);
                const float u_dl2 = u_d * (2.0f * (float)((u_sw >> sh2) & 7) + 1.0f);
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int qh_pick    = (l < 2) ? (2 * sub) : (2 * sub + 1);
                    const uint8_t g_qh   = gbx->qh[qh_pick];
                    const uint8_t u_qh   = ubx->qh[qh_pick];
                    const int shift_idx  = 8 - 4 * (l & 1);
                    const int g_idx = (int) gbx->qs[4 * sub + l]
                                    | (((int) g_qh << shift_idx) & 0x700);
                    const int u_idx = (int) ubx->qs[4 * sub + l]
                                    | (((int) u_qh << shift_idx) & 0x700);
                    const int delta_bit  = (l & 1) == 0 ? 0x08 : 0x80;
                    g_dl_l[l]    = (l < 2) ? g_dl1 : g_dl2;
                    u_dl_l[l]    = (l < 2) ? u_dl1 : u_dl2;
                    g_delta_l[l] = (g_qh & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;
                    u_delta_l[l] = (u_qh & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;
                    const uint64_t g_u64 = IQ1S_GRID[g_idx];
                    const uint64_t u_u64 = IQ1S_GRID[u_idx];
                    g_v[2 * l + 0] = (int)(g_u64 & 0xFFFFFFFF);
                    g_v[2 * l + 1] = (int)((g_u64 >> 32) & 0xFFFFFFFF);
                    u_v[2 * l + 0] = (int)(u_u64 & 0xFFFFFFFF);
                    u_v[2 * l + 1] = (int)((u_u64 >> 32) & 0xFFFFFFFF);
                }
            }

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;
                float g_partial = 0.0f, u_partial = 0.0f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int g_sumi = 0, u_sumi = 0, sumi_y = 0;
                    g_sumi = dp4a(g_v[2 * l + 0], y_packed[2 * l + 0], g_sumi);
                    g_sumi = dp4a(g_v[2 * l + 1], y_packed[2 * l + 1], g_sumi);
                    u_sumi = dp4a(u_v[2 * l + 0], y_packed[2 * l + 0], u_sumi);
                    u_sumi = dp4a(u_v[2 * l + 1], y_packed[2 * l + 1], u_sumi);
                    sumi_y = dp4a(0x01010101, y_packed[2 * l + 0], sumi_y);
                    sumi_y = dp4a(0x01010101, y_packed[2 * l + 1], sumi_y);
                    g_partial += g_dl_l[l] * d8 * ((float) g_sumi + g_delta_l[l] * (float) sumi_y);
                    u_partial += u_dl_l[l] * d8 * ((float) u_sumi + u_delta_l[l] * (float) sumi_y);
                }
                sums_gate[c] += g_partial;
                sums_up[c]   += u_partial;
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
