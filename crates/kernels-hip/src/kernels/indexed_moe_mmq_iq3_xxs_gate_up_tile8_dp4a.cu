// indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a — IQ3_XXS MoE MMQ tile8 gate+up.
// Decode mirrors mmq_iq3_xxs_wave64.cu: sub-block scale
// is (0.5 + (aux32 >> 28)) * 0.5; signs come from KSIGNS_IQ2XS LUT.

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
static __device__ __forceinline__ int apply_signs_packed(int g_u32, int sign4) {
    int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int mag = (g_u32 >> (8 * j)) & 0xFF;
        const int neg = (sign4 >> j) & 1;
        const int v   = neg ? -mag : mag;
        out |= (v & 0xFF) << (8 * j);
    }
    return out;
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_iq3_xxs_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_iq3_xxs* __restrict__ gate_w,
    const flambeau_block_iq3_xxs* __restrict__ up_w,
    const flambeau_block_q8_1*    __restrict__ y,
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
        const flambeau_block_iq3_xxs* gbx = nullptr;
        const flambeau_block_iq3_xxs* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            int g_v[8] = {0};
            int u_v[8] = {0};
            float g_sf = 0.0f, u_sf = 0.0f;
            if (row_ok) {
                const uint8_t* g_scs = gbx->qs + QK_K / 4;
                const uint8_t* u_scs = ubx->qs + QK_K / 4;
                const uint8_t* g_sp = g_scs + 4 * sub;
                const uint8_t* u_sp = u_scs + 4 * sub;
                const uint32_t g_aux32 =
                      (uint32_t) g_sp[0] | ((uint32_t) g_sp[1] << 8)
                    | ((uint32_t) g_sp[2] << 16) | ((uint32_t) g_sp[3] << 24);
                const uint32_t u_aux32 =
                      (uint32_t) u_sp[0] | ((uint32_t) u_sp[1] << 8)
                    | ((uint32_t) u_sp[2] << 16) | ((uint32_t) u_sp[3] << 24);
                g_sf = (0.5f + (float)(g_aux32 >> 28)) * 0.5f;
                u_sf = (0.5f + (float)(u_aux32 >> 28)) * 0.5f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int g1_idx = (int) gbx->qs[sub * 8 + 2 * l + 0];
                    const int g2_idx = (int) gbx->qs[sub * 8 + 2 * l + 1];
                    const int u1_idx = (int) ubx->qs[sub * 8 + 2 * l + 0];
                    const int u2_idx = (int) ubx->qs[sub * 8 + 2 * l + 1];
                    const int g1 = (int) IQ3XXS_GRID[g1_idx];
                    const int g2 = (int) IQ3XXS_GRID[g2_idx];
                    const int uu1 = (int) IQ3XXS_GRID[u1_idx];
                    const int uu2 = (int) IQ3XXS_GRID[u2_idx];
                    const uint8_t g_sb = KSIGNS_IQ2XS[(g_aux32 >> (7 * l)) & 0x7F];
                    const uint8_t u_sb = KSIGNS_IQ2XS[(u_aux32 >> (7 * l)) & 0x7F];
                    g_v[2 * l + 0] = apply_signs_packed(g1,  g_sb        & 0x0F);
                    g_v[2 * l + 1] = apply_signs_packed(g2, (g_sb >> 4)  & 0x0F);
                    u_v[2 * l + 0] = apply_signs_packed(uu1, u_sb        & 0x0F);
                    u_v[2 * l + 1] = apply_signs_packed(uu2, (u_sb >> 4) & 0x0F);
                }
            }

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
                sums_gate[c] += g_d * d8 * (float) g_sumi * g_sf;
                sums_up[c]   += u_d * d8 * (float) u_sumi * u_sf;
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
