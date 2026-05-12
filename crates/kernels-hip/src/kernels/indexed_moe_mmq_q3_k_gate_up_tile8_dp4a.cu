// Q3_K MoE MMQ tile8 (gate_up). Fused gate + up projection over a shared
// activation slab. Same Q3_K decode as mmq_q3_K_wave64.cu — byte-wise u32
// loads (Q3_K block size 110 B alternates 4-byte / 2-byte aligned across
// blocks). Output indexed via `pair = sorted_pair_idx_padded[tile_n + c]`,
// then `(token, slot)` decoded for the canonical
// `[n_tokens, top_k, n_rows]` output layout.

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

static __device__ __forceinline__ uint32_t q3k_t8gu_load_u32(const uint8_t* p) {
    return (uint32_t) p[0]
         | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16)
         | ((uint32_t) p[3] << 24);
}

static __device__ __forceinline__ void q3k_t8gu_unpack_scales(
    const uint8_t* __restrict__ scales,
    int8_t out[16]
) {
    const uint32_t k1 = 0x0303'0303u;
    const uint32_t k2 = 0x0f0f'0f0fu;
    uint32_t aux[4];
    aux[0] = q3k_t8gu_load_u32(scales);
    aux[1] = q3k_t8gu_load_u32(scales + 4);
    const uint32_t tmp = q3k_t8gu_load_u32(scales + 8);
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    const uint8_t* bytes = (const uint8_t*) aux;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = (int8_t) bytes[i];
    }
}

static __device__ __forceinline__ void q3k_t8gu_decode_sub(
    bool row_ok,
    const flambeau_block_q3_K* __restrict__ bx,
    int blk128,
    int shift_iter,
    int v[8]
) {
    const int shift         = 2 * shift_iter;
    const int hmask_bit_pos = shift_iter + 4 * blk128;
    if (!row_ok) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = 0;
        return;
    }
    const uint8_t* qs_base = bx->qs + blk128 * 32;
    const uint8_t* hm_base = bx->hmask;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t ql_word = q3k_t8gu_load_u32(qs_base + j * 4);
        const uint32_t qh_word = q3k_t8gu_load_u32(hm_base + j * 4);
        const int raw2 = (int) ((ql_word >> shift) & 0x03030303u);
        const int hi   = (int) (((qh_word >> hmask_bit_pos) << 2) & 0x04040404u);
        v[j] = raw2 | hi;
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 2)
void flambeau_indexed_moe_mmq_q3_k_gate_up_tile8_dp4a_q8_1(
    const flambeau_block_q3_K* __restrict__ gate_w,
    const flambeau_block_q3_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
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

    constexpr int q8_per_super = QK_K / QK8_1;

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    (void) n_tokens;

    for (int ib = 0; ib < n_sb_per_row; ++ib) {
        float g_d = 0.0f, u_d = 0.0f;
        int8_t g_sc[16] = {0};
        int8_t u_sc[16] = {0};
        const flambeau_block_q3_K* gbx = nullptr;
        const flambeau_block_q3_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d = (float) gbx->d;
            u_d = (float) ubx->d;
            q3k_t8gu_unpack_scales(gbx->scales, g_sc);
            q3k_t8gu_unpack_scales(ubx->scales, u_sc);
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int blk128     = sub >> 2;
            const int shift_iter = sub & 3;

            int g_v[8], u_v[8];
            q3k_t8gu_decode_sub(row_ok, gbx, blk128, shift_iter, g_v);
            q3k_t8gu_decode_sub(row_ok, ubx, blk128, shift_iter, u_v);

            const int g_sca = (int) g_sc[2 * sub + 0] - 32;
            const int g_scb = (int) g_sc[2 * sub + 1] - 32;
            const int u_sca = (int) u_sc[2 * sub + 0] - 32;
            const int u_scb = (int) u_sc[2 * sub + 1] - 32;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_token[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int g_sumi_a = 0, g_sumi_b = 0;
                int u_sumi_a = 0, u_sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g_sumi_a = dp4a(g_v[j], y_packed[j], g_sumi_a);
                    u_sumi_a = dp4a(u_v[j], y_packed[j], u_sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_packed[j], sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    g_sumi_b = dp4a(g_v[j], y_packed[j], g_sumi_b);
                    u_sumi_b = dp4a(u_v[j], y_packed[j], u_sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_packed[j], sumi_y_b);
                }

                const int g_corr_a = g_sumi_a - 4 * sumi_y_a;
                const int g_corr_b = g_sumi_b - 4 * sumi_y_b;
                const int u_corr_a = u_sumi_a - 4 * sumi_y_a;
                const int u_corr_b = u_sumi_b - 4 * sumi_y_b;

                sums_gate[c] += g_d * d8 *
                    ((float)(g_sca * g_corr_a + g_scb * g_corr_b));
                sums_up[c] += u_d * d8 *
                    ((float)(u_sca * u_corr_a + u_scb * u_corr_b));
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
