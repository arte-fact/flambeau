// indexed_moe_mmq_q5_k_down_tile8_dp4a — 1.a Q5_K down-projection MoE MMQ.
// Closes the MMVQ-at-prefill hole in Qwen3-Coder-30B-A3B-UD-Q4_K_XL. In the
// UD mixed-quant GGUF, 13 of 48 layers have Q5_K `ffn_down_exps` (the rest
// Q4_K / Q6_K). 0.b rocprofv3 at Mesh<4> pp=512 attributed 640 ms
// (31.56 % of prefill wall, 9.8 ms / call × 65 calls) to
// `indexed_moe_mmvq_q5_k` — classic prefill-on-MMVQ latency-bound pattern.
// Structure: fuses `indexed_moe_mmq_q4_k_down_tile8_dp4a` tile
// layout (Q4_K + dmin — same scales header) with the 5th-bit `qh` decode
// from `mmvq_q5_k.cu` (bit `2*il + half` of `qh[lane]` contributes 16 to
// each raw_q). No byte-borrow risk (Q4/Q5 nibbles stay unsigned).
// Per-block invariant (padded sort): all 8 slots share the same
// expert → weight slab loaded per-thread exactly once per sub-block,
// reused across 8 activation columns via the 8-dp4a inner loop.
// Launch:
// grid = (ceil(n_rows / 64), padded_total / 8, 1)
// block = (64, 1, 1)

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

extern "C" __global__ __launch_bounds__(WARP_SIZE, 2)
void flambeau_indexed_moe_mmq_q5_k_down_tile8_dp4a_q8_1(
    const flambeau_block_q5_K* __restrict__ down_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,          // [n_experts + 1]
    float*      __restrict__ dst,
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

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        slot_pair[c] = sorted_pair_idx_padded[tile_n + c];
    }

    const int blocks_per_row_x = n_sb_per_row;
    constexpr int q8_per_super = QK_K / QK8_1;  // = 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    (void) n_tokens; (void) top_k;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float d = 0.0f, dmin = 0.0f;
        uint8_t sub_sc[8] = {0};
        uint8_t sub_m [8] = {0};
        const flambeau_block_q5_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            d    = (float) bx->d;
            dmin = (float) bx->dmin;
            // Same 12-byte scales layout as Q4_K.
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, bx->scales, &sub_sc[j], &sub_m[j]);
            }
        }

        float sumf_d[TILE_N];
        float sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il    = sub >> 1;
            const int half  = sub & 1;
            const int s_bit = 2 * il + half;   // qh bit position (see mmvq_q5_k.cu)

            // Decode 32 Q5_K weights for this sub-block into 8 packed int32
            // (4 unsigned values each, byte range [0, 31]).
            // nib4 = low or high 4-bit nibble of qs packed across 4 bytes.
            // hi4 = bit s_bit of qh, shifted left by 4 to add 16 when set.
            int v[8] = {0};
            if (row_ok) {
                const int* ql_ptr = (const int*) (bx->qs + 32 * il);
                const int* qh_ptr = (const int*) bx->qh;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_ptr[j];
                    const int qh_word = qh_ptr[j];

                    const int nib4 = (half == 0)
                        ? (ql_word & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);

                    // Extract bit s_bit from each byte → 0 or 1 per byte → ×16.
                    const int hi4 = ((qh_word >> s_bit) & 0x01010101) << 4;

                    v[j] = nib4 | hi4;
                }
            }

            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_d = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
                    sumi_d = dp4a(v[j], y_j, sumi_d);
                    sumi_y = dp4a(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * ((float) sumi_d) * sc_f;
                sumf_m[c] += d8 * ((float) sumi_y) * m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += d * sumf_d[c] - dmin * sumf_m[c];
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
}
