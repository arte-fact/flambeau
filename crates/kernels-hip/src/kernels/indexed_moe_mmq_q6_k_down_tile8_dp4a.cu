// indexed_moe_mmq_q6_k_down_tile8_dp4a — V2.8.b Q6_K down-projection MoE MMQ.
//
// Closes the last MMVQ-at-prefill hole in Qwen3.6-35B-A3B-UD-Q4_K_S. In the
// UD mixed-quant GGUF, 6 of the 40 layers have Q6_K `ffn_down_exps` (the
// rest are Q4_K). V2.8.a rocprofv3 at Mesh<2> pp=512 attributed 97 ms
// (10.2 % of prefill, 16 ms / call × 6 calls) to `indexed_moe_mmvq_q6_k`,
// with PMC showing MemBusy 46 % / VALUBusy 21 % / 16 k waves per call —
// the classic prefill-on-MMVQ latency-bound pattern. MMVQ emits 1 output
// per block; a wave64 MMQ tile emits 64 × 8 = 512.
//
// Structure: fuses the V2.6.b `indexed_moe_mmq_q4_k_down_tile8_dp4a` tile
// layout with the V2.3.b.3 `mmq_q6_K_wave64` Q6_K decode path (raw·y
// -32·Σy bias-correction — avoids the byte-borrow bug that bit V2.3.d.1).
//
// Per-block invariant (from V2.6.a padded sort):
//   All 8 slots in a block map to the same expert. Weight slab loaded
//   per-thread exactly once per sub-block, reused across all 8
//   activation columns via the 8-dp4a inner loop.
//
// Launch:
//   grid = (ceil(n_rows / 64), padded_total / 8, 1)
//   block = (64, 1, 1)

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

// V2.9.b: see Q4_K tile8 siblings — same occupancy-floor fix.
extern "C" __global__ __launch_bounds__(WARP_SIZE, 2)
void flambeau_indexed_moe_mmq_q6_k_down_tile8_dp4a_q8_1(
    const flambeau_block_q6_K* __restrict__ down_w,
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
        float super_d = 0.0f;
        int8_t sc_buf[16] = {0};

        const flambeau_block_q6_K* bx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            bx = &down_w[w_row_off];
            super_d = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                sc_buf[j] = bx->scales[j];
            }
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int h     = sub >> 2;     // 0 or 1
            const int q_idx = sub & 3;      // 0..3
            const int qh_shift = 2 * q_idx;

            // Decode 32 Q6_K weights into 8 packed int32 (4 unsigned values each,
            // byte range [0, 63]). See mmq_q6_K_wave64.cu for the layout
            // derivation. We DO NOT apply the -32 shift byte-wise; we compensate
            // after the DP4A via the (raw-32)·y = raw·y - 32·Σy identity.
            int v[8] = {0};
            if (row_ok) {
                const int ql_base = 64 * h + ((q_idx & 1) ? 32 : 0);
                const int qh_base = 32 * h;
                const int* ql_words = (const int*) (bx->ql + ql_base);
                const int* qh_words = (const int*) (bx->qh + qh_base);

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_words[j];
                    const int qh_word = qh_words[j];

                    const int nib4 = (q_idx < 2)
                        ? ((ql_word >> 0) & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);

                    const int hi4 = ((qh_word >> qh_shift) << 4) & 0x30303030;

                    v[j] = nib4 | hi4;
                }
            }

            // Q6_K scales: 16 scales per super-block, one per 16 elements.
            // A 32-element sub-block covers 2 scale slots.
            const int sc_a = (int) sc_buf[8 * h + 2 * q_idx + 0];
            const int sc_b = (int) sc_buf[8 * h + 2 * q_idx + 1];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const flambeau_block_q8_1* by =
                    &y[(size_t) slot_pair[c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + sub];
                const float d8 = (float) by->d;
                const int* y_packed = (const int*) by->qs;

                int sumi_a = 0, sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi_a = dp4a(v[j], y_packed[j], sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_packed[j], sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    sumi_b = dp4a(v[j], y_packed[j], sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_packed[j], sumi_y_b);
                }

                const int corrected_a = sumi_a - 32 * sumi_y_a;
                const int corrected_b = sumi_b - 32 * sumi_y_b;

                sums[c] += super_d * d8 *
                    ((float)(sc_a * corrected_a + sc_b * corrected_b));
            }
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
}
