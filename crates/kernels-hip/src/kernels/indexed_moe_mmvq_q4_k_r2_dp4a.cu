// indexed_moe_mmvq_q4_k_r2_dp4a — Q4_K MoE MMVQ with DP4A inner loop.
// Drop-in replacement for `indexed_moe_mmvq_q4_k_r2.cu`. Same block/grid
// shape (64 threads = 1 wave64, 2 rows per block), same output layout.
// Inner loop swaps per-element scalar `d*sc*raw_q*y - dmin*m*y` for llama.cpp's
// `vec_dot_q4_K_q8_1_impl_vmmq` DP4A pattern:
// - pack 4 nibbles into int32
// - dp4a(q_nibbles, u_q8, 0) → sum of 4 raw_q * raw_y products
// - dp4a(0x01010101, u_q8, 0) → sum of 4 raw_y values (for dmin*m subtraction)
// Per super-block, 32 lanes (half-warp per row) cover all 128 bytes of
// `bk->qs` as 32 int32s. Each lane owns ONE int32 = 4 low nibbles going to
// the even sub-block of a pair, + 4 high nibbles going to the odd sub-block.
// Lane layout:
// pair_idx = lane_lo >> 3 — 0..3 — which pair of sub-blocks (0:(0,1), 1:(2,3), …)
// iqs = lane_lo & 7 — 0..7 — which int32 of the pair's 32-byte slice
// Each lane does 4 dp4a calls → 16 int8×int8 MACs per super-block per lane,
// matching the scalar kernel's 8 scalar FMAs per super-block per lane but with
// ~2× better inner-loop throughput.
// Reference: /artefact/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:501-527.

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_r2_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_pair = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane     = threadIdx.x;              // 0..63
    const int row_hi   = lane >> 5;                // 0 → row R, 1 → row R+1
    const int lane_lo  = lane & 31;                // 0..31 — half-warp position

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int pair_idx = lane_lo >> 3;             // 0..3 — which sub-block pair
    const int iqs      = lane_lo & 7;              // 0..7 — which int32 in the pair's slice
    const int sub_lo   = pair_idx * 2;             // 0, 2, 4, 6 — low-nibble sub-block
    const int sub_hi   = sub_lo + 1;               // 1, 3, 5, 7 — high-nibble sub-block

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        // 4 bytes of qs = 4 low nibbles (→ sub_lo) + 4 high nibbles (→ sub_hi).
        const int qs_int = ((const int*) bk->qs)[pair_idx * 8 + iqs];
        const int q_lo = qs_int & 0x0F0F0F0F;
        const int q_hi = (qs_int >> 4) & 0x0F0F0F0F;

        // Per-sub-block 6-bit scale/min pairs.
        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

        const flambeau_block_q8_1* ya_lo = y_row + (b * 8 + sub_lo);
        const flambeau_block_q8_1* ya_hi = y_row + (b * 8 + sub_hi);

        const int u_lo = ((const int*) ya_lo->qs)[iqs];
        const int u_hi = ((const int*) ya_hi->qs)[iqs];

        const int sumi_lo = flambeau_dp4a_q4k(q_lo, u_lo, 0);
        const int sumi_hi = flambeau_dp4a_q4k(q_hi, u_hi, 0);
        const int summ_lo = flambeau_dp4a_q4k(0x01010101, u_lo, 0);
        const int summ_hi = flambeau_dp4a_q4k(0x01010101, u_hi, 0);

        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        // contrib = d_y * (d * sc * Σ(raw_q·raw_y) - dmin * m * Σ(raw_y))
        acc += d_y_lo *
               (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo);
        acc += d_y_hi *
               (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
