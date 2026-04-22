// indexed_moe_mmvq_q4_k_r4_dp4a — MoE Q4_K down-projection MMVQ, 4 rows per block.
//
// r4 extension of `indexed_moe_mmvq_q4_k_r2_dp4a.cu`. Used for the down
// projection in `forward_moe_ffn_{prefill,decode}`. Halves block count
// again vs r2 (and quarters vs the 1-row baseline):
//   - r2: grid = (n_rows / 2, n_tokens * top_k)
//   - r4: grid = (n_rows / 4, n_tokens * top_k)
//
// Lane layout (wave64, 4 rows simultaneously):
//   row_idx  = lane >> 4            — 0..3: which row in the quad
//   lane_lo  = lane & 15            — 0..15: quarter-warp position
//
// Each 16-lane quarter-warp handles 1 row. Q4_K super-block has 32 int32s
// of qs (128 bytes); 16 lanes cover them with 2 ints/lane per super-block
// → inner loop does 2 dp4a chains instead of 1 (r2) per super-block.
//
// Expected win: V2.4.a measured +19 % prefill from the same transition
// (baseline→r4) on the gate_up kernel; this kernel takes ~400 ms / 21 %
// of prefill, so proportional savings give another +4–5 % end-to-end.

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ int flambeau_dp4a_q4k_r4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_r4_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row_quad = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (token >= n_tokens) return;

    const int lane    = threadIdx.x;               // 0..63
    const int row_idx = lane >> 4;                 // 0..3
    const int lane_lo = lane & 15;                 // 0..15

    const int row = row_quad * 4 + row_idx;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q4_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    // Each lane handles 2 iqs positions (iqs_base and iqs_base + 4).
    const int pair_idx = lane_lo >> 2;             // 0..3
    const int iqs_base = lane_lo & 3;              // 0..3
    const int sub_lo   = pair_idx * 2;             // 0, 2, 4, 6
    const int sub_hi   = sub_lo + 1;               // 1, 3, 5, 7

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        uint8_t sc_lo = 0, m_lo = 0, sc_hi = 0, m_hi = 0;
        flambeau_q4k_scale_min(sub_lo, bk->scales, &sc_lo, &m_lo);
        flambeau_q4k_scale_min(sub_hi, bk->scales, &sc_hi, &m_hi);

        const flambeau_block_q8_1* ya_lo = y_row + (b * 8 + sub_lo);
        const flambeau_block_q8_1* ya_hi = y_row + (b * 8 + sub_hi);
        const float d_y_lo = (float) ya_lo->d;
        const float d_y_hi = (float) ya_hi->d;

        #pragma unroll
        for (int ipair = 0; ipair < 2; ++ipair) {
            const int iqs = iqs_base + ipair * 4;

            const int qs_int = ((const int*) bk->qs)[pair_idx * 8 + iqs];
            const int q_lo = qs_int & 0x0F0F0F0F;
            const int q_hi = (qs_int >> 4) & 0x0F0F0F0F;

            const int u_lo = ((const int*) ya_lo->qs)[iqs];
            const int u_hi = ((const int*) ya_hi->qs)[iqs];

            const int sumi_lo = flambeau_dp4a_q4k_r4(q_lo, u_lo, 0);
            const int sumi_hi = flambeau_dp4a_q4k_r4(q_hi, u_hi, 0);
            const int summ_lo = flambeau_dp4a_q4k_r4(0x01010101, u_lo, 0);
            const int summ_hi = flambeau_dp4a_q4k_r4(0x01010101, u_hi, 0);

            acc += d_y_lo *
                   (d * (float) sc_lo * (float) sumi_lo - dmin * (float) m_lo * (float) summ_lo);
            acc += d_y_hi *
                   (d * (float) sc_hi * (float) sumi_hi - dmin * (float) m_hi * (float) summ_hi);
        }
    }

    acc = gfx906_quarter_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
