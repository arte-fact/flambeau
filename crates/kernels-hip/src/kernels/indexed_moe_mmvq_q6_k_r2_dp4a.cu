// indexed_moe_mmvq_q6_k_r2_dp4a — Q6_K MoE MMVQ with DP4A inner loop.
//
// Launch: blockDim = { 64 } (wave64), gridDim = { n_row_pairs,
// n_tokens * top_k, 1 }. 2 output rows per block (lanes 0..31 own row R,
// lanes 32..63 own row R+1). 32 iqs positions per super-block per row.
//
// Q6_K bias correction: raw_q is stored as unsigned [0, 63]; the −32 bias
// is folded out via the dp4a identity (raw − 32) * y = raw * y − 32 * Σ y
// (gfx906 has no per-byte saturating int8 subtract).

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"

#define Q6K_QR_M   2
#define Q6K_QI_M  32

static __device__ __forceinline__ int flambeau_indexed_moe_q6_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q6_k_r2_dp4a_q8_1(
    const flambeau_block_q6_K* __restrict__ x,
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

    const int lane    = threadIdx.x;
    const int row_hi  = lane >> 5;
    const int lane_lo = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const flambeau_block_q6_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    const int iqs = lane_lo;

    const int bq8_offset   = Q6K_QR_M * (iqs / (Q6K_QI_M / 2)) * 2
                           + (iqs % (Q6K_QI_M / 2)) / (Q6K_QI_M / 4);
    const int scale_offset = (Q6K_QI_M / 4) * (iqs / (Q6K_QI_M / 2))
                           + (iqs % (Q6K_QI_M / 2)) / (Q6K_QI_M / 8);
    const int vh_shift     = 2 * ((iqs % (Q6K_QI_M / 2)) / (Q6K_QI_M / 4));
    const int qh_int_idx   = (Q6K_QI_M / 4) * (iqs / (Q6K_QI_M / 2))
                           + iqs % (Q6K_QI_M / 4);

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q6_K* bk = xrow + b;

        const int vl = ((const int*) bk->ql)[iqs];
        const int vh = ((const int*) bk->qh)[qh_int_idx] >> vh_shift;

        const int8_t* sc_ptr = bk->scales + scale_offset;

        const flambeau_block_q8_1* ya0 = y_row + (b * 8 + bq8_offset + 0);
        const flambeau_block_q8_1* ya1 = y_row + (b * 8 + bq8_offset + 2);
        const int u0 = ((const int*) ya0->qs)[iqs % 8];
        const int u1 = ((const int*) ya1->qs)[iqs % 8];
        const float d8_0 = (float) ya0->d;
        const float d8_1 = (float) ya1->d;

        float sumf = 0.0f;
        {
            const int sc   = (int) sc_ptr[0];
            const int vil  = (vl >> 0) & 0x0F0F0F0F;
            const int vih  = ((vh >> 0) << 4) & 0x30303030;
            const int raw  = vil | vih;
            const int dot  = flambeau_indexed_moe_q6_k_dp4a(raw, u0, 0);
            const int sumy = flambeau_indexed_moe_q6_k_dp4a(0x01010101, u0, 0);
            sumf += d8_0 * ((float) (dot - 32 * sumy) * (float) sc);
        }
        {
            const int sc   = (int) sc_ptr[4];
            const int vil  = (vl >> 4) & 0x0F0F0F0F;
            const int vih  = ((vh >> 4) << 4) & 0x30303030;
            const int raw  = vil | vih;
            const int dot  = flambeau_indexed_moe_q6_k_dp4a(raw, u1, 0);
            const int sumy = flambeau_indexed_moe_q6_k_dp4a(0x01010101, u1, 0);
            sumf += d8_1 * ((float) (dot - 32 * sumy) * (float) sc);
        }

        const float d = (float) bk->d;
        acc += d * sumf;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
