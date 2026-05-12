// indexed_moe_mmvq_q3_k — Q3_K MMVQ with per-token expert routing.
// Single-row MMVQ pattern (64-thread wave64, 1 row/block) extended for MoE:
//   blockDim = { 64 }
//   gridDim  = { n_rows, n_tokens * top_k, 1 }
// Per block computes dst[token, slot_idx, row] = <weights[expert, row], act[token]>.
// Decode logic identical to mmvq_q3_k.cu including the byte-wise scales
// unpack idiom (110 B block stride leaves scales[] misaligned for every other
// super-block).

#include "block_quant.cuh"
#include "gfx906.cuh"

static __device__ __forceinline__ uint32_t q3k_moe_load_u32_unaligned(const uint8_t* p) {
    return (uint32_t) p[0]
         | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16)
         | ((uint32_t) p[3] << 24);
}

static __device__ __forceinline__ void q3k_moe_unpack_scales(
    const uint8_t* __restrict__ scales,
    int8_t out[16]
) {
    const uint32_t k1 = 0x0303'0303u;
    const uint32_t k2 = 0x0f0f'0f0fu;
    uint32_t aux[4];
    aux[0] = q3k_moe_load_u32_unaligned(scales);
    aux[1] = q3k_moe_load_u32_unaligned(scales + 4);
    const uint32_t tmp = q3k_moe_load_u32_unaligned(scales + 8);
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

extern "C" __global__ void flambeau_indexed_moe_mmvq_q3_k_q8_1(
    const flambeau_block_q3_K* __restrict__ x,     // [n_experts, n_rows, n_sb]
    const flambeau_block_q8_1* __restrict__ y,     // [n_tokens, n_sb * 8]
    const int* __restrict__ expert_ids,            // [n_tokens, top_k]
    float* __restrict__ dst,                       // [n_tokens, top_k, n_rows]
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;

    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int lane = threadIdx.x;             // 0..63

    const flambeau_block_q3_K* xrow =
        x + (((size_t) expert * n_rows) + row) * n_sb_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_sb_per_row * 8;

    float acc = 0.0f;

    for (int b = 0; b < n_sb_per_row; ++b) {
        const flambeau_block_q3_K* bk = xrow + b;
        const float d_all = (float) bk->d;
        int8_t scales[16];
        q3k_moe_unpack_scales(bk->scales, scales);

        const flambeau_block_q8_1* y_sb = y_row + b * 8;

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int y_idx         = lane + k * 64;
            const int blk128        = y_idx >> 7;
            const int within_128    = y_idx & 127;
            const int shift_iter    = within_128 >> 5;
            const int within_32     = within_128 & 31;
            const int scale_idx     = within_32 >> 4;
            const int l             = within_32 & 15;
            const int qi            = l + 16 * scale_idx;
            const int is            = 8 * blk128 + 2 * shift_iter + scale_idx;
            const int shift         = 2 * shift_iter;
            const int hmask_bit_pos = shift_iter + 4 * blk128;

            const int hmask_bit = (bk->hmask[qi] >> hmask_bit_pos) & 1;
            const int sub       = (hmask_bit == 0) ? 4 : 0;
            const int qs_byte   = bk->qs[blk128 * 32 + qi];
            const int q_val     = ((qs_byte >> shift) & 3) - sub;

            const float x_val = d_all * (float) (scales[is] - 32) * (float) q_val;

            const int y_block = y_idx >> 5;
            const int y_off   = y_idx & 31;
            const flambeau_block_q8_1* ya = y_sb + y_block;
            const float d_y = (float) ya->d;
            const int   qi8 = (int) ya->qs[y_off];
            acc += x_val * d_y * (float) qi8;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[((size_t) token * top_k + slot_idx) * n_rows + row] = acc;
    }
}
