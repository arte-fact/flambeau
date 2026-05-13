#include "block_quant.cuh"
#include "gfx906.cuh"

#define IMQ5_1_BLOCK_THREADS 256
#define IMQ5_1_WARPS (IMQ5_1_BLOCK_THREADS / WARP_SIZE)
#define IMQ5_1_INT32_PER_BLOCK 4
#define IMQ5_1_BLOCKS_PER_ITER (IMQ5_1_BLOCK_THREADS / IMQ5_1_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_imq5_1_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int expand_bits4(unsigned int qh, int start) {
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q5_1_q8_1(
    const flambeau_block_q5_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ dst,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_blocks_per_row
) {
    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q5_1* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += IMQ5_1_BLOCKS_PER_ITER) {
        const flambeau_block_q5_1* bx = xrow + b;
        const flambeau_block_q8_1* by = y_row + b;

        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        const unsigned int qh = *((const unsigned int*) bx->qh);
        const int bit_lo = expand_bits4(qh, lane4 * 4);
        const int bit_hi = expand_bits4(qh, lane4 * 4 + 16);

        int sumi_nib = 0;
        sumi_nib = flambeau_imq5_1_dp4a(vi_lo, u_lo, sumi_nib);
        sumi_nib = flambeau_imq5_1_dp4a(vi_hi, u_hi, sumi_nib);
        int sumi_bit = 0;
        sumi_bit = flambeau_imq5_1_dp4a(bit_lo, u_lo, sumi_bit);
        sumi_bit = flambeau_imq5_1_dp4a(bit_hi, u_hi, sumi_bit);

        const float d_x = (float) bx->d;
        const float m_x = (float) bx->m;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        acc += (sumi_nib + 16 * sumi_bit) * (d_x * d_y) + m_x * s_y * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[IMQ5_1_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < IMQ5_1_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = IMQ5_1_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            const size_t dst_idx =
                ((size_t) token * top_k + slot_idx) * n_rows + row;
            dst[dst_idx] = v;
        }
    }
}
