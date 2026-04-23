// indexed_moe_mmvq_q4_0_gate_up_dp4a — V2.23.b.1 fused gate+up MoE MMVQ
// for Q4_0 expert weights. Sibling of `indexed_moe_mmvq_q4_0.cu` that
// reads the Q8_1 activation once per block and produces two outputs
// (gate and up), halving the launch count at decode where the tile8 MMQ
// path does not fire (n_tokens < 32).
//
// Same per-block shape as the single-weight variant: blockDim=256,
// gridDim={n_rows, n_tokens*top_k, 1}. Each thread owns one
// `lane4`-indexed Q4_0 block slice; the inner loop accumulates both
// gate and up partial sums from the same activation word.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define IMQ4_0GU_BLOCK_THREADS 256
#define IMQ4_0GU_WARPS (IMQ4_0GU_BLOCK_THREADS / WARP_SIZE)
#define IMQ4_0GU_INT32_PER_BLOCK 4
#define IMQ4_0GU_BLOCKS_PER_ITER (IMQ4_0GU_BLOCK_THREADS / IMQ4_0GU_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_imq4_0gu_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float*      __restrict__ gate_out,
    float*      __restrict__ up_out,
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

    const flambeau_block_q4_0* g_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q4_0* u_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += IMQ4_0GU_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* g_bx = g_row + b;
        const flambeau_block_q4_0* u_bx = u_row + b;
        const flambeau_block_q8_1* by   = y_row + b;

        const int vg = ((const int*) g_bx->qs)[lane4];
        const int vu = ((const int*) u_bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vg_lo = (vg >> 0) & 0x0F0F0F0F;
        const int vg_hi = (vg >> 4) & 0x0F0F0F0F;
        const int vu_lo = (vu >> 0) & 0x0F0F0F0F;
        const int vu_hi = (vu >> 4) & 0x0F0F0F0F;

        int sumi_g = 0, sumi_u = 0;
        sumi_g = flambeau_imq4_0gu_dp4a(vg_lo, u_lo, sumi_g);
        sumi_g = flambeau_imq4_0gu_dp4a(vg_hi, u_hi, sumi_g);
        sumi_u = flambeau_imq4_0gu_dp4a(vu_lo, u_lo, sumi_u);
        sumi_u = flambeau_imq4_0gu_dp4a(vu_hi, u_hi, sumi_u);

        const float d_g = (float) g_bx->d;
        const float d_u = (float) u_bx->d;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        acc_g += sumi_g * (d_g * d_y) - 8.0f * d_g * s_y * 0.25f;
        acc_u += sumi_u * (d_u * d_y) - 8.0f * d_u * s_y * 0.25f;
    }

    acc_g = gfx906_warp_reduce_sum(acc_g);
    acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_warp_g[IMQ4_0GU_WARPS];
    __shared__ float s_warp_u[IMQ4_0GU_WARPS];
    if (lane == 0) {
        s_warp_g[warp] = acc_g;
        s_warp_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float vg = (lane < IMQ4_0GU_WARPS) ? s_warp_g[lane] : 0.0f;
        float vu = (lane < IMQ4_0GU_WARPS) ? s_warp_u[lane] : 0.0f;
        #pragma unroll
        for (int off = IMQ4_0GU_WARPS / 2; off > 0; off >>= 1) {
            vg += __shfl_xor(vg, off, WARP_SIZE);
            vu += __shfl_xor(vu, off, WARP_SIZE);
        }
        if (lane == 0) {
            const size_t idx = ((size_t) token * top_k + slot_idx) * n_rows + row;
            gate_out[idx] = vg;
            up_out[idx]   = vu;
        }
    }
}
