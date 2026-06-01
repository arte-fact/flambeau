// indexed_moe_mmvq_q8_0_gate_up_dp4a — fused gate+up Q8_0 indexed-MoE
// MMVQ. Sibling of `indexed_moe_mmvq_q8_0.cu` (single-weight) and
// `indexed_moe_mmvq_q4_0_gate_up_dp4a.cu` (Q4_0 gate+up). Reads each
// Q8_1 activation int32 word once per block and produces both gate and
// up outputs, halving the MoE decode launch count for Q8_0 expert
// weights (gemma4-26B-A4B, qwen3-moe Q8_K UD families).
//
// Same per-block shape as the single-weight variant: blockDim=256,
// gridDim={n_rows, n_tokens*top_k, 1}, VDR=2 DP4A inner loop.
// Layout:
//   gate_w / up_w  [n_experts, n_rows, n_blocks_per_row] Q8_0
//   y              [n_tokens, n_blocks_per_row]          Q8_1 (QK8_0==QK8_1)
//   expert_ids     [n_tokens, top_k]                     i32
//   gate_out / up_out [n_tokens, top_k, n_rows]          F32

#include "block_quant.cuh"
#include "gfx906.cuh"

#define IMQ8GU_BLOCK_THREADS 256
#define IMQ8GU_WARPS (IMQ8GU_BLOCK_THREADS / WARP_SIZE)
#define IMQ8GU_THREADS_PER_QBLK 4
#define IMQ8GU_BLOCKS_PER_ITER (IMQ8GU_BLOCK_THREADS / IMQ8GU_THREADS_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_imq8gu(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q8_0_gate_up_dp4a_q8_1(
    const flambeau_block_q8_0* __restrict__ gate_w,
    const flambeau_block_q8_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
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

    const int tid         = threadIdx.x;
    const int warp        = tid / WARP_SIZE;
    const int lane        = tid & (WARP_SIZE - 1);
    const int lane_in_grp = tid & 3;
    const int block_idx   = tid >> 2;

    const flambeau_block_q8_0* g_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_0* u_row =
        up_w   + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += IMQ8GU_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* gx = g_row + b;
        const flambeau_block_q8_0* ux = u_row + b;
        const flambeau_block_q8_1* by = y_row + b;

        // VDR=2: 2 int32s (8 Q8 quants) per thread per inner iter.
        const int gi0 = ((const int*) gx->qs)[lane_in_grp * 2 + 0];
        const int gi1 = ((const int*) gx->qs)[lane_in_grp * 2 + 1];
        const int uxi0 = ((const int*) ux->qs)[lane_in_grp * 2 + 0];
        const int uxi1 = ((const int*) ux->qs)[lane_in_grp * 2 + 1];
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];

        int sumi_g = flambeau_dp4a_imq8gu(gi0, yi0, 0);
        sumi_g     = flambeau_dp4a_imq8gu(gi1, yi1, sumi_g);
        int sumi_u = flambeau_dp4a_imq8gu(uxi0, yi0, 0);
        sumi_u     = flambeau_dp4a_imq8gu(uxi1, yi1, sumi_u);

        acc_g += (float) gx->d * (float) by->d * (float) sumi_g;
        acc_u += (float) ux->d * (float) by->d * (float) sumi_u;
    }

    acc_g = gfx906_warp_reduce_sum(acc_g);
    acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_warp_g[IMQ8GU_WARPS];
    __shared__ float s_warp_u[IMQ8GU_WARPS];
    if (lane == 0) {
        s_warp_g[warp] = acc_g;
        s_warp_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float vg = (lane < IMQ8GU_WARPS) ? s_warp_g[lane] : 0.0f;
        float vu = (lane < IMQ8GU_WARPS) ? s_warp_u[lane] : 0.0f;
        #pragma unroll
        for (int off = IMQ8GU_WARPS / 2; off > 0; off >>= 1) {
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
