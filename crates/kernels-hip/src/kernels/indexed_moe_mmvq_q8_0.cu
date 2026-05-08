// indexed_moe_mmvq_q8_0 — Q8_0 MMVQ with per-token expert routing.
// Sibling of `indexed_moe_mmvq_q4_k` (the Q4_K r1) and
// `indexed_moe_mmvq_q6_k` (the Q6_K) kernels; unblocks UD-Q8_K_XL GGUFs
// where MoE expert weights stay Q8_0 instead of the usual Q4_K.
// Inner math identical to `mmvq_q8_0_dp4a_vdr2.cu`: 256 threads/block,
// VDR=2 pattern (8 elements per thread per inner iter via two dp4a).
// Differences from the non-MoE kernel:
// - weight rows are selected by `expert_ids[token, slot_idx]`
// - activation is `y[token, :]` (shared across all top_k slots)
// - output is `dst[token, slot_idx, row]` layout, matching the combine
// kernel contract.
// Layout:
// weights [n_experts, n_rows, n_blocks_per_row] Q8_0 blocks
// activations [n_tokens, n_blocks_per_row] Q8_1 blocks (QK8_0 == QK8_1)
// expert_ids [n_tokens, top_k] i32
// output [n_tokens, top_k, n_rows] F32
// Launch:
// blockDim = { 256 } (same as mmvq_q8_0_dp4a_vdr2)
// gridDim = { n_rows, n_tokens * top_k, 1 }

#include "block_quant.cuh"
#include "gfx906.cuh"

#define IMQ8_BLOCK_THREADS 256
#define IMQ8_WARPS_PER_BLOCK (IMQ8_BLOCK_THREADS / WARP_SIZE)
#define IMQ8_THREADS_PER_QBLK 4
#define IMQ8_BLOCKS_PER_ITER (IMQ8_BLOCK_THREADS / IMQ8_THREADS_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_imq8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q8_0_dp4a_q8_1(
    const flambeau_block_q8_0* __restrict__ x,     // [n_experts, n_rows, n_blocks_per_row]
    const flambeau_block_q8_1* __restrict__ y,     // [n_tokens, n_blocks_per_row]
    const int* __restrict__ expert_ids,            // [n_tokens, top_k]
    float* __restrict__ dst,                       // [n_tokens, top_k, n_rows]
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

    const flambeau_block_q8_0* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += IMQ8_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y_row + b;

        // VDR=2: load 2 consecutive int32s (8 bytes = 8 Q8 quants) per thread.
        const int xi0 = ((const int*) bx->qs)[lane_in_grp * 2 + 0];
        const int xi1 = ((const int*) bx->qs)[lane_in_grp * 2 + 1];
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];

        int sumi = flambeau_dp4a_imq8(xi0, yi0, 0);
        sumi     = flambeau_dp4a_imq8(xi1, yi1, sumi);

        acc += (float) bx->d * (float) by->d * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_partials[IMQ8_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_partials[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < IMQ8_WARPS_PER_BLOCK) ? s_partials[lane] : 0.0f;
        #pragma unroll
        for (int off = IMQ8_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            const size_t dst_idx =
                ((size_t) token * top_k + slot_idx) * n_rows + row;
            dst[dst_idx] = v;
        }
    }
}
