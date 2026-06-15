// indexed_moe_mmvq_q4_1 — Q4_1 MMVQ with per-token expert routing.
// Unblocks Qwen-published Qwen3.6-35B-A3B-Q4_0 whose
// `ffn_down_exps` are Q4_1 (gate/up are Q4_0; down is Q4_1 because affine
// is friendlier to the down direction's wider dynamic range, a common
// quant-by-tensor-class choice).
// Sibling of `indexed_moe_mmvq_q4_0.cu` — same per-token expert routing
// + 256-thread block + (lane4, lane4 + 4) DP4A nibble pairing. Only the
// inner reconstruction differs:
// Q4_0 uses (q - 8) · y → sumi · (d_x · d_y) - 8 · d_x · s_y · 0.25
// Q4_1 uses q · y + m → sumi · (d_x · d_y) + m_x · s_y · 0.25
// Layout:
// weights [n_experts, n_rows, n_blocks_per_row] Q4_1 blocks (20 B)
// activations [n_tokens, n_blocks_per_row] Q8_1 blocks
// expert_ids [n_tokens, top_k] i32
// output [n_tokens, top_k, n_rows] F32
// Launch: blockDim=256, gridDim={n_rows, n_tokens*top_k, 1}.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define IMQ4_1_BLOCK_THREADS 256
#define IMQ4_1_WARPS (IMQ4_1_BLOCK_THREADS / WARP_SIZE)
#define IMQ4_1_INT32_PER_BLOCK 4
#define IMQ4_1_BLOCKS_PER_ITER (IMQ4_1_BLOCK_THREADS / IMQ4_1_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_imq4_1_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_1_q8_1(
    const flambeau_block_q4_1* __restrict__ x,
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

    const flambeau_block_q4_1* xrow =
        x + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row =
        y + (size_t) token * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += IMQ4_1_BLOCKS_PER_ITER) {
        const flambeau_block_q4_1* bx = xrow + b;
        const flambeau_block_q8_1* by = y_row + b;

        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = flambeau_imq4_1_dp4a(vi_lo, u_lo, sumi);
        sumi = flambeau_imq4_1_dp4a(vi_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float m_x = (float) bx->m;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // Per-block (m_x * s_y) constant divided by IMQ4_1_INT32_PER_BLOCK
        // so the warp reduce sums to exactly one (m_x * s_y) per Q4_1 block.
        acc += sumi * (d_x * d_y) + (m_x * s_y) * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[IMQ4_1_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < IMQ4_1_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = IMQ4_1_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            const size_t dst_idx =
                ((size_t) token * top_k + slot_idx) * n_rows + row;
            dst[dst_idx] = v;
        }
    }
}
