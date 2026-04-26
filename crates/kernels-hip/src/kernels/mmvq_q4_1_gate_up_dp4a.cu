// mmvq_q4_1_gate_up_dp4a — fused gate+up Q4_1 dense MMVQ with DP4A.
//
// C8-i1. Sibling of `mmvq_q4_0_gate_up_dp4a` for Q4_1 weights. Closes the
// dense-FFN gate+up fusion gap on Qwen3.5-9B-Q4_1 / Qwen3.5-27B-Q4_1, where
// `ffn_gate` + `ffn_up` are both Q4_1 and currently take two unfused MMVQ
// launches per layer per rank (visible in cross-model bench: 9B-Q4_1
// 70 tok/s on Mesh<2>, with Q4_1 single-row MMVQ as the top kernel).
//
// Reads each Q8_1 activation word once per block; halves the launch count
// for any layer that calls the unfused pair. Same per-pointer +
// asymmetric-row contract as the Q4_0 fused kernel.
//
// Q4_1 reconstruction (vs Q4_0's `(q - 8)`): affine `y = d·q + m`, so the
// per-block correction is `+ m_x · s_y · 0.25` (split across 4 lanes per
// block) instead of `- 8 · d_x · s_y · 0.25`.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU41_BLOCK_THREADS 256
#define GU41_WARPS_PER_BLOCK (GU41_BLOCK_THREADS / WARP_SIZE)
#define GU41_INT32_PER_QBLK 4
#define GU41_BLOCKS_PER_ITER (GU41_BLOCK_THREADS / GU41_INT32_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_gu41(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(GU41_BLOCK_THREADS)
void flambeau_mmvq_q4_1_gate_up_dp4a_q8_1(
    const flambeau_block_q4_1* __restrict__ gate_w,
    const flambeau_block_q4_1* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;
    if (!do_gate && !do_up) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_1* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_1* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += GU41_BLOCKS_PER_ITER) {
        const flambeau_block_q4_1* gbk = g_row + b;
        const flambeau_block_q4_1* ubk = u_row + b;
        const flambeau_block_q8_1* by  = y + b;

        // Shared activation read — once per block per thread.
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        if (do_gate) {
            const int v = ((const int*) gbk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_gu41(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu41(vi_hi, u_hi, sumi);
            const float d_x = (float) gbk->d;
            const float m_x = (float) gbk->m;
            // Q4_1 affine: `+ m_x · s_y · 0.25` per block, split 4 ways.
            acc_g += sumi * (d_x * d_y) + (m_x * s_y) * 0.25f;
        }
        if (do_up) {
            const int v = ((const int*) ubk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_gu41(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu41(vi_hi, u_hi, sumi);
            const float d_x = (float) ubk->d;
            const float m_x = (float) ubk->m;
            acc_u += sumi * (d_x * d_y) + (m_x * s_y) * 0.25f;
        }
    }

    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_g[GU41_WARPS_PER_BLOCK];
    __shared__ float s_u[GU41_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < GU41_WARPS_PER_BLOCK) ? s_g[lane] : 0.0f;
        float u = (lane < GU41_WARPS_PER_BLOCK) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = GU41_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
