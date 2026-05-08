// mmvq_q4_0_gate_up_t128_dp4a — fused gate+up Q4_0 with t128 schedule.
// Pairs the style 128-thread thin-block schedule (mmvq_q4_1_t128 /
// mmvq_q4_0_t128) with the gate+up activation-sharing pattern from cycle 1
// (mmvq_q4_0_gate_up_dp4a). Aimed at the same gfx906 latency-bound regime
// at decode (~10 % HBM utilisation): 128 t/block = 2 wave64s/CU lets gfx906
// run two concurrent blocks per CU, packing more in-flight work to hide
// HBM latency than the 256t baseline (1 block/CU at occupancy ceiling).
// Same per-pointer + asymmetric-row contract as mmvq_q4_0_gate_up_dp4a;
// only the thread count and per-thread per-iter loop bound change.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU4T128_BLOCK_THREADS 128
#define GU4T128_WARPS_PER_BLOCK (GU4T128_BLOCK_THREADS / WARP_SIZE)
#define GU4T128_BLOCKS_PER_ITER (GU4T128_BLOCK_THREADS / 4)

static __device__ __forceinline__ int flambeau_dp4a_gu4t128(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(GU4T128_BLOCK_THREADS)
void flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
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

    const flambeau_block_q4_0* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_0* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += GU4T128_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* gbk = g_row + b;
        const flambeau_block_q4_0* ubk = u_row + b;
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
            sumi = flambeau_dp4a_gu4t128(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu4t128(vi_hi, u_hi, sumi);
            const float d_x = (float) gbk->d;
            acc_g += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        if (do_up) {
            const int v = ((const int*) ubk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_gu4t128(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu4t128(vi_hi, u_hi, sumi);
            const float d_x = (float) ubk->d;
            acc_u += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_g[GU4T128_WARPS_PER_BLOCK];
    __shared__ float s_u[GU4T128_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < GU4T128_WARPS_PER_BLOCK) ? s_g[lane] : 0.0f;
        float u = (lane < GU4T128_WARPS_PER_BLOCK) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = GU4T128_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
