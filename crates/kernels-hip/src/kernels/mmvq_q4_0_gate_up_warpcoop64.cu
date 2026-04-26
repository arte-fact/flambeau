// mmvq_q4_0_gate_up_warpcoop64 — fused gate+up Q4_0, single-warp schedule.
// C6-i1 sibling of `mmvq_q4_0_warpcoop64`.
//
// 64 t/block (one wave64) = 16 Q4_0 blocks/iter × DP4A. Same gate+up
// activation-sharing pattern as `mmvq_q4_0_gate_up_t128_dp4a` (each thread
// reads y once per block, runs DP4A against gate-row weights, then up-row
// weights). With one warp per block, the cross-warp LDS reduce of the t128
// kernel collapses to a single in-place gfx906 DPP butterfly.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU4WC64_BLOCK_THREADS WARP_SIZE  // == 64 on gfx906
#define GU4WC64_BLOCKS_PER_ITER (GU4WC64_BLOCK_THREADS / 4)

static __device__ __forceinline__ int flambeau_dp4a_gu4wc64(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(GU4WC64_BLOCK_THREADS)
void flambeau_mmvq_q4_0_gate_up_warpcoop64_q8_1(
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
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_0* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_0* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += GU4WC64_BLOCKS_PER_ITER) {
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
            sumi = flambeau_dp4a_gu4wc64(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu4wc64(vi_hi, u_hi, sumi);
            const float d_x = (float) gbk->d;
            acc_g += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        if (do_up) {
            const int v = ((const int*) ubk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_gu4wc64(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu4wc64(vi_hi, u_hi, sumi);
            const float d_x = (float) ubk->d;
            acc_u += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    // Single warp — no LDS round-trip.
    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    if (tid == 0) {
        if (do_gate) gate_out[row] = acc_g;
        if (do_up)   up_out[row]   = acc_u;
    }
}
