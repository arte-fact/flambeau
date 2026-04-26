// mmvq_q8_0_gate_up_t128_vdr2 — fused gate+up Q8_0 with combined t128 + VDR=2.
//
// C9-followup-2 sibling of `mmvq_q8_0_gate_up_dp4a` (256t, VDR=2). Same
// activation-sharing fusion (one Q8_1 read per block, both gate and up
// outputs), but at 128 t/block to pack 2 wave64s/CU = 2 in-flight blocks/CU
// at the gfx906 occupancy ceiling. C9-followup proved this combo wins +3 %
// on 27B-Q8_0 single-row decode; the same lever should compose with the
// gate+up fusion for the dense FFN call sites.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU8T128_BLOCK_THREADS 128
#define GU8T128_WARPS_PER_BLOCK (GU8T128_BLOCK_THREADS / WARP_SIZE)
#define GU8T128_THREADS_PER_QBLK 4
#define GU8T128_BLOCKS_PER_ITER (GU8T128_BLOCK_THREADS / GU8T128_THREADS_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_gu8t128(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(GU8T128_BLOCK_THREADS)
void flambeau_mmvq_q8_0_gate_up_t128_vdr2_q8_1(
    const flambeau_block_q8_0* __restrict__ gate_w,
    const flambeau_block_q8_0* __restrict__ up_w,
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

    const int tid         = threadIdx.x;
    const int warp        = tid / WARP_SIZE;
    const int lane        = tid & (WARP_SIZE - 1);
    const int lane_in_grp = tid & 3;
    const int block_idx   = tid >> 2;

    const flambeau_block_q8_0* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q8_0* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += GU8T128_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* gbk = g_row + b;
        const flambeau_block_q8_0* ubk = u_row + b;
        const flambeau_block_q8_1* by  = y + b;

        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];
        const float d_y = (float) by->d;

        if (do_gate) {
            const int gi0 = ((const int*) gbk->qs)[lane_in_grp * 2 + 0];
            const int gi1 = ((const int*) gbk->qs)[lane_in_grp * 2 + 1];
            int sumi = flambeau_dp4a_gu8t128(gi0, yi0, 0);
            sumi     = flambeau_dp4a_gu8t128(gi1, yi1, sumi);
            acc_g += (float) gbk->d * d_y * (float) sumi;
        }
        if (do_up) {
            const int ui0 = ((const int*) ubk->qs)[lane_in_grp * 2 + 0];
            const int ui1 = ((const int*) ubk->qs)[lane_in_grp * 2 + 1];
            int sumi = flambeau_dp4a_gu8t128(ui0, yi0, 0);
            sumi     = flambeau_dp4a_gu8t128(ui1, yi1, sumi);
            acc_u += (float) ubk->d * d_y * (float) sumi;
        }
    }

    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_g[GU8T128_WARPS_PER_BLOCK];
    __shared__ float s_u[GU8T128_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < GU8T128_WARPS_PER_BLOCK) ? s_g[lane] : 0.0f;
        float u = (lane < GU8T128_WARPS_PER_BLOCK) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = GU8T128_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
