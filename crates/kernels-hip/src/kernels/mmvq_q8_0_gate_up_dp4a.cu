// mmvq_q8_0_gate_up_dp4a — fused gate+up Q8_0 dense MMVQ with DP4A.
// For the shared-expert FFN (Qwen3.6 et al) the `ffn_gate_shexp` and
// `ffn_up_shexp` matmuls share the same activation. A naive dispatch
// launches two independent `mmvq_q8_0_dp4a` kernels, reading the same
// Q8_1 activation twice from HBM.
// This kernel reads the activation ONCE per block and computes BOTH
// matmuls in the same inner loop. Same block/grid/thread layout as the
// VDR=2 Q8_0 DP4A kernel (256 threads, 1 row/block).
// Per-block: 2 dp4a against x.qs, 2 dp4a against gate_w.qs, 2 dp4a against
// up_w.qs ... wait, we only need 2 dp4a per block for the activation side
// (VDR=2), then split: gate_sumi = dp4a(gate_v, u, 0) + dp4a(gate_v+1, u+1),
// up_sumi = dp4a(up_v, u, 0) + dp4a(up_v+1, u+1). Shared: u loads.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU_BLOCK_THREADS 256
#define GU_WARPS_PER_BLOCK (GU_BLOCK_THREADS / WARP_SIZE)
#define GU_THREADS_PER_QBLK 4
#define GU_BLOCKS_PER_ITER (GU_BLOCK_THREADS / GU_THREADS_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_gu8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q8_0_gate_up_dp4a_q8_1(
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
    // max of the two n_rows is the grid dimension; each output is a no-op for
    // rows past its own limit. Allows asymmetric fusion (e.g. attn_qkv 8192 +
    // attn_gate 4096): shared activation read still saves HBM where both
    // outputs apply, and per-block launch overhead is halved everywhere.
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

    for (int b = block_idx; b < n_blocks_per_row; b += GU_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* gbk = g_row + b;
        const flambeau_block_q8_0* ubk = u_row + b;
        const flambeau_block_q8_1* by  = y + b;

        // Shared activation read — ONCE per block per thread (halves HBM
        // traffic on the Q8_1 side vs two independent mmvq launches).
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];
        const float d_y = (float) by->d;

        if (do_gate) {
            const int gi0 = ((const int*) gbk->qs)[lane_in_grp * 2 + 0];
            const int gi1 = ((const int*) gbk->qs)[lane_in_grp * 2 + 1];
            int sumi = flambeau_dp4a_gu8(gi0, yi0, 0);
            sumi     = flambeau_dp4a_gu8(gi1, yi1, sumi);
            acc_g += (float) gbk->d * d_y * (float) sumi;
        }
        if (do_up) {
            const int ui0 = ((const int*) ubk->qs)[lane_in_grp * 2 + 0];
            const int ui1 = ((const int*) ubk->qs)[lane_in_grp * 2 + 1];
            int sumi = flambeau_dp4a_gu8(ui0, yi0, 0);
            sumi     = flambeau_dp4a_gu8(ui1, yi1, sumi);
            acc_u += (float) ubk->d * d_y * (float) sumi;
        }
    }

    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_g[GU_WARPS_PER_BLOCK];
    __shared__ float s_u[GU_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < GU_WARPS_PER_BLOCK) ? s_g[lane] : 0.0f;
        float u = (lane < GU_WARPS_PER_BLOCK) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = GU_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
