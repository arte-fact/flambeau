#pragma once
// Fused gate+up Q4_0 decode MMVQ, t128 schedule (128 threads/block). One launch
// produces both the gate and up projections of a decode token from the shared
// Q8_1 activation; asymmetric rows (`do_gate`/`do_up`) cover FFNs whose gate and
// up differ in row count. Backend-neutral: `dp4a` + `warp_reduce_sum` from the
// arch header, int loads via `load_int_b2` (Q4_0 `qs` at block offset 2 →
// 4-byte read faults on NVIDIA), `WARP_SIZE` the arch constant.
#include "quant_traits.cuh"

static __device__ __forceinline__ void mmvq_q4_0_gate_up_t128_row(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    int n_rows_gate, int n_rows_up, int n_blocks_per_row)
{
    constexpr int THREADS = 128;
    constexpr int WARPS = THREADS / WARP_SIZE;
    constexpr int BLOCKS_PER_ITER = THREADS / 4;

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

    for (int b = block_idx; b < n_blocks_per_row; b += BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* gbk = g_row + b;
        const flambeau_block_q4_0* ubk = u_row + b;
        const flambeau_block_q8_1* by  = y + b;

        const int u_lo = load_int_b2(by->qs, lane4);
        const int u_hi = load_int_b2(by->qs, lane4 + 4);
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        if (do_gate) {
            const int v = load_int_b2((const int8_t*) gbk->qs, lane4);
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) gbk->d;
            acc_g += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        if (do_up) {
            const int v = load_int_b2((const int8_t*) ubk->qs, lane4);
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) ubk->d;
            acc_u += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    if (do_gate) acc_g = warp_reduce_sum(acc_g);
    if (do_up)   acc_u = warp_reduce_sum(acc_u);

    __shared__ float s_g[WARPS];
    __shared__ float s_u[WARPS];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < WARPS) ? s_g[lane] : 0.0f;
        float u = (lane < WARPS) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = WARPS / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
