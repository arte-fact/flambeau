#pragma once
// Fused K+V Q4_0 decode MMVQ with F16 destination: one launch produces both the
// K and V projections of a single decode token (they share shape + the Q8_1
// activation), writing F16 straight out (no F32 scratch + cast). 256 threads,
// groups of 4 lanes cooperate on one Q4_0 block per iter; two F32 accumulators
// reduce per warp then across warps. Backend-neutral: `dp4a` + `warp_reduce_sum`
// come from the arch header, int loads go through `load_int_b2` (Q4_0 `qs` sits
// at block offset 2 → a 4-byte read faults on NVIDIA), `WARP_SIZE` is the arch
// constant. The Q4_0 bias correction (`q∈[0,15]` centred at 8) matches the
// hand-written gfx906 kernel.
#include "quant_traits.cuh"

static __device__ __forceinline__ void mmvq_q4_0_kv_f16dst_row(
    const flambeau_block_q4_0* __restrict__ k_w,
    const flambeau_block_q4_0* __restrict__ v_w,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ k_out,
    fb_fp16_t* __restrict__ v_out,
    int n_rows_kv, int n_blocks_per_row)
{
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / WARP_SIZE;
    constexpr int BLOCKS_PER_ITER = THREADS / 4;

    const int row = blockIdx.x;
    if (row >= n_rows_kv) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_0* k_row = k_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_0* v_row = v_w + (size_t) row * n_blocks_per_row;

    float acc_k = 0.0f;
    float acc_v = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* kbk = k_row + b;
        const flambeau_block_q4_0* vbk = v_row + b;
        const flambeau_block_q8_1* by  = y + b;

        const int u_lo = load_int_b2(by->qs, lane4);
        const int u_hi = load_int_b2(by->qs, lane4 + 4);
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        {
            const int v = load_int_b2((const int8_t*) kbk->qs, lane4);
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) kbk->d;
            acc_k += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        {
            const int v = load_int_b2((const int8_t*) vbk->qs, lane4);
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) vbk->d;
            acc_v += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    acc_k = warp_reduce_sum(acc_k);
    acc_v = warp_reduce_sum(acc_v);

    __shared__ float s_k[WARPS];
    __shared__ float s_v[WARPS];
    if (lane == 0) {
        s_k[warp] = acc_k;
        s_v[warp] = acc_v;
    }
    __syncthreads();

    if (warp == 0) {
        float k = (lane < WARPS) ? s_k[lane] : 0.0f;
        float v = (lane < WARPS) ? s_v[lane] : 0.0f;
        #pragma unroll
        for (int off = WARPS / 2; off > 0; off >>= 1) {
            k += __shfl_xor(k, off, WARP_SIZE);
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            k_out[row] = (fb_fp16_t) k;
            v_out[row] = (fb_fp16_t) v;
        }
    }
}
