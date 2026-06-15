// mmvq_q5_1 — Q5_1 weight × Q8_1 activation → {F32, F16} dst.
// Q5_1 layout: 2-byte d + 2-byte m + 4-byte qh + 16 bytes nibbles. Each
// element reconstructs as `y = d · q5 + m` where `q5 = (qh_bit_i << 4) |
// nibble_i` (∈ [0, 31]).
// Dot with Q8_1 activation `z` (where `d_y · Σ z_i = s_y`):
// Σ y_i · d_y · z_i = d · d_y · Σ q5_i · z_i + m · s_y
// = d · d_y · (sumi_nib + 16·sumi_bit) + m · s_y
// where sumi_nib = Σ nibble_i · z_i (packed DP4A on low nibbles) and
// sumi_bit = Σ bit_i · z_i (packed DP4A on expanded 5th bits).
// Same threading + nibble-pairing pattern as mmvq_q5_0 / mmvq_q4_1:
// lane4 = tid & 3, block_idx = tid >> 2
// u_lo = y.qs[lane4] (elements 4·lane4 .. +3)
// u_hi = y.qs[lane4 + 4] (elements 4·lane4+16 .. +19)
// Per block: 4 DP4A (nibble_lo + nibble_hi + bit_lo + bit_hi) + `m·s_y/4`.
// Block/grid: blockDim=256, gridDim=n_rows.
//
// Two output dtypes via templated __device__ body:
//   flambeau_mmvq_q5_1_q8_1      → F32 dst (legacy scratch-then-cast)
//   flambeau_mmvq_q5_1_q8_1_f16  → F16 dst (saturating; direct store)

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q5_1_THREADS 256
#define MMVQ_Q5_1_WARPS (MMVQ_Q5_1_THREADS / WARP_SIZE)
#define MMVQ_Q5_1_INT32_PER_BLOCK 4
#define MMVQ_Q5_1_BLOCKS_PER_ITER (MMVQ_Q5_1_THREADS / MMVQ_Q5_1_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q5_1_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Same expand_bits4 pattern as mmvq_q5_0: turn 4 consecutive qh bits
// starting at `start` into 4 bytes (each 0 or 1), packed into an int32.
static __device__ __forceinline__ int expand_bits4(unsigned int qh, int start) {
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

template<typename OutT>
__device__ void mmvq_q5_1_q8_1_body(
    const flambeau_block_q5_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q5_1* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q5_1_BLOCKS_PER_ITER) {
        const flambeau_block_q5_1* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        const int v    = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        const unsigned int qh = *((const unsigned int*) bx->qh);
        const int bit_lo = expand_bits4(qh, lane4 * 4);
        const int bit_hi = expand_bits4(qh, lane4 * 4 + 16);

        int sumi_nib = 0;
        sumi_nib = flambeau_q5_1_dp4a(vi_lo, u_lo, sumi_nib);
        sumi_nib = flambeau_q5_1_dp4a(vi_hi, u_hi, sumi_nib);
        int sumi_bit = 0;
        sumi_bit = flambeau_q5_1_dp4a(bit_lo, u_lo, sumi_bit);
        sumi_bit = flambeau_q5_1_dp4a(bit_hi, u_hi, sumi_bit);

        const float d_x = (float) bx->d;
        const float m_x = (float) bx->m;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // d_x · d_y · (sumi_nib + 16·sumi_bit) + m_x · s_y.
        // Correction m_x·s_y split across 4 lanes → ×0.25 per thread.
        acc += (sumi_nib + 16 * sumi_bit) * (d_x * d_y)
             + m_x * s_y * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q5_1_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q5_1_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q5_1_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q5_1_q8_1(
    const flambeau_block_q5_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q5_1_q8_1_body<float>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q5_1_q8_1_f16(
    const flambeau_block_q5_1* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q5_1_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_blocks_per_row);
}
