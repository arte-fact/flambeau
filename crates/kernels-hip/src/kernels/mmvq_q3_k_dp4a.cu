// mmvq_q3_k_dp4a — Q3_K weight × Q8_1 activation → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_q3_K_q8_1_impl_mmvq`
// (vecdotq.cuh:447). Same launch shape as `mmvq_q5_k_dp4a.cu`: 256
// threads/block, single output row, BLOCKS_PER_ITER super-blocks per
// outer iter.
//
// Q3_K block layout (256 elements, 16 scale groups × 16 elements):
//   hmask[32]    — 1 bit per element (high bit of the 3-bit value)
//   qs[64]       — 2 bits per element (low 2 bits, 4 per byte)
//   scales[12]   — 16 signed 6-bit scales (per 16-element group), packed
//   d (F16)      — super-block scale
//
// Threading: 64 threads per super-block. Each handles ONE (iqs, i) pair
// where iqs ∈ [0,QI3_K) = [0,16) and i ∈ [0,QR3_K) = [0,4). One DP4A
// covers 4 quants. 16 × 4 × 4 = 256 quants / super-block.
//
// Per-byte subtraction trap: the reference does
//   `vi = __vsubss4(vil, vih)` then `dp4a(vi, u, 0)`. On gfx906 there's
// no per-byte SIMD-subtract intrinsic, and an int32 `vil - vih` borrows
// across byte lanes when vil < vih. We split into two DP4A calls
// (one for `vil`, one for `vih`) and subtract scalar results before
// scaling — algebraically identical, no borrow.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q3_K_THREADS 256
#define MMVQ_Q3_K_WARPS (MMVQ_Q3_K_THREADS / WARP_SIZE)
#define MMVQ_Q3_K_THREADS_PER_BLOCK 64
#define MMVQ_Q3_K_BLOCKS_PER_ITER (MMVQ_Q3_K_THREADS / MMVQ_Q3_K_THREADS_PER_BLOCK)

#define Q3K_QI3_K  16
#define Q3K_QR3_K   4
#define Q3K_QI8_1   8

static __device__ __forceinline__ int flambeau_q3_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q3_k_dp4a_body(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid           = threadIdx.x;
    const int warp          = tid / WARP_SIZE;
    const int lane          = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = tid / MMVQ_Q3_K_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_Q3_K_THREADS_PER_BLOCK - 1);
    const int iqs           = lane_in_sb >> 2;          // 0..15
    const int i             = lane_in_sb & 3;           // 0..3

    const int bq8_offset    = Q3K_QR3_K * (iqs / (Q3K_QI3_K / 2));   // 0 or 4
    const int scale_offset  = iqs - (iqs & (Q3K_QI8_1 - 1))
                            + ((iqs & (Q3K_QI8_1 - 1)) / (Q3K_QI8_1 / 2));
    const int isc           = scale_offset + 2 * i;
    const int isc_low       = isc & 7;                  // % (QK_K/32) = % 8
    const int sc_shift_low  = 4 * (isc >> 3);
    const int isc_high      = isc & 3;                  // % (QK_K/64) = % 4
    const int sc_shift_high = 2 * (isc >> 2);

    const flambeau_block_q3_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row; sb += MMVQ_Q3_K_BLOCKS_PER_ITER) {
        const flambeau_block_q3_K* bk = xrow + sb;

        const float d = (float) bk->d;

        const int sc_low  = (bk->scales[isc_low] >> sc_shift_low) & 0xF;
        const int sc_high = ((bk->scales[(Q3K_QI3_K / 2) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc      = (sc_low | sc_high) - 32;

        const int vl      = ((const int*) bk->qs)[iqs];
        const int vh_full = ~((const int*) bk->hmask)[iqs & ((Q3K_QI3_K / 2) - 1)];
        const int vh      = vh_full >> bq8_offset;

        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + bq8_offset + i;
        const int u_i = ((const int*) ya->qs)[iqs & (Q3K_QI8_1 - 1)];
        const float d_y = (float) ya->d;

        const int sumi_l = flambeau_q3_k_dp4a(vil, u_i, 0);
        const int sumi_h = flambeau_q3_k_dp4a(vih, u_i, 0);
        const int sumi   = sumi_l - sumi_h;

        acc += d * (float) sc * d_y * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q3_K_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q3_K_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q3_K_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q3_k_dp4a_q8_1(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q3_k_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q3_k_dp4a_q8_1_f16(
    const flambeau_block_q3_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q3_k_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
