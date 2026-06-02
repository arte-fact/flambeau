// mmvq_q2_k_dp4a — Q2_K weight × Q8_1 activation → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_q2_K_q8_1_impl_mmvq` (vecdotq.cuh:364).
// 256 threads/block, single output row, 64 threads per super-block,
// BLOCKS_PER_ITER = 4. Same launch shape as `mmvq_q3_k_dp4a.cu`
// (QI2_K = QI3_K = 16, QR2_K = QR3_K = 4).
//
// Q2_K block layout (256 elements, 16 scale groups × 16 elements):
//   scales[16]   — packed (4-bit sc | 4-bit m) per 16-element group
//   qs[64]       — 2 bits per element (4 per byte)
//   d (F16)      — super-block scale
//   dmin (F16)   — super-block min scale (applied to the m-term)
//
// Per (iqs, i) lane:
//   vi = (qs[iqs] >> (2*i)) & 0x03030303    — 4 packed 2-bit quants
//   sc_byte = scales[scale_offset + 2*i]
//   d-term: d · sc_lo · d_y · dp4a(vi, u[i], 0)
//   m-term: dmin · m_lo · d_y · dp4a(0x01010101, u[i], 0)   (= Σu[i])
// d and dmin are per-super-block, so the inner subtraction happens
// inside the loop.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q2_K_THREADS 256
#define MMVQ_Q2_K_WARPS (MMVQ_Q2_K_THREADS / WARP_SIZE)
#define MMVQ_Q2_K_THREADS_PER_BLOCK 64
#define MMVQ_Q2_K_BLOCKS_PER_ITER (MMVQ_Q2_K_THREADS / MMVQ_Q2_K_THREADS_PER_BLOCK)

#define Q2K_QI2_K  16
#define Q2K_QR2_K   4
#define Q2K_QI8_1   8

static __device__ __forceinline__ int flambeau_q2_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q2_k_dp4a_body(
    const flambeau_block_q2_K* __restrict__ x,
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
    const int sb_idx_in_iter = tid / MMVQ_Q2_K_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_Q2_K_THREADS_PER_BLOCK - 1);
    const int iqs           = lane_in_sb >> 2;          // 0..15
    const int i             = lane_in_sb & 3;           // 0..3

    const int bq8_offset    = Q2K_QR2_K * (iqs / Q2K_QI8_1);  // 0 or 4
    const int scale_offset  = iqs - (iqs & (Q2K_QI8_1 - 1))
                            + ((iqs & (Q2K_QI8_1 - 1)) / (Q2K_QI8_1 / 2));

    const flambeau_block_q2_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_Q2_K_BLOCKS_PER_ITER) {
        const flambeau_block_q2_K* bk = xrow + sb;

        const float d_sb    = (float) bk->d;
        const float dmin_sb = (float) bk->dmin;

        const int sc_byte = bk->scales[scale_offset + 2 * i];
        const int sc_lo   = sc_byte & 0x0F;
        const int m_4bit  = (sc_byte >> 4) & 0x0F;
        const int m_packed = m_4bit * 0x01010101;

        const int v = ((const int*) bk->qs)[iqs];
        const int vi = (v >> (2 * i)) & 0x03030303;

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + bq8_offset + i;
        const int u_i = ((const int*) ya->qs)[iqs & (Q2K_QI8_1 - 1)];
        const float d_y = (float) ya->d;

        const int sumi_d = flambeau_q2_k_dp4a(vi, u_i, 0);
        const int sumi_m = flambeau_q2_k_dp4a(m_packed, u_i, 0);

        acc += d_y * (d_sb * (float) sc_lo * (float) sumi_d
                      - dmin_sb * (float) sumi_m);
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q2_K_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q2_K_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q2_K_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q2_k_dp4a_q8_1(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q2_k_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q2_k_dp4a_q8_1_f16(
    const flambeau_block_q2_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q2_k_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
