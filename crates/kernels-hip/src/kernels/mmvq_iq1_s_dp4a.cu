// mmvq_iq1_s_dp4a — IQ1_S weight × Q8_1 → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq1_s_q8_1` (vecdotq.cuh:1190).
// IQ1_S is structurally different from IQ2/IQ3:
//   - No sign mask; codebook entries are *signed* i8 magnitudes
//     stored directly in `IQ1S_GRID[2048]` as uint64 of 8 packed i8.
//   - Each weight is `dl * (grid_byte_i + delta)` where:
//       dl    = d * (2*scale_3bit + 1)
//       delta = ±IQ1_DELTA (sign from qh bit 15)
//     The delta offset introduces a bias term that is computed
//     separately from the main dp4a dot product.
//   - VDR = 1 (no x2 step on iqs; iqs ∈ [0, QI1_S) = [0, 8) directly
//     maps to the sub-block index).
//
// Per (iqs, l_grp) lane:
//   qh_u16    = bk->qh[2*iqs] | (bk->qh[2*iqs+1] << 8)
//   scale_3b  = (qh_u16 >> 12) & 7
//   dl        = d_super * (2 * scale_3b + 1)
//   delta     = (qh_u16 & 0x8000) ? -IQ1_DELTA : +IQ1_DELTA
//   idx_lo    = bk->qs[4*iqs + l_grp]
//   high3     = (qh_u16 >> (3 * l_grp)) & 7
//   idx       = idx_lo | (high3 << 8)            (11-bit IQ1S_GRID index)
//   grid_u64  = IQ1S_GRID[idx]                   — 8 signed i8 magnitudes
//   grid_lo, grid_hi = low/high 4 bytes (as int32 of signed i8)
//   sumi_lane = dp4a(grid_lo, u0, 0) + dp4a(grid_hi, u1, 0)
//   sum_q8_lane = dp4a(0x01010101, u0, 0) + dp4a(0x01010101, u1, 0)
//   acc      += dl * d_y * (sumi_lane + delta * sum_q8_lane)
//
// The `delta * sum_q8_lane` term distributes correctly: across the 4 l_grp
// lanes per sub-block, the partials sum to `delta * sum_q8_full_sub_block`.
//
// Launch: 256 threads/block, single row, 32 threads per super-block
// (8 iqs × 4 l_grp), BLOCKS_PER_ITER = 8.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ1_S_THREADS 256
#define MMVQ_IQ1_S_WARPS (MMVQ_IQ1_S_THREADS / WARP_SIZE)
#define MMVQ_IQ1_S_THREADS_PER_BLOCK 32
#define MMVQ_IQ1_S_BLOCKS_PER_ITER (MMVQ_IQ1_S_THREADS / MMVQ_IQ1_S_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq1_s_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_iq1_s_dp4a_body(
    const flambeau_block_iq1_s* __restrict__ x,
    const flambeau_block_q8_1*  __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid           = threadIdx.x;
    const int warp          = tid / WARP_SIZE;
    const int lane          = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = tid / MMVQ_IQ1_S_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ1_S_THREADS_PER_BLOCK - 1);
    const int iqs           = lane_in_sb >> 2;          // 0..7 — sub-block index
    const int l_grp         = lane_in_sb & 3;           // 0..3
    const int l0            = 2 * l_grp;

    const flambeau_block_iq1_s* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ1_S_BLOCKS_PER_ITER) {
        const flambeau_block_iq1_s* bk = xrow + sb;

        const float d_sb = (float) bk->d;

        // qh is u16 array (stored as u8 pairs). Reconstruct.
        const int qh_u16 = (int) bk->qh[2 * iqs]
                         | ((int) bk->qh[2 * iqs + 1] << 8);

        const float dl    = d_sb * (2.0f * (float)((qh_u16 >> 12) & 7) + 1.0f);
        const float delta = (qh_u16 & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;

        const int idx_lo = (int) bk->qs[4 * iqs + l_grp];
        const int high3  = (qh_u16 >> (3 * l_grp)) & 7;
        const int idx    = idx_lo | (high3 << 8);

        // IQ1S_GRID is uint64 with 8 signed i8 magnitudes per entry.
        const uint64_t grid_u64 = IQ1S_GRID[idx];
        const int grid_lo = (int) (uint32_t) grid_u64;
        const int grid_hi = (int) (uint32_t) (grid_u64 >> 32);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + iqs;
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq1_s_dp4a(grid_lo, u0, 0);
        sumi     = flambeau_iq1_s_dp4a(grid_hi, u1, sumi);

        int sum_q8 = flambeau_iq1_s_dp4a(0x01010101, u0, 0);
        sum_q8     = flambeau_iq1_s_dp4a(0x01010101, u1, sum_q8);

        acc += dl * d_y * ((float) sumi + delta * (float) sum_q8);
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ1_S_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ1_S_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ1_S_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq1_s_dp4a_q8_1(
    const flambeau_block_iq1_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq1_s_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq1_s_dp4a_q8_1_f16(
    const flambeau_block_iq1_s* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq1_s_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
