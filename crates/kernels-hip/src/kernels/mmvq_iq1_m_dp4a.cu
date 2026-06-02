// mmvq_iq1_m_dp4a — IQ1_M weight × Q8_1 → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq1_m_q8_1` (vecdotq.cuh:1223).
// IQ1_M extends IQ1_S:
//   - Same IQ1S_GRID (2048 × signed-i8 codebook, no sign mask).
//   - No per-block d in bytes — d (F16) is reassembled from the top
//     nibble of each of 4 u16 scale words (iq1m_reassemble_d helper
//     in mmvq_iq1_m.cu — duplicated below).
//   - Two scales per sub-block (sc0 for l_grp<2, sc1 for l_grp>=2),
//     each 3 bits within the corresponding u16 scale word.
//   - Per-l_grp delta sign (qh byte bit 3 or 7 based on which nibble).
//
// Per (iqs, l_grp) lane: same dp4a + bias structure as IQ1_S, but
// with the dl picked from the appropriate half-sub-block scale.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "iq_grid.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ1_M_THREADS 256
#define MMVQ_IQ1_M_WARPS (MMVQ_IQ1_M_THREADS / WARP_SIZE)
#define MMVQ_IQ1_M_THREADS_PER_BLOCK 32
#define MMVQ_IQ1_M_BLOCKS_PER_ITER (MMVQ_IQ1_M_THREADS / MMVQ_IQ1_M_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq1_m_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ float flambeau_iq1_m_reassemble_d(
    const uint8_t* __restrict__ scales
) {
    const int sc0 = (int) scales[0] | ((int) scales[1] << 8);
    const int sc1 = (int) scales[2] | ((int) scales[3] << 8);
    const int sc2 = (int) scales[4] | ((int) scales[5] << 8);
    const int sc3 = (int) scales[6] | ((int) scales[7] << 8);
    const int d_bits = (sc0 >> 12)
                     | ((sc1 >> 8) & 0x00F0)
                     | ((sc2 >> 4) & 0x0F00)
                     | (sc3 & 0xF000);
    fb_fp16_t d_fp16 = *reinterpret_cast<const fb_fp16_t*>(&d_bits);
    return (float) d_fp16;
}

template<typename OutT>
__device__ void mmvq_iq1_m_dp4a_body(
    const flambeau_block_iq1_m* __restrict__ x,
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
    const int sb_idx_in_iter = tid / MMVQ_IQ1_M_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ1_M_THREADS_PER_BLOCK - 1);
    const int iqs           = lane_in_sb >> 2;          // 0..7
    const int l_grp         = lane_in_sb & 3;           // 0..3
    const int l0            = 2 * l_grp;

    const flambeau_block_iq1_m* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ1_M_BLOCKS_PER_ITER) {
        const flambeau_block_iq1_m* bk = xrow + sb;

        const float d_super = flambeau_iq1_m_reassemble_d(bk->scales);

        // Per-half-sub-block 3-bit scales: sc[iqs/2] (one u16) holds both.
        const int sc_word = (int) bk->scales[2 * (iqs >> 1)]
                          | ((int) bk->scales[2 * (iqs >> 1) + 1] << 8);
        const int shift1 = 6 * (iqs & 1);
        const int dl_scale = (l_grp < 2)
            ? ((sc_word >> shift1)       & 7)
            : ((sc_word >> (shift1 + 3)) & 7);
        const float dl = d_super * (2.0f * (float) dl_scale + 1.0f);

        // qh byte select + nibble extraction.
        const int qh_pick = (l_grp < 2) ? (2 * iqs) : (2 * iqs + 1);
        const uint8_t qh_byte = bk->qh[qh_pick];
        const int shift_idx = 8 - 4 * (l_grp & 1);          // 8 for l_grp 0,2; 4 for l_grp 1,3
        const int idx = (int) bk->qs[4 * iqs + l_grp]
                      | (((int) qh_byte << shift_idx) & 0x700);
        const int delta_bit = (l_grp & 1) == 0 ? 0x08 : 0x80;
        const float delta = (qh_byte & delta_bit) ? -IQ1_DELTA : IQ1_DELTA;

        const uint64_t grid_u64 = IQ1S_GRID[idx];
        const int grid_lo = (int) (uint32_t) grid_u64;
        const int grid_hi = (int) (uint32_t) (grid_u64 >> 32);

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + iqs;
        const int u0 = ((const int*) ya->qs)[l0 + 0];
        const int u1 = ((const int*) ya->qs)[l0 + 1];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq1_m_dp4a(grid_lo, u0, 0);
        sumi     = flambeau_iq1_m_dp4a(grid_hi, u1, sumi);

        int sum_q8 = flambeau_iq1_m_dp4a(0x01010101, u0, 0);
        sum_q8     = flambeau_iq1_m_dp4a(0x01010101, u1, sum_q8);

        acc += dl * d_y * ((float) sumi + delta * (float) sum_q8);
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ1_M_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ1_M_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ1_M_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq1_m_dp4a_q8_1(
    const flambeau_block_iq1_m* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq1_m_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq1_m_dp4a_q8_1_f16(
    const flambeau_block_iq1_m* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq1_m_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
