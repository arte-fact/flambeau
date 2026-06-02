// mmvq_iq4_xs_dp4a — IQ4_XS weight × Q8_1 activation → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq4_xs_q8_1` (vecdotq.cuh:1294).
// 256 threads/block, single output row, BLOCKS_PER_ITER super-blocks
// per outer iter. Same shell as `mmvq_q5_k_dp4a.cu`.
//
// IQ4_XS block layout (256 elements, 8 sub-blocks × 32 elements):
//   d (F16)        — super-block scale
//   scales_h (u16) — high 2 bits of each of 8 6-bit per-sub-block scales
//   scales_l[4]    — low 4 bits of the 8 scales, two per byte
//   qs[128]        — 4-bit codebook indices into `kvalues_iq4nl` (LUT)
//
// LUT lookup uses gfx906's `__builtin_amdgcn_perm` — 4 indices per call,
// matching llama.cpp's `get_int_from_table_16` HIP branch.
//
// Threading: 32 threads per super-block. 4 threads per sub-block × 8
// sub-blocks. Each thread handles one int32 of qs (8 packed nibbles).
// dp4a inner: 4 LUT-mapped low nibbles × 4 Q8 elements, then same for
// high nibbles → 2 dp4as / thread / super-block.

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ4_XS_THREADS 256
#define MMVQ_IQ4_XS_WARPS (MMVQ_IQ4_XS_THREADS / WARP_SIZE)
#define MMVQ_IQ4_XS_THREADS_PER_BLOCK 32
#define MMVQ_IQ4_XS_BLOCKS_PER_ITER (MMVQ_IQ4_XS_THREADS / MMVQ_IQ4_XS_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq4_xs_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// 4-byte LUT lookup into `kvalues_iq4nl`. `q4` packs 4 indices, one per
// byte, in the low 4 bits of each byte (mask 0x0F0F0F0F applied before
// calling). Returns the 4 i8 LUT values packed into one int32.
// Mirrors the HIP branch of llama.cpp's `get_int_from_table_16` but
// returns one int32 (the v.x or v.y half) rather than int2.
static __device__ __forceinline__ uint32_t flambeau_iq4_xs_table_4(
    uint32_t q4_nibbles
) {
    // kvalues_iq4nl as 4 uint32s of 16 packed i8 (little-endian).
    constexpr uint32_t kv0 = 0xBFAD9881u; // {-127, -104, -83, -65}
    constexpr uint32_t kv1 = 0xF6EADDCFu; // {-49, -35, -22, -10}
    constexpr uint32_t kv2 = 0x26190D01u; // {1, 13, 25, 38}
    constexpr uint32_t kv3 = 0x71594535u; // {53, 69, 89, 113}

    const uint32_t v_low  = __builtin_amdgcn_perm(kv1, kv0, q4_nibbles & 0x07070707);
    const uint32_t v_high = __builtin_amdgcn_perm(kv3, kv2, q4_nibbles & 0x07070707);
    // Mask 0x03020100 = identity perm; OR in bit 3 of each nibble (×8) to
    // pick between low / high halves of the table.
    const uint32_t mask   = 0x03020100u | ((q4_nibbles & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(v_high, v_low, mask);
}

template<typename OutT>
__device__ void mmvq_iq4_xs_dp4a_body(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid           = threadIdx.x;
    const int warp          = tid / WARP_SIZE;
    const int lane          = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = tid / MMVQ_IQ4_XS_THREADS_PER_BLOCK;
    const int lane_in_sb    = tid & (MMVQ_IQ4_XS_THREADS_PER_BLOCK - 1);
    const int sub           = lane_in_sb >> 2;            // 0..7
    const int lane4         = lane_in_sb & 3;             // 0..3

    const flambeau_block_iq4_xs* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row;
         sb += MMVQ_IQ4_XS_BLOCKS_PER_ITER) {
        const flambeau_block_iq4_xs* bk = xrow + sb;

        const float d  = (float) bk->d;
        const int   ls = flambeau_iq4_xs_scale(sub, bk->scales_h, bk->scales_l);

        // One int32 of qs: 4 bytes × 2 nibbles = 8 packed quants.
        // sub-block `sub` lives in qs[sub*16 .. sub*16 + 16); lane4 picks
        // one of 4 int32s within that 16-byte region.
        const int aux_q4 = ((const int*) bk->qs)[sub * 4 + lane4];
        const uint32_t q_low  = (uint32_t) aux_q4 & 0x0F0F0F0Fu;
        const uint32_t q_high = ((uint32_t) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const uint32_t v_lo = flambeau_iq4_xs_table_4(q_low);
        const uint32_t v_hi = flambeau_iq4_xs_table_4(q_high);

        // Q8_1 sub-block `sub` has 32 quants packed as 8 int32s.
        // Low nibbles → first 4 q8 int32s; high nibbles → next 4.
        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = ((const int*) ya->qs)[lane4];
        const int u_hi = ((const int*) ya->qs)[lane4 + 4];
        const float d_y = (float) ya->d;

        int sumi = flambeau_iq4_xs_dp4a((int) v_lo, u_lo, 0);
        sumi     = flambeau_iq4_xs_dp4a((int) v_hi, u_hi, sumi);

        acc += d * (float) ls * d_y * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ4_XS_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ4_XS_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ4_XS_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq4_xs_dp4a_q8_1(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq4_xs_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq4_xs_dp4a_q8_1_f16(
    const flambeau_block_iq4_xs* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_iq4_xs_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
