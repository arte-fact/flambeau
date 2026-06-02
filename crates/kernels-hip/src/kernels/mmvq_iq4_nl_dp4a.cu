// mmvq_iq4_nl_dp4a — IQ4_NL weight × Q8_1 activation → {F32, F16}, DP4A.
//
// Ported from llama.cpp's `vec_dot_iq4_nl_q8_1` (vecdotq.cuh:1270).
// 32-element block (like Q4_0) with d (F16) + 16 bytes packed qs. Each
// nibble indexes the 16-entry `kvalues_iq4nl` codebook for the
// reconstructed signed i8 value.
//
// Launch shape mirrors `mmvq_q4_0.cu`: 256 threads/block, single output
// row, 4 lanes per IQ4_NL block, 64 blocks per outer iter. Per thread:
// one int32 of qs (= 8 nibbles, 4 low + 4 high), 2 dp4as covering 8
// elements.
//
// LUT lookup uses gfx906's `__builtin_amdgcn_perm` (4 indices per call)
// — same helper as `mmvq_iq4_xs_dp4a.cu`. No bias correction term
// (codebook is signed; no Q4_0-style q - 8 subtraction).

#include "block_quant.cuh"
#include "../arch_primitives/gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_IQ4_NL_THREADS 256
#define MMVQ_IQ4_NL_WARPS (MMVQ_IQ4_NL_THREADS / WARP_SIZE)
#define MMVQ_IQ4_NL_INT32_PER_BLOCK 4
#define MMVQ_IQ4_NL_BLOCKS_PER_ITER (MMVQ_IQ4_NL_THREADS / MMVQ_IQ4_NL_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_iq4_nl_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ uint32_t flambeau_iq4_nl_table_4(
    uint32_t q4_nibbles
) {
    constexpr uint32_t kv0 = 0xBFAD9881u; // {-127, -104, -83, -65}
    constexpr uint32_t kv1 = 0xF6EADDCFu; // {-49, -35, -22, -10}
    constexpr uint32_t kv2 = 0x26190D01u; // {1, 13, 25, 38}
    constexpr uint32_t kv3 = 0x71594535u; // {53, 69, 89, 113}

    const uint32_t v_low  = __builtin_amdgcn_perm(kv1, kv0, q4_nibbles & 0x07070707);
    const uint32_t v_high = __builtin_amdgcn_perm(kv3, kv2, q4_nibbles & 0x07070707);
    const uint32_t mask   = 0x03020100u | ((q4_nibbles & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(v_high, v_low, mask);
}

template<typename OutT>
__device__ void mmvq_iq4_nl_dp4a_body(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1*   __restrict__ y,
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

    const flambeau_block_iq4_nl* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_IQ4_NL_BLOCKS_PER_ITER) {
        const flambeau_block_iq4_nl* bx = xrow + b;
        const flambeau_block_q8_1*   by = y + b;

        const int aux_q4 = ((const int*) bx->qs)[lane4];
        const uint32_t q_low  = (uint32_t) aux_q4 & 0x0F0F0F0Fu;
        const uint32_t q_high = ((uint32_t) aux_q4 >> 4) & 0x0F0F0F0Fu;

        const uint32_t v_lo = flambeau_iq4_nl_table_4(q_low);
        const uint32_t v_hi = flambeau_iq4_nl_table_4(q_high);

        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        int sumi = flambeau_iq4_nl_dp4a((int) v_lo, u_lo, 0);
        sumi     = flambeau_iq4_nl_dp4a((int) v_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        acc += d_x * d_y * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_IQ4_NL_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_IQ4_NL_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_IQ4_NL_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_iq4_nl_dp4a_q8_1(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_iq4_nl_dp4a_body<float>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_iq4_nl_dp4a_q8_1_f16(
    const flambeau_block_iq4_nl* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_iq4_nl_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_blocks_per_row);
}
