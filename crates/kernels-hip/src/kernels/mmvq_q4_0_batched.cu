// mmvq_q4_0_batched — Q4_0 weight × N Q8_1 activation rows → N F32 dst rows.
//
// K1 — Q4_0 batched MMVQ with compile-time N specialization. The Q4_1
// batched sibling (`mmvq_q4_1_batched`) uses a runtime n_slots loop bounded
// by MAX_N=8 with `if (s >= n_slots) break;` inside `#pragma unroll`. The
// compiler keeps all 8 iterations' VGPRs allocated regardless of actual
// n_slots → measured 59 VGPRs vs 17 for the single-row baseline → ~60 %
// occupancy drop on gfx906 → weight-amortization win drowns in latency-
// hiding loss.
//
// Fix: emit three `extern "C"` entry points (N=2, 3, 4) that each
// instantiate `flambeau_mmvq_q4_0_batched_body<N>` with compile-time N.
// The compiler then allocates only N slots' VGPRs → expected ~20/22/24
// VGPRs → preserved occupancy (≥ 8 waves/SIMD on gfx906, same regime as
// the single-row kernel). Caller picks the right entry by n_slots.
//
// Math: per-block contribution `sumi · (d_x · d_y) - 8 · d_x · s_y`
// where s_y = d_y · sum(y_qs) is baked into the Q8_1 block header. The
// `-8·d_x·s_y` correction is split across the 4 lane4 ∈ [0,4) lanes via
// `* 0.25f` so the warp reduce sums to exactly one correction per Q4_0
// block.
//
// Output layout: `dst[N, n_rows]` slot-major F32 — matches the `qmatmul`
// ABI's `[m, n] = [batch, output]` convention.
//
// Block/grid: blockDim = 256, gridDim = n_rows (one block per output
// row). Threading mirrors `mmvq_q4_0`.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4_0_BATCHED_THREADS 256
#define MMVQ_Q4_0_BATCHED_WARPS (MMVQ_Q4_0_BATCHED_THREADS / WARP_SIZE)
#define MMVQ_Q4_0_BATCHED_INT32_PER_BLOCK 4
#define MMVQ_Q4_0_BATCHED_BLOCKS_PER_ITER \
    (MMVQ_Q4_0_BATCHED_THREADS / MMVQ_Q4_0_BATCHED_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q4_0_batched_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Body templated on compile-time N. Each instantiation allocates exactly
// N slots' worth of accumulators + reduction shared-memory.
template <int N>
__device__ __forceinline__ void flambeau_mmvq_q4_0_batched_body(
    const flambeau_block_q4_0* __restrict__ x,    // [n_rows, n_blocks_per_row]
    const flambeau_block_q8_1* __restrict__ y,    // [N, n_blocks_per_row]
    float* __restrict__ dst,                       // [N, n_rows]
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

    const flambeau_block_q4_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) {
        acc[s] = 0.0f;
    }

    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_0_BATCHED_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* bx = xrow + b;
        // Weight: read ONCE per (block, thread) — amortization lever.
        const int v     = ((const int*) bx->qs)[lane4];
        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;
        const float d_x = (float) bx->d;

        // Inner slot loop — N is compile-time, fully unrolled, no
        // runtime conditional. Each slot reads its own activation block.
        #pragma unroll
        for (int s = 0; s < N; ++s) {
            const flambeau_block_q8_1* by =
                y + (size_t) s * n_blocks_per_row + b;
            const int u_lo = ((const int*) by->qs)[lane4];
            const int u_hi = ((const int*) by->qs)[lane4 + 4];
            int sumi = 0;
            sumi = flambeau_q4_0_batched_dp4a(vi_lo, u_lo, sumi);
            sumi = flambeau_q4_0_batched_dp4a(vi_hi, u_hi, sumi);
            const float d_y = (float) by->d;
            const float s_y = (float) by->s;
            acc[s] += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    // Per-slot reduction. LDS sized exactly for N.
    __shared__ float s_warp[MMVQ_Q4_0_BATCHED_WARPS * N];

    #pragma unroll
    for (int s = 0; s < N; ++s) {
        float v_red = gfx906_warp_reduce_sum(acc[s]);
        if (lane == 0) {
            s_warp[s * MMVQ_Q4_0_BATCHED_WARPS + warp] = v_red;
        }
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int s = 0; s < N; ++s) {
            float v_red = (lane < MMVQ_Q4_0_BATCHED_WARPS)
                ? s_warp[s * MMVQ_Q4_0_BATCHED_WARPS + lane]
                : 0.0f;
            #pragma unroll
            for (int off = MMVQ_Q4_0_BATCHED_WARPS / 2; off > 0; off >>= 1) {
                v_red += __shfl_xor(v_red, off, WARP_SIZE);
            }
            if (lane == 0) {
                dst[(size_t) s * n_rows + row] = v_red;
            }
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q4_0_q8_1_batched_n2(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_batched_body<2>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q4_0_q8_1_batched_n3(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_batched_body<3>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q4_0_q8_1_batched_n4(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_batched_body<4>(x, y, dst, n_rows, n_blocks_per_row);
}
