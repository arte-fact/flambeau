// mmvq_q4_1_batched — Q4_1 weight × N Q8_1 activation rows → N F32 dst rows.
//
// **#288** — batched-MMVQ kernel that AMORTIZES weight HBM reads across N
// activation rows. The single-row `mmvq_q4_1_q8_1` reads the full weight tile
// from HBM once per output row × per launch; calling it N times for N batched
// slots costs N× the HBM bandwidth (see `feedback_qmatmul_small_m_no_amortize.md`).
//
// This kernel uses gridDim=(n_rows,) — same as the single-row variant — but
// loops over all N slots within the inner K-block iteration, reusing the
// loaded weight register values N times. Per (output row, K-block) pair:
//   - Weight bytes: read ONCE per thread regardless of N.
//   - Activation bytes: read N times (each slot has its own row).
//   - DP4A: N invocations.
// HBM bandwidth scales: weight reads stay constant in N (the lever); activation
// reads are negligible vs weight (Q4_1 weight is ~9 KB/row × n_rows ≈ 36 MB at
// k=4096; activation is ~144 KB/row × N ≈ 1 MB at N=8).
//
// Mirrors `attention_decode_f16_batched`'s pattern (see #266): single launch
// covers N "rows" (batch dim) with shared resource reuse.
//
// Output layout: dst[N, n_rows] F32, slot-major (matches qmatmul ABI's
// [m, n] = [batch, output] convention).
//
// Block/grid:
//   blockDim = 256, gridDim = n_rows (one block per output row).
//   Same threading as `mmvq_q4_1_q8_1`: lane4 ∈ [0,4) = which int32 of qs;
//   block_idx = which Q4_1 block (0..63 across 256 threads). N slots inner-
//   looped per (block_idx, K-iter).
//
// VGPR budget: per-slot accumulator adds 1 F32 register per slot. At N=8
// that's +8 VGPRs over the N=1 baseline. Single-row mmvq_q4_1 uses ~20
// VGPRs (estimated from kernel size); N=8 batched lands at ~28 — still
// well under the gfx906 64-VGPR-per-wave threshold.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4_1_BATCHED_THREADS 256
#define MMVQ_Q4_1_BATCHED_WARPS (MMVQ_Q4_1_BATCHED_THREADS / WARP_SIZE)
#define MMVQ_Q4_1_BATCHED_INT32_PER_BLOCK 4
#define MMVQ_Q4_1_BATCHED_BLOCKS_PER_ITER \
    (MMVQ_Q4_1_BATCHED_THREADS / MMVQ_Q4_1_BATCHED_INT32_PER_BLOCK)
// Maximum batch slots supported per kernel call. Bounded for VGPR + LDS
// budget. Caller validates n_slots ∈ [1, MMVQ_Q4_1_BATCHED_MAX_N].
#define MMVQ_Q4_1_BATCHED_MAX_N 8

static __device__ __forceinline__ int flambeau_q4_1_batched_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Numerical-equivalence note: the per-K-block math in this kernel is
// algebraically identical to `mmvq_q4_1_q8_1`'s, but the surrounding
// slot loop changes hipcc's FMA-contraction choices. As a result,
// per-element output drift versus the single-row kernel sits at f32 LSB
// scale (max_abs_err ≈ 1.5e-6 at k=4096, well within the existing
// batched-GDN tolerance). The parity test enforces abs_err < 1e-5
// rather than bit-equal — matches the batched-GDN tolerance accepted
// on the model-level forward path.

extern "C" __global__ void flambeau_mmvq_q4_1_q8_1_batched(
    const flambeau_block_q4_1* __restrict__ x,    // [n_rows, n_blocks_per_row]
    const flambeau_block_q8_1* __restrict__ y,    // [n_slots, n_blocks_per_row]
    float* __restrict__ dst,                       // [n_slots, n_rows]
    const int n_rows,
    const int n_blocks_per_row,
    const int n_slots
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_1* xrow = x + (size_t) row * n_blocks_per_row;

    // Per-slot register accumulators. Compile-time bounded MAX_N so the
    // compiler keeps these in registers (no LDS spill).
    float acc[MMVQ_Q4_1_BATCHED_MAX_N];
    #pragma unroll
    for (int s = 0; s < MMVQ_Q4_1_BATCHED_MAX_N; ++s) {
        acc[s] = 0.0f;
    }

    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_1_BATCHED_BLOCKS_PER_ITER) {
        const flambeau_block_q4_1* bx = xrow + b;
        // Load weight bytes ONCE — this is the amortization lever.
        const int v     = ((const int*) bx->qs)[lane4];
        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;
        const float d_x = (float) bx->d;
        const float m_x = (float) bx->m;

        // Inner loop over slots, reusing the loaded weights.
        #pragma unroll
        for (int s = 0; s < MMVQ_Q4_1_BATCHED_MAX_N; ++s) {
            if (s >= n_slots) break;
            const flambeau_block_q8_1* by =
                y + (size_t) s * n_blocks_per_row + b;
            const int u_lo = ((const int*) by->qs)[lane4];
            const int u_hi = ((const int*) by->qs)[lane4 + 4];
            int sumi = 0;
            sumi = flambeau_q4_1_batched_dp4a(vi_lo, u_lo, sumi);
            sumi = flambeau_q4_1_batched_dp4a(vi_hi, u_hi, sumi);
            const float d_y = (float) by->d;
            const float s_y = (float) by->s;
            // Same per-block math as mmvq_q4_1: sumi · (d_x · d_y) + (m_x · s_y).
            // The constant term is split across the 4 int32-lanes (lane4 ∈ [0,4))
            // by the *0.25f factor so the warp reduce sums to one (m_x · s_y)
            // per Q4_1 block.
            acc[s] += sumi * (d_x * d_y) + (m_x * s_y) * 0.25f;
        }
    }

    // Per-slot reduction. Warp-reduce each slot's accumulator independently;
    // stash lane-0 results in LDS; warp 0 finishes the cross-warp reduction
    // and writes dst.
    __shared__ float s_warp[MMVQ_Q4_1_BATCHED_WARPS * MMVQ_Q4_1_BATCHED_MAX_N];

    #pragma unroll
    for (int s = 0; s < MMVQ_Q4_1_BATCHED_MAX_N; ++s) {
        if (s >= n_slots) break;
        float v_red = gfx906_warp_reduce_sum(acc[s]);
        if (lane == 0) {
            s_warp[s * MMVQ_Q4_1_BATCHED_WARPS + warp] = v_red;
        }
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int s = 0; s < MMVQ_Q4_1_BATCHED_MAX_N; ++s) {
            if (s >= n_slots) break;
            float v_red = (lane < MMVQ_Q4_1_BATCHED_WARPS)
                ? s_warp[s * MMVQ_Q4_1_BATCHED_WARPS + lane]
                : 0.0f;
            #pragma unroll
            for (int off = MMVQ_Q4_1_BATCHED_WARPS / 2; off > 0; off >>= 1) {
                v_red += __shfl_xor(v_red, off, WARP_SIZE);
            }
            if (lane == 0) {
                dst[(size_t) s * n_rows + row] = v_red;
            }
        }
    }
}
