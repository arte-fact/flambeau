// rmsnorm_q8_1_fused — RMSNorm + Q8_1 activation quantize in one kernel.
//
// Candle D1 pattern: the decode path's hot loop is `... → RMSNorm → Q8_1
// quantize → MMVQ → ...`, and every RMSNorm→Q8_1 edge goes through HBM
// twice (F16 write, F16 read). Fusing it saves one full pass of hidden-
// sized traffic per transformer layer.
//
// Output is Q8_1 blocks of 32 elements each. For each row (length K, must
// be a multiple of QK8_1=32):
//   1. mean_sq = Σ(x[i] * weight[i])² / K     (the full RMSNorm output
//                                              has weight applied before
//                                              quant — not just x)
//   2. rsqrt   = 1 / sqrt(mean_sq + eps)       (broadcast)
//   3. normed[i] = x[i] * weight[i] * rsqrt   (kept in registers)
//   4. amax    = max|normed[j]| over j ∈ block (per-block Q8_1 scale)
//   5. d       = amax / 127
//   6. qs[j]   = round(normed[j] / d)           (per-block int8 quant)
//   7. s       = d * Σ qs[j]                    (Q8_1 "sum" field)
//
// Layout choices:
//   blockDim  = { 256 }                         (4 wave64 warps)
//   gridDim   = { n_rows }
//   shared    = 4 floats for cross-warp sum + a per-warp scratch for block
//               max-abs reductions.
//
// NOTE: the RMSNorm step uses the TRUE sum-of-squares (NOT the weighted
// product squared) per ggml convention — weight is applied after the
// rsqrt in the standard definition. We follow that; the test harness
// mirrors.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>

#define RMSQ8_THREADS 256
#define RMSQ8_WARPS (RMSQ8_THREADS / 64)

extern "C" __global__ void flambeau_rmsnorm_q8_1_fused(
    const fb_fp16_t* __restrict__ x,               // [n_rows, k]
    const fb_fp16_t* __restrict__ weight,          // [k]
    flambeau_block_q8_1* __restrict__ y,           // [n_rows, k / QK8_1]
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const fb_fp16_t* xrow = x + (size_t) row * k;
    flambeau_block_q8_1* yrow = y + (size_t) row * (k / QK8_1);

    // --- Phase 1: Σ x² ---
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSQ8_THREADS) {
        const float v = (float) xrow[i];
        sum_sq += v * v;
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[RMSQ8_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < RMSQ8_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSQ8_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) k;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    // --- Phase 2: per-block quantise ---
    // Block layout: 32 consecutive output quants per Q8_1 block. With
    // 256 threads and 32 threads per block, there are 8 blocks processed
    // per iteration. We loop over (k / QK8_1) blocks, 8 at a time.
    const int nblocks = k / QK8_1;
    const int block_lane  = tid & 31;               // 0..31 — position within Q8_1 block
    const int block_group = tid >> 5;               // 0..7  — which Q8_1 block in this step

    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int b = b0 + block_group;
        const int base = b * QK8_1 + block_lane;
        float normed = 0.0f;
        if (b < nblocks && base < k) {
            const float xv = (float) xrow[base];
            const float wv = (float) weight[base];
            normed = xv * rsqrt * wv;
        }

        // Per-block max-abs reduce over the 32 lanes that own this block
        // (one bank of lanes in the wave64). Uses shfl_xor within the
        // 32-lane half-warp.
        float amax = fabsf(normed);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor(amax, off, 32);
            amax = fmaxf(amax, other);
        }
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

        const int qi = max(-127, min(127, (int) rintf(normed * id)));

        // Per-block sum reduce for the `s` field.
        int sum_qi = qi;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_qi += __shfl_xor(sum_qi, off, 32);
        }

        if (b < nblocks) {
            yrow[b].qs[block_lane] = (int8_t) qi;
            if (block_lane == 0) {
                yrow[b].d = (fb_fp16_t) d;
                yrow[b].s = (fb_fp16_t) (d * (float) sum_qi);
            }
        }
    }
}
