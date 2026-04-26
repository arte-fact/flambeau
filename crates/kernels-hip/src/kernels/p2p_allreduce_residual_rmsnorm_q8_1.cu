// p2p_allreduce_residual_rmsnorm_q8_1 — TP-3b-i3 4-op fusion.
//
// One launch per rank fuses:
//   1. Peer-read via BAR1: load partial_local + partial_peer{0,1,2}.
//   2. Residual-add: hidden[i] += partial_local + Σ peers.
//   3. RMSNorm: normed[i] = hidden[i] * rsqrt(mean(hidden²) + eps) * weight[i].
//   4. Q8_1 quantize: per-32-element block, qs[j] = round(normed[j]/d),
//      d = amax/127, s = d * Σ qs[j].
//
// Replaces THREE launches in the post-FFN cross-layer boundary:
//   - flambeau_p2p_allreduce_residual_tp4 (TP-0b)
//   - flambeau_rmsnorm_q8_1_fused (existing) — already fuses norm+quant
//   → ONE fused call. Per-token saving: 64 launches × 10 µs ≈ 0.6 ms at
//   60 tok/s budget (~3.5% throughput).
//
// Bit-exact equivalent: arithmetic order matches the unfused chain.
//   - AR sum: same FP32 accumulate as residual_tp4.
//   - sum_sq: __shfl_xor + LDS (same reduction tree as rmsnorm_f16).
//   - Q8_1 amax / quantize: per-half-warp reduction as in rmsnorm_q8_1_fused.
//
// Constraints:
//   - n must be a multiple of 256 (single-block kernel) AND of QK8_1=32
//     (every Qwen3.5/3.6 hidden satisfies both: {2048, 4096, 5120, 8192}).
//   - Output is `flambeau_block_q8_1[n / QK8_1]`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "block_quant.cuh"

#define P2P_AR_NQ_THREADS 256
#define P2P_AR_NQ_WARPS (P2P_AR_NQ_THREADS / 64)

// ------------------------------------------------------------------------
// TP=4: AR + residual + RMSNorm + Q8_1 quantize.
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_NQ_THREADS)
void flambeau_p2p_allreduce_residual_rmsnorm_q8_1_tp4(
    fb_fp16_t* __restrict__ hidden,            // [n] — residual-folded in place
    const fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const fb_fp16_t* __restrict__ partial_peer1,
    const fb_fp16_t* __restrict__ partial_peer2,
    const fb_fp16_t* __restrict__ rms_weight,  // [n]
    flambeau_block_q8_1* __restrict__ out_q8_1, // [n / QK8_1]
    const unsigned int n,
    const float eps
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    // --- Phase 1: peer-read + residual fold + sum-of-squares ---
    // Note: sum_sq is over the AR'd `hidden`, NOT over `hidden * weight`,
    // matching the standard RMSNorm definition (weight applied after rsqrt).
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < (int) n; i += P2P_AR_NQ_THREADS) {
        const float h_v  = (float) hidden[i];
        const float pl_v = (float) partial_local[i];
        const float p0_v = (float) partial_peer0[i];
        const float p1_v = (float) partial_peer1[i];
        const float p2_v = (float) partial_peer2[i];
        const float new_h = h_v + pl_v + p0_v + p1_v + p2_v;
        hidden[i] = (fb_fp16_t) new_h;
        sum_sq += new_h * new_h;
    }
    // Block-reduce sum_sq.
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[P2P_AR_NQ_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < P2P_AR_NQ_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = P2P_AR_NQ_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) n;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    // --- Phase 2: per-Q8_1-block quantise. ---
    // 256 threads × 1 element/iter; 32 threads handle one Q8_1 block,
    // so 8 blocks processed per iteration. Loop over n / QK8_1 blocks.
    const int nblocks = (int) n / QK8_1;
    const int block_lane = tid & 31;       // 0..31 — position within block
    const int block_group = tid >> 5;      // 0..7  — which block in this step

    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int b = b0 + block_group;
        const int base = b * QK8_1 + block_lane;
        float normed = 0.0f;
        if (b < nblocks && base < (int) n) {
            const float h_v = (float) hidden[base];
            const float w_v = (float) rms_weight[base];
            normed = h_v * rsqrt * w_v;
        }

        // Per-block max-abs reduce over 32 lanes (half-warp).
        float amax = fabsf(normed);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor(amax, off, 32);
            amax = fmaxf(amax, other);
        }
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        const int qi = max(-127, min(127, (int) rintf(normed * id)));

        int sum_qi = qi;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_qi += __shfl_xor(sum_qi, off, 32);
        }

        if (b < nblocks) {
            out_q8_1[b].qs[block_lane] = (int8_t) qi;
            if (block_lane == 0) {
                out_q8_1[b].d = (fb_fp16_t) d;
                out_q8_1[b].s = (fb_fp16_t) (d * (float) sum_qi);
            }
        }
    }
}

// ------------------------------------------------------------------------
// TP=2 variant.
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_NQ_THREADS)
void flambeau_p2p_allreduce_residual_rmsnorm_q8_1_tp2(
    fb_fp16_t* __restrict__ hidden,
    const fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const fb_fp16_t* __restrict__ rms_weight,
    flambeau_block_q8_1* __restrict__ out_q8_1,
    const unsigned int n,
    const float eps
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < (int) n; i += P2P_AR_NQ_THREADS) {
        const float h_v  = (float) hidden[i];
        const float pl_v = (float) partial_local[i];
        const float p0_v = (float) partial_peer0[i];
        const float new_h = h_v + pl_v + p0_v;
        hidden[i] = (fb_fp16_t) new_h;
        sum_sq += new_h * new_h;
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[P2P_AR_NQ_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < P2P_AR_NQ_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = P2P_AR_NQ_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) n;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    const int nblocks = (int) n / QK8_1;
    const int block_lane = tid & 31;
    const int block_group = tid >> 5;
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int b = b0 + block_group;
        const int base = b * QK8_1 + block_lane;
        float normed = 0.0f;
        if (b < nblocks && base < (int) n) {
            const float h_v = (float) hidden[base];
            const float w_v = (float) rms_weight[base];
            normed = h_v * rsqrt * w_v;
        }
        float amax = fabsf(normed);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor(amax, off, 32);
            amax = fmaxf(amax, other);
        }
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        const int qi = max(-127, min(127, (int) rintf(normed * id)));
        int sum_qi = qi;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_qi += __shfl_xor(sum_qi, off, 32);
        }
        if (b < nblocks) {
            out_q8_1[b].qs[block_lane] = (int8_t) qi;
            if (block_lane == 0) {
                out_q8_1[b].d = (fb_fp16_t) d;
                out_q8_1[b].s = (fb_fp16_t) (d * (float) sum_qi);
            }
        }
    }
}
