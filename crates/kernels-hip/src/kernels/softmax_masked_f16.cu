// softmax_masked_f16 — per-row masked softmax for attention scores.
//
//   v[i]   = scale * scores[i] + mask[i]      (mask may be null)
//   m      = max_i v[i]
//   d      = Σ_i exp(v[i] - m)
//   out[i] = exp(v[i] - m) / d
//
// "Online softmax" two-pass layout:
//   Pass 1 (global → registers): compute running (m, d) stats per thread,
//          reduce across threads with __shfl_xor + LDS.
//   Pass 2 (global → global): re-read scores, compute exp((v - m)) * inv_d,
//          write to output. No intermediate global write — saves one HBM
//          pass vs the textbook 3-pass version.
//
// Launch shape:
//   blockDim  = { 256 }                         (4 wave64 warps)
//   gridDim   = { n_rows }
//   shared    = 4 floats (cross-warp reductions)

#include <hip/hip_runtime.h>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

typedef _Float16 fb_fp16_t;

#define SOFTMAX_THREADS 256
#define SOFTMAX_WARPS (SOFTMAX_THREADS / 64)

extern "C" __global__ void flambeau_softmax_masked_f16(
    const fb_fp16_t* __restrict__ scores,    // [n_rows, k]
    const fb_fp16_t* __restrict__ mask,      // [n_rows, k] or null
    fb_fp16_t* __restrict__ out,             // [n_rows, k]
    const int n_rows,
    const int k,
    const float scale
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const fb_fp16_t* s_row = scores + (size_t) row * k;
    const fb_fp16_t* m_row = mask ? (mask + (size_t) row * k) : nullptr;
    fb_fp16_t*       o_row = out + (size_t) row * k;

    // --- Pass 1: online max + exp-sum ---
    //
    // Per thread: run the recurrence (local_max, local_sum) over this
    // thread's strided slice. After the inner loop, reduce both stats
    // across the warp / block.
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += SOFTMAX_THREADS) {
        float v = scale * (float) s_row[i];
        if (m_row) {
            v += (float) m_row[i];
        }
        if (v > local_max) {
            // Rescale the running sum to the new max.
            local_sum = local_sum * __expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += __expf(v - local_max);
        }
    }

    // Warp reduce (wave64). We need a pair-wise merge that rescales the
    // sum when maxes differ: `(m, d) ⊕ (m', d') = (max(m,m'), d*e^{m-M} + d'*e^{m'-M})`.
    // __shfl_xor lets us do the pair exchange; we carry both stats.
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        float other_max = __shfl_xor(local_max, off, 64);
        float other_sum = __shfl_xor(local_sum, off, 64);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * __expf(local_max - new_max)
                  + other_sum * __expf(other_max - new_max);
        local_max = new_max;
    }

    // Cross-warp merge via LDS. Each warp writes (max, sum) to a slot;
    // warp 0 reduces across the 4 warps.
    __shared__ float s_max[SOFTMAX_WARPS];
    __shared__ float s_sum[SOFTMAX_WARPS];
    if (lane == 0) {
        s_max[warp] = local_max;
        s_sum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float wm = (lane < SOFTMAX_WARPS) ? s_max[lane] : -INFINITY;
        float ws = (lane < SOFTMAX_WARPS) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = SOFTMAX_WARPS / 2; off > 0; off >>= 1) {
            float other_m = __shfl_xor(wm, off, 64);
            float other_s = __shfl_xor(ws, off, 64);
            float new_m = fmaxf(wm, other_m);
            ws = ws * __expf(wm - new_m) + other_s * __expf(other_m - new_m);
            wm = new_m;
        }
        if (lane == 0) {
            s_max[0] = wm;
            s_sum[0] = ws;
        }
    }
    __syncthreads();
    const float row_max = s_max[0];
    const float inv_sum = 1.0f / s_sum[0];

    // --- Pass 2: normalise ---
    #pragma unroll 4
    for (int i = tid; i < k; i += SOFTMAX_THREADS) {
        float v = scale * (float) s_row[i];
        if (m_row) {
            v += (float) m_row[i];
        }
        const float e = __expf(v - row_max);
        o_row[i] = (fb_fp16_t) (e * inv_sum);
    }
}
