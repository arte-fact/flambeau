#pragma once

// Per-row masked softmax for attention scores, F16 in/out. Two-pass online
// softmax: pass 1 computes (max, sum) via block_reduce_softmax over the
// WARP_SIZE seam; pass 2 re-reads scores and normalizes. mask may be null.
// out[i] = exp(scale*scores[i] + mask[i] - max) / sum.
template <int THREADS>
static __device__ void softmax_masked_f16_row(
    const fb_fp16_t* __restrict__ scores,
    const fb_fp16_t* __restrict__ mask,
    fb_fp16_t* __restrict__ out,
    const int n_rows,
    const int k,
    const float scale
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const fb_fp16_t* s_row = scores + (size_t) row * k;
    const fb_fp16_t* m_row = mask ? (mask + (size_t) row * k) : nullptr;
    fb_fp16_t* o_row = out + (size_t) row * k;

    float local_max = FB_NEG_INF;
    float local_sum = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        float v = scale * (float) s_row[i];
        if (m_row) {
            v += (float) m_row[i];
        }
        if (v > local_max) {
            local_sum = local_sum * __expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += __expf(v - local_max);
        }
    }

    __shared__ float scratch_max[THREADS / WARP_SIZE];
    __shared__ float scratch_sum[THREADS / WARP_SIZE];
    block_reduce_softmax<THREADS>(local_max, local_sum, scratch_max, scratch_sum);
    const float row_max = local_max;
    const float inv_sum = 1.0f / local_sum;

    for (int i = tid; i < k; i += THREADS) {
        float v = scale * (float) s_row[i];
        if (m_row) {
            v += (float) m_row[i];
        }
        o_row[i] = (fb_fp16_t) (__expf(v - row_max) * inv_sum);
    }
}
