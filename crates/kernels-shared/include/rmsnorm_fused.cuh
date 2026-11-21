#pragma once

// Fused RMSNorm variants (gemma4 cast/residual paths + the decode RMSNorm→Q8_1
// edge). All share the phase-1 sum-of-squares block reduce (block_reduce_sum
// over the WARP_SIZE seam); they differ only in input dtype and epilogue, so
// each is its own device body (rule 14 — no forced single template). One block
// per row, THREADS threads. Saturating clamp at ±F16_MAX on F16 residual adds.

static __device__ __forceinline__ float fb_sat_f16(float v) {
    if (v > 65504.0f) return 65504.0f;
    if (v < -65504.0f) return -65504.0f;
    return v;
}

// F32 input → F16 weight → F16 output (gemma4 post-norm: cast+rmsnorm fused).
template <int THREADS>
static __device__ void rmsnorm_f32_to_f16_row(
    const float* __restrict__ x,
    const fb_fp16_t* __restrict__ weight,
    fb_fp16_t* __restrict__ y,
    const int n_rows, const int k, const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const float* x_row = x + (size_t) row * k;
    fb_fp16_t* y_row = y + (size_t) row * k;

    float sum_sq = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = x_row[i];
        sum_sq += v * v;
    }
    __shared__ float scratch[THREADS / WARP_SIZE];
    const float rsqrt = 1.0f / sqrtf(block_reduce_sum<THREADS>(sum_sq, scratch) / (float) k + eps);

    for (int i = tid; i < k; i += THREADS) {
        y_row[i] = (fb_fp16_t) (x_row[i] * (float) weight[i] * rsqrt);
    }
}

// F32 delta → F16 weight, added into the F16 residual stream (in-place safe).
template <int THREADS>
static __device__ void rmsnorm_f32_to_f16_add_residual_row(
    const float* __restrict__ x,
    const fb_fp16_t* __restrict__ weight,
    const fb_fp16_t* __restrict__ resid_in,
    fb_fp16_t* __restrict__ resid_out,
    const int n_rows, const int k, const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const float* x_row = x + (size_t) row * k;
    const fb_fp16_t* resid_row = resid_in + (size_t) row * k;
    fb_fp16_t* out_row = resid_out + (size_t) row * k;

    float sum_sq = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = x_row[i];
        sum_sq += v * v;
    }
    __shared__ float scratch[THREADS / WARP_SIZE];
    const float rsqrt = 1.0f / sqrtf(block_reduce_sum<THREADS>(sum_sq, scratch) / (float) k + eps);

    for (int i = tid; i < k; i += THREADS) {
        const float v = x_row[i] * (float) weight[i] * rsqrt;
        out_row[i] = (fb_fp16_t) fb_sat_f16((float) resid_row[i] + v);
    }
}

// F16 delta → F16 weight, added into the F16 residual stream (in-place safe).
template <int THREADS>
static __device__ void rmsnorm_f16_to_f16_add_residual_row(
    const fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ weight,
    const fb_fp16_t* __restrict__ resid_in,
    fb_fp16_t* __restrict__ resid_out,
    const int n_rows, const int k, const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const fb_fp16_t* x_row = x + (size_t) row * k;
    const fb_fp16_t* resid_row = resid_in + (size_t) row * k;
    fb_fp16_t* out_row = resid_out + (size_t) row * k;

    float sum_sq = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = (float) x_row[i];
        sum_sq += v * v;
    }
    __shared__ float scratch[THREADS / WARP_SIZE];
    const float rsqrt = 1.0f / sqrtf(block_reduce_sum<THREADS>(sum_sq, scratch) / (float) k + eps);

    for (int i = tid; i < k; i += THREADS) {
        const float v = (float) x_row[i] * (float) weight[i] * rsqrt;
        out_row[i] = (fb_fp16_t) fb_sat_f16((float) resid_row[i] + v);
    }
}

// Fused add + rmsnorm: mid = x_in + delta; mid_norm = rmsnorm(mid)*weight. Both
// outputs are consumed downstream (decode residual + FFN input).
template <int THREADS>
static __device__ void rmsnorm_f16_add_residual_row(
    const fb_fp16_t* __restrict__ x_in,
    const fb_fp16_t* __restrict__ delta,
    const fb_fp16_t* __restrict__ weight,
    fb_fp16_t* __restrict__ mid,
    fb_fp16_t* __restrict__ mid_norm,
    const int n_rows, const int k, const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const fb_fp16_t* x_row = x_in + (size_t) row * k;
    const fb_fp16_t* d_row = delta + (size_t) row * k;
    fb_fp16_t* m_row = mid + (size_t) row * k;
    fb_fp16_t* n_row = mid_norm + (size_t) row * k;

    float sum_sq = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = (float) x_row[i] + (float) d_row[i];
        m_row[i] = (fb_fp16_t) v;
        sum_sq += v * v;
    }
    __shared__ float scratch[THREADS / WARP_SIZE];
    const float rsqrt = 1.0f / sqrtf(block_reduce_sum<THREADS>(sum_sq, scratch) / (float) k + eps);

    for (int i = tid; i < k; i += THREADS) {
        n_row[i] = (fb_fp16_t) ((float) m_row[i] * (float) weight[i] * rsqrt);
    }
}

// Fused RMSNorm → Q8_1 quantize (decode hot edge). Phase 1: Σx² block reduce.
// Phase 2: per-Q8_1-block (32 lanes) amax/sum via width-32 __shfl_xor — already
// wave-agnostic (the 32 is the block width, not the hardware warp).
template <int THREADS>
static __device__ void rmsnorm_q8_1_fused_row(
    const fb_fp16_t* __restrict__ x,
    const fb_fp16_t* __restrict__ weight,
    flambeau_block_q8_1* __restrict__ y,
    const int n_rows, const int k, const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;
    const fb_fp16_t* x_row = x + (size_t) row * k;
    flambeau_block_q8_1* y_row = y + (size_t) row * (k / QK8_1);

    float sum_sq = 0.0f;
    for (int i = tid; i < k; i += THREADS) {
        const float v = (float) x_row[i];
        sum_sq += v * v;
    }
    __shared__ float scratch[THREADS / WARP_SIZE];
    const float rsqrt = 1.0f / sqrtf(block_reduce_sum<THREADS>(sum_sq, scratch) / (float) k + eps);

    const int nblocks = k / QK8_1;
    const int groups = THREADS / 32;
    const int block_lane = tid & 31;
    const int block_group = tid >> 5;
    for (int b0 = 0; b0 < nblocks; b0 += groups) {
        const int b = b0 + block_group;
        const int base = b * QK8_1 + block_lane;
        float normed = 0.0f;
        if (b < nblocks && base < k) {
            normed = (float) x_row[base] * rsqrt * (float) weight[base];
        }
        float amax = fabsf(normed);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, off, 32));
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        const int qi = max(-127, min(127, (int) rintf(normed * id)));
        int sum_qi = qi;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_qi += __shfl_xor(sum_qi, off, 32);
        }
        if (b < nblocks) {
            y_row[b].qs[block_lane] = (int8_t) qi;
            if (block_lane == 0) {
                y_row[b].d = (fb_fp16_t) d;
                y_row[b].s = (fb_fp16_t) (d * (float) sum_qi);
            }
        }
    }
}
