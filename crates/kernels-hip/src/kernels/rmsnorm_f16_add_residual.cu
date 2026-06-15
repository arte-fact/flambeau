// rmsnorm_f16_add_residual — fused add+rmsnorm.
// Pattern in `forward_layer_decode`:
// mid = x_in + attn_delta (add_f16)
// mid_norm = rmsnorm(mid) * weight (rmsnorm_f16)
// Both mid and mid_norm are consumed downstream (mid → moe_residual path,
// mid_norm → FFN input), so we emit both. Single kernel, single pass.
// Launch identical to rmsnorm_f16.cu: blockDim={256}, gridDim={n_rows},
// 4 wave64 warps/block. K must be multiple of 256 (Qwen3.6's hidden sizes
// 2048/5120/15360 all are).

#include <hip/hip_runtime.h>

#define RMSNORM_THREADS 256
#define RMSNORM_WARPS (RMSNORM_THREADS / 64)

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_rmsnorm_f16_add_residual(
    const fb_fp16_t* __restrict__ x_in,     // [n_rows, k] — residual to add
    const fb_fp16_t* __restrict__ delta,    // [n_rows, k] — attention delta
    const fb_fp16_t* __restrict__ weight,   // [k]
    fb_fp16_t*       __restrict__ mid,      // [n_rows, k] — x_in + delta
    fb_fp16_t*       __restrict__ mid_norm, // [n_rows, k] — rmsnorm(mid) * weight
    const int n_rows,
    const int k,
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const fb_fp16_t* xrow  = x_in  + (size_t) row * k;
    const fb_fp16_t* drow  = delta + (size_t) row * k;
    fb_fp16_t*       mrow  = mid   + (size_t) row * k;
    fb_fp16_t*       nrow  = mid_norm + (size_t) row * k;

    // Phase 1: compute mid = x_in + delta, accumulate sum-of-squares on mid.
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_THREADS) {
        const float v = (float) xrow[i] + (float) drow[i];
        mrow[i] = (fb_fp16_t) v;
        sum_sq += v * v;
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }

    __shared__ float s_warp[RMSNORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < RMSNORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = RMSNORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();

    const float total_sq = s_warp[0];
    const float mean_sq  = total_sq / (float) k;
    const float rsqrt    = 1.0f / sqrtf(mean_sq + eps);

    // Phase 2: elementwise scale — re-read mid (now in HBM) since registers
    // across all threads don't span the row.
    #pragma unroll 4
    for (int i = tid; i < k; i += RMSNORM_THREADS) {
        const float v = (float) mrow[i];
        const float w = (float) weight[i];
        nrow[i] = (fb_fp16_t) (v * w * rsqrt);
    }
}
