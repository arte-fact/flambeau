// p2p_allreduce_residual_rmsnorm — fused AR + residual-add + RMSNorm.
// In one launch per rank:
// 1. Read peer partials directly via BAR1.
// 2. hidden[i] += partial_local[i] + Σ partial_peer{0,1,2}[i]
// (residual-fold AR — same arithmetic as flambeau_p2p_allreduce_residual_tp{2,4})
// 3. Compute mean(hidden^2) across the full vector of length n.
// 4. out_norm[i] = hidden[i] * rsqrt(mean_sq + eps) * weight[i]
// Replaces TWO launches in the TP forward path (residual-tp4 + rmsnorm_f16)
// with ONE. Bit-exact equivalent because:
// - Same AR arithmetic (FP32 accumulate of the same operands in the same order).
// - Same RMSNorm formula on the post-AR hidden value, identical to running
// flambeau_rmsnorm_f16 on the AR'd buffer.
// - Single-block (one block per call, no inter-block reduction races).
// Constraints:
// - n must be a multiple of THREADS=256 AND fit in one block's element budget.
// Per thread: ELEMS_PER_THREAD = n / 256. For Qwen3.5/3.6 hidden sizes
// {2048, 4096, 5120, 8192} this is 8/16/20/32 elements/thread — all
// comfortable within VGPR budget (≤ 32 floats × 4 bytes = 128 bytes/lane).
// - Outer grid = 1 block (the kernel is row-internal — the global mean-sq
// reduce is a single-block reduction over LDS, no inter-block sync).
// gfx906 occupancy: 256 threads/block × 1 block = 4 wave64s on a single CU.
// One CU per rank running this means peak occupancy isn't a concern; the
// kernel is fundamentally serial against itself.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "block_quant.cuh"

#define P2P_AR_NORM_THREADS 256
#define P2P_AR_NORM_WARPS (P2P_AR_NORM_THREADS / 64)

// ------------------------------------------------------------------------
// TP=4: hidden += partial_local + Σ partial_peer{0,1,2}; then RMSNorm.
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_NORM_THREADS)
void flambeau_p2p_allreduce_residual_rmsnorm_tp4(
    fb_fp16_t* __restrict__ hidden,            // [n] — residual-folded in place
    const fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const fb_fp16_t* __restrict__ partial_peer1,
    const fb_fp16_t* __restrict__ partial_peer2,
    const fb_fp16_t* __restrict__ rms_weight,  // [n]
    fb_fp16_t* __restrict__ out_norm,          // [n]
    const unsigned int n,
    const float eps
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    // --- Phase 1: peer-read + residual fold + sum-of-squares ---
    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < (int) n; i += P2P_AR_NORM_THREADS) {
        const float h_v  = (float) hidden[i];
        const float pl_v = (float) partial_local[i];
        const float p0_v = (float) partial_peer0[i];
        const float p1_v = (float) partial_peer1[i];
        const float p2_v = (float) partial_peer2[i];
        const float new_h = h_v + pl_v + p0_v + p1_v + p2_v;
        // Write the residual-folded hidden so the caller's downstream
        // residual ops see the AR'd value.
        hidden[i] = (fb_fp16_t) new_h;
        sum_sq += new_h * new_h;
    }

    // --- Reduce sum_sq across the block ---
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[P2P_AR_NORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < P2P_AR_NORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = P2P_AR_NORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) n;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    // --- Phase 2: elementwise weight × rsqrt × residual-folded hidden ---
    #pragma unroll 4
    for (int i = tid; i < (int) n; i += P2P_AR_NORM_THREADS) {
        const float h_v = (float) hidden[i];
        const float w_v = (float) rms_weight[i];
        out_norm[i] = (fb_fp16_t) (h_v * w_v * rsqrt);
    }
}

// ------------------------------------------------------------------------
// TP=2: hidden += partial_local + partial_peer0; then RMSNorm.
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_NORM_THREADS)
void flambeau_p2p_allreduce_residual_rmsnorm_tp2(
    fb_fp16_t* __restrict__ hidden,
    const fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const fb_fp16_t* __restrict__ rms_weight,
    fb_fp16_t* __restrict__ out_norm,
    const unsigned int n,
    const float eps
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < (int) n; i += P2P_AR_NORM_THREADS) {
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
    __shared__ float s_warp[P2P_AR_NORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < P2P_AR_NORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = P2P_AR_NORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) n;
    const float rsqrt = 1.0f / sqrtf(mean_sq + eps);

    #pragma unroll 4
    for (int i = tid; i < (int) n; i += P2P_AR_NORM_THREADS) {
        const float h_v = (float) hidden[i];
        const float w_v = (float) rms_weight[i];
        out_norm[i] = (fb_fp16_t) (h_v * w_v * rsqrt);
    }
}

// ------------------------------------------------------------------------
// Gemma4 post-attn / post-ffn fused path. Different shape from the
// `_residual_rmsnorm_tp*` kernels above (which fold the AR'd partial
// INTO `hidden` and then norm `hidden`):
//
//   resid_out = resid_in + rmsnorm(Σ proj_partial, post_norm_w, eps)
//
// Inputs are F32 projection outputs (saves the upstream
// `cast_f32_to_f16` launch that S1's split-launch path needed). The
// AR-summed partial is rmsnormed and added to a SEPARATE input
// residual; the new residual is written to a fresh output buffer
// (gemma4's pool advances residual slots per layer).
//
// Replaces a 2-launch sequence
//   `ar_sum_f32(proj_local)` + `rmsnorm_f32_to_f16_add_residual(...)`
// with one launch.
//
// Layout: gridDim={n_rows}, blockDim={256}. Each block handles one
// row of `n` elements. AR sums held in per-thread registers (max
// 32 elems/thread × 256 threads = hidden ≤ 8192 supported). Peer
// BAR1 read happens once per element.
// ------------------------------------------------------------------------

#define P2P_POSTNORM_MAX_ELEMS_PER_THREAD 32

extern "C" __global__ __launch_bounds__(P2P_AR_NORM_THREADS)
void flambeau_p2p_allreduce_postattn_residual_rmsnorm_f32_to_f16_tp2(
    const float*     __restrict__ proj_local,
    const float*     __restrict__ proj_peer0,
    const fb_fp16_t* __restrict__ post_norm_w,
    const fb_fp16_t* __restrict__ resid_in,
    fb_fp16_t*       __restrict__ resid_out,
    const unsigned int n,
    const float eps
) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float*     local_row     = proj_local  + (size_t) row * n;
    const float*     peer_row      = proj_peer0  + (size_t) row * n;
    const fb_fp16_t* resid_in_row  = resid_in    + (size_t) row * n;
    fb_fp16_t*       resid_out_row = resid_out   + (size_t) row * n;

    float ar_sums[P2P_POSTNORM_MAX_ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int k = 0; k < P2P_POSTNORM_MAX_ELEMS_PER_THREAD; ++k) {
        const int i = tid + k * P2P_AR_NORM_THREADS;
        if (i < (int) n) {
            const float s = local_row[i] + peer_row[i];
            ar_sums[k] = s;
            sum_sq += s * s;
        } else {
            ar_sums[k] = 0.0f;
        }
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[P2P_AR_NORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < P2P_AR_NORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = P2P_AR_NORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) n;
    const float rsqrt   = 1.0f / sqrtf(mean_sq + eps);

    #pragma unroll
    for (int k = 0; k < P2P_POSTNORM_MAX_ELEMS_PER_THREAD; ++k) {
        const int i = tid + k * P2P_AR_NORM_THREADS;
        if (i < (int) n) {
            const float w = (float) post_norm_w[i];
            const float r = (float) resid_in_row[i];
            float out = r + ar_sums[k] * w * rsqrt;
            if (out > 65504.0f) out = 65504.0f;
            else if (out < -65504.0f) out = -65504.0f;
            resid_out_row[i] = (fb_fp16_t) out;
        }
    }
}

extern "C" __global__ __launch_bounds__(P2P_AR_NORM_THREADS)
void flambeau_p2p_allreduce_postattn_residual_rmsnorm_f32_to_f16_tp4(
    const float*     __restrict__ proj_local,
    const float*     __restrict__ proj_peer0,
    const float*     __restrict__ proj_peer1,
    const float*     __restrict__ proj_peer2,
    const fb_fp16_t* __restrict__ post_norm_w,
    const fb_fp16_t* __restrict__ resid_in,
    fb_fp16_t*       __restrict__ resid_out,
    const unsigned int n,
    const float eps
) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 6;
    const int lane = tid & 63;

    const float*     local_row     = proj_local  + (size_t) row * n;
    const float*     peer0_row     = proj_peer0  + (size_t) row * n;
    const float*     peer1_row     = proj_peer1  + (size_t) row * n;
    const float*     peer2_row     = proj_peer2  + (size_t) row * n;
    const fb_fp16_t* resid_in_row  = resid_in    + (size_t) row * n;
    fb_fp16_t*       resid_out_row = resid_out   + (size_t) row * n;

    float ar_sums[P2P_POSTNORM_MAX_ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int k = 0; k < P2P_POSTNORM_MAX_ELEMS_PER_THREAD; ++k) {
        const int i = tid + k * P2P_AR_NORM_THREADS;
        if (i < (int) n) {
            const float s = local_row[i] + peer0_row[i] + peer1_row[i] + peer2_row[i];
            ar_sums[k] = s;
            sum_sq += s * s;
        } else {
            ar_sums[k] = 0.0f;
        }
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 64);
    }
    __shared__ float s_warp[P2P_AR_NORM_WARPS];
    if (lane == 0) {
        s_warp[warp] = sum_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < P2P_AR_NORM_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = P2P_AR_NORM_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if (lane == 0) {
            s_warp[0] = v;
        }
    }
    __syncthreads();
    const float mean_sq = s_warp[0] / (float) n;
    const float rsqrt   = 1.0f / sqrtf(mean_sq + eps);

    #pragma unroll
    for (int k = 0; k < P2P_POSTNORM_MAX_ELEMS_PER_THREAD; ++k) {
        const int i = tid + k * P2P_AR_NORM_THREADS;
        if (i < (int) n) {
            const float w = (float) post_norm_w[i];
            const float r = (float) resid_in_row[i];
            float out = r + ar_sums[k] * w * rsqrt;
            if (out > 65504.0f) out = 65504.0f;
            else if (out < -65504.0f) out = -65504.0f;
            resid_out_row[i] = (fb_fp16_t) out;
        }
    }
}
