#pragma once

// Per-token top-k over F32 logits + softmax of the k winners. One block per
// token, blockDim = n_experts. k rounds of "block argmax + mask"; the argmax
// butterfly rides the WARP_SIZE seam (value+index, lower-index-wins tiebreak),
// so it is correct on wave64 and wave32. Scratch arrays are sized by the
// per-backend warp count (TOPK_MAX_EXPERTS / WARP_SIZE).

#ifndef TOPK_INFINITY
#define TOPK_INFINITY __builtin_huge_valf()
#endif
#define TOPK_MAX_K 16
#define TOPK_MAX_EXPERTS 512
#define TOPK_MAX_WARPS (TOPK_MAX_EXPERTS / WARP_SIZE)

static __device__ void topk_softmax_f32_row(
    const float* __restrict__ logits,
    int* __restrict__ out_indices,
    float* __restrict__ out_weights,
    const int n_tokens,
    const int n_experts,
    const int k
) {
    const int token = blockIdx.x;
    if (token >= n_tokens) return;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;

    __shared__ float s_logits[TOPK_MAX_EXPERTS];
    __shared__ float top_vals[TOPK_MAX_K];
    __shared__ int   top_idxs[TOPK_MAX_K];
    __shared__ float s_max_v[TOPK_MAX_WARPS];
    __shared__ int   s_max_i[TOPK_MAX_WARPS];

    if (tid < TOPK_MAX_EXPERTS) {
        s_logits[tid] = (tid < n_experts)
            ? logits[(size_t) token * n_experts + tid]
            : -TOPK_INFINITY;
    }
    __syncthreads();

    for (int round = 0; round < k; ++round) {
        float local_val = (tid < TOPK_MAX_EXPERTS) ? s_logits[tid] : -TOPK_INFINITY;
        int local_idx = tid;

        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const float other_v = __shfl_xor(local_val, off, WARP_SIZE);
            const int other_idx = __shfl_xor(local_idx, off, WARP_SIZE);
            if (other_v > local_val || (other_v == local_val && other_idx < local_idx)) {
                local_val = other_v;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_max_v[warp] = local_val;
            s_max_i[warp] = local_idx;
        }
        __syncthreads();

        float best_v = s_max_v[0];
        int best_i = s_max_i[0];
        for (int w = 1; w < TOPK_MAX_WARPS; ++w) {
            if (w >= n_warps) continue;
            const float v = s_max_v[w];
            const int i = s_max_i[w];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }

        if (tid == 0) {
            top_vals[round] = best_v;
            top_idxs[round] = best_i;
        }
        __syncthreads();
        if (tid == best_i) {
            s_logits[tid] = -TOPK_INFINITY;
        }
        __syncthreads();
    }

    if (tid < k) {
        float m = top_vals[0];
        for (int i = 1; i < k; ++i) {
            if (top_vals[i] > m) m = top_vals[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += __expf(top_vals[i] - m);
        }
        out_indices[(size_t) token * k + tid] = top_idxs[tid];
        out_weights[(size_t) token * k + tid] = __expf(top_vals[tid] - m) / sum;
    }
}
