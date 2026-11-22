// apply_per_expert_scale_f32 — fold each selected expert's scalar into the
// per-token routing weight (in place). One thread per (token, k). No reduce.
// Launch: gridDim.x = n_tokens, blockDim.x = top_k.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_apply_per_expert_scale_f32(
    float* __restrict__ expert_weights,
    const int* __restrict__ expert_ids,
    const float* __restrict__ expert_scales,
    const int top_k
) {
    const int t = blockIdx.x;
    const int k = threadIdx.x;
    if (k >= top_k) return;
    const int row = t * top_k + k;
    const int eid = expert_ids[row];
    expert_weights[row] = expert_weights[row] * expert_scales[eid];
}
