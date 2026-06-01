// apply_per_expert_scale_f32 — multiply per-token routing weights by
// each selected expert's scalar. Gemma4 26B-A4B's `ffn_down_exps.scale`
// (F32 [n_experts]) is applied post down-projection per llama.cpp's
// `build_moe_ffn`:
//   experts[t, k, h] *= down_exps_s[topk_ids[t, k]]
//   weighted          = experts * topk_weights[t, k]
//   moe_out[t, h]     = Σ_k weighted[t, k, h]
// Identical to folding the per-expert scale into the routing weights
// BEFORE the combine — moe_out[t, h] = Σ_k (s[k] * w[k]) * down[k, h].
// This kernel does the fold so the existing `moe_combine_*` kernels
// pick up the scale for free.
//
// Layout:
//   weights      F32 [n_tokens, top_k]  (modified in place)
//   expert_ids   i32 [n_tokens, top_k]
//   scale        F32 [n_experts]
// Launch: gridDim.x = n_tokens, blockDim.x = top_k threads.
// `n_tokens = 1` recovers the single-token decode case; the prefill
// path passes `n_tokens > 1` so the per-token scale fold is one
// launch instead of N.

#include <hip/hip_runtime.h>

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
