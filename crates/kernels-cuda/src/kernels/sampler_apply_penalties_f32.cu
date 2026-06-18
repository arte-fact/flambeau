// sampler_apply_penalties_f32 — repetition / presence / frequency penalties in
// place on F32 logits. Input is a host-deduped (token_id, count) array, so each
// thread owns a unique token (no atomics). One thread per pair, no reduce.
// Launch: 1D, ceil(n_pairs/256) blocks × 256 threads.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_sampler_apply_penalties_f32(
    float* __restrict__ logits,                  // [V] in/out
    const unsigned int* __restrict__ token_counts, // [n_pairs * 2] (tok, count)
    const int n_pairs,
    const int vocab,
    const float repetition_penalty,
    const float presence_penalty,
    const float frequency_penalty
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;

    const unsigned int tok   = token_counts[(size_t) idx * 2 + 0];
    const unsigned int count = token_counts[(size_t) idx * 2 + 1];
    if ((int) tok >= vocab) return;

    float l = logits[tok];
    if (repetition_penalty != 1.0f && repetition_penalty > 0.0f) {
        l = (l > 0.0f) ? (l / repetition_penalty) : (l * repetition_penalty);
    }
    if (presence_penalty != 0.0f) {
        l -= presence_penalty;
    }
    if (frequency_penalty != 0.0f) {
        l -= frequency_penalty * (float) count;
    }
    logits[tok] = l;
}
