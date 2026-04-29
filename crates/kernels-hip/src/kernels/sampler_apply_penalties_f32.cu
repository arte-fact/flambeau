// sampler_apply_penalties_f32 — apply repetition / presence / frequency
// penalties in place on F32 logits. Input is a packed
// `(token_id, count)` array of length `n_pairs` (already deduped on
// host via Sampler-F's sort+dedup). One thread per pair; each thread
// reads `logits[tok]`, applies the three penalties, writes back. No
// cross-thread coordination since deduped pairs guarantee unique
// `tok` per thread (no atomic-add needed).
//
// Mirrors the host-side `apply_penalty_kernel` in
// `crates/runtime/src/sampling.rs` so a future cert can compare GPU
// output bit-for-bit against the host reference.
//
// Layout:
//   blockDim = { 256 }                          (4 wave64 warps)
//   gridDim  = { ceil(n_pairs / 256) }
//
// Penalty conventions (llama.cpp / OpenAI):
//   * `repetition_penalty`: `logit /= penalty` when logit > 0,
//                           `logit *= penalty` otherwise. Preserves
//                           sign. No-op at 1.0.
//   * `presence_penalty`:   `logit -= penalty`. No-op at 0.0.
//   * `frequency_penalty`:  `logit -= penalty * count`. No-op at 0.0.

#include <hip/hip_runtime.h>

extern "C" __global__ void flambeau_sampler_apply_penalties_f32(
    float* __restrict__ logits,                  // [V] in/out
    const unsigned int* __restrict__ token_counts, // [n_pairs * 2] (tok, count) interleaved
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

    // Repetition penalty (sign-preserving). Active iff != 1.0 && > 0.
    if (repetition_penalty != 1.0f && repetition_penalty > 0.0f) {
        if (l > 0.0f) {
            l /= repetition_penalty;
        } else {
            l *= repetition_penalty;
        }
    }
    // Presence penalty.
    if (presence_penalty != 0.0f) {
        l -= presence_penalty;
    }
    // Frequency penalty.
    if (frequency_penalty != 0.0f) {
        l -= frequency_penalty * (float) count;
    }

    logits[tok] = l;
}
