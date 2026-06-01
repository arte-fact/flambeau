// apply_softcap_f32 — elementwise final-logit softcap.
// y[i] = tanh(x[i] / cap) * cap. Used by Gemma4 on the LM-head logits
// (hparams.f_final_logit_softcapping = 30 across all 5 audited GGUFs).
// In-place safe: x and y may alias.
// Launch: 1D, ceil(n/256) blocks × 256 threads. One element per thread.

#include <hip/hip_runtime.h>

#ifndef SOFTCAP_THREADS
#define SOFTCAP_THREADS 256
#endif

extern "C" __global__ void flambeau_apply_softcap_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n,
    const float cap
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv = 1.0f / cap;
    y[i] = tanhf(x[i] * inv) * cap;
}
