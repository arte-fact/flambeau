// swiglu_f16 — SwiGLU activation for FFN gate+up projection output.
// y[i] = silu(gate[i]) * up[i]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Pure pointwise; one thread per element. gfx906's `__frcp_rn` + `__expf`
// are fast-path enough that we don't need a specialised transcendental.
// Launch shape:
// blockDim = { 256 }
// gridDim = { ceil(n / 256) }
// shared = 0
// Inputs and outputs are flat F16 buffers of length `n` — the caller is
// responsible for laying out [m, hidden] as a contiguous `n = m * hidden`
// stream.

#include <hip/hip_runtime.h>

typedef _Float16 fb_fp16_t;

extern "C" __global__ void flambeau_swiglu_f16(
    const fb_fp16_t* __restrict__ gate,
    const fb_fp16_t* __restrict__ up,
    fb_fp16_t* __restrict__ y,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g = (float) gate[i];
    const float u = (float) up[i];

    // silu(x) = x / (1 + e^{-x}). Compute in F32 so large-negative x doesn't
    // overflow the F16 exponent range (min normal f16 ≈ 6.1e-5).
    const float sig = 1.0f / (1.0f + __expf(-g));
    const float silu = g * sig;

    y[i] = (fb_fp16_t) (silu * u);
}
