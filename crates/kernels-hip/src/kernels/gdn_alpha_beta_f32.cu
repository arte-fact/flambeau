// gdn_alpha_beta_f32 — fused GDN α/β/gate compute on [num_v_heads] F32.
//
// Replaces the V1.7.3-c2 host-roundtrip placeholder. Per element i:
//   gate_out[i] = softplus(alpha_in[i] + ssm_dt_bias[i]) * ssm_a[i]
//   beta_out[i] = sigmoid(beta_in[i])
//
// Numerically-stable forms:
//   softplus(x) = max(x, 0) + log1p(exp(-|x|))
//   sigmoid(x)  = { 1 / (1 + exp(-x)) if x >= 0,  exp(x) / (1 + exp(x)) otherwise }
//
// Launch: one block of 64 threads (covers num_v_heads ≤ 64, which is the
// Qwen3.6 budget of 32). Generalising to >64 is a grid-dim bump but no
// current model hits it.

#include <hip/hip_runtime.h>

extern "C" __global__ void flambeau_gdn_alpha_beta_f32(
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float a = alpha_in[i] + ssm_dt_bias[i];
    const float abs_a = fabsf(a);
    const float max_a = fmaxf(a, 0.0f);
    const float softplus = max_a + __logf(1.0f + __expf(-abs_a));
    gate_out[i] = softplus * ssm_a[i];

    const float b = beta_in[i];
    float sig;
    if (b >= 0.0f) {
        sig = 1.0f / (1.0f + __expf(-b));
    } else {
        const float z = __expf(b);
        sig = z / (1.0f + z);
    }
    beta_out[i] = sig;
}
