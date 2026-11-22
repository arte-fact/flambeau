// gdn_alpha_beta_f32 — fused GDN gate = softplus(alpha+dt_bias)*ssm_a, beta =
// sigmoid(beta_in). Pointwise per (token, v-head). grid (n_tokens,1,1), block
// num_v_heads.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_gdn_alpha_beta_f32(
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_dt_bias,
    const float* __restrict__ ssm_a,
    float*       __restrict__ gate_out,
    float*       __restrict__ beta_out,
    const int num_v_heads,
    const int n_tokens
) {
    const int t = blockIdx.x;
    const int i = threadIdx.x;
    if (t >= n_tokens || i >= num_v_heads) return;

    const size_t row = (size_t) t * num_v_heads;
    const float a = alpha_in[row + i] + ssm_dt_bias[i];
    const float abs_a = fabsf(a);
    const float max_a = fmaxf(a, 0.0f);
    const float softplus = max_a + __logf(1.0f + __expf(-abs_a));
    gate_out[row + i] = softplus * ssm_a[i];

    const float b = beta_in[row + i];
    float sig;
    if (b >= 0.0f) {
        sig = 1.0f / (1.0f + __expf(-b));
    } else {
        const float z = __expf(b);
        sig = z / (1.0f + z);
    }
    beta_out[row + i] = sig;
}
