// gdn_alpha_beta_f32 — fused GDN α/β/gate compute.
//
// Per token t, per v-head i:
//   gate_out[t, i] = softplus(alpha_in[t, i] + ssm_dt_bias[i]) * ssm_a[i]
//   beta_out[t, i] = sigmoid(beta_in[t, i])
//
// `ssm_dt_bias` and `ssm_a` are per-head constants (shape [num_v_heads]),
// shared across all L tokens.
//
// V2.2.d fix 3: kernel now takes `n_tokens`; grid = (n_tokens, 1, 1),
// block = (num_v_heads, 1, 1). Previously the caller looped L times at
// one-token-per-launch; at pp512 × 16 GDN layers this fired 12312 tiny
// launches dominated by launch overhead (4 µs/call × 12k = 50 ms, plus
// ~150 ms of corresponding rocclr_copyBuffer arg-marshalling). One
// launch per layer now.
//
// Decode (n_tokens = 1) works unchanged — grid = (1, 1, 1).
//
// Numerically-stable forms:
//   softplus(x) = max(x, 0) + log1p(exp(-|x|))
//   sigmoid(x)  = { 1 / (1 + exp(-x)) if x >= 0,  exp(x) / (1 + exp(x)) otherwise }

#include <hip/hip_runtime.h>

extern "C" __global__ void flambeau_gdn_alpha_beta_f32(
    const float* __restrict__ alpha_in,    // [n_tokens, num_v_heads]
    const float* __restrict__ beta_in,     // [n_tokens, num_v_heads]
    const float* __restrict__ ssm_dt_bias, // [num_v_heads]
    const float* __restrict__ ssm_a,       // [num_v_heads]
    float*       __restrict__ gate_out,    // [n_tokens, num_v_heads]
    float*       __restrict__ beta_out,    // [n_tokens, num_v_heads]
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
