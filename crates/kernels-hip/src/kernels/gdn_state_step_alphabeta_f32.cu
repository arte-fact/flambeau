// gdn_state_step_alphabeta_f32 — C10 fused GDN recurrent step + α/β/gate
// Same kernel structure as `gdn_state_step_f32_s128`, but absorbs the
// preceding `flambeau_gdn_alpha_beta_f32` launch (one launch saved per
// GDN layer per token). Instead of reading already-processed gate/beta
// scalars, this kernel reads the raw mmvq outputs `alpha_in[B,L,H]` and
// `beta_in[B,L,H]` plus the per-head constants `ssm_dt_bias[H]`,
// `ssm_a[H]`, and computes:
// gate_eff = exp( softplus(alpha[t,i] + ssm_dt_bias[i]) * ssm_a[i] )
// beta_eff = sigmoid( beta_in[t,i] )
// inline. The unfused chain ran:
// alpha_beta(...) → gate_device, beta_device (kernel 1)
// state_step(... gate_device, beta_device ...) (kernel 2)
// On Qwen3.6 with 30 GDN layers, that's 60 launches per decode token;
// the fusion drops it to 30 — saves ~5 µs × 30 ≈ 150 µs/token (~3-5 %
// of decode wall on a hybrid 9B/35B run).
// Numerically identical to the unfused chain at FP32: same op order
// (softplus → __expf), same sigmoid branch, same warp-reduce
// signatures. Caller no longer needs to allocate `gate_device` /
// `beta_device` scratch buffers, but those slots can stay (used only
// by callers that haven't migrated to this fused variant).

#include <hip/hip_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#ifndef GDN_WARPS_PER_BLOCK
#define GDN_WARPS_PER_BLOCK 4
#endif

static __device__ __forceinline__ float gdn_warp_reduce_sum_f32_ab(float x) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, WARP_SIZE);
    }
    return x;
}

template <int S_v>
static __device__ __forceinline__ void gdn_state_step_alphabeta_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ alpha_in,
    const float * __restrict__ beta_in,
    const float * __restrict__ ssm_dt_bias,
    const float * __restrict__ ssm_a,
    const float * __restrict__ state_in,
    float * __restrict__ state_out,
    float * __restrict__ attn_out,
    int B, int H, int L, int n_rep, int rep_inner_layout)
{
    constexpr int warp_size     = WARP_SIZE;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of WARP_SIZE");

    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const int col   = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane  = threadIdx.x;

    if (col >= S_v) {
        return;
    }

    // GQA k-head mapping. See gdn_state_step_f32.cu for the full
    // explanation: rep_inner_layout=0 (rep-OUTER, qwen35moe / candle)
    // uses cyclic `h_idx % H_kv`; rep_inner_layout=1 (rep-INNER,
    // qwen3next) uses interleaved `h_idx / n_rep`.
    const int H_kv   = H / n_rep;
    const int h_kv   = (rep_inner_layout != 0) ? (h_idx / n_rep) : (h_idx % H_kv);
    const int bh     = b_idx * H    + h_idx;
    const int64_t q_stride_bl  = (int64_t) L * H_kv * S_v;
    const int64_t v_stride_bl  = (int64_t) L * H    * S_v;
    const int64_t gt_stride_bl = (int64_t) L * H;
    const float * state_in_bh  = state_in  + (int64_t)bh * S_v * S_v;
    float *       state_out_bh = state_out + (int64_t)bh * S_v * S_v;

    const float * state_in_col  = state_in_bh  + (int64_t)col * S_v;
    float *       state_out_col = state_out_bh + (int64_t)col * S_v;

    // Per-head constants — same value for every token; load once into
    // registers (the broadcast across L is lane-uniform, so a single
    // scalar load is fine).
    const float dt_bias_h = ssm_dt_bias[h_idx];
    const float a_h       = ssm_a[h_idx];

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = state_in_col[i];
    }

    for (int t = 0; t < L; t++) {
        const float * alpha_bt = alpha_in + b_idx * gt_stride_bl + (int64_t) t * H;
        const float * beta_bt  = beta_in  + b_idx * gt_stride_bl + (int64_t) t * H;
        const float * q_bt     = q        + b_idx * q_stride_bl  + ((int64_t) t * H_kv + h_kv) * S_v;
        const float * k_bt     = k        + b_idx * q_stride_bl  + ((int64_t) t * H_kv + h_kv) * S_v;
        const float * v_bt     = v        + b_idx * v_stride_bl  + ((int64_t) t * H    + h_idx) * S_v;

        // Inline alpha/beta compute — exact numerical match with
        // gdn_alpha_beta_f32. The state-step then does __expf(gate)
        // (legacy semantic), so we emit the same softplus*ssm_a as
        // gate-input and sigmoid as beta-input.
        const float a_raw    = alpha_bt[h_idx] + dt_bias_h;
        const float abs_a    = fabsf(a_raw);
        const float max_a    = fmaxf(a_raw, 0.0f);
        const float softplus = max_a + __logf(1.0f + __expf(-abs_a));
        const float gate_pre = softplus * a_h;
        const float g_val    = __expf(gate_pre);

        const float b_raw = beta_bt[h_idx];
        float beta_val;
        if (b_raw >= 0.0f) {
            beta_val = 1.0f / (1.0f + __expf(-b_raw));
        } else {
            const float z = __expf(b_raw);
            beta_val = z / (1.0f + z);
        }

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_bt[i];
            q_reg[r] = q_bt[i];
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] *= g_val;
        }

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = gdn_warp_reduce_sum_f32_ab(kv_shard);

        const float v_col     = v_bt[col];
        const float delta_col = (v_col - kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]   += k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = gdn_warp_reduce_sum_f32_ab(attn_partial);

        if (lane == 0) {
            attn_out[(((int64_t) b_idx * L + t) * H + h_idx) * S_v + col] = attn_col;
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        state_out_col[i] = s_shard[r];
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE * GDN_WARPS_PER_BLOCK, 1)
void flambeau_gdn_state_step_alphabeta_f32_s128(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ alpha_in,
    const float * __restrict__ beta_in,
    const float * __restrict__ ssm_dt_bias,
    const float * __restrict__ ssm_a,
    const float * __restrict__ state_in,
    float * __restrict__ state_out,
    float * __restrict__ attn_out,
    int B, int H, int L, int n_rep, int rep_inner_layout)
{
    gdn_state_step_alphabeta_impl<128>(
        q, k, v, alpha_in, beta_in, ssm_dt_bias, ssm_a,
        state_in, state_out, attn_out, B, H, L, n_rep, rep_inner_layout);
}
