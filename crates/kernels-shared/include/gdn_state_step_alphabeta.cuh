#pragma once

// GDN recurrent step with inline alpha/beta (fused gdn_alpha_beta + state-step).
// Same register-sharded column recurrence as gdn_state_step.cuh (reuses
// gdn_warp_reduce_sum_f32 + the WARP_SIZE seam); the decay g is folded into the
// state update. Two addressing modes — contiguous state, or per-batch (per-slot)
// state pointer arrays. Include gdn_state_step.cuh first for the reduce.

// One token of the column recurrence: advances `s_shard` (this lane's rows of
// column `col`) by token `t` and writes the column's attention output.
template <int S_v>
static __device__ __forceinline__ void gdn_ab_token_step(
    float (&s_shard)[S_v / WARP_SIZE],
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_in,
    float* __restrict__ attn_out,
    const float dt_bias_h, const float a_h,
    const int b_idx, const int t, const int L, const int H, const int H_kv,
    const int h_idx, const int h_kv, const int col, const int lane,
    const int64_t q_stride_bl, const int64_t v_stride_bl, const int64_t gt_stride_bl)
{
    constexpr int warp_size = WARP_SIZE;
    constexpr int rows_per_lane = S_v / warp_size;
    const float* alpha_bt = alpha_in + b_idx * gt_stride_bl + (int64_t) t * H;
    const float* beta_bt = beta_in + b_idx * gt_stride_bl + (int64_t) t * H;
    const float* q_bt = q + b_idx * q_stride_bl + ((int64_t) t * H_kv + h_kv) * S_v;
    const float* k_bt = k + b_idx * q_stride_bl + ((int64_t) t * H_kv + h_kv) * S_v;
    const float* v_bt = v + b_idx * v_stride_bl + ((int64_t) t * H + h_idx) * S_v;

    const float a_raw = alpha_bt[h_idx] + dt_bias_h;
    const float softplus = fmaxf(a_raw, 0.0f) + __logf(1.0f + __expf(-fabsf(a_raw)));
    const float g_val = __expf(softplus * a_h);
    const float b_raw = beta_bt[h_idx];
    const float beta_val = (b_raw >= 0.0f)
        ? (1.0f / (1.0f + __expf(-b_raw)))
        : (__expf(b_raw) / (1.0f + __expf(b_raw)));

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        k_reg[r] = k_bt[i];
        q_reg[r] = q_bt[i];
    }
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        kv_shard += s_shard[r] * k_reg[r];
    }
    const float kv_col = gdn_warp_reduce_sum_f32(kv_shard);

    const float delta_col = (v_bt[col] - g_val * kv_col) * beta_val;
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = gdn_warp_reduce_sum_f32(attn_partial);
    if (lane == 0) {
        attn_out[(((int64_t) b_idx * L + t) * H + h_idx) * S_v + col] = attn_col;
    }
}

template <int S_v>
static __device__ __forceinline__ void gdn_state_step_alphabeta_impl(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_dt_bias,
    const float* __restrict__ ssm_a,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    float* __restrict__ attn_out,
    int B, int H, int L, int n_rep, int rep_inner_layout)
{
    constexpr int warp_size = WARP_SIZE;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of WARP_SIZE");

    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= S_v) return;

    const int H_kv = H / n_rep;
    const int h_kv = (rep_inner_layout != 0) ? (h_idx / n_rep) : (h_idx % H_kv);
    const int bh = b_idx * H + h_idx;
    const int64_t q_stride_bl = (int64_t) L * H_kv * S_v;
    const int64_t v_stride_bl = (int64_t) L * H * S_v;
    const int64_t gt_stride_bl = (int64_t) L * H;
    const float* state_in_col = state_in + (int64_t) bh * S_v * S_v + (int64_t) col * S_v;
    float* state_out_col = state_out + (int64_t) bh * S_v * S_v + (int64_t) col * S_v;

    const float dt_bias_h = ssm_dt_bias[h_idx];
    const float a_h = ssm_a[h_idx];

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = state_in_col[r * warp_size + lane];
    }

    for (int t = 0; t < L; t++) {
        gdn_ab_token_step<S_v>(s_shard, q, k, v, alpha_in, beta_in, attn_out, dt_bias_h, a_h,
                               b_idx, t, L, H, H_kv, h_idx, h_kv, col, lane,
                               q_stride_bl, v_stride_bl, gt_stride_bl);
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        state_out_col[r * warp_size + lane] = s_shard[r];
    }
}

// Per-slot variant: state lives in per-batch buffers addressed by pointer arrays.
template <int S_v>
static __device__ __forceinline__ void gdn_state_step_ab_batched_slots_impl(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_dt_bias,
    const float* __restrict__ ssm_a,
    const float* const* __restrict__ state_in_ptrs,
    float* const* __restrict__ state_out_ptrs,
    float* __restrict__ attn_out,
    int B, int H, int L, int n_rep, int rep_inner_layout)
{
    constexpr int warp_size = WARP_SIZE;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of WARP_SIZE");

    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= S_v) return;

    const int H_kv = H / n_rep;
    const int h_kv = (rep_inner_layout != 0) ? (h_idx / n_rep) : (h_idx % H_kv);
    const int64_t q_stride_bl = (int64_t) L * H_kv * S_v;
    const int64_t v_stride_bl = (int64_t) L * H * S_v;
    const int64_t gt_stride_bl = (int64_t) L * H;
    const float* state_in_col = state_in_ptrs[b_idx] + (int64_t) h_idx * S_v * S_v + (int64_t) col * S_v;
    float* state_out_col = state_out_ptrs[b_idx] + (int64_t) h_idx * S_v * S_v + (int64_t) col * S_v;

    const float dt_bias_h = ssm_dt_bias[h_idx];
    const float a_h = ssm_a[h_idx];

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = state_in_col[r * warp_size + lane];
    }

    for (int t = 0; t < L; t++) {
        gdn_ab_token_step<S_v>(s_shard, q, k, v, alpha_in, beta_in, attn_out, dt_bias_h, a_h,
                               b_idx, t, L, H, H_kv, h_idx, h_kv, col, lane,
                               q_stride_bl, v_stride_bl, gt_stride_bl);
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        state_out_col[r * warp_size + lane] = s_shard[r];
    }
}

// Speculative-verify variant (one sequence, B = 1): reads the input state from
// `state_live` when `rs_idx[0] == 0`, else from `kstates` slot `rs_idx[0] - 1`,
// and writes the state after every token t into `kstates` slot t instead of
// writing a final state back. A rejection then only moves `rs_idx`. Each thread
// loads its own state elements before storing any, so reading and writing the
// same slot is safe.
template <int S_v>
static __device__ __forceinline__ void gdn_state_step_ab_kstates_impl(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_dt_bias,
    const float* __restrict__ ssm_a,
    const float* state_live,
    float* kstates,
    const int* __restrict__ rs_idx,
    float* __restrict__ attn_out,
    int H, int L, int n_rep, int rep_inner_layout)
{
    constexpr int warp_size = WARP_SIZE;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of WARP_SIZE");

    const int h_idx = blockIdx.x;
    const int b_idx = 0;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= S_v) return;

    const int H_kv = H / n_rep;
    const int h_kv = (rep_inner_layout != 0) ? (h_idx / n_rep) : (h_idx % H_kv);
    const int64_t q_stride_bl = (int64_t) L * H_kv * S_v;
    const int64_t v_stride_bl = (int64_t) L * H * S_v;
    const int64_t gt_stride_bl = (int64_t) L * H;
    const int64_t slot_stride = (int64_t) H * S_v * S_v;
    const int64_t col_off = (int64_t) h_idx * S_v * S_v + (int64_t) col * S_v;
    const int idx = rs_idx[0];
    const float* state_in_col = (idx == 0 ? state_live : kstates + (int64_t) (idx - 1) * slot_stride) + col_off;

    const float dt_bias_h = ssm_dt_bias[h_idx];
    const float a_h = ssm_a[h_idx];

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = state_in_col[r * warp_size + lane];
    }

    for (int t = 0; t < L; t++) {
        gdn_ab_token_step<S_v>(s_shard, q, k, v, alpha_in, beta_in, attn_out, dt_bias_h, a_h,
                               b_idx, t, L, H, H_kv, h_idx, h_kv, col, lane,
                               q_stride_bl, v_stride_bl, gt_stride_bl);
        float* out_col = kstates + (int64_t) t * slot_stride + col_off;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            out_col[r * warp_size + lane] = s_shard[r];
        }
    }
}
