// gdn_state_step_alphabeta_f32_batched_slots — batched GDN recurrent step
// across N independent decode slots, each with its own per-slot state
// buffer.
//
// Same compute as `gdn_state_step_alphabeta_f32_s128`. The only
// difference: the existing kernel assumes all B batches' states live
// in one contiguous buffer at `state_in + b * H * S_v * S_v`. In the
// batched-decode driver each slot's `GdnLayerState::state` is an
// independent device allocation, so this variant takes a [B] device
// array of state-base pointers (one per slot) and dereferences
// `state_in_ptrs[b_idx]` instead.
//
// Caller plumbs a small `[B] u64` pointer array (32 bytes at B=4)
// per call; rest of the I/O (q, k, v, alpha_in, beta_in, attn_out)
// remains slot-major contiguous in `[B, L, H, S_v]` layout, exactly
// as the existing kernel expects.

#include <hip/hip_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#ifndef GDN_WARPS_PER_BLOCK
#define GDN_WARPS_PER_BLOCK 4
#endif

static __device__ __forceinline__ float gdn_warp_reduce_sum_f32_bs(float x) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, WARP_SIZE);
    }
    return x;
}

template <int S_v>
static __device__ __forceinline__ void gdn_state_step_ab_batched_slots_impl(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ alpha_in,
    const float * __restrict__ beta_in,
    const float * __restrict__ ssm_dt_bias,
    const float * __restrict__ ssm_a,
    const float * const * __restrict__ state_in_ptrs,
    float * const * __restrict__ state_out_ptrs,
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

    const int H_kv  = H / n_rep;
    const int h_kv  = (rep_inner_layout != 0) ? (h_idx / n_rep) : (h_idx % H_kv);
    const int64_t q_stride_bl  = (int64_t) L * H_kv * S_v;
    const int64_t v_stride_bl  = (int64_t) L * H    * S_v;
    const int64_t gt_stride_bl = (int64_t) L * H;

    const float * state_in_base  = state_in_ptrs[b_idx];
    float *       state_out_base = state_out_ptrs[b_idx];
    const float * state_in_bh    = state_in_base  + (int64_t) h_idx * S_v * S_v;
    float *       state_out_bh   = state_out_base + (int64_t) h_idx * S_v * S_v;

    const float * state_in_col  = state_in_bh  + (int64_t) col * S_v;
    float *       state_out_col = state_out_bh + (int64_t) col * S_v;

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

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = gdn_warp_reduce_sum_f32_bs(kv_shard);

        const float v_col     = v_bt[col];
        const float delta_col = (v_col - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = gdn_warp_reduce_sum_f32_bs(attn_partial);

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

extern "C" __global__ __launch_bounds__(WARP_SIZE * GDN_WARPS_PER_BLOCK, 2)
void flambeau_gdn_state_step_alphabeta_f32_s128_batched_slots(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ alpha_in,
    const float * __restrict__ beta_in,
    const float * __restrict__ ssm_dt_bias,
    const float * __restrict__ ssm_a,
    const float * const * __restrict__ state_in_ptrs,
    float * const * __restrict__ state_out_ptrs,
    float * __restrict__ attn_out,
    int B, int H, int L, int n_rep, int rep_inner_layout)
{
    gdn_state_step_ab_batched_slots_impl<128>(
        q, k, v, alpha_in, beta_in, ssm_dt_bias, ssm_a,
        state_in_ptrs, state_out_ptrs, attn_out, B, H, L, n_rep, rep_inner_layout);
}
