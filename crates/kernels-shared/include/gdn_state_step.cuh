#pragma once

// Gated-Delta-Net recurrent step (S_v = 128). Each warp owns one output column
// of the per-head state; a lane holds S_v/WARP_SIZE state rows in registers and
// reduces its column over the warp. Backend-neutral: WARP_SIZE comes from the
// arch seam (64 on gfx906 → 2 rows/lane; 32 on sm_8x → 4 rows/lane), so the
// register-sharded recurrence scales without a wave64 assumption. The caller
// launches block = (WARP_SIZE, GDN_WARPS_PER_BLOCK).
#ifndef GDN_WARPS_PER_BLOCK
#define GDN_WARPS_PER_BLOCK 4
#endif

static __device__ __forceinline__ float gdn_warp_reduce_sum_f32(float x) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, WARP_SIZE);
    }
    return x;
}

template <int S_v>
static __device__ __forceinline__ void gdn_state_step_impl(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
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
    const float* state_in_bh = state_in + (int64_t) bh * S_v * S_v;
    float* state_out_bh = state_out + (int64_t) bh * S_v * S_v;
    const float* state_in_col = state_in_bh + (int64_t) col * S_v;
    float* state_out_col = state_out_bh + (int64_t) col * S_v;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = state_in_col[r * warp_size + lane];
    }

    for (int t = 0; t < L; t++) {
        const float* gate_bt = gate + b_idx * gt_stride_bl + (int64_t) t * H;
        const float* beta_bt = beta + b_idx * gt_stride_bl + (int64_t) t * H;
        const float* q_bt = q + b_idx * q_stride_bl + ((int64_t) t * H_kv + h_kv) * S_v;
        const float* k_bt = k + b_idx * q_stride_bl + ((int64_t) t * H_kv + h_kv) * S_v;
        const float* v_bt = v + b_idx * v_stride_bl + ((int64_t) t * H + h_idx) * S_v;

        const float g_val = __expf(gate_bt[h_idx]);
        const float beta_val = beta_bt[h_idx];

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
        const float kv_col = gdn_warp_reduce_sum_f32(kv_shard);

        const float v_col = v_bt[col];
        const float delta_col = (v_col - kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] += k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = gdn_warp_reduce_sum_f32(attn_partial);

        if (lane == 0) {
            attn_out[(((int64_t) b_idx * L + t) * H + h_idx) * S_v + col] = attn_col;
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        state_out_col[r * warp_size + lane] = s_shard[r];
    }
}
