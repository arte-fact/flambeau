// gdn_state_step_f32 — fused Gated-Delta-Net recurrent step (S_v = 128).
// Replaces the ~8 tensor-op launches of candle's `delta_net_single_step`
// with one kernel that consumes q, k, v, gate, beta, and an in-place state
// and emits `L` output rows plus the updated state. One launch handles both
// decode (L = 1) and prefill (L > 1) — state stays register-resident across
// all L tokens in the recurrence loop.
// Ported from candle-hip-kernels/src/gated_delta_net.cu, itself ported from
// llamacpp-turbo's `ggml-cuda/gated_delta_net.cu::gated_delta_net_cuda`.
// Kept minimal: Wave64 (WARP_SIZE=64), KDA=false (scalar gate per head),
// single template instantiation at S_v = 128 to cover qwen35moe / qwen36moe
// (head_k_dim = head_v_dim = 128). Extending to S_v ∈ {16, 32, 64} later is
// mechanical — copy the extern "C" wrapper with a new name.
// Layout (row-major, contiguous) — all L-outer so the kernel integrates
// cleanly with upstream ops that produce per-token rows (`qmatmul(M=L)`,
// `gather_qkv_strided`, etc.), and so attn_out reshapes directly to
// `[L, d_inner]` for the downstream `ssm_norm + silu(z)*x + ssm_out` chain:
// q, k: (B, L, H_kv, S_v) — H_kv = H / n_rep (GQA-shared heads)
// v: (B, L, H_v, S_v)
// gate,beta: (B, L, H_v) — one scalar per (b, t, h)
// state_in: (B, H_v, S_v, S_v) — stored as [col][row]: the col-outer
// transpose lets a warp reading column
// `col` hit S_v contiguous floats (8
// cache lines at S_v=128) instead of
// S_v cache lines strided by 512 B.
// state_out: (B, H_v, S_v, S_v) — same col-outer layout. state_in and
// state_out may alias (the kernel loads
// state_in into registers once at the
// start and writes state_out at the end).
// attn_out: (B, L, H_v, S_v) — **L outer, H_v middle, S_v inner**.
// Matches llama.cpp's output shape
// `[S_v, H_v, n_tokens, n_seqs]` (which
// then reshapes to `[d_inner, L]` — one
// contiguous `d_inner` row per token),
// and lets our downstream `rmsnorm_f32`,
// `swiglu_f32(z, ...)` and `ssm_out`
// mmvq read `[L, d_inner]` directly.
// At L=1 this coincides with the
// head-outer layout, which is why // decode landed correctly while // prefill at L>1 was silently wrong
// until (this commit).
// Per token t, each warp owns one output column `col`:
// state[col, i] *= exp(gate[t])
// sk[col] = Σ_i state[col, i] * k[t, i]
// delta[col] = (v[t, col] - sk[col]) * beta[t]
// state[col, i] += k[t, i] * delta[col]
// attn[t, col] = Σ_i state[col, i] * q[t, i]
// Grid: (H_v, B, ceil(S_v / WARPS_PER_BLOCK))
// Block: (WARP_SIZE = 64, WARPS_PER_BLOCK = 4, 1)
// → 256 threads = 4 warps per block, 4 columns of the same (b, h) issuing
// together. __launch_bounds__(256, 1) keeps the VGPR budget generous
// (we want S_v/WARP_SIZE = 2 state rows + 2 k-rows + 2 q-rows in regs).

#include <hip/hip_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

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
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ gate,
    const float * __restrict__ beta,
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

    // V / state / attn_out / gate / beta are indexed by (b, h_idx).
    // Q / K are indexed by (b, h_kv), the GQA-shared k-head for this
    // v-head. Two repeat conventions exist in the wild:
    // rep-OUTER (rep_inner_layout = 0): candle / llama.cpp qwen35moe
    // uses `ggml_repeat_4d(Q, num_v_heads)` which CYCLES through
    // k-heads — v-head h_idx → k-head `h_idx % H_kv`. Pattern at
    // n_rep=2: v-heads {0..H_kv-1} cover k-heads {0..H_kv-1}, then
    // v-heads {H_kv..2*H_kv-1} cover them again.
    // rep-INNER (rep_inner_layout = 1): llama.cpp qwen3next
    // does an `ggml_reshape_4d → repeat_4d → reshape` that
    // INTERLEAVES — each k-head is duplicated `n_rep` times in a
    // row, giving v-head h_idx → k-head `h_idx / n_rep`. Pattern
    // at n_rep=2: v-heads {0,1} share k-head 0, {2,3} share
    // k-head 1, etc. (See `qwen3next.cpp:418-431` for the explicit
    // reshape-interleave.) This is the natural `kv = v / n_rep`
    // mapping used by most modern GQA models.
    // The two layouts are NOT compatible; using the wrong one produces
    // a recurrent state that evolves with the wrong q/k for half the
    // v-heads, manifesting at the model output as a degenerate-token
    // attractor (e.g. always argmax `**` or `\n`). qwen35moe
    // (Qwen3.6-35B-A3B) ships rep-OUTER; qwen3next (Coder-Next-80B,
    // future qwen3 hybrid releases) ships rep-INNER. Caller selects.
    const int H_kv   = H / n_rep;
    const int h_kv   = (rep_inner_layout != 0) ? (h_idx / n_rep) : (h_idx % H_kv);
    const int bh     = b_idx * H    + h_idx;
    const int bh_kv  = b_idx * H_kv + h_kv;
    // L-outer layout: all inputs are `[B, L, H, S_v]` (or `[B, L, H]` for
    // gate/beta). Index: `q[((b*L + t) * H_kv + h_kv) * S_v + i]`. State
    // stays `[B, H, S_v, S_v]` (per-head, time-independent).
    const int64_t q_stride_bl  = (int64_t) L * H_kv * S_v;  // per-batch block for q/k
    const int64_t v_stride_bl  = (int64_t) L * H    * S_v;
    const int64_t gt_stride_bl = (int64_t) L * H;
    const float * state_in_bh  = state_in  + (int64_t)bh * S_v * S_v;
    float *       state_out_bh = state_out + (int64_t)bh * S_v * S_v;

    // Col-outer transposed layout: a warp reading column `col` hits S_v
    // contiguous floats. Since the state is zero-init in Rust and only ever
    // touched by this kernel, flipping both reads and writes symmetrically
    // preserves semantics without any Rust-side change.
    const float * state_in_col  = state_in_bh  + (int64_t)col * S_v;
    float *       state_out_col = state_out_bh + (int64_t)col * S_v;

    // Load column `col` of state_in into registers. Warp reads S_v
    // contiguous floats → coalesced.
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = state_in_col[i];
    }

    for (int t = 0; t < L; t++) {
        // Row bases for this (b, t): gate[b, t, h], q[b, t, h_kv, :],
        // k[b, t, h_kv, :], v[b, t, h, :].
        const float * gate_bt = gate + b_idx * gt_stride_bl + (int64_t) t * H;
        const float * beta_bt = beta + b_idx * gt_stride_bl + (int64_t) t * H;
        const float * q_bt    = q    + b_idx * q_stride_bl  + ((int64_t) t * H_kv + h_kv) * S_v;
        const float * k_bt    = k    + b_idx * q_stride_bl  + ((int64_t) t * H_kv + h_kv) * S_v;
        const float * v_bt    = v    + b_idx * v_stride_bl  + ((int64_t) t * H    + h_idx) * S_v;

        // Per-token scalars. `__expf` lowers to v_exp_f32 on HIP (SFU,
        // ~23-bit precision).
        const float g_val    = __expf(gate_bt[h_idx]);
        const float beta_val = beta_bt[h_idx];

        // Per-token row shards of k and q in registers. Same lane sharding
        // as s_shard.
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_bt[i];
            q_reg[r] = q_bt[i];
        }

        // Apply decay: state[col, i] *= g_val. Same op order as the CPU
        // reference — preserves numerical parity.
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] *= g_val;
        }

        // sk[col] = Σ_i state[col, i] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = gdn_warp_reduce_sum_f32(kv_shard);

        // delta[col] = (v[col] - sk[col]) * beta
        const float v_col     = v_bt[col];
        const float delta_col = (v_col - kv_col) * beta_val;

        // Fused: state[col, i] += k[i] * delta_col
        // attn[col] = Σ_i state[col, i] * q[i] (new state)
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]   += k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = gdn_warp_reduce_sum_f32(attn_partial);

        // One writer per column. `attn_out` layout is (B, L, H, S_v):
        // `[((b*L + t) * H + h) * S_v + col]`. llama.cpp's downstream
        // `ssm_norm + silu(z)*x + ssm_out_proj` expects `[L, d_inner]`
        // contiguous per token (d_inner = H * S_v), which this L-outer-H-
        // middle-S_v-innermost layout provides directly.
        if (lane == 0) {
            attn_out[(((int64_t) b_idx * L + t) * H + h_idx) * S_v + col] = attn_col;
        }
    }

    // Write state column to state_out (same col-outer layout — contiguous).
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        state_out_col[i] = s_shard[r];
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE * GDN_WARPS_PER_BLOCK, 1)
void flambeau_gdn_state_step_f32_s128(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ gate,
    const float * __restrict__ beta,
    const float * __restrict__ state_in,
    float * __restrict__ state_out,
    float * __restrict__ attn_out,
    int B, int H, int L, int n_rep, int rep_inner_layout)
{
    gdn_state_step_impl<128>(q, k, v, gate, beta, state_in, state_out, attn_out, B, H, L, n_rep, rep_inner_layout);
}
