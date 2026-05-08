// mmvq_q4_0_kv_f16dst_dp4a — fused K+V Q4_0 dense MMVQ with F16 destination.
// Cycle 3 lever (post-review). In a TP full-attn layer, attn_k and
// attn_v are both Q4_0, share the same Q8_1 activation, and have the
// same per-rank shape `[local_n_kv_heads · head_dim, hidden]`. The
// existing path runs THREE serialised launches per K and per V:
// 1. mmvq_q4_0_q8_1 → scratch.mmvq_f32 (shared!)
// 2. cast_f32_to_f16 → scratch.k_f16
// 3. mmvq_q4_0_q8_1 → scratch.mmvq_f32 (overwrites!)
// 4. cast_f32_to_f16 → scratch.v_f16
// This kernel collapses both into ONE launch:
// - Single activation read per block per thread (the gate+up pattern)
// - Two F32 accumulators per output row
// - Final write straight to F16 destinations — no intermediate F32 buffer
// Saves 1 MMVQ launch + 2 cast launches per full-attn layer per rank.
// Symmetric n_rows (K and V always share shape) → no asymmetric branches.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define KV4_BLOCK_THREADS 256
#define KV4_WARPS_PER_BLOCK (KV4_BLOCK_THREADS / WARP_SIZE)
#define KV4_INT32_PER_QBLK 4
#define KV4_BLOCKS_PER_ITER (KV4_BLOCK_THREADS / KV4_INT32_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_kv4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(KV4_BLOCK_THREADS)
void flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ k_w,
    const flambeau_block_q4_0* __restrict__ v_w,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ k_out,            // F16 destination
    fb_fp16_t* __restrict__ v_out,            // F16 destination
    const int n_rows_kv,                       // same for both
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows_kv) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_0* k_row = k_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_0* v_row = v_w + (size_t) row * n_blocks_per_row;

    float acc_k = 0.0f;
    float acc_v = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += KV4_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* kbk = k_row + b;
        const flambeau_block_q4_0* vbk = v_row + b;
        const flambeau_block_q8_1* by  = y + b;

        // Shared activation read — once per block per thread (the win).
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // K — same Q4_0 (q-8) DP4A bias-correction as the gate+up kernel.
        {
            const int v = ((const int*) kbk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_kv4(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_kv4(vi_hi, u_hi, sumi);
            const float d_x = (float) kbk->d;
            acc_k += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        // V
        {
            const int v = ((const int*) vbk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_kv4(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_kv4(vi_hi, u_hi, sumi);
            const float d_x = (float) vbk->d;
            acc_v += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    acc_k = gfx906_warp_reduce_sum(acc_k);
    acc_v = gfx906_warp_reduce_sum(acc_v);

    __shared__ float s_k[KV4_WARPS_PER_BLOCK];
    __shared__ float s_v[KV4_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_k[warp] = acc_k;
        s_v[warp] = acc_v;
    }
    __syncthreads();

    if (warp == 0) {
        float k = (lane < KV4_WARPS_PER_BLOCK) ? s_k[lane] : 0.0f;
        float v = (lane < KV4_WARPS_PER_BLOCK) ? s_v[lane] : 0.0f;
        #pragma unroll
        for (int off = KV4_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            k += __shfl_xor(k, off, WARP_SIZE);
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            // F16 destination — skip the F32 round-trip + cast launch.
            k_out[row] = (fb_fp16_t) k;
            v_out[row] = (fb_fp16_t) v;
        }
    }
}
