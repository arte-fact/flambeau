// mmvq_q4_0_gate_up_dp4a — fused gate+up Q4_0 dense MMVQ with DP4A.
//
// TP-perf-c1 lever — Qwen3.6-27B-Q4_0 decode is dominated by
// `flambeau_mmvq_q4_0_q8_1` (50% of decode wall). Most of those calls
// come in pairs that share the activation: `attn_qkv` + `attn_gate`
// in GDN layers, and `ffn_gate` + `ffn_up` in dense FFN layers (both
// Q4_0 in this model). This kernel merges each pair into one launch
// and reads the Q8_1 activation once.
//
// Mirrors `mmvq_q8_0_gate_up_dp4a` exactly except:
//   - reads Q4_0 weights (18 B / block: half d + 16 nibble bytes),
//   - uses the (q - 8) DP4A bias-correction identity from `mmvq_q4_0`:
//       sumi  = dp4a(vi_lo, u_lo, 0) + dp4a(vi_hi, u_hi, sumi)
//       acc  += sumi · d_x · d_y - 8 · d_x · s_y · 0.25  (per-block)
//   - asymmetric `n_rows_gate` vs `n_rows_up` supported via per-row
//     short-circuit (same pattern as the Q8_0 fusion).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU4_BLOCK_THREADS 256
#define GU4_WARPS_PER_BLOCK (GU4_BLOCK_THREADS / WARP_SIZE)
#define GU4_INT32_PER_QBLK 4
#define GU4_BLOCKS_PER_ITER (GU4_BLOCK_THREADS / GU4_INT32_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_gu4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q4_0_gate_up_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;
    if (!do_gate && !do_up) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q4_0* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_0* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int b = block_idx; b < n_blocks_per_row; b += GU4_BLOCKS_PER_ITER) {
        const flambeau_block_q4_0* gbk = g_row + b;
        const flambeau_block_q4_0* ubk = u_row + b;
        const flambeau_block_q8_1* by  = y + b;

        // Shared activation read — once per block per thread (the win).
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // **Cycle-1 review null**: tried explicit gate/up interleaving
        // (load both v's, all 4 DP4A back-to-back) — measured -1% on
        // Qwen3.6-27B-Q4_0 TP w=2. The compiler already orders the two
        // sequential branches optimally; interleaving adds VGPR pressure
        // (more live values per thread) which costs more than ILP gains.
        // Sequential branches are the local optimum.
        if (do_gate) {
            const int v = ((const int*) gbk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_gu4(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu4(vi_hi, u_hi, sumi);
            const float d_x = (float) gbk->d;
            // Per-block: sumi · d_x · d_y - 8 · d_x · s_y. The bias-correction
            // term is constant per block; split across the 4 lanes (lane4 ∈ [0,4))
            // by · 0.25 so the warp-reduce sums to exactly one correction per block.
            acc_g += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        if (do_up) {
            const int v = ((const int*) ubk->qs)[lane4];
            const int vi_lo = (v >> 0) & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = 0;
            sumi = flambeau_dp4a_gu4(vi_lo, u_lo, sumi);
            sumi = flambeau_dp4a_gu4(vi_hi, u_hi, sumi);
            const float d_x = (float) ubk->d;
            acc_u += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_g[GU4_WARPS_PER_BLOCK];
    __shared__ float s_u[GU4_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < GU4_WARPS_PER_BLOCK) ? s_g[lane] : 0.0f;
        float u = (lane < GU4_WARPS_PER_BLOCK) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = GU4_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
