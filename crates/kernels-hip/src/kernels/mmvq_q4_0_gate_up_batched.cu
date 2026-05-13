// mmvq_q4_0_gate_up_batched — fused (gate, up) Q4_0 MMVQ × N activation cols.
//
// K5 — extends `mmvq_q4_0_gate_up_dp4a` (the fused gate+up Q4_0 kernel) with
// an inner activation-column loop bounded by compile-time N ∈ {2, 3, 4}.
// Each block reads ONE gate weight row + ONE up weight row + ONE shared
// activation strip per col, computing 2N output values (gate_out[c, row] +
// up_out[c, row] for c in 0..N).
//
// Lever: GDN's per-token bottleneck is the gate+up fusion at m=1 called
// twice for L=2 spec verify. This kernel folds the L=2 verify's gate+up
// into ONE launch, amortizing both weight reads (shared with the fused
// kernel) AND across the N activation cols. K1's pattern, applied to the
// 2-weight fused shape.
//
// Output layout (matches the qmatmul ABI's slot-major convention):
//   gate_out[N, n_rows_gate] F32
//   up_out  [N, n_rows_up]   F32
//
// Block/grid: blockDim = 256 (4 × wave64), gridDim.x = max(n_rows_gate,
// n_rows_up). Asymmetric-row support preserved via per-row do_gate /
// do_up short-circuit. Bias correction `-8·d_x·s_y * 0.25f` per Q4_0
// block per col (split across 4 lane4 lanes).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define GU4B_THREADS 256
#define GU4B_WARPS (GU4B_THREADS / WARP_SIZE)
#define GU4B_INT32_PER_QBLK 4
#define GU4B_BLOCKS_PER_ITER (GU4B_THREADS / GU4B_INT32_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_gu4b(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q4_0_gate_up_batched_body(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,         // [N, n_blocks_per_row]
    float* __restrict__ gate_out,                       // [N, n_rows_gate]
    float* __restrict__ up_out,                         // [N, n_rows_up]
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

    const flambeau_block_q4_0* g_row = do_gate ? (gate_w + (size_t) row * n_blocks_per_row) : gate_w;
    const flambeau_block_q4_0* u_row = do_up   ? (up_w   + (size_t) row * n_blocks_per_row) : up_w;

    // Per-column accumulators — 2 per col (gate + up).
    float acc_g[N];
    float acc_u[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) {
        acc_g[c] = 0.0f;
        acc_u[c] = 0.0f;
    }

    for (int b = block_idx; b < n_blocks_per_row; b += GU4B_BLOCKS_PER_ITER) {
        int g_vi_lo = 0, g_vi_hi = 0;
        float g_dx = 0.0f;
        if (do_gate) {
            const flambeau_block_q4_0* gbk = g_row + b;
            const int g_v = ((const int*) gbk->qs)[lane4];
            g_vi_lo = (g_v >> 0) & 0x0F0F0F0F;
            g_vi_hi = (g_v >> 4) & 0x0F0F0F0F;
            g_dx    = (float) gbk->d;
        }
        int u_vi_lo = 0, u_vi_hi = 0;
        float u_dx = 0.0f;
        if (do_up) {
            const flambeau_block_q4_0* ubk = u_row + b;
            const int u_v = ((const int*) ubk->qs)[lane4];
            u_vi_lo = (u_v >> 0) & 0x0F0F0F0F;
            u_vi_hi = (u_v >> 4) & 0x0F0F0F0F;
            u_dx    = (float) ubk->d;
        }

        // Inner col loop — fully unrolled at compile time.
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* by =
                y + (size_t) c * n_blocks_per_row + b;
            const int yu_lo = ((const int*) by->qs)[lane4];
            const int yu_hi = ((const int*) by->qs)[lane4 + 4];
            const float d_y = (float) by->d;
            const float s_y = (float) by->s;

            if (do_gate) {
                int sumi = 0;
                sumi = flambeau_dp4a_gu4b(g_vi_lo, yu_lo, sumi);
                sumi = flambeau_dp4a_gu4b(g_vi_hi, yu_hi, sumi);
                acc_g[c] += sumi * (g_dx * d_y) - 8.0f * g_dx * s_y * 0.25f;
            }
            if (do_up) {
                int sumi = 0;
                sumi = flambeau_dp4a_gu4b(u_vi_lo, yu_lo, sumi);
                sumi = flambeau_dp4a_gu4b(u_vi_hi, yu_hi, sumi);
                acc_u[c] += sumi * (u_dx * d_y) - 8.0f * u_dx * s_y * 0.25f;
            }
        }
    }

    // Per-col reduction. LDS sized exactly for N × 2 lanes.
    __shared__ float s_g[GU4B_WARPS * N];
    __shared__ float s_u[GU4B_WARPS * N];

    #pragma unroll
    for (int c = 0; c < N; ++c) {
        float vg = do_gate ? gfx906_warp_reduce_sum(acc_g[c]) : 0.0f;
        float vu = do_up   ? gfx906_warp_reduce_sum(acc_u[c]) : 0.0f;
        if (lane == 0) {
            s_g[c * GU4B_WARPS + warp] = vg;
            s_u[c * GU4B_WARPS + warp] = vu;
        }
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            float g = (lane < GU4B_WARPS) ? s_g[c * GU4B_WARPS + lane] : 0.0f;
            float u = (lane < GU4B_WARPS) ? s_u[c * GU4B_WARPS + lane] : 0.0f;
            #pragma unroll
            for (int off = GU4B_WARPS / 2; off > 0; off >>= 1) {
                g += __shfl_xor(g, off, WARP_SIZE);
                u += __shfl_xor(u, off, WARP_SIZE);
            }
            if (lane == 0) {
                if (do_gate) gate_out[(size_t) c * n_rows_gate + row] = g;
                if (do_up)   up_out  [(size_t) c * n_rows_up   + row] = u;
            }
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n2(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_gate_up_batched_body<2>(
        gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row
    );
}

extern "C" __global__ void flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n3(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_gate_up_batched_body<3>(
        gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row
    );
}

extern "C" __global__ void flambeau_mmvq_q4_0_gate_up_dp4a_q8_1_batched_n4(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_gate_up_batched_body<4>(
        gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row
    );
}
