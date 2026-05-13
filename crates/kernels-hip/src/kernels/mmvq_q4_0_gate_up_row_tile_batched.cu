// mmvq_q4_0_gate_up_row_tile_batched — Q4_0 fused gate+up MMVQ with
// row-tiled activation reuse. R=4 output rows per block share one
// LDS-resident Q8_1 activation strip across all N decode slots.
//
// Why: K5 (`mmvq_q4_0_gate_up_batched`) processes 1 row per block, so the
// same Q8_1 activation is HBM-fetched n_rows times across blocks.
// gfx906 L2 (4 MB) can't hold a many-block activation strip when n_rows
// is large (~14k for GDN gate/up at intermediate=14336), and the
// activation traffic dominated kernel time at n_rows ≫ N.
//
// Row-tile design:
//   - Block: 256 threads = 4 wave64. Warp w ∈ {0..3} owns row = row_base + w.
//   - Outer iter strides 16 Q4_0 super-blocks (= 16 × 32 = 512 elements of K).
//   - LDS holds 16 × N Q8_1 blocks (qs[32], d, s) per outer iter, loaded by
//     the first N×16 ≤ 64 threads — 1 block per thread (8 i32 qs + d + s).
//   - Each warp's 64 lanes then run dp4a across the 16 staged blocks against
//     its row's gate/up weight bytes, 1 i32 of weight per lane, 4 lanes per
//     block — same per-lane work as K5 but the activation comes from LDS,
//     not HBM, and is reused across 4 rows.
//
// Compile-time N ∈ {2, 3, 4} kept for ABI parity with K5.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define RTGU_WARPS 4
#define RTGU_THREADS (RTGU_WARPS * WARP_SIZE)
#define RTGU_OUTER_BLOCKS 16

static __device__ __forceinline__ int flambeau_dp4a_rtgu(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template <int N>
__device__ __forceinline__ void flambeau_mmvq_q4_0_gate_up_row_tile_batched_body(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    const int tid     = threadIdx.x;
    const int warp    = tid / WARP_SIZE;
    const int lane    = tid & (WARP_SIZE - 1);
    const int lane4   = lane & 3;
    const int bi_in   = lane >> 2;

    const int row_base = blockIdx.x * RTGU_WARPS;
    const int row      = row_base + warp;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;

    const flambeau_block_q4_0* g_row = do_gate ? (gate_w + (size_t) row * n_blocks_per_row) : gate_w;
    const flambeau_block_q4_0* u_row = do_up   ? (up_w   + (size_t) row * n_blocks_per_row) : up_w;

    __shared__ int   s_y_qs[N][RTGU_OUTER_BLOCKS][8];
    __shared__ float s_y_d [N][RTGU_OUTER_BLOCKS];
    __shared__ float s_y_s [N][RTGU_OUTER_BLOCKS];

    float acc_g[N], acc_u[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) {
        acc_g[c] = 0.0f;
        acc_u[c] = 0.0f;
    }

    for (int b_outer = 0; b_outer < n_blocks_per_row; b_outer += RTGU_OUTER_BLOCKS) {
        // Cooperative LDS load. tid in 0..(N*16-1) loads slot c=tid/16,
        // block bi=tid%16. Others idle this phase.
        if (tid < N * RTGU_OUTER_BLOCKS) {
            const int c  = tid / RTGU_OUTER_BLOCKS;
            const int bi = tid % RTGU_OUTER_BLOCKS;
            const int block_idx = b_outer + bi;
            if (block_idx < n_blocks_per_row) {
                const flambeau_block_q8_1* py =
                    y + (size_t) c * n_blocks_per_row + block_idx;
                const int* py_i = (const int*) py->qs;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_y_qs[c][bi][j] = py_i[j];
                }
                s_y_d[c][bi] = (float) py->d;
                s_y_s[c][bi] = (float) py->s;
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_y_qs[c][bi][j] = 0;
                }
                s_y_d[c][bi] = 0.0f;
                s_y_s[c][bi] = 0.0f;
            }
        }
        __syncthreads();

        if (bi_in < RTGU_OUTER_BLOCKS) {
            const int block_idx = b_outer + bi_in;
            if (block_idx < n_blocks_per_row) {
                int g_vi_lo = 0, g_vi_hi = 0;
                float g_dx = 0.0f;
                if (do_gate) {
                    const flambeau_block_q4_0* gbk = g_row + block_idx;
                    const int g_v = ((const int*) gbk->qs)[lane4];
                    g_vi_lo = (g_v >> 0) & 0x0F0F0F0F;
                    g_vi_hi = (g_v >> 4) & 0x0F0F0F0F;
                    g_dx    = (float) gbk->d;
                }
                int u_vi_lo = 0, u_vi_hi = 0;
                float u_dx = 0.0f;
                if (do_up) {
                    const flambeau_block_q4_0* ubk = u_row + block_idx;
                    const int u_v = ((const int*) ubk->qs)[lane4];
                    u_vi_lo = (u_v >> 0) & 0x0F0F0F0F;
                    u_vi_hi = (u_v >> 4) & 0x0F0F0F0F;
                    u_dx    = (float) ubk->d;
                }

                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    const int yu_lo = s_y_qs[c][bi_in][lane4];
                    const int yu_hi = s_y_qs[c][bi_in][lane4 + 4];
                    const float d_y = s_y_d[c][bi_in];
                    const float s_y = s_y_s[c][bi_in];

                    if (do_gate) {
                        int sumi = 0;
                        sumi = flambeau_dp4a_rtgu(g_vi_lo, yu_lo, sumi);
                        sumi = flambeau_dp4a_rtgu(g_vi_hi, yu_hi, sumi);
                        acc_g[c] += sumi * (g_dx * d_y) - 8.0f * g_dx * s_y * 0.25f;
                    }
                    if (do_up) {
                        int sumi = 0;
                        sumi = flambeau_dp4a_rtgu(u_vi_lo, yu_lo, sumi);
                        sumi = flambeau_dp4a_rtgu(u_vi_hi, yu_hi, sumi);
                        acc_u[c] += sumi * (u_dx * d_y) - 8.0f * u_dx * s_y * 0.25f;
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < N; ++c) {
        float vg = do_gate ? gfx906_warp_reduce_sum(acc_g[c]) : 0.0f;
        float vu = do_up   ? gfx906_warp_reduce_sum(acc_u[c]) : 0.0f;
        if (lane == 0) {
            if (do_gate) gate_out[(size_t) c * n_rows_gate + row] = vg;
            if (do_up)   up_out  [(size_t) c * n_rows_up   + row] = vu;
        }
    }
}

extern "C" __global__ __launch_bounds__(RTGU_THREADS, 1)
void flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n2(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_gate_up_row_tile_batched_body<2>(
        gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row
    );
}

extern "C" __global__ __launch_bounds__(RTGU_THREADS, 1)
void flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n3(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_gate_up_row_tile_batched_body<3>(
        gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row
    );
}

extern "C" __global__ __launch_bounds__(RTGU_THREADS, 1)
void flambeau_mmvq_q4_0_gate_up_row_tile_dp4a_q8_1_batched_n4(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    flambeau_mmvq_q4_0_gate_up_row_tile_batched_body<4>(
        gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row
    );
}
