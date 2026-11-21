#pragma once
// Templated Q4_0 fused gate+up row-tiled batched MMVQ — the triple fusion: two
// weight matrices (gate_w, up_w) share one LDS-staged Q8_1 activation strip
// across `WARPS = THREADS/WARP_SIZE` output rows AND N decode columns. Same
// 4-weight-int32-lanes-per-block nibble + `(q-8)` bias layout as the Q4_0
// row-tile, with gate and up accumulated separately and asymmetric
// `n_rows_gate`/`n_rows_up` (the GDN/FFN gate vs up may differ in row count).
// Warp-size agnostic (rule 5): block-coverage strides `WARP_SIZE/4` so all 16
// staged blocks are covered on wave64 (one pass) and wave32 (two). Backend-
// neutral: `dp4a` / `warp_reduce_sum` from the arch header, all int loads
// through `load_int_b2`. `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q4_0_gate_up_row_tile_batched_row(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows_gate, int n_rows_up, int n_blocks_per_row)
{
    constexpr int WARPS           = THREADS / WARP_SIZE;
    constexpr int OUTER_BLOCKS    = 16;
    constexpr int Y_INT32         = 8;
    constexpr int COVERAGE_LANES  = 4;
    constexpr int BLOCKS_PER_PASS = WARP_SIZE / COVERAGE_LANES;

    const int tid   = threadIdx.x;
    const int warp  = tid / WARP_SIZE;
    const int lane  = tid & (WARP_SIZE - 1);
    const int lane4 = lane & (COVERAGE_LANES - 1);

    const int row      = blockIdx.x * WARPS + warp;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;

    const flambeau_block_q4_0* g_row = do_gate ? (gate_w + (size_t) row * n_blocks_per_row) : gate_w;
    const flambeau_block_q4_0* u_row = do_up   ? (up_w   + (size_t) row * n_blocks_per_row) : up_w;

    __shared__ int   s_y_qs[N][OUTER_BLOCKS][Y_INT32];
    __shared__ float s_y_d [N][OUTER_BLOCKS];
    __shared__ float s_y_s [N][OUTER_BLOCKS];

    float acc_g[N], acc_u[N];
    #pragma unroll
    for (int c = 0; c < N; ++c) { acc_g[c] = 0.0f; acc_u[c] = 0.0f; }

    for (int b_outer = 0; b_outer < n_blocks_per_row; b_outer += OUTER_BLOCKS) {
        if (tid < N * OUTER_BLOCKS) {
            const int c  = tid / OUTER_BLOCKS;
            const int bi = tid % OUTER_BLOCKS;
            const int block_idx = b_outer + bi;
            if (block_idx < n_blocks_per_row) {
                const flambeau_block_q8_1* py = y + (size_t) c * n_blocks_per_row + block_idx;
                #pragma unroll
                for (int j = 0; j < Y_INT32; ++j) s_y_qs[c][bi][j] = load_int_b2(py->qs, j);
                s_y_d[c][bi] = (float) py->d;
                s_y_s[c][bi] = (float) py->s;
            } else {
                #pragma unroll
                for (int j = 0; j < Y_INT32; ++j) s_y_qs[c][bi][j] = 0;
                s_y_d[c][bi] = 0.0f;
                s_y_s[c][bi] = 0.0f;
            }
        }
        __syncthreads();

        for (int bi = lane / COVERAGE_LANES; bi < OUTER_BLOCKS; bi += BLOCKS_PER_PASS) {
            const int block_idx = b_outer + bi;
            if (block_idx < n_blocks_per_row) {
                int g_vi_lo = 0, g_vi_hi = 0; float g_dx = 0.0f;
                if (do_gate) {
                    const int g_v = load_int_b2((const int8_t*) (g_row[block_idx].qs), lane4);
                    g_vi_lo = g_v & 0x0F0F0F0F;
                    g_vi_hi = (g_v >> 4) & 0x0F0F0F0F;
                    g_dx    = (float) g_row[block_idx].d;
                }
                int u_vi_lo = 0, u_vi_hi = 0; float u_dx = 0.0f;
                if (do_up) {
                    const int u_v = load_int_b2((const int8_t*) (u_row[block_idx].qs), lane4);
                    u_vi_lo = u_v & 0x0F0F0F0F;
                    u_vi_hi = (u_v >> 4) & 0x0F0F0F0F;
                    u_dx    = (float) u_row[block_idx].d;
                }

                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    const int yu_lo = s_y_qs[c][bi][lane4];
                    const int yu_hi = s_y_qs[c][bi][lane4 + 4];
                    const float d_y = s_y_d[c][bi];
                    const float s_y = s_y_s[c][bi];
                    if (do_gate) {
                        int sumi = dp4a(g_vi_lo, yu_lo, 0);
                        sumi     = dp4a(g_vi_hi, yu_hi, sumi);
                        acc_g[c] += sumi * (g_dx * d_y) - 8.0f * g_dx * s_y * 0.25f;
                    }
                    if (do_up) {
                        int sumi = dp4a(u_vi_lo, yu_lo, 0);
                        sumi     = dp4a(u_vi_hi, yu_hi, sumi);
                        acc_u[c] += sumi * (u_dx * d_y) - 8.0f * u_dx * s_y * 0.25f;
                    }
                }
            }
        }
        __syncthreads();
    }

    mmvq_reduce_store_rowtile_gate_up_batched<N, OutT>(
        acc_g, acc_u, gate_out, up_out, row, n_rows_gate, n_rows_up, do_gate, do_up, lane);
}
