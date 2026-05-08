// mmq_q5_K_wave64 — wave64 MMQ for Q5_K × Q8_1 activation (standard
// BlockQ8_1 layout, NOT DS4 MMQ layout).
// Port of candle's `mul_mat_q5_K_gfx906_v2`
// (/artefact/candle/candle-hip-kernels/src/quantized.cu:7760-7912).
// Without an MMQ kernel for Q5_K the ssm_out + related GDN projections
// fall through to per-row MMVQ at prefill — ~12k launches per pp512
// (~1145 ms of GPU-busy wasted on launch overhead).
// Unlike the turbo MMQ variants, this kernel reads the **standard**
// `flambeau_block_q8_1` (36-byte-per-row blocks, not the 144-byte DS4
// MMQ layout). No activation quantise change needed at call sites — the
// same `scratch.x_q8_1` buffer that MMVQ consumes works here.
// Tile shape:
// MMQ_Y = 64 (one wave64 per output row; each thread = 1 row)
// TILE_N = 8 (8 output cols per tile, loop-unrolled per thread)
// Grid = (ceil(nrows_x / 64), ceil(ncols_y / 8))
// Block = 64 threads (one warp)
// Q5_K dequant math (same as MMVQ kernel):
// y_j = super_d * sub_sc * (ql_nibble | (qh_bit << 4))
// - super_dmin * sub_m
// sumi_d over DP4A of (v, y_quant), sumi_y for min correction.
// Args (8 scalar + 3 ptr — different shape than the `qmatmul_q4_1_mmq_*`
// DS4 kernel because the activation layout is different):
// vx, vy, dst,
// ncols_x = K (elements),
// nrows_x = N (weight rows),
// ncols_y = M (batch rows),
// nrows_y = K (elements per column of Y — nrows_y / QK8_1 = nb_per_row),
// nrows_dst = N
// Correctness oracle: CPU dequant(weights) × Q8_1-roundtrip(act), via the
// sweep harness (crates/bench/src/sweep_mmq.rs, new `Q5K_Wave64` variant).

#include "block_quant.cuh"
#include "mmq_prefetch.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK_K
#define QK_K 256
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

// 4-way packed signed int8 dot using the gfx906 intrinsic.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Unpack the packed 6-bit (scale, min) pair for Q5_K sub-block `j` from
// the 12-byte `scales` array. Bit-identical mirror of `get_scale_min_k4`
// in candle and `flambeau_q4k_scale_min` in flambeau's block_quant.cuh.
static __device__ __forceinline__ void q5k_scale_min(
    int j, const uint8_t* __restrict__ q, uint8_t& sc, uint8_t& m
) {
    if (j < 4) {
        sc = q[j]     & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q5_K_wave64_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float*      __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst
) {
    const int tile_m = blockIdx.x * WARP_SIZE;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row       = tile_m + tid;
    const bool row_ok   = (row < nrows_x);

    const flambeau_block_q5_K* x = (const flambeau_block_q5_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;    // Q5_K super-blocks per row
    const int blocks_per_col_y = nrows_y / QK8_1;   // Q8_1 blocks per col
    constexpr int q8_per_super = QK_K / QK8_1;      // 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        // L2 prefetch of next super-block's X and Y data.
        // Wave64 single-warp block → use the 1-D variant of the prefetch
        // helper (no threadIdx.y gate). Prefetches ~1 KB per axis.
        const int ib_next = ib + 1;
        int x_pf = 0, y_pf = 0;
        if (ib_next < blocks_per_row_x) {
            // X: this row's next Q5_K super-block (176 B).
            x_pf = gfx906_prefetch_next_1d(
                (const int*) &x[(size_t) row * blocks_per_row_x + ib_next]);
            // Y: tile col 0's next super-block worth of Q8_1 blocks
            // (8 blocks × 36 B = 288 B of the 8 cols in this tile).
            y_pf = gfx906_prefetch_next_1d(
                (const int*) &y[(size_t) tile_n * blocks_per_col_y + ib_next * q8_per_super]);
        }

        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sub_sc[8] = {0};
        uint8_t sub_m [8] = {0};
        int qh_word[8] = {0};

        const flambeau_block_q5_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d    = (float) bx->d;
            super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q5k_scale_min(j, bx->scales, sub_sc[j], sub_m[j]);
            }
            const int* qh_ptr = (const int*) bx->qh;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                qh_word[j] = qh_ptr[j];
            }
        }

        float sumf_d[TILE_N];
        float sumf_m[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) { sumf_d[c] = 0.0f; sumf_m[c] = 0.0f; }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int v[8] = {0};
            if (row_ok) {
                const int* ql_ptr = (const int*) (bx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_ptr[j];
                    const int vl = (half == 0)
                        ? (ql_word & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);
                    const int vh = ((qh_word[j] >> sub) << 4) & 0x10101010;
                    v[j] = vl | vh;
                }
            }

            const float sc_f = (float) sub_sc[sub];
            const float m_f  = (float) sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;

                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                const int* y_packed = (const int*) by->qs;

                int sumi_d = 0;
                int sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
                    sumi_d = dp4a(v[j], y_j, sumi_d);
                    sumi_y = dp4a(0x01010101, y_j, sumi_y);
                }
                sumf_d[c] += d8 * ((float) sumi_d) * sc_f;
                sumf_m[c] += d8 * ((float) sumi_y) * m_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c] - super_dmin * sumf_m[c];
        }

        // Consume prefetch dummies so the compiler can't DCE the
        // global_load_dword instructions at the top of this iter.
        gfx906_prefetch_consume(x_pf);
        gfx906_prefetch_consume(y_pf);
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) {
            dst[(size_t) col * nrows_dst + row] = sums[c];
        }
    }
}
