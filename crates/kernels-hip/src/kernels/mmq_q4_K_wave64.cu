// mmq_q4_K_wave64 — wave64 MMQ for Q4_K × Q8_1 activation.
// port of candle's `mul_mat_q4_K_gfx906_impl_fast`
// (/artefact/candle/candle-hip-kernels/src/quantized.cu:8203-8324).
// Mirrors `mmq_q5_K_wave64` almost verbatim — Q4_K differs only
// in the weight dequant math: flat 4-bit nibbles (no high-bit merge from
// `qh` that Q5_K needs).
// Why: `mmq_q4_K_4warp` is an F32-tile placeholder — it
// dequantises to F32, copies a 16×32 F32 tile to LDS, and MMs via scalar
// FMA. At prefill, Q4_K-weighted mats in Qwen3.6-35B MoE (ffn_gate / up
// exps) fall through to per-row MMVQ once m≥128 because MMQ is slower
// per-call than MMVQ-looped. DP4A port lifts per-call time into the
// turbo band; m≥128 dispatch can switch to MMQ.
// Activation layout: standard `flambeau_block_q8_1` (36 B), NOT the DS4
// 144 B MMQ layout. Reuses `scratch.x_q8_1` at the call site — same
// buffer as MMVQ and Q5_K MMQ.
// Tile: MMQ_Y = 64 (one wave64 per output row), TILE_N = 8 (8 output
// cols per thread, loop-unrolled). Grid = (⌈nrows_x / 64⌉, ⌈ncols_y / 8⌉),
// Block = 64 threads (one warp).
// Args (8 scalar + 3 ptr — same shape as Q5_K wave64):
// vx, vy, dst,
// ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst
// Correctness oracle: CPU dequant(weights) × Q8_1-roundtrip(act), via
// `crates/bench/src/sweep_mmq.rs` new `Q4KWave64` variant.

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

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_q4_K_wave64_q8_1(
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

    const int row     = tile_m + tid;
    const bool row_ok = (row < nrows_x);

    const flambeau_block_q4_K* x = (const flambeau_block_q4_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // = 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f, super_dmin = 0.0f;
        uint8_t sub_sc[8] = {0};
        uint8_t sub_m [8] = {0};

        const flambeau_block_q4_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d    = (float) bx->d;
            super_dmin = (float) bx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, bx->scales, &sub_sc[j], &sub_m[j]);
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

            // Q4_K weight decode — flat 4-bit nibbles, no qh merge.
            int v[8] = {0};
            if (row_ok) {
                const int* ql_ptr = (const int*) (bx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_ptr[j];
                    v[j] = (half == 0)
                        ? (ql_word & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);
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
