// mmq_iq4_xs_wave64 — wave64 MMQ for IQ4_XS × Q8_1 activation.
// Mirrors `mmq_q4_K_wave64.cu` structurally:
//   - 64 threads = one wave64 = 64 output rows per block (MMQ_Y = 64).
//   - Each thread owns one row, computes TILE_N=8 output columns.
//   - Inner loop is dp4a across 8 i32 weight words × 8 i32 activation words
//     per sub-block (= 32 quants paired against 32 Q8_1 i8s).
//
// IQ4_XS differs from Q4_K in two places:
//   1) Weight decode goes through the 16-entry signed-i8 LUT
//      (`KVALUES_IQ4NL`) instead of using the raw nibble value. Each 4-bit
//      code becomes a signed i8 ∈ [-127, 113]; we pack 4 of them per i32
//      so dp4a sees signed-i8 × signed-i8 directly.
//   2) No `dmin` / per-sub-block `m` correction — IQ4_XS reconstruction is
//      symmetric (`y = d * ls * lut[code]`), so the kernel only accumulates
//      the `sumi_d` term (no `sumi_y` partial).
//
// Sub-block scale: signed 6-bit packed across `scales_l[4]` (low nibble)
// and `scales_h` (top 2 bits × 8), biased -32. See
// `flambeau_iq4_xs_scale()` in `block_quant.cuh`.
//
// Activation: standard `flambeau_block_q8_1` (36 B / 32 elems), reused
// from the MMVQ path. No DS4 MMQ-tile activation needed.

#include "block_quant.cuh"
#include "iq_grid.cuh"
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

// Apply the IQ4_NL signed-i8 LUT to each of 4 nibbles packed in `nibbles`
// (one nibble per byte, high nibble of each byte is 0). Returns 4 signed
// i8 values packed into a single i32, suitable for `dp4a`.
static __device__ __forceinline__ int pack_iq4_lut(int nibbles) {
    const int b0 = (int) flambeau_iq4nl_lut(nibbles & 0xFF);
    const int b1 = (int) flambeau_iq4nl_lut((nibbles >>  8) & 0xFF);
    const int b2 = (int) flambeau_iq4nl_lut((nibbles >> 16) & 0xFF);
    const int b3 = (int) flambeau_iq4nl_lut((nibbles >> 24) & 0xFF);
    return (b0 & 0xFF)
         | ((b1 & 0xFF) <<  8)
         | ((b2 & 0xFF) << 16)
         | ((b3 & 0xFF) << 24);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_mmq_iq4_xs_wave64_q8_1(
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

    const flambeau_block_iq4_xs* x = (const flambeau_block_iq4_xs*) vx;
    const flambeau_block_q8_1*   y = (const flambeau_block_q8_1*)   vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // 8 sub-blocks

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float    super_d  = 0.0f;
        uint16_t scales_h = 0;
        const flambeau_block_iq4_xs* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d  = (float) bx->d;
            scales_h = bx->scales_h;
        }

        float sumf_d[TILE_N];
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) sumf_d[c] = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            // Decode IQ4_XS weights for this sub-block into 8 i32 of
            // packed signed-i8 LUT values.
            //   qs layout: 16 bytes per sub-block at `bx->qs + sub*16`.
            //   Low nibbles → elements 0..15 of the sub-block.
            //   High nibbles → elements 16..31 of the sub-block.
            // Pair with the 8 i32 of Q8_1 activation (= 32 i8) for the
            // matching Q8_1 block.
            int v[8] = {0};
            int ls   = 0;
            if (row_ok) {
                const int* ql_ptr = (const int*) (bx->qs + 16 * sub);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int word = ql_ptr[j];
                    v[j]     = pack_iq4_lut(word        & 0x0F0F0F0F);
                    v[j + 4] = pack_iq4_lut((word >> 4) & 0x0F0F0F0F);
                }
                ls = flambeau_iq4_xs_scale(sub, scales_h, bx->scales_l);
            }

            const float ls_f = (float) ls;

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;

                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                const int* y_packed = (const int*) by->qs;

                int sumi_d = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sumi_d = dp4a(v[j], y_packed[j], sumi_d);
                }
                sumf_d[c] += d8 * ((float) sumi_d) * ls_f;
            }
        }

        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            sums[c] += super_d * sumf_d[c];
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
