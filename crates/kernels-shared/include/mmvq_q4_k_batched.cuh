#pragma once
// Templated Q4_K batched MMVQ (decode GEMV, N>1 slots). Scalar-FP32 super-block
// path (the single-row `mmvq_q4_k` structure, not the r2 dp4a kernel): 64
// threads / one output row per block, `byte_off` (0..31) the element within a
// 32-wide sub-block, `hi_half` (0/1) selecting even/odd sub-blocks. The weight
// super-block is dequantized once per (grp, hi_half) — `x_val = d·sc·q − dmin·m`
// — then multiplied against the matching activation byte of each of the N slots
// (slot-major `y` / `dst`). Backend-neutral: `flambeau_q4k_scale_min` + the
// N-way reduce come from the shared headers; every load is a single byte or
// fp16 (no 4-byte misalignment), so no `load_int_b2` is needed. `OutT` is the
// destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, int N, class OutT = float>
static __device__ __forceinline__ void mmvq_q4_k_batched_row(
    const flambeau_block_q4_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_superblocks_per_row)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane     = threadIdx.x;
    const int byte_off = lane & 31;
    const int hi_half  = lane >> 5;

    const flambeau_block_q4_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc[N];
    #pragma unroll
    for (int s = 0; s < N; ++s) acc[s] = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q4_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        #pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int sub = 2 * grp + hi_half;

            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(sub, bk->scales, &sc, &m);

            const int byte_v = (int) bk->qs[grp * 32 + byte_off];
            const int raw_q  = hi_half ? (byte_v >> 4) : (byte_v & 0x0F);
            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            #pragma unroll
            for (int s = 0; s < N; ++s) {
                const flambeau_block_q8_1* ya =
                    y + (size_t) s * n_superblocks_per_row * 8 + b * 8 + sub;
                acc[s] += x_val * (float) ya->d * (float) (int) ya->qs[byte_off];
            }
        }
    }

    mmvq_reduce_store_batched<THREADS, N, OutT>(acc, dst, row, n_rows);
}
