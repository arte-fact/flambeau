// mmvq_q5_k_r2_f16dst — Q5_K MMVQ r2 with F16 destination output.
//
// Cycle 4 lever (post-review). The TP GDN path's `ssm_out` projection
// is Q5_K and lands every layer per rank: `mmvq_q5_k_r2_q8_1` writes
// F32 to scratch.ssm_out_f32, followed by a `cast_f32_to_f16` into
// scratch.partial_attn_out. The intermediate F32 buffer is consumed
// by nothing else — collapsing the cast into the kernel epilogue
// saves one launch + one HBM round-trip per GDN layer per rank.
//
// Mathematically identical to mmvq_q5_k_r2_q8_1 except the final
// `dst[row] = acc` becomes `dst[row] = (fb_fp16_t) acc`.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q5_k_r2_f16dst_q8_1(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,                  // F16 destination
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;
    const int row_hi   = lane >> 5;
    const int lane_lo  = lane & 31;

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;

    const flambeau_block_q5_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_superblocks_per_row; ++b) {
        const flambeau_block_q5_K* bk = xrow + b;
        const float d    = (float) bk->d;
        const float dmin = (float) bk->dmin;

        const flambeau_block_q8_1* y_sb = y + (size_t) b * 8;
        const uint8_t qh_byte = bk->qh[lane_lo];

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc = 0, m = 0;
            flambeau_q4k_scale_min(s, bk->scales, &sc, &m);

            const int byte_idx = (s >> 1) * 32 + lane_lo;
            const int byte_v   = (int) bk->qs[byte_idx];
            const int raw_q4   = (s & 1) ? (byte_v >> 4) : (byte_v & 0x0F);

            const int mask = 1 << s;
            const int raw_q = raw_q4 + ((qh_byte & mask) ? 16 : 0);

            const float x_val = d * (float) sc * (float) raw_q - dmin * (float) m;

            const flambeau_block_q8_1* ya = y_sb + s;
            const float d_y = (float) ya->d;
            const int   qi  = (int) ya->qs[lane_lo];
            const float y_val = d_y * (float) qi;

            acc += x_val * y_val;
        }
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = (fb_fp16_t) acc;
    }
}
