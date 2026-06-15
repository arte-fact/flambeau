// mmq_f16_q8_1 — F16 weight × Q8_1 activation, multi-activation-row variant
// of `mmvq_f16_q8_1`. Replaces the row-by-row prefill
// fallback (m independent MMVQ launches) with ONE launch of grid =
// (n_rows, m) blocks.
// Kernel math is byte-identical to `mmvq_f16_q8_1`: same 256-thread block,
// same lane_in_grp=tid&3 int32-pair slot pattern, same Q8_1 dequant →
// F32 FMA. The only change is picking up the activation row via
// blockIdx.y and offsetting the y pointer + dst pointer accordingly.
// Why this helps: row-by-row costs ~1 µs launch overhead per row
// on top of the ~30 µs HBM-roofline per-call time. At L=512 × ~100 F16
// tensors/layer × 40 layers = 2M launches/prefill — launch overhead
// dominates. One launch amortises the overhead across all m rows;
// per-block GPU work is unchanged.
// Does not use LDS (the per-block math doesn't benefit — each block's
// weight row is unique, and 256 threads within a block broadcast-share
// Y from L1 already).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define F16_MMQ_THREADS 256
#define F16_MMQ_WARPS (F16_MMQ_THREADS / WARP_SIZE)
#define F16_MMQ_INT32_PER_BLOCK 4
#define F16_MMQ_BLOCKS_PER_ITER (F16_MMQ_THREADS / F16_MMQ_INT32_PER_BLOCK)

extern "C" __global__ void flambeau_mmq_f16_q8_1(
    const fb_fp16_t* __restrict__ x_f16,              // [n_rows, n_cols] row-major
    const flambeau_block_q8_1* __restrict__ y,        // [n_tokens, n_blocks_per_row] row-major
    float* __restrict__ dst,                          // [n_tokens, n_rows] row-major
    const int n_rows,
    const int n_tokens,
    const int n_blocks_per_row                        // = n_cols / 32
) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    if (row >= n_rows || tok >= n_tokens) return;

    const int tid         = threadIdx.x;
    const int warp        = tid / WARP_SIZE;
    const int lane        = tid & (WARP_SIZE - 1);
    const int lane_in_grp = tid & 3;
    const int block_idx   = tid >> 2;

    const size_t n_cols = (size_t) n_blocks_per_row * 32;
    const fb_fp16_t* xrow = x_f16 + (size_t) row * n_cols;
    const flambeau_block_q8_1* y_row = y + (size_t) tok * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += F16_MMQ_BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* by = y_row + b;
        const int elem_off = lane_in_grp * 8;

        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];
        const float d_y = (float) by->d;

        signed char q[8];
        q[0] = (signed char) ((yi0      ) & 0xff);
        q[1] = (signed char) ((yi0 >>  8) & 0xff);
        q[2] = (signed char) ((yi0 >> 16) & 0xff);
        q[3] = (signed char) ((yi0 >> 24) & 0xff);
        q[4] = (signed char) ((yi1      ) & 0xff);
        q[5] = (signed char) ((yi1 >>  8) & 0xff);
        q[6] = (signed char) ((yi1 >> 16) & 0xff);
        q[7] = (signed char) ((yi1 >> 24) & 0xff);

        const size_t x_off = (size_t) b * 32 + elem_off;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float xv = (float) xrow[x_off + j];
            acc += xv * (float) q[j] * d_y;
        }
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_partials[F16_MMQ_WARPS];
    if (lane == 0) {
        s_partials[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < F16_MMQ_WARPS) ? s_partials[lane] : 0.0f;
        #pragma unroll
        for (int off = F16_MMQ_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[(size_t) tok * n_rows + row] = v;
        }
    }
}
