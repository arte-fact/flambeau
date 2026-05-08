// mmvq_f16_q8_1 — F16 weight × Q8_1 activation MMVQ for F16 weights in
// UD-Q8_K_XL GGUFs (Unsloth dynamic quant reserves F16 for layers flagged
// precision-sensitive by the i-matrix).
// Same threading pattern as `mmvq_q8_0_dp4a_vdr2` (256 threads/block, 1 row
// per block, lane_in_grp=tid&3 → int32-pair slot, block_idx=tid>>2). Per
// inner step each thread processes 8 elements — 8 F16 weights × 8 dequantised
// Q8_1 quants, all accumulated into one F32 per thread, then warp-reduce.
// The weight side is F16 so DP4A doesn't apply — we use plain F32 FMAs.
// On gfx906 a 40-CU MI50 at 1 TB/s HBM, reading the 30 MiB (6144×5120 F16)
// weight matrix is 30 µs roofline per call; 1 expects ~100 F16 MMVQs per
// 27B-UD-Q8_K_XL decode step (48 attn_gate + 48 ssm_out + a few scatter),
// so ~3 ms/decode-step lower bound.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define F16_VDR2_BLOCK_THREADS 256
#define F16_VDR2_WARPS_PER_BLOCK (F16_VDR2_BLOCK_THREADS / WARP_SIZE)
#define F16_VDR2_THREADS_PER_QBLK 4
#define F16_VDR2_BLOCKS_PER_ITER (F16_VDR2_BLOCK_THREADS / F16_VDR2_THREADS_PER_QBLK)

extern "C" __global__ void flambeau_mmvq_f16_q8_1(
    const fb_fp16_t* __restrict__ x_f16,              // [n_rows, n_cols] row-major
    const flambeau_block_q8_1* __restrict__ y,        // [n_blocks_per_row] Q8_1 activation
    float* __restrict__ dst,                          // [n_rows] F32 output
    const int n_rows,
    const int n_blocks_per_row                        // = n_cols / 32
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid         = threadIdx.x;
    const int warp        = tid / WARP_SIZE;
    const int lane        = tid & (WARP_SIZE - 1);
    const int lane_in_grp = tid & 3;                  // 0..3: int32-pair slot within the Q8_1 block (8 int32 per 32-int8 block → 4 pairs)
    const int block_idx   = tid >> 2;                 // 0..63

    const size_t n_cols = (size_t) n_blocks_per_row * 32;
    const fb_fp16_t* xrow = x_f16 + (size_t) row * n_cols;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += F16_VDR2_BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* by = y + b;

        // Offset inside this 32-int8 block, 8 elements per thread (2 int32s of
        // quants, 8 F16 weights). `lane_in_grp * 8` gives the element offset
        // within the block.
        const int elem_off = lane_in_grp * 8;

        // Read 8 consecutive int8 quants as 2 int32s.
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];
        const float d_y = (float) by->d;

        // Extract 8 int8s from the 2 int32s.
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

    // Warp reduce + cross-warp reduce via LDS, same pattern as the
    // mmvq_q8_0_dp4a_vdr2 kernel.
    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_partials[F16_VDR2_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_partials[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < F16_VDR2_WARPS_PER_BLOCK) ? s_partials[lane] : 0.0f;
        #pragma unroll
        for (int off = F16_VDR2_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
