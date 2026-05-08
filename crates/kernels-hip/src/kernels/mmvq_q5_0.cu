// mmvq_q5_0 — Q5_0 weight × Q8_1 activation → F32 dst.
// Q5_0 layout: 2-byte F16 scale + 4-byte qh (32 "5th bits") + 16 bytes
// nibble-packed unsigned low-4-bits. Reconstruction per element i:
// q5_i = ((qh >> i) & 1) << 4 | nibble_i;
// y_i = d * (q5_i - 16)
// DP4A identity (same as Q4_0 but with offset 16 and 5th bits):
// (q5 - 16) · y = (nibble · y) + (bit << 4 · y) - 16 · sum(y)
// = nibble_dp4a + 16 · bit_dp4a - 16 · sum_y
// where `bit_dp4a` is the DP4A of the 5th-bit pattern (packed as 0/1 bytes)
// against the y quant int32. Per 4-byte slot the 5th bits are 4 disjoint
// bits of the `qh` word; we expand them to 4 bytes `(bit?1:0)` in an int.
// For each thread handling lane4 ∈ [0, 4):
// - low half (elements 4·lane4 .. +3): nibble = (v >> 0) & 0x0F0F0F0F,
// bit = (qh >> (4·lane4)) & 0x0F, expanded to 4 bytes {bit0,bit1,bit2,bit3}
// - high half (elements 4·lane4 + 16 .. +19): nibble = (v >> 4) & 0x0F0F0F0F,
// bit = (qh >> (4·lane4 + 16)) & 0x0F, expanded likewise
// Bit expansion: given `b = 4 bits packed into bits [0..4) of a byte`, produce
// `b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)` — each byte is 0 or 1.
// Block/grid: blockDim=256, gridDim=n_rows.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q5_0_THREADS 256
#define MMVQ_Q5_0_WARPS (MMVQ_Q5_0_THREADS / WARP_SIZE)
#define MMVQ_Q5_0_INT32_PER_BLOCK 4
#define MMVQ_Q5_0_BLOCKS_PER_ITER (MMVQ_Q5_0_THREADS / MMVQ_Q5_0_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q5_0_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Expand 4 consecutive bits of qh starting at bit `start` into 4 bytes
// (each byte 0 or 1), packed into an int32. `qh` is the full 32-bit qh
// field for this block.
static __device__ __forceinline__ int expand_bits4(unsigned int qh, int start) {
    // Each nibble of the result is either 0x01 or 0x00 depending on the
    // corresponding qh bit. Build byte-by-byte.
    int out = 0;
    out |= ((qh >> (start + 0)) & 1u);
    out |= ((qh >> (start + 1)) & 1u) << 8;
    out |= ((qh >> (start + 2)) & 1u) << 16;
    out |= ((qh >> (start + 3)) & 1u) << 24;
    return out;
}

extern "C" __global__ void flambeau_mmvq_q5_0_q8_1(
    const flambeau_block_q5_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int lane4     = tid & 3;
    const int block_idx = tid >> 2;

    const flambeau_block_q5_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q5_0_BLOCKS_PER_ITER) {
        const flambeau_block_q5_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        // Read nibbles (same layout as Q4_0/Q4_1).
        const int v = ((const int*) bx->qs)[lane4];
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        // Read qh as a single uint32 (4-byte little-endian field).
        const unsigned int qh = *((const unsigned int*) bx->qh);

        // 5th bits for low half (elements 4·lane4 .. +3) start at bit 4·lane4,
        // for high half (elements 4·lane4 + 16 .. +19) at bit 4·lane4 + 16.
        const int bit_lo = expand_bits4(qh, lane4 * 4);
        const int bit_hi = expand_bits4(qh, lane4 * 4 + 16);

        // DP4A: (nibble + 16·bit) · y = nibble·y + 16 · bit·y
        int sumi_nib = 0;
        sumi_nib = flambeau_q5_0_dp4a(vi_lo, u_lo, sumi_nib);
        sumi_nib = flambeau_q5_0_dp4a(vi_hi, u_hi, sumi_nib);
        int sumi_bit = 0;
        sumi_bit = flambeau_q5_0_dp4a(bit_lo, u_lo, sumi_bit);
        sumi_bit = flambeau_q5_0_dp4a(bit_hi, u_hi, sumi_bit);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // (q5 - 16) · y = nibble·y + 16·bit·y - 16·sum(y)
        // dot = d_x · d_y · (sumi_nib + 16·sumi_bit) - 16 · d_x · s_y
        // Per-block correction split across 4 lanes → × 0.25.
        acc += (sumi_nib + 16 * sumi_bit) * (d_x * d_y)
             - 16.0f * d_x * s_y * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q5_0_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q5_0_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q5_0_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
