// mmvq_q4_1 — Q4_1 weight × Q8_1 activation → F32 dst, DP4A inner loop.
//
// V2.2.b first kernel for `arch=qwen35` (Qwen3.5-9B-Q4_1). Q4_1 is the
// legacy 4-bit quant with a per-block min offset (block = 32 elements,
// 18 bytes: d fp16 + m fp16 + 16 bytes nibble-packed quants).
//
// DP4A inner loop, one Q4_1 block per 4 threads (lane4 ∈ [0, 4)):
//   v = ((int*)qs)[lane4];            // one int32 = 4 packed bytes
//   vi_lo = (v >> 0) & 0x0F0F0F0F;    // 4 low nibbles  → elements [4·lane4 .. +3]
//   vi_hi = (v >> 4) & 0x0F0F0F0F;    // 4 high nibbles → elements [4·lane4+16 .. +19]
//   u_lo  = ((int*)y.qs)[lane4];      // y int32 covering elements [4·lane4 .. +3]
//   u_hi  = ((int*)y.qs)[lane4 + 4];  // y int32 covering elements [4·lane4+16 .. +19]
//   sumi  = dp4a(vi_lo, u_lo, 0)
//         + dp4a(vi_hi, u_hi, 0);
//   result = sumi * (d_x * d_y) + (m_x * s_y)
//
// Reference: ggml nibble layout (see `flambeau_quant::dequantize::dequant_q4_1`):
//   byte i (i ∈ [0, 16)): low nibble = element i; high nibble = element i + 16.
//
// Note: llama.cpp's own `vec_dot_q4_1_q8_1_impl` calls dp4a with `u[2*i+0]`
// and `u[2*i+1]`. That works on their Q4 layout because they use a
// pre-shuffled format (`get_int_b2`) that reorders nibbles for contiguous
// DP4A pairing. We use the on-disk ggml Q4_1 layout directly (no pre-shuffle)
// so the pairing is `lane4` + `lane4 + 4`, not `2·lane4` + `2·lane4 + 1`.
//
// Block/grid:
//   blockDim = 256, gridDim = n_rows (one block per output row).
//   Each of the 256 threads handles (block_idx, int_idx_in_block) such that
//   block_idx ∈ [0, 64) and int_idx ∈ [0, 4) — 64 * 4 = 256 threads cover
//   4 Q4_1 blocks worth of int32s at a time, then stride by
//   BLOCKS_PER_ITER.

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q4_1_THREADS 256
#define MMVQ_Q4_1_WARPS (MMVQ_Q4_1_THREADS / WARP_SIZE)
#define MMVQ_Q4_1_INT32_PER_BLOCK 4           // 16 bytes of qs → 4 int32
#define MMVQ_Q4_1_BLOCKS_PER_ITER (MMVQ_Q4_1_THREADS / MMVQ_Q4_1_INT32_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q4_1_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q4_1_q8_1(
    const flambeau_block_q4_1* __restrict__ x,
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
    // Within a Q4_1 block, which of the 4 int32s of qs we own.
    const int lane4     = tid & 3;
    // Which Q4_1 block (0..63 across the 256 threads).
    const int block_idx = tid >> 2;

    const flambeau_block_q4_1* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_Q4_1_BLOCKS_PER_ITER) {
        const flambeau_block_q4_1* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        // Load one int32 (4 packed Q4_1 bytes) from this thread's slot.
        const int v = ((const int*) bx->qs)[lane4];
        // Low nibbles of v[lane4] pair with y int32 at index `lane4`
        // (elements [4·lane4 .. +3]). High nibbles pair with index
        // `lane4 + 4` (elements [4·lane4 + 16 .. +19]).
        const int u_lo = ((const int*) by->qs)[lane4];
        const int u_hi = ((const int*) by->qs)[lane4 + 4];

        const int vi_lo = (v >> 0) & 0x0F0F0F0F;
        const int vi_hi = (v >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = flambeau_q4_1_dp4a(vi_lo, u_lo, sumi);
        sumi = flambeau_q4_1_dp4a(vi_hi, u_hi, sumi);

        const float d_x = (float) bx->d;
        const float m_x = (float) bx->m;
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        // Each thread contributes its share of sumi * (d_x*d_y). The
        // `m_x * s_y` constant is per-block — divide by the 4 int32-lanes so
        // the warp reduce sums to exactly one `m_x * s_y` per Q4_1 block.
        acc += sumi * (d_x * d_y) + (m_x * s_y) * 0.25f;
    }

    // Warp-wide sum.
    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q4_1_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q4_1_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q4_1_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
