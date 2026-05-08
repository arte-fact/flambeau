// mmvq_bf16_bf16 — BF16 weight × BF16 activation MMVQ.
// the load-bearing kernel of the BF16-throughout MTP forward.
// vLLM/sglang's BF16 path keeps activations in BF16 across the matmul
// chain (no Q8_1 quantize between layers); this kernel makes that
// possible on flambeau.
// gfx906 has no native BF16 arithmetic — CDNA2/MI200's
// `v_dot2_f32_bf16` is the first BF16-capable hardware. We emulate by
// up-casting both operands to F32 (lossless bit-shift) and accumulating
// in F32. Compared to F16×Q8_1 mmvq, we avoid the 1-pass activation
// quantize (eliminates ~0.78 % per-block-of-32 noise) and gain ~3 bits
// of exponent range, at the cost of 2× the activation HBM bandwidth
// (BF16 = 2 B/elem vs Q8_1 ≈ 1.125 B/elem).
// Layout / launch:
// weight `x` : `[n_rows, k]` BF16 row-major
// act `y` : `[k]` BF16
// dst : `[n_rows]` F32
// grid : (n_rows, 1, 1)
// block : (256, 1, 1) — 4 wave64 warps; one row per block,
// threads stride k by 256 with coalesced loads.
// `k` must be a multiple of 256 (we don't tail-handle here; in practice
// every Qwen3.6 hidden / projection size satisfies this).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define BF16_MMVQ_BLOCK_THREADS 256
#define BF16_MMVQ_WARPS_PER_BLOCK (BF16_MMVQ_BLOCK_THREADS / WARP_SIZE)

extern "C" __global__ void flambeau_mmvq_bf16_bf16(
    const fb_bf16_t* __restrict__ x_bf16,    // weight [n_rows, k]
    const fb_bf16_t* __restrict__ y_bf16,    // activation [k]
    float* __restrict__ dst,                 // [n_rows]
    const int n_rows,
    const int k
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid & (WARP_SIZE - 1);

    const fb_bf16_t* xrow = x_bf16 + (size_t) row * (size_t) k;

    float acc = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < k; i += BF16_MMVQ_BLOCK_THREADS) {
        const float w = fb_bf16_to_f32(xrow[i]);
        const float a = fb_bf16_to_f32(y_bf16[i]);
        acc += w * a;
    }

    // Warp reduce.
    acc = gfx906_warp_reduce_sum(acc);

    // Cross-warp reduce via LDS (4 warps → 1).
    __shared__ float s_partials[BF16_MMVQ_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_partials[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < BF16_MMVQ_WARPS_PER_BLOCK) ? s_partials[lane] : 0.0f;
        #pragma unroll
        for (int off = BF16_MMVQ_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst[row] = v;
        }
    }
}
