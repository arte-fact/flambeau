// Device-position F16 -> Q8_0 quantize-and-append. CUDA-only sibling of
// quantize_f16_q8_0: the destination cache row is read on-device from
// `pos_ptr[0]` (the decode-graph position) instead of being folded into a host
// destination pointer, so a captured decode graph advances the Q8 KV write
// cursor with no per-token rebind. Decode-only: one row of `n_elems` F16
// (n_elems = kv_width), quantized into `n_elems / 32` Q8_0 blocks written at
// physical row `ring_depth > 0 ? position % ring_depth : position`. `y` is the
// UNSLID cache slab base. One thread block per 32 elements, 32 threads.

#include "sm_80.cuh"
#include "block_quant.cuh"

extern "C" __global__ void flambeau_quantize_row_f16_q8_0_dpos(
    const fb_fp16_t* __restrict__ x,
    flambeau_block_q8_0* __restrict__ y,
    const int n_elems,
    const int ring_depth,
    const int* __restrict__ pos_ptr
) {
    const int ib = blockIdx.x;
    const int t  = threadIdx.x;
    const int base = ib * QK8_0;
    if (base + t >= n_elems) return;

    const int position = pos_ptr[0];
    const int blocks_per_row = n_elems / QK8_0;
    const int dst_row = (ring_depth > 0) ? (position % ring_depth) : position;
    const int out_block = dst_row * blocks_per_row + ib;

    const float xi = (float) x[base + t];
    float amax = fabsf(xi);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor(amax, offset, 32);
        amax = fmaxf(amax, other);
    }

    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    const int qi = min(127, max(-127, (int) rintf(xi * id)));

    if (t == 0) {
        y[out_block].d = (fb_fp16_t) d;
    }
    y[out_block].qs[t] = (int8_t) qi;
}
