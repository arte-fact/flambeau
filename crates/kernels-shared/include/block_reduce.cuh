#pragma once

// Block-wide sum reduction, broadcast to every thread. Backend-neutral: warp
// width and the per-warp shuffle come from the arch-primitives seam
// (`WARP_SIZE`, `warp_reduce_sum`), so the same body is correct on wave64
// (gfx906) and wave32 (sm_8x / RDNA). THREADS must be a multiple of WARP_SIZE
// and `THREADS / WARP_SIZE <= WARP_SIZE`. `scratch` is caller-owned LDS of at
// least `THREADS / WARP_SIZE` floats.
template <int THREADS>
static __device__ __forceinline__ float block_reduce_sum(float val, float* scratch) {
    const int warps = THREADS / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        scratch[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float s = (lane < warps) ? scratch[lane] : 0.0f;
        s = warp_reduce_sum(s);
        if (lane == 0) {
            scratch[0] = s;
        }
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}
