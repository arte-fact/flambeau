#pragma once

// Block-wide online-softmax reduction: merge per-thread (max, sum_exp) pairs
// with the rescaling recurrence
//   (m, d) ⊕ (m', d') = (M, d·e^{m-M} + d'·e^{m'-M}),  M = max(m, m')
// and broadcast the row (max, sum) to every thread. Backend-neutral: the warp
// width + shuffle come from the arch seam (`WARP_SIZE`, `__shfl_xor`), so the
// same body is correct on wave64 and wave32. `scratch_max`/`scratch_sum` are
// caller-owned LDS of at least `THREADS / WARP_SIZE` floats each. `-INF` is the
// identity max (padding lanes contribute nothing).
#ifndef FB_NEG_INF
#define FB_NEG_INF (-__builtin_huge_valf())
#endif

template <int THREADS>
static __device__ __forceinline__ void block_reduce_softmax(
    float& vmax, float& vsum, float* scratch_max, float* scratch_sum) {
    const int warps = THREADS / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        const float om = __shfl_xor(vmax, off, WARP_SIZE);
        const float os = __shfl_xor(vsum, off, WARP_SIZE);
        const float nm = fmaxf(vmax, om);
        // Factor 1 when a max already equals the merged max — avoids
        // __expf(-INF - -INF) = NaN when both inputs are empty (-INF, 0).
        const float rs = (vmax == nm) ? 1.0f : __expf(vmax - nm);
        const float ro = (om == nm) ? 1.0f : __expf(om - nm);
        vsum = vsum * rs + os * ro;
        vmax = nm;
    }

    if (lane == 0) {
        scratch_max[warp] = vmax;
        scratch_sum[warp] = vsum;
    }
    __syncthreads();

    if (warp == 0) {
        float wm = (lane < warps) ? scratch_max[lane] : FB_NEG_INF;
        float ws = (lane < warps) ? scratch_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const float om = __shfl_xor(wm, off, WARP_SIZE);
            const float os = __shfl_xor(ws, off, WARP_SIZE);
            const float nm = fmaxf(wm, om);
            const float rs = (wm == nm) ? 1.0f : __expf(wm - nm);
            const float ro = (om == nm) ? 1.0f : __expf(om - nm);
            ws = ws * rs + os * ro;
            wm = nm;
        }
        if (lane == 0) {
            scratch_max[0] = wm;
            scratch_sum[0] = ws;
        }
    }
    __syncthreads();
    vmax = scratch_max[0];
    vsum = scratch_sum[0];
    __syncthreads();
}
