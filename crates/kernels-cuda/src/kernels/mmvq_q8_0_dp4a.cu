// Q8_0 weight × Q8_1 activation -> F32 decode GEMV, dp4a inner loop.
// A template instantiation of the backend-neutral `mmvq_dp4a_row` skeleton —
// the per-dtype code is `quant_traits<Q8_0>`. Same entry-point name + ABI as
// the HIP kernel so the registry/dispatch address it identically.
// Launch: gridDim = { n_rows }, blockDim = { 256 }, shared = 0.

#include "sm_80.cuh"
#include "quant_traits.cuh"
#include "mmvq.cuh"

extern "C" __global__ void flambeau_mmvq_q8_0_dp4a_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_dp4a_row<Q8_0, 256>(x, y, dst, n_rows, n_blocks_per_row);
}
