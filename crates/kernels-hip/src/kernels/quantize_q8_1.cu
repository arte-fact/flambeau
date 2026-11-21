// quantize_q8_1 — F32 activation -> Q8_1 blocks. Thin wrapper over the
// backend-neutral kernels-shared device function; same entry-point name + ABI
// as the kernels-cuda twin so the registry/dispatch address it identically.
// Launch: gridDim = { n_elems / QK8_1 }, blockDim = { 32 }, shared = 0.

#include "quantize_q8_1.cuh"

extern "C" __global__ void flambeau_quantize_row_q8_1(
    const float* __restrict__ x,
    flambeau_block_q8_1* __restrict__ y,
    const int n_elems
) {
    quantize_q8_1_block(x, y, n_elems);
}
