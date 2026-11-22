// CUDA entry for the fused gate+up Q4_0 decode MMVQ (t128). Body is the
// backend-neutral `mmvq_q4_0_gate_up_t128_row` in the shared header; HIP builds
// the same stem from its hand-written gfx906 .cu.
#include "sm_80.cuh"
#include "mmvq_q4_0_gate_up_t128.cuh"

extern "C" __global__ __launch_bounds__(128)
void flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_blocks_per_row
) {
    mmvq_q4_0_gate_up_t128_row(gate_w, up_w, y, gate_out, up_out, n_rows_gate, n_rows_up, n_blocks_per_row);
}
