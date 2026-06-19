// CUDA entry for the fused K+V Q4_0 decode MMVQ (F16 dst). Body is the
// backend-neutral `mmvq_q4_0_kv_f16dst_row` in the shared header; HIP builds the
// same stem from its hand-written gfx906 .cu (unification deferred — would need
// a HIP re-cert).
#include "sm_80.cuh"
#include "mmvq_q4_0_kv_f16dst.cuh"

extern "C" __global__ __launch_bounds__(256)
void flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1(
    const flambeau_block_q4_0* __restrict__ k_w,
    const flambeau_block_q4_0* __restrict__ v_w,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ k_out,
    fb_fp16_t* __restrict__ v_out,
    const int n_rows_kv,
    const int n_blocks_per_row
) {
    mmvq_q4_0_kv_f16dst_row(k_w, v_w, y, k_out, v_out, n_rows_kv, n_blocks_per_row);
}
