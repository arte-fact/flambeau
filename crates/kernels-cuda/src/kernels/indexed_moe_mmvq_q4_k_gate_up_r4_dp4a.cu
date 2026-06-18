// CUDA entry for the fused gate+up Q4_K MoE MMVQ (r4, 16-lane groups). The body
// is the backend-neutral `indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_row` in the
// shared header; HIP builds the same stem from its hand-written gfx906 .cu. The
// HIP→CUDA unification (one manifest cell, drop the static pair) is deferred —
// it needs a HIP re-cert since the manifest body would replace the hand kernel.
#include "sm_80.cuh"
#include "indexed_moe_mmvq_q4_k.cuh"

extern "C" __global__ void flambeau_indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_q8_1(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row
) {
    indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_row<float>(
        gate_w, up_w, y, expert_ids, gate_out, up_out,
        n_rows, n_tokens, top_k, n_sb_per_row);
}
