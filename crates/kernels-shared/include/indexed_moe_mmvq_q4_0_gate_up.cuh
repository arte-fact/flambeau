#pragma once
// Templated fused gate+up Q4_0 indexed-MoE MMVQ. The MoE sibling of
// `mmvq_q4_0_gate_up_dp4a_row`: gate and up expert rows share one per-token Q8_1
// activation read; each thread owns one int32 of nibble quants (4 threads per
// 32-element block), unpacks low/high nibble planes, and runs the `(q - 8)`
// dp4a bias identity `sumi·d_x·d_y − 8·d_x·s_y` (the constant split ×0.25 across
// the 4 lanes so the block reduce sums it once). MoE addressing: weight rows by
// `expert_ids[token,slot]`, per-token activation `y[token,:]`, outputs
// `gate_out`/`up_out`[token,slot,row]. Backend-neutral: `dp4a` + the two-output
// reduce from the arch+shared headers; int loads through `load_int_b2`.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmvq_q4_0_gate_up_dp4a_row(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row)
{
    constexpr int INT32_PER_QBLK  = QK4_0 / 8;
    constexpr int BLOCKS_PER_ITER = THREADS / INT32_PER_QBLK;

    const int row      = blockIdx.x;
    const int slot     = blockIdx.y;
    const int token    = slot / top_k;
    const int slot_idx = slot - token * top_k;
    if (row >= n_rows || token >= n_tokens) return;

    const int expert = expert_ids[(size_t) token * top_k + slot_idx];

    const int tid       = threadIdx.x;
    const int lane4     = tid & (INT32_PER_QBLK - 1);
    const int block_idx = tid / INT32_PER_QBLK;

    const flambeau_block_q4_0* g_row =
        gate_w + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q4_0* u_row =
        up_w + (((size_t) expert * n_rows) + row) * n_blocks_per_row;
    const flambeau_block_q8_1* y_row = y + (size_t) token * n_blocks_per_row;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* by = y_row + b;
        const int u_lo = load_int_b2(by->qs, lane4);
        const int u_hi = load_int_b2(by->qs, lane4 + 4);
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        {
            const int v = load_int_b2((const int8_t*) (g_row[b].qs), lane4);
            const int vi_lo = v & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi     = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) g_row[b].d;
            acc_g += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        {
            const int v = load_int_b2((const int8_t*) (u_row[b].qs), lane4);
            const int vi_lo = v & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi     = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) u_row[b].d;
            acc_u += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    const int dst_idx = (int) (((size_t) token * top_k + slot_idx) * n_rows + row);
    mmvq_reduce_store_gate_up<THREADS, OutT>(acc_g, acc_u, gate_out, up_out, dst_idx, true, true);
}
