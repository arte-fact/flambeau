#pragma once
// Templated fused gate+up Q4_0 dense MMVQ (decode GEMV). Like the Q8_0 fusion,
// the gate and up weight rows share one Q8_1 activation read; here each thread
// owns one int32 of nibble quants (4 threads per 32-element block), unpacks the
// low/high nibble planes, and runs the `(q - 8)` dp4a bias-correction identity
// from the single-row Q4_0 kernel: per block `sumi·d_x·d_y − 8·d_x·s_y`, with
// the constant bias split ×0.25 across the 4 lanes so the block reduce sums it
// exactly once. Backend-neutral: `dp4a` / the two-output reduce come from the
// arch + shared headers; the weight and activation int loads go through
// `load_int_b2` (Q4_0 `qs` sits at block-offset 2 — a 4-byte read faults on
// NVIDIA). `OutT` is the destination dtype.
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<int THREADS, class OutT = float>
static __device__ __forceinline__ void mmvq_q4_0_gate_up_dp4a_row(
    const flambeau_block_q4_0* __restrict__ gate_w,
    const flambeau_block_q4_0* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows_gate, int n_rows_up, int n_blocks_per_row)
{
    constexpr int INT32_PER_QBLK  = QK4_0 / 8;          // 4 nibble-int32 per block
    constexpr int BLOCKS_PER_ITER = THREADS / INT32_PER_QBLK;

    const int row = blockIdx.x;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;
    if (!do_gate && !do_up) return;

    const int tid       = threadIdx.x;
    const int lane4     = tid & (INT32_PER_QBLK - 1);
    const int block_idx = tid / INT32_PER_QBLK;

    const flambeau_block_q4_0* g_row = gate_w + (size_t) row * n_blocks_per_row;
    const flambeau_block_q4_0* u_row = up_w   + (size_t) row * n_blocks_per_row;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += BLOCKS_PER_ITER) {
        const flambeau_block_q8_1* by = y + b;
        const int u_lo = load_int_b2(by->qs, lane4);
        const int u_hi = load_int_b2(by->qs, lane4 + 4);
        const float d_y = (float) by->d;
        const float s_y = (float) by->s;

        if (do_gate) {
            const int v = load_int_b2((const int8_t*) (g_row[b].qs), lane4);
            const int vi_lo = v & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi     = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) g_row[b].d;
            acc_g += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
        if (do_up) {
            const int v = load_int_b2((const int8_t*) (u_row[b].qs), lane4);
            const int vi_lo = v & 0x0F0F0F0F;
            const int vi_hi = (v >> 4) & 0x0F0F0F0F;
            int sumi = dp4a(vi_lo, u_lo, 0);
            sumi     = dp4a(vi_hi, u_hi, sumi);
            const float d_x = (float) u_row[b].d;
            acc_u += sumi * (d_x * d_y) - 8.0f * d_x * s_y * 0.25f;
        }
    }

    mmvq_reduce_store_gate_up<THREADS, OutT>(acc_g, acc_u, gate_out, up_out, row, do_gate, do_up);
}
