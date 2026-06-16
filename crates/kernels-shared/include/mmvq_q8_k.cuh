#pragma once
// Templated single-row Q8_K MMVQ (decode GEMV). Q8_K is the simplest K-quant:
// a 256-element super-block of int8 quants with one f32 super-block scale `d`
// and no sub-block scales. One super-block covers 8 Q8_1 activation sub-blocks;
// per sub-block the partial is d_x·d_y·dp4a(x,y) with d_x factored out. 256
// threads, 8 int32 per sub-block → 32 sub-blocks per iter. Backend-neutral:
// `dp4a` / the reduce from the arch header + shared headers; int loads go
// through `load_int_b2`. `OutT` is the destination dtype (F32 or F16-dst).
#include "quant_traits.cuh"
#include "mmvq_reduce.cuh"

template<class OutT>
static __device__ __forceinline__ void mmvq_q8_k_row(
    const flambeau_block_q8_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    int n_rows, int n_super_blocks_per_row)
{
    constexpr int THREADS       = 256;
    constexpr int INT32_PER_SUB = 8;
    constexpr int SUBS_PER_ITER = THREADS / INT32_PER_SUB;

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid     = threadIdx.x;
    const int lane8   = tid & 7;
    const int sub_idx = tid >> 3;

    const int n_sub_per_row = n_super_blocks_per_row * 8;
    const flambeau_block_q8_K* xrow = x + (size_t) row * n_super_blocks_per_row;

    float acc = 0.0f;
    for (int s = sub_idx; s < n_sub_per_row; s += SUBS_PER_ITER) {
        const int super_idx    = s >> 3;
        const int sub_in_super = s & 7;
        const flambeau_block_q8_K* bx = xrow + super_idx;
        const flambeau_block_q8_1* by = y + s;

        const int xi = load_int_b2((const int8_t*) (bx->qs + sub_in_super * 32), lane8);
        const int yi = load_int_b2(by->qs, lane8);
        const int sumi = dp4a(xi, yi, 0);

        acc += bx->d * (float) by->d * (float) sumi;
    }

    mmvq_reduce_store<THREADS, OutT>(acc, dst, row);
}
