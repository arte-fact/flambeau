#pragma once
// Q4_K embedding-row gather, F16 out, token ids read on device. Matches the
// host load path (flambeau-quant `dequant_q4_k`, then f32 -> f16 round to
// nearest even) bit for bit: `d1 = d * sc` and `m1 = dmin * m` are rounded
// separately, then `d1 * q - m1` is evaluated without FMA contraction
// (`fb_mul_sub_rn`, from the backend's fp_rn.cuh).
//
// Grid: x = output row, y = super-block within the row (hidden / QK_K).
// Block: QK_K threads, one element each.

#include "block_quant.cuh"

static __device__ __forceinline__ void embed_gather_q4_k_f16_body(
    const flambeau_block_q4_K* __restrict__ embd,
    fb_fp16_t* __restrict__ dst,
    const int* __restrict__ ids,
    const int hidden)
{
    const int row = blockIdx.x;
    const int sb = blockIdx.y;
    const int e = threadIdx.x;
    const int blocks_per_row = hidden / QK_K;
    const flambeau_block_q4_K* b = embd + (size_t) ids[row] * blocks_per_row + sb;

    const int chunk = e >> 6;
    const int high = (e >> 5) & 1;
    const int l = e & 31;
    uint8_t sc, m;
    flambeau_q4k_scale_min(2 * chunk + high, b->scales, &sc, &m);
    const float d1 = (float) b->d * (float) sc;
    const float m1 = (float) b->dmin * (float) m;
    const uint8_t byte = b->qs[chunk * 32 + l];
    const float q = (float) (high ? (byte >> 4) : (byte & 0xF));

    dst[(size_t) row * hidden + (size_t) sb * QK_K + e] = (fb_fp16_t) fb_mul_sub_rn(d1, q, m1);
}
