#pragma once
// quantize_q8_1_few — F16 or F32 activation → Q8_1 in the int8 MMA B-fragment
// layout of the few-column MMQ (m16n8k32, 8 columns per group).
//
// Per unit of 256 values (8 sub-blocks of 32) and group of 8 columns, 144
// int4 (2304 B); the groups of a unit are adjacent, units follow in k order:
//   qs: int4[4][32], [j][lane] for lane = 4*g + t holds values [4t, 4t + 4)
//       and [16 + 4t, 16 + 4t + 4) of sub-block 2j, then of sub-block 2j + 1,
//       for column g of the group.
//   ds: int4[4][4], [t][j] holds the scale of columns 2t and 2t + 1 for
//       sub-block 2j, then 2j + 1: d (D4, float) or (d, d * sum q) (DS4,
//       half2).
// Columns past total_b are zero (q = 0, d = 0).
//
// One block per (column, unit), 64 threads of 4 values each; the 8 threads of
// a sub-block reduce with width-8 shuffles.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define FEW_NCOLS 8
#define FEW_INT4_PER_UNIT_GROUP (4 * 32 + 4 * 4)

template <bool DS4, typename T>
__device__ __forceinline__ void quantize_few_unit(
    const T* __restrict__ x, int4* __restrict__ vy, int ncols, int total_b) {
    const int col = blockIdx.x;
    const int unit = blockIdx.y;
    const int n_groups = gridDim.x / FEW_NCOLS;
    const int iu = 4 * threadIdx.x;

    float xi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (col < total_b) {
        const T* src = x + (size_t) col * ncols + (size_t) unit * 256 + iu;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            xi[i] = (float) src[i];
        }
    }

    float amax = fmaxf(fmaxf(fabsf(xi[0]), fabsf(xi[1])), fmaxf(fabsf(xi[2]), fabsf(xi[3])));
#pragma unroll
    for (int m = 4; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m, 8));
    }

    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    int8_t q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = (int8_t) roundf(xi[i] * d_inv);
    }
    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;

    float qsum = 0.0f;
    if (DS4) {
        qsum = (float) (q[0] + q[1] + q[2] + q[3]);
#pragma unroll
        for (int m = 4; m > 0; m >>= 1) {
            qsum += __shfl_xor_sync(0xffffffffu, qsum, m, 8);
        }
    }

    const int g = col % FEW_NCOLS;
    const int s = iu / 32;
    const int k = iu % 32;
    int4* yg = vy + ((size_t) unit * n_groups + col / FEW_NCOLS) * FEW_INT4_PER_UNIT_GROUP;

    int packed;
    memcpy(&packed, q, sizeof(packed));
    ((int*) (yg + (s / 2) * 32 + 4 * g + (k % 16) / 4))[2 * (s % 2) + k / 16] = packed;
    if (k == 0) {
        int4* ds = yg + 4 * 32 + 4 * (g / 2) + s / 2;
        if (DS4) {
            ((__half2*) ds)[2 * (s % 2) + g % 2] = __floats2half2_rn(d, d * qsum);
        } else {
            ((float*) ds)[2 * (s % 2) + g % 2] = d;
        }
    }
}

