#pragma once
// Few-column MMQ on sm_80+ int8 tensor cores (mma.sync), for 1..24
// activation columns (NCG groups of 8). Port of llama.cpp's mul_mat_q_few
// (ggml-cuda/mmq-few.cu, the correctness oracle); one decoder per weight type.
//
// A block holds 4 warps over 16·NMT weight rows; the warps split K in units of
// 256 values (one super-block). Each warp copies the raw super-blocks of its
// rows into its own 2-stage cp.async ring in shared memory, then each lane
// decodes the A fragments of rows g and g + 8 (lane = 4g + t: values
// [4t, 4t + 4) and [16 + 4t, ...) of each 32-value sub-block) straight from
// the ring. The activation is pre-quantised in the B-fragment layout
// (quantize_q8_1_few.cuh), so a lane's B fragments are plain int4 loads. The
// 4 warp partials are summed in shared memory at the end.
//
// A type whose super-block is not a multiple of 16 bytes is
// copied from the 16-byte boundary before it, the last chunk cut to the bytes
// needed (no read past them), and decoded at the row's offset in the ring.
//
// Requirements (checked by the launcher): nunits = k / 256 a multiple of the
// units per ring stage; y in the few layout with ceil(ncols / 8) groups and the
// decoder's scale record.
//
// dst is slot-major: dst[j * stride_col_dst + i] for column j, weight row i.

#include <cuda_fp16.h>
#include <stdint.h>
#include "mma_int8.cuh"
#include "cp_async.cuh"

namespace mmq_few {

constexpr int NWARPS = 4;
constexpr int NROWS = 16;
constexpr int NCOLS = 8;
constexpr int Y_INT4 = 4 * 32 + 4 * 4;
constexpr int NSTAGES = 2;

// Units per ring stage: 2, or 1 with 3 column groups (fewer free registers)
// or 32-row tiles.
__host__ __device__ constexpr int units_per_stage(int ncg, int nmt) {
    return ncg > 2 || nmt > 1 ? 1 : 2;
}

// 16-byte chunks a ring row copies: whole units, or for unaligned units the
// worst case over the 8 even start offsets.
__host__ __device__ constexpr int nchunks(int unit_bytes, int nus) {
    return (nus * unit_bytes) % 16 == 0 ? nus * unit_bytes / 16 : (14 + nus * unit_bytes + 15) / 16;
}

// Ring row stride: unaligned rows padded to an odd number of chunks against
// bank conflicts.
__host__ __device__ constexpr int sram_stride(int unit_bytes, int nus) {
    return (nus * unit_bytes) % 16 != 0 && nchunks(unit_bytes, nus) % 2 == 0
        ? 16 * (nchunks(unit_bytes, nus) + 1)
        : 16 * nchunks(unit_bytes, nus);
}

struct YFrag {
    int4 qs[4];
    int4 ds[4];
};

__device__ __forceinline__ void load_y(YFrag& yf, const int4* __restrict__ y) {
    const int lane = threadIdx.x;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.qs[j] = y[j * 32 + lane];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yf.ds[j] = y[4 * 32 + 4 * (lane % 4) + j];
    }
}

// B fragment of sub-block s: values [4t, 4t + 4) and [16 + 4t, ...) of
// column g of the group.
__device__ __forceinline__ int2 get_b(const YFrag& yf, int s) {
    const int4 q = yf.qs[s / 2];
    return s % 2 == 0 ? make_int2(q.x, q.y) : make_int2(q.z, q.w);
}

// DS4 record: d of sub-block s for columns 2t and 2t + 1.
__device__ __forceinline__ float2 get_d_ds4(const YFrag& yf, int s) {
    const __half2* ds = (const __half2*) &yf.ds[s / 2];
    return make_float2(__low2float(ds[2 * (s % 2) + 0]), __low2float(ds[2 * (s % 2) + 1]));
}

// DS4 record: d·Σq of sub-block s for columns 2t and 2t + 1 (the min term).
__device__ __forceinline__ float2 get_sum_ds4(const YFrag& yf, int s) {
    const __half2* ds = (const __half2*) &yf.ds[s / 2];
    return make_float2(__high2float(ds[2 * (s % 2) + 0]), __high2float(ds[2 * (s % 2) + 1]));
}

// D4 record: d of sub-block s for columns 2t and 2t + 1.
__device__ __forceinline__ float2 get_d_d4(const YFrag& yf, int s) {
    const float* d = (const float*) &yf.ds[s / 2];
    return make_float2(d[2 * (s % 2) + 0], d[2 * (s % 2) + 1]);
}

// 4 bytes from a 2-byte aligned address.
__device__ __forceinline__ int load_b2(const void* p, int i32) {
    const uint16_t* x16 = (const uint16_t*) p;
    return (int) (x16[2 * i32] | ((uint32_t) x16[2 * i32 + 1] << 16));
}

// ---------------------------------------------------------------------------
// Decoders: one unit of 256 values of rows g and g + 8 (x[0], x[1]: their
// super-blocks in the ring) against every column group. C fragment l of
// group c: row g + 8·(l / 2), column c·8 + 2t + l % 2.

// The 6-bit scales (ksc 0, 1) and mins (ksc 2, 3) of a Q4_K / Q5_K
// super-block as 4 bytes each: ksc 0 = sc0..sc3, 1 = sc4..sc7, 2 = m0..m3,
// 3 = m4..m7.
__device__ __forceinline__ int unpack_scales_q45_k(const int* scales, int ksc) {
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F) |
           ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);
}

// Q4_K (Q5 = false, 144-B super-block: dm, scales[12], qs[128]) and Q5_K
// (Q5 = true, 176 B: dm, scales[12], qh[32], qs[128]); DS4 activation.
template <bool Q5>
struct DecQ45K {
    static constexpr int UNIT_BYTES = Q5 ? 176 : 144;
    static constexpr int QS = Q5 ? 48 : 16;

    template <int NCG>
    static __device__ __forceinline__ void vec_dot(
        const char* const (&x)[2], const YFrag (&yf)[NCG], float (&sum)[NCG][4]) {
        const int t = threadIdx.x % 4;

        int qs[2][4][2];
        int qh[2][2];
        int sc[2][2];
        int mn[2][2];
        float d[2];
        float dmin[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int* q = (const int*) (x[r] + QS);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                qs[r][j][0] = q[8 * j + t];
                qs[r][j][1] = q[8 * j + 4 + t];
            }
            if constexpr (Q5) {
                const int* h = (const int*) (x[r] + 16);
                qh[r][0] = h[t];
                qh[r][1] = h[4 + t];
            }
            const int4 hdr = *(const int4*) x[r];
            const int scales[3] = {hdr.y, hdr.z, hdr.w};
            sc[r][0] = unpack_scales_q45_k(scales, 0);
            sc[r][1] = unpack_scales_q45_k(scales, 1);
            mn[r][0] = unpack_scales_q45_k(scales, 2);
            mn[r][1] = unpack_scales_q45_k(scales, 3);
            const float2 dm = __half22float2(*(const __half2*) &hdr.x);
            d[r] = dm.x;
            dmin[r] = dm.y;
        }

        float sum_d[NCG][4] = {{0.0f}};
        float sum_m[NCG][4] = {{0.0f}};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int j = s / 2;
            const int shift = 4 * (s % 2);
            flambeau_mma::tile_i16 a;
            a.x[0] = (qs[0][j][0] >> shift) & 0x0F0F0F0F;
            a.x[1] = (qs[1][j][0] >> shift) & 0x0F0F0F0F;
            a.x[2] = (qs[0][j][1] >> shift) & 0x0F0F0F0F;
            a.x[3] = (qs[1][j][1] >> shift) & 0x0F0F0F0F;
            if constexpr (Q5) {
                a.x[0] |= ((qh[0][0] >> s) & 0x01010101) << 4;
                a.x[1] |= ((qh[1][0] >> s) & 0x01010101) << 4;
                a.x[2] |= ((qh[0][1] >> s) & 0x01010101) << 4;
                a.x[3] |= ((qh[1][1] >> s) & 0x01010101) << 4;
            }
            const int ls0 = (sc[0][s / 4] >> (8 * (s % 4))) & 0xFF;
            const int ls1 = (sc[1][s / 4] >> (8 * (s % 4))) & 0xFF;
            const float m0 = (float) ((mn[0][s / 4] >> (8 * (s % 4))) & 0xFF);
            const float m1 = (float) ((mn[1][s / 4] >> (8 * (s % 4))) & 0xFF);
#pragma unroll
            for (int c = 0; c < NCG; ++c) {
                const int2 bq = get_b(yf[c], s);
                flambeau_mma::tile_i8 b;
                b.x[0] = bq.x;
                b.x[1] = bq.y;
                flambeau_mma::tile_i16 acc;
                flambeau_mma::mma(acc, a, b);
                const float2 dy = get_d_ds4(yf[c], s);
                sum_d[c][0] += dy.x * (float) (acc.x[0] * ls0);
                sum_d[c][1] += dy.y * (float) (acc.x[1] * ls0);
                sum_d[c][2] += dy.x * (float) (acc.x[2] * ls1);
                sum_d[c][3] += dy.y * (float) (acc.x[3] * ls1);
            }
#pragma unroll
            for (int c = 0; c < NCG; ++c) {
                const float2 sy = get_sum_ds4(yf[c], s);
                sum_m[c][0] += m0 * sy.x;
                sum_m[c][1] += m0 * sy.y;
                sum_m[c][2] += m1 * sy.x;
                sum_m[c][3] += m1 * sy.y;
            }
        }
#pragma unroll
        for (int c = 0; c < NCG; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                sum[c][l] += d[l / 2] * sum_d[c][l] - dmin[l / 2] * sum_m[c][l];
            }
        }
    }
};

// ---------------------------------------------------------------------------

template <class Dec, int NCG, int NMT>
__device__ __forceinline__ void mmq_few_rows(
    const char* __restrict__ x, const int4* __restrict__ y, float* __restrict__ dst,
    const int nunits, const int nrows_x, const int ncols, const int stride_row_x,
    const int stride_col_dst) {
    constexpr int unit = Dec::UNIT_BYTES;
    constexpr int nrows = NMT * NROWS;
    constexpr int nus = units_per_stage(NCG, NMT);
    constexpr bool unaligned = (nus * unit) % 16 != 0;
    constexpr int nchk = nchunks(unit, nus);
    constexpr int stride = sram_stride(unit, nus);
    constexpr int nbytes_stage = nrows * stride;
    constexpr int ncopies = nrows * nchk;

    extern __shared__ __align__(16) char smem[];
    char* ring = smem + threadIdx.y * NSTAGES * nbytes_stage;

    const int row0 = blockIdx.x * nrows;
    const int iu_begin = nus * threadIdx.y;
    const int iu_step = nus * NWARPS;

    auto load_stage = [&](int stage, int iu0) {
        const unsigned int dst_stage = flambeau_cp_async::to_shared(ring + stage * nbytes_stage);
#pragma unroll
        for (int c0 = 0; c0 < ncopies; c0 += 32) {
            const int c = c0 + threadIdx.x;
            if (ncopies % 32 == 0 || c < ncopies) {
                const int i = c / nchk;
                const int ic = c % nchk;
                const int row = min(row0 + i, nrows_x - 1);
                const char* src = x + (size_t) row * stride_row_x + (size_t) iu0 * unit;
                if constexpr (unaligned) {
                    const int off = (int) ((uintptr_t) src % 16);
                    const int need = off + nus * unit - 16 * ic;
                    if (need > 0) {
                        flambeau_cp_async::cg_16_partial(
                            dst_stage + i * stride + 16 * ic, src - off + 16 * ic, need < 16 ? need : 16);
                    }
                } else {
                    flambeau_cp_async::cg_16(dst_stage + i * stride + 16 * ic, src + 16 * ic);
                }
            }
        }
    };

    // Offsets of this lane's rows from their 16-byte boundary.
    int off_row[2 * NMT] = {0};
    if constexpr (unaligned) {
#pragma unroll
        for (int r = 0; r < 2 * NMT; ++r) {
            const int row = min(row0 + (int) threadIdx.x / 4 + 8 * r, nrows_x - 1);
            off_row[r] = (int) ((uintptr_t) (x + (size_t) row * stride_row_x) % 16);
        }
    }

#pragma unroll
    for (int stage = 0; stage < NSTAGES - 1; ++stage) {
        const int iu0 = iu_begin + stage * iu_step;
        if (iu0 < nunits) {
            load_stage(stage, iu0);
        }
        flambeau_cp_async::commit_group();
    }

    auto load_y_unit = [&](YFrag (&yf)[NCG], int iu) {
#pragma unroll
        for (int c = 0; c < NCG; ++c) {
            load_y(yf[c], y + ((size_t) iu * NCG + c) * Y_INT4);
        }
    };

    // The next activation fragments are loaded right after the MMAs that use
    // the current ones (an earlier load is slower in llama.cpp's tuning).
    YFrag yf[nus][NCG];
    if (iu_begin < nunits) {
#pragma unroll
        for (int k = 0; k < nus; ++k) {
            load_y_unit(yf[k], iu_begin + k);
        }
    }

    float sum[NMT][NCG][4] = {{{0.0f}}};

    for (int iter = 0, iu0 = iu_begin; iu0 < nunits; ++iter, iu0 += iu_step) {
        const int iu_load = iu0 + (NSTAGES - 1) * iu_step;
        if (iu_load < nunits) {
            load_stage((iter + NSTAGES - 1) % NSTAGES, iu_load);
        }
        flambeau_cp_async::commit_group();

        const bool has_next = iu0 + iu_step < nunits;

        flambeau_cp_async::wait_group<NSTAGES - 1>();
        __syncwarp();

        const char* xs = ring + (iter % NSTAGES) * nbytes_stage;
        const int off_stage = unaligned ? (iu0 * unit) % 16 : 0;
#pragma unroll
        for (int k = 0; k < nus; ++k) {
#pragma unroll
            for (int mt = 0; mt < NMT; ++mt) {
                const char* const xr[2] = {
                    xs + (NROWS * mt + threadIdx.x / 4 + 0) * stride + (off_row[2 * mt + 0] + off_stage) % 16
                        + k * unit,
                    xs + (NROWS * mt + threadIdx.x / 4 + 8) * stride + (off_row[2 * mt + 1] + off_stage) % 16
                        + k * unit,
                };
                Dec::template vec_dot<NCG>(xr, yf[k], sum[mt]);
            }
            if (has_next) {
                load_y_unit(yf[k], iu0 + iu_step + k);
            }
        }

        __syncwarp();  // the stage is overwritten in the next iteration
    }
    flambeau_cp_async::wait_all();

    // Sum the warp partials, as one flat array per lane.
    constexpr int nsum = NMT * NCG * 4;
    float* sum_flat = &sum[0][0][0];
    float* buf = (float*) smem;
    __syncthreads();
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < nsum; ++i) {
            buf[((threadIdx.y - 1) * nsum + i) * 32 + threadIdx.x] = sum_flat[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int w = 0; w < NWARPS - 1; ++w) {
#pragma unroll
        for (int i = 0; i < nsum; ++i) {
            sum_flat[i] += buf[(w * nsum + i) * 32 + threadIdx.x];
        }
    }

#pragma unroll
    for (int mt = 0; mt < NMT; ++mt) {
#pragma unroll
        for (int c = 0; c < NCG; ++c) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + NROWS * mt + flambeau_mma::tile_i16::get_i(l);
                const int j = c * NCOLS + flambeau_mma::tile_i16::get_j(l);
                if (i < nrows_x && j < ncols) {
                    dst[(size_t) j * stride_col_dst + i] = sum[mt][c][l];
                }
            }
        }
    }
}

}  // namespace mmq_few

// One entry per column-group count. Dynamic shared memory:
// NWARPS · NSTAGES · 16·NMT · sram_stride(UNIT_BYTES, units_per_stage) bytes,
// block (32, 4), grid ceil(nrows_x / (16·NMT)).
#define FLAMBEAU_MMQ_FEW_ENTRY(NAME, DEC, NCG, NMT)                                                 \
    extern "C" __global__ void __launch_bounds__(32 * mmq_few::NWARPS, 1)                         \
    flambeau_mmq_few_##NAME##_mma_c##NCG##_t##NMT(                                                \
        const char* __restrict__ x, const int4* __restrict__ y, float* __restrict__ dst,          \
        const int nunits, const int nrows_x, const int ncols, const int stride_row_x,             \
        const int stride_col_dst) {                                                               \
        mmq_few::mmq_few_rows<DEC, NCG, NMT>(x, y, dst, nunits, nrows_x, ncols, stride_row_x,     \
                                             stride_col_dst);                                     \
    }
