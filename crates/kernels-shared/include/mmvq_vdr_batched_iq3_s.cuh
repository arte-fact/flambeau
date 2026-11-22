#pragma once
// IQ3_S trait of the batched VDR MMVQ (`mmvq_vdr_batched.cuh`), in its own
// header so the other batched cells do not pull the IQ3_S / IQ2 grids.
// Backend-neutral: the grid, sign mask and dp4a come from the shared headers
// and the arch seam.
#include "mmvq_vdr_batched.cuh"
#include "mmvq_iq3_s.cuh"

namespace mmvq_vdr {

// IQ3_S, VDR 2 (`vec_dot_iq3_s_q8_1`, as `mmvq_iq3_s_vdr2_row`): 8 threads
// per super-block, one 32-weight sub-block each — 2 ints of `qs`, one `qh`
// byte and one int of `signs` give 8 signed grid ints, decoded once per row
// and dotted against the N columns.
struct Iq3S {
    typedef flambeau_block_iq3_s block_t;
    static constexpr int TPB = 8;
    static constexpr int YPB = 8;
    template<int N, int R>
    static __device__ __forceinline__ void step(
        const block_t* const (&bk)[R], const flambeau_block_q8_1* y, size_t ys, int lane,
        float (&acc)[R][N])
    {
        const int iqs = 2 * lane;
        int g[R][8];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned int qs_w[2] = {
                (unsigned int) load_int_b2((const int8_t*) bk[r]->qs, iqs + 0),
                (unsigned int) load_int_b2((const int8_t*) bk[r]->qs, iqs + 1),
            };
            const int qh = bk[r]->qh[iqs / 2];
            const unsigned int signs = (unsigned int) load_int_b2((const int8_t*) bk[r]->signs, iqs / 2);
            #pragma unroll
            for (int l0 = 0; l0 < 8; l0 += 2) {
                const unsigned int q0 = (qs_w[l0 / 4] >> (8 * (l0 % 4))) & 0xFF;
                const unsigned int q1 = (qs_w[l0 / 4] >> (8 * (l0 % 4 + 1))) & 0xFF;
                const unsigned int g0 = IQ3S_GRID[q0 | ((qh << (8 - l0)) & 0x100)];
                const unsigned int g1 = IQ3S_GRID[q1 | ((qh << (7 - l0)) & 0x100)];
                const unsigned int s  = (signs >> (4 * l0)) & 0xFF;
                const unsigned int m0 = iq3s_sign_mask(s & 0x0F);
                const unsigned int m1 = iq3s_sign_mask(s >> 4);
                g[r][l0]     = (int) ((g0 ^ m0) + (m0 & 0x01010101u));
                g[r][l0 + 1] = (int) ((g1 ^ m1) + (m1 & 0x01010101u));
            }
            const int scale = 1 + 2 * ((bk[r]->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
            d[r] = (float) bk[r]->d * (float) scale;
        }
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            const flambeau_block_q8_1* ya = y + c * ys + lane;
            int u[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) u[j] = load_int_b4(ya->qs, j);
            const float dy = (float) ya->d;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int sumi = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sumi = dp4a(g[r][j], u[j], sumi);
                acc[r][c] += d[r] * dy * (float) sumi;
            }
        }
    }
};

}  // namespace mmvq_vdr
