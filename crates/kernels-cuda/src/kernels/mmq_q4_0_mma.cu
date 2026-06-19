// Q4_0 dense-prefill MMQ on sm_80+ int8 tensor cores (mma.sync m16n8k32).
// CUDA-only counterpart of the dp4a mmq_q4_0_4warp_lds; the -8 zero-point folds
// into the A dequant (__vsubss4) so no Σx bias term is needed. Weights and the
// DS4 144-B Q8_1 activation are staged to shared in canonical [row][K/4 int32]
// order, then consumed by the mma fragments (flambeau_mma): A via ldmatrix (the
// 16x8 input fragment differs from a row-major read), B via load_generic.
// Float-accumulates d_x·d_y·Σ(q-8)·q_y per 32-K Q4_0 block. Requires
// ncols_x % 128 == 0 (the dispatch k%128 guard routes the remainder to MMVQ).
#include "sm_80.cuh"
#include "block_quant.cuh"
#include "quant_traits.cuh"
#include "mma_int8.cuh"

template<int MMQ_Y, int MMQ_X, int WN, int WM>
static __device__ __forceinline__ void mmq_q4_0_mma_row(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    int ncols_x, int nrows_x, int ncols_y,
    int stride_col_y, int stride_row_x, int nrows_dst)
{
    constexpr int NWARPS   = WN * WM;
    constexpr int NTHREADS = 32 * NWARPS;
    constexpr int K_INTS   = 8;     // 32 K-elements / 4 = 8 int32 per row
    constexpr int MMQ_INTS = 36;    // 144-B activation block / 4
    constexpr int tnN = (MMQ_Y / WN) / 16;
    constexpr int tnM = (MMQ_X / WM) / 8;

    __shared__ __align__(16) int a_sh[MMQ_Y * K_INTS];
    __shared__ int   b_sh[MMQ_X * K_INTS];
    __shared__ float dx_sh[MMQ_Y];
    __shared__ float dy_sh[MMQ_X];

    const int it  = blockIdx.x;
    const int jt  = blockIdx.y;
    const int tx  = threadIdx.x;          // lane 0..31
    const int ty  = threadIdx.y;          // warp 0..NWARPS-1
    const int tid = ty * 32 + tx;

    const int n_blocks = ncols_x / QK4_0;
    const flambeau_block_q4_0* x = (const flambeau_block_q4_0*) vx;
    const int* y = (const int*) vy;

    const int wn = ty % WN;
    const int wm = ty / WN;
    const int row_base = wn * (MMQ_Y / WN);
    const int tok_base = wm * (MMQ_X / WM);

    float acc[tnN][tnM][4];
    #pragma unroll
    for (int nn = 0; nn < tnN; ++nn)
        #pragma unroll
        for (int mm = 0; mm < tnM; ++mm)
            #pragma unroll
            for (int l = 0; l < 4; ++l) acc[nn][mm][l] = 0.0f;

    for (int kb = 0; kb < n_blocks; ++kb) {
        __syncthreads();

        for (int r = tid; r < MMQ_Y; r += NTHREADS) {
            int row_g = it * MMQ_Y + r;
            if (row_g >= nrows_x) row_g = nrows_x - 1;
            const flambeau_block_q4_0* bxi = x + (size_t) row_g * stride_row_x + kb;
            dx_sh[r] = (float) bxi->d;
            #pragma unroll
            for (int jc = 0; jc < 4; ++jc) {
                const int q = load_int_b2((const int8_t*) bxi->qs, jc);
                a_sh[r * K_INTS + jc]     = __vsubss4((q >> 0) & 0x0F0F0F0F, 0x08080808);
                a_sh[r * K_INTS + jc + 4] = __vsubss4((q >> 4) & 0x0F0F0F0F, 0x08080808);
            }
        }

        const int bigblock = kb / 4;
        const int sub = kb % 4;
        for (int idx = tid; idx < MMQ_X * K_INTS; idx += NTHREADS) {
            const int token = idx / K_INTS;
            const int jc = idx % K_INTS;
            int tok_g = jt * MMQ_X + token;
            if (tok_g >= ncols_y) tok_g = ncols_y - 1;
            const size_t base = ((size_t) bigblock * stride_col_y + tok_g) * MMQ_INTS;
            b_sh[token * K_INTS + jc] = y[base + 4 + sub * 8 + jc];
            if (jc == 0) dy_sh[token] = (float) ((const fb_fp16_t*) (y + base))[2 * sub];
        }
        __syncthreads();

        flambeau_mma::tile_i16 af[tnN];
        flambeau_mma::tile_i8  bf[tnM];
        #pragma unroll
        for (int nn = 0; nn < tnN; ++nn)
            flambeau_mma::load_ldmatrix(af[nn], &a_sh[(row_base + nn * 16) * K_INTS], K_INTS);
        #pragma unroll
        for (int mm = 0; mm < tnM; ++mm)
            flambeau_mma::load_generic(bf[mm], &b_sh[(tok_base + mm * 8) * K_INTS], K_INTS);

        #pragma unroll
        for (int nn = 0; nn < tnN; ++nn) {
            #pragma unroll
            for (int mm = 0; mm < tnM; ++mm) {
                flambeau_mma::tile_i16 c;
                flambeau_mma::mma(c, af[nn], bf[mm]);
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int r = row_base + nn * 16 + flambeau_mma::tile_i16::get_i(l);
                    const int t = tok_base + mm * 8  + flambeau_mma::tile_i16::get_j(l);
                    acc[nn][mm][l] += (float) c.x[l] * dx_sh[r] * dy_sh[t];
                }
            }
        }
    }

    #pragma unroll
    for (int nn = 0; nn < tnN; ++nn) {
        #pragma unroll
        for (int mm = 0; mm < tnM; ++mm) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int r = row_base + nn * 16 + flambeau_mma::tile_i16::get_i(l);
                const int t = tok_base + mm * 8  + flambeau_mma::tile_i16::get_j(l);
                const int row_g = it * MMQ_Y + r;
                const int tok_g = jt * MMQ_X + t;
                if (row_g < nrows_x && tok_g < ncols_y)
                    dst[(size_t) tok_g * nrows_dst + row_g] = acc[nn][mm][l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void flambeau_mmq_q4_0_mma_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y,
    const int stride_col_y, const int stride_row_x, const int nrows_dst)
{
    mmq_q4_0_mma_row<128, 64, 8, 1>(vx, vy, dst, ncols_x, nrows_x, ncols_y,
                                    stride_col_y, stride_row_x, nrows_dst);
}
