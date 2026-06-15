// mmq_q4_1_4warp_lds — 4-warp LDS-tiled MMQ for Q4_1 × Q8_1_MMQ.
// Port of candle's turbo Q4_1 MMQ kernel
// (/artefact/candle/candle-hip-kernels/src/mmq_turbo.cu:276-1563),
// which itself ports llamacpp-turbo's `mmq.cuh` DP4A path. The
// (MMQ_Y=128, MMQ_X=64, DS4 Q8_1 layout, 2-phase Y-load per K-iter)
// structure is needed for full perf — hand-rolling lands at ~32% of
// turbo wall-clock.
// Differences from candle's source:
// - No `MMQ_TURBO_EXPORT` macro-parameterised variants. We ship exactly
// one kernel: MMQ_X=64, need_check=false ("unchecked"). MMQ_X=8/16/32
// variants and the checked (need_check=true) variant land later if
// needed — right now the dispatch table pins m>=128 at MMQ_X=64.
// - No MoE variants (the gather-quantise path).
// - No L2 prefetch (D3) — pending S3-class follow-up.
// - Uses flambeau's block structs (flambeau_block_q4_1,
// flambeau_block_q8_1_mmq) rather than candle-local mirrors.
// Y tile input layout (from quantize_q8_1_mmq):
// `vy` is `block_q8_1_mmq` × (n_big_blocks, ncols_y) row-major.
// Each 144-B block holds 128 K-elements; the outer K-loop advances by
// `blocks_per_iter = MMQ_ITER_K / QK4_1 = 8` Q4_1 blocks = 256 K-elems
// = 2 MMQ big-blocks per iter.
// X tile input layout: `vx` is `block_q4_1` × (nrows_x, ncols_x/QK4_1)
// row-major. 20 B per block, as loaded from GGUF.
// Launch:
// grid = (ceil(nrows_x / MMQ_Y=128), ceil(ncols_y / MMQ_X=64))
// block = (WARP_SIZE=64, MMQ_NWARPS=4, 1) — 2D thread index
// shared_bytes = computed by host (see `ops/src/hip/qmatmul.rs`)

#include "block_quant.cuh"
#include "mmq_prefetch.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// --- MMQ geometry (mirrors candle mmq_turbo.cu:175-211) ---
#ifndef MMQ_Y
#define MMQ_Y 128
#endif
#ifndef MMQ_NWARPS
#define MMQ_NWARPS 4
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define MMQ_TILE_NE_K 32                                     // K elements per vec_dot call
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1) // = 36 ints/col per Y half-tile
#define MMQ_ITER_K    256                                    // K elements per outer iter
#define VDR_Q4_0_Q8_1_MMQ 4                                  // used by Q4_1 too (shared shape)

#ifndef QK4_1
#define QK4_1 32
#endif
#ifndef QR4_1
#define QR4_1 2
#endif
#define QI4_1 (QK4_1 / (4 * QR4_1))  // = 4 ints per Q4_1 block

#ifndef QK8_1
#define QK8_1 32
#endif
#ifndef QR8_1
#define QR8_1 1
#endif
#define QI8_1 (QK8_1 / (4 * QR8_1))  // = 8 ints per Q8_1 block

#define Q8_1_MMQ_BYTES 144
#define Q8_1_MMQ_INTS  (Q8_1_MMQ_BYTES / 4)  // = 36

// LDS sizing (DP4A path, from turbo MMQ_DP4A_TXS_Q4_1):
// x_qs: mmq_y * (MMQ_TILE_NE_K + 1) ints (33 per row, +1 bank pad)
// x_dm: mmq_y * (MMQ_TILE_NE_K/QI4_1) + mmq_y/QI4_1 half2 (8 + 32 per row)
#define X_QS_INTS (MMQ_Y * (MMQ_TILE_NE_K + 1))
#define X_DM_H2S  (MMQ_Y * (MMQ_TILE_NE_K / QI4_1) + MMQ_Y / QI4_1)

#define GGML_PAD(x, n) (((x) + ((n) - 1)) & ~((n) - 1))

// --- Helpers ---

static __device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    // Q4_1's qs is 4-byte aligned (after the 4-byte dm header), so a direct
    // int load is safe.
    return ((const int*) x)[i32];
}

static __device__ __forceinline__ int dp4a_sdot4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Port of turbo vecdotq.cuh:162-190 (vec_dot_q4_1_q8_1_impl<vdr=4>).
static __device__ __forceinline__ float q4_1_q8_1_dp4a_4(
    const int* v,            // 4 ints of Q4_1 qs (packed nibbles)
    const int* u,            // 8 ints of Q8_1 qs (DS4 layout)
    const __half2 dm4,       // Q4_1 (delta, min) packed as half2
    const __half2 ds8        // Q8_1 (d, Σxi) packed as half2
) {
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = dp4a_sdot4(vi0, u[2 * i + 0], sumi);
        sumi = dp4a_sdot4(vi1, u[2 * i + 1], sumi);
    }
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    return sumi * (dm4f.x * ds8f.x) + (dm4f.y * ds8f.y);
}

// --- Load X tile (Q4_1 nibbles + per-block (d, m) half2) ---

template <bool need_check>
static __device__ __forceinline__ void load_tiles_q4_1(
    const char* __restrict__ x,
    int*         __restrict__ x_qs,
    __half2*     __restrict__ x_dm,
    const int kbx0,
    const int i_max,
    const int kb_remaining,            // # of valid blocks in this K-iter (1..8)
    const int stride_row_x
) {
    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_1); // = 32
    constexpr int nrows           = WARP_SIZE / threads_per_row; // = 2
    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI4_1;
    const int kqsx = txi % QI4_1;
    const int kbx_safe = (kbx < kb_remaining) ? kbx : (kb_remaining - 1);

    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const flambeau_block_q4_1* bxi =
            (const flambeau_block_q4_1*) x + kbx0 + i * stride_row_x + kbx_safe;
        const int qs0 = get_int_b4(bxi->qs, kqsx);
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_1;  // = 8
    constexpr int rows_per_warp         = WARP_SIZE / blocks_per_tile_x_row; // = 8
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    const bool kbxd_valid = (kbxd < kb_remaining);
    const int kbxd_safe = kbxd_valid ? kbxd : 0;

    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = (i < i_max) ? i : i_max;
        }
        const flambeau_block_q4_1* bxi =
            (const flambeau_block_q4_1*) x + kbx0 + i * stride_row_x + kbxd_safe;
        // flambeau_block_q4_1 stores d, m as two separate fb_fp16_t (4 bytes).
        // Pack into a half2 to match candle's storage convention for the dot.
        const __half2 dm = kbxd_valid
            ? __floats2half2_rn((float) bxi->d, (float) bxi->m)
            : __floats2half2_rn(0.0f, 0.0f);
        x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + kbxd] = dm;
    }
}

// --- Vector dot (consumes one MMQ Y big-block at offset k00 ∈ {0, 32}) ---

template <int mmq_x>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int*     __restrict__ x_qs,
    const __half2* __restrict__ x_dm,
    const int*     __restrict__ tile_y,
    float*         __restrict__ sum,
    const int k00
) {
    const int*     y_qs = tile_y + 4;
    const __half2* y_ds = (const __half2*) tile_y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1 * VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;
        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
            const int j = j0 + threadIdx.y;
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2))
                               + (k01 / 2) % (QI8_1 / 2);

                const int4 vec0 = *((const int4*) &y_qs[j * MMQ_TILE_Y_K + kyqs]);
                const int4 vec1 = *((const int4*) &y_qs[j * MMQ_TILE_Y_K + kyqs + QI4_1]);
                int u[2 * VDR_Q4_0_Q8_1_MMQ];
                u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
                u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;

                const __half2 dm4 =
                    x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + k0 / (QR4_1 * QI4_1)];
                const __half2 ds8 = y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1];

                const int* v = &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / QR4_1];
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE] +=
                    q4_1_q8_1_dp4a_4(v, u, dm4, ds8);
            }
        }
    }
}

// --- Main kernel ---

template <int mmq_x, bool need_check>
static __device__ void mul_mat_q4_1_turbo_impl(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float*      __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int stride_col_y,
    const int stride_row_x,
    const int nrows_dst
) {
    extern __shared__ int shared_buf[];
    int*     tile_y = shared_buf;
    int*     x_qs   = tile_y + GGML_PAD(mmq_x * MMQ_TILE_Y_K, MMQ_NWARPS * WARP_SIZE);
    __half2* x_dm   = (__half2*) (x_qs + X_QS_INTS);

    const int it = blockIdx.x;
    const int jt = blockIdx.y;

    constexpr int sum_slots = mmq_x * MMQ_Y / (MMQ_NWARPS * WARP_SIZE);
    float sum[sum_slots];
    #pragma unroll
    for (int s = 0; s < sum_slots; ++s) sum[s] = 0.0f;

    const int* y_base = (const int*) vy + jt * mmq_x * Q8_1_MMQ_INTS;
    constexpr int blocks_per_iter = MMQ_ITER_K / QK4_1;  // = 8
    const int n_blocks_x = ncols_x / QK4_1;
    const int kb0_stop = (n_blocks_x + blocks_per_iter - 1)
                         / blocks_per_iter * blocks_per_iter;
    const int i_max = nrows_x - it * MMQ_Y - 1;

    // L2 prefetch: each K-iter issues global_load_dword
    // hints for the NEXT iter's Y-tile and X-tile. Loads are async —
    // the real cooperative tile loads one iter later hit warm
    // cachelines. Hides ~90 % of HBM latency when K is large enough
    // that compute is bandwidth-bound.
    constexpr int qk4_1_ints_per_block = QK4_1 / 8;   // = 4 (int32s of Q4_1 qs per block)
    constexpr int q4_1_block_bytes = 20;              // sizeof(flambeau_block_q4_1)

    for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        const int kb_remaining =
            (n_blocks_x - kb0 < blocks_per_iter) ? (n_blocks_x - kb0)
                                                 : blocks_per_iter;

        // --- Prefetch next iter's Y tile (first DS4 block) into L2 ---
        // Address = y_base + stride_col_y * ((kb0 + iter) / 4) * Q8_1_MMQ_INTS
        const int kb0_next = kb0 + blocks_per_iter;
        const bool prefetch_valid = (kb0_next < kb0_stop);
        int y_prefetch_dummy = 0;
        int x_prefetch_dummy = 0;
        if (prefetch_valid) {
            const int* by_next0 =
                y_base + (size_t) stride_col_y * (kb0_next / 4) * Q8_1_MMQ_INTS;
            y_prefetch_dummy = gfx906_prefetch_y_next(by_next0);
            // X tile for next iter: base + (it * MMQ_Y + lane) * stride_row_x
            // + kb0_next * sizeof(block_q4_1). Offset_x in bytes.
            const int64_t offset_x_bytes_next =
                ((int64_t) it * MMQ_Y * stride_row_x + kb0_next) * q4_1_block_bytes;
            x_prefetch_dummy = gfx906_prefetch_x_next(
                (const char*) vx,
                (int) offset_x_bytes_next,
                stride_row_x * q4_1_block_bytes);
        }

        load_tiles_q4_1<need_check>(
            (const char*) vx,
            x_qs, x_dm,
            it * MMQ_Y * stride_row_x + kb0,
            (need_check ? i_max : 0),
            kb_remaining,
            stride_row_x);

        // Consume the prefetch dummies — compiler can't elide the
        // global_load_dword once their return is used by v_mov_b32.
        if (prefetch_valid) {
            gfx906_prefetch_consume(y_prefetch_dummy);
            gfx906_prefetch_consume(x_prefetch_dummy);
        }

        const int* by0 =
            y_base + (size_t) stride_col_y * (kb0 / 4) * Q8_1_MMQ_INTS;
        #pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by0[l];
        }
        __syncthreads();

        vec_dot_q4_1_q8_1_dp4a<mmq_x>(x_qs, x_dm, tile_y, sum, 0);

        __syncthreads();

        const int* by1 =
            y_base + (size_t) stride_col_y * ((kb0 / 4) * Q8_1_MMQ_INTS + Q8_1_MMQ_INTS);
        #pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += MMQ_NWARPS * WARP_SIZE) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            tile_y[l] = by1[l];
        }
        __syncthreads();

        vec_dot_q4_1_q8_1_dp4a<mmq_x>(x_qs, x_dm, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += MMQ_NWARPS) {
        const int j = j0 + threadIdx.y;
        const int col_g = jt * mmq_x + j;
        if (col_g >= ncols_y) return;
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            const int row_g = it * MMQ_Y + i;
            if (need_check && row_g >= nrows_x) continue;
            dst[(size_t) col_g * nrows_dst + row_g] =
                sum[(j0 / MMQ_NWARPS) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE];
        }
    }
}

// --- Single kernel export: MMQ_X=64, need_check=false ---
// 2 waves/SIMD with 128 VGPR is the right config for this kernel.
// MMQ_X=32 (VGPR 116→84, waves/SIMD 2→3) and __launch_bounds__(256, 3)
// occupancy-override variants are both slower at realistic
// prefill shapes — extra VGPR lets the compiler unroll the DP4A chain
// and hold sumi accumulators in registers, and dropping below that
// hurts per-thread throughput more than occupancy gains can recover.

extern "C" __global__
__launch_bounds__(WARP_SIZE * MMQ_NWARPS, 2)
void flambeau_mmq_q4_1_4warp_lds_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float*      __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int stride_col_y,
    const int stride_row_x,
    const int nrows_dst
) {
    mul_mat_q4_1_turbo_impl</*mmq_x=*/64, /*need_check=*/false>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y,
        stride_col_y, stride_row_x, nrows_dst);
}

