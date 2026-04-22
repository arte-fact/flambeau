// indexed_moe_mmq_q4_k_gate_up_turbo — V2.14.c indexed-MoE port of the
// V2.14.b turbo-style 4-warp LDS-tiled Q4_K MMQ.
//
// Two changes vs the dense `mmq_q4_K_turbo` kernel:
//   1. Weight pointer is expert-indexed.
//      All MMQ_X slots in a block share the same expert (V2.6.a padded-sort
//      invariant). Resolve `expert = expert_ids[sorted_pair_idx_padded[tile_n]]`
//      once per block, then address
//        gate_w[expert * n_rows * n_sb_per_row + row * n_sb_per_row + kb0]
//      as usual.
//   2. Y activation is per-pair gathered, not contiguous along col axis.
//      Dense turbo loads `tile_y[l] = by0[l]` with by0 pointing at the mmq_x
//      col slice after the baked `tile_n * sz_mmq_int` offset. For indexed
//      MoE the "cols" are pair-indices from sorted_pair_idx_padded, NOT
//      contiguous in memory. Each tile_y[l] load now computes
//        pair = slot_pair[l / MMQ_TILE_Y_K_LDS]
//        tile_y[l] = y_int[big_block * n_pairs * sz + pair * sz + (l % MMQ_TILE_Y_K_LDS)]
//      The 2-wave indirect gather adds some overhead vs the dense contiguous
//      load, but the LDS tile reuse across 128 rows × 8 dp4a columns per
//      vec_dot dwarfs it.
//
// Dual outputs (gate + up): both weight matrices share the same activation.
// Per-block LDS holds gate's tile_x AND up's tile_x side-by-side (2× LDS
// budget for x), loaded in sequence per K-iter. Each vec_dot pass runs twice
// (once with gate tile, once with up tile) against the same tile_y.
//
// Padding slots (V2.6.a): tail slots in each expert range repeat the last
// real pair_idx → redundant compute + duplicate writeback, no branch.
//
// Launch:
//   grid  = (⌈n_rows / MMQ_Y⌉, ⌈padded_total / MMQ_X⌉, 1)
//   block = (64, 4, 1) = 256 threads = 4 warps × 64
//   shared = tile_y (MMQ_X×MMQ_TILE_Y_K_LDS)
//          + 2 × (TXS_QS + TXS_DM + TXS_SC) for gate_x + up_x
//
// Activation format: `block_q8_1_mmq` (DS4). Must be pre-quantised via
// `quantize_q8_1_mmq` on an F32 activation (F16 input: cast then quantise).

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define MMQ_ITER_K       256
#define MMQ_TILE_NE_K    32
#define QR4_K            2
#define QI4_K            32
#define VDR_Q4_K_Q8_1_MMQ 8
#define QK_K             256
#define QK8_1_FLAMBEAU   32
#define QI8_1            8
#define MMQ_TILE_Y_K_LDS (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)   // 36

// MMQ_X=8 matches V2.6.a padded-sort alignment. Turbo dense uses 16 but the
// indexed-MoE path's per-expert padding is pad-to-8 — using 16 would read
// uninitialised sorted_pair_idx_padded slots for the 9th-16th entries in a
// block straddling an expert boundary, producing OOB Y loads (HIP 700).
#define MMQ_X 8
#define MMQ_Y 128
#define NWARPS 4
#define BLOCK_THREADS (WARP_SIZE * NWARPS)   // 256

#define TXS_QS (MMQ_Y * MMQ_TILE_NE_K + MMQ_Y)           // 4224 ints
#define TXS_DM (MMQ_Y * MMQ_TILE_NE_K / QI4_K)            // 128 half2 slots
#define TXS_SC (MMQ_Y * MMQ_TILE_NE_K / 8 + MMQ_Y / 8)    // 528 ints
#define TILE_X_TOTAL (TXS_QS + TXS_DM + TXS_SC)           // 4880 ints per weight tile

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m,
    const __half2 & dm4, const __half2 * __restrict__ ds8
) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR4_K * VDR_Q4_K_Q8_1_MMQ / QI8_1; ++i) {
        int sumi_d = 0;
        #pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * QI8_1 + j], sumi_d);
        }
        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i];
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// Helper: cooperative load of one Q4_K super-block for MMQ_Y rows into an
// LDS tile (x_qs, x_dm, x_sc) following turbo's load_tiles_q4_K DP4A branch.
// `w` must already be offset to `+ expert * n_rows * n_sb_per_row`.
static __device__ __forceinline__ void load_x_q4_K_super_block(
    const flambeau_block_q4_K * __restrict__ w,
    const int kb0,
    const int tile_m,
    const int n_rows,
    const int n_sb_per_row,
    int * __restrict__ x_qs,
    __half2 * __restrict__ x_dm,
    int * __restrict__ x_sc
) {
    // qs load: 32 threads/row × 2 rows/warp × 4 warps → 256 threads cover 8 rows/iter × 16 iters = 128 rows
    {
        constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);    // 32
        constexpr int nrows           = WARP_SIZE / threads_per_row;  // 2
        const int txi = threadIdx.x % threads_per_row;

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
            const int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const flambeau_block_q4_K* bxi = w + row * n_sb_per_row + kb0;
                    const int qs0 = ((const int*) bxi->qs)[txi];
                    x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
                }
            }
        }
    }

    // dm load: one half2 per row; 256 threads redundantly write 128 slots
    {
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * WARP_SIZE) {
            const int i = (i0 + threadIdx.y * WARP_SIZE + threadIdx.x) % MMQ_Y;
            const int row = tile_m + i;
            if (row < n_rows) {
                const flambeau_block_q4_K* bxi = w + row * n_sb_per_row + kb0;
                const __half* dm_ptr = (const __half*) bxi;
                x_dm[i] = __halves2half2(dm_ptr[0], dm_ptr[1]);
            }
        }
    }

    // scales pass: 4 ksc positions per row × 128 rows = 512 writes
    {
        constexpr int rows_per_warp_sc = WARP_SIZE / 4;    // 16
        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_sc) {
            const int i_raw = i0 + threadIdx.y * rows_per_warp_sc + threadIdx.x / (MMQ_TILE_NE_K / 8);
            const int i = i_raw % MMQ_Y;
            const int ksc = threadIdx.x % (MMQ_TILE_NE_K / 8);
            const int row = tile_m + i;
            if (row < n_rows) {
                const flambeau_block_q4_K* bxi =
                    w + row * n_sb_per_row + kb0
                    + (threadIdx.x % (MMQ_TILE_NE_K / 8)) / (QI4_K / 8);
                const int* scales = (const int*) bxi->scales;
                const int scales8 = unpack_scales_q45_K(scales, ksc);
                x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = scales8;
            }
        }
    }
}

// Helper: one DP4A vec_dot pass against (x_qs, x_dm, x_sc, tile_y) at k00.
// Accumulates into `sum[]`. Same indexing as dense turbo.
static __device__ __forceinline__ void vec_dot_pass(
    const int * __restrict__ x_qs,
    const __half2 * __restrict__ x_dm,
    const int * __restrict__ x_sc,
    const int * __restrict__ tile_y,
    float * __restrict__ sum,
    const int k00
) {
    const int   * y_qs = tile_y + 4;
    const __half2 * y_ds = (const __half2 *) tile_y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
            const int j = j0 + threadIdx.y;

            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i < MMQ_Y) {
                    const uint8_t* sc = (const uint8_t*) &x_sc[
                        i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32
                    ] + 2 * (k01 / 16);

                    sum[j0 / NWARPS * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE) + i0 / WARP_SIZE]
                      += vec_dot_q4_K_q8_1_impl_mmq(
                        &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2],
                        &y_qs[j * MMQ_TILE_Y_K_LDS + k01],
                        sc, sc + 8,
                        x_dm[i],
                        &y_ds[j * MMQ_TILE_Y_K_LDS + k01 / QI8_1]);
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(BLOCK_THREADS, 1)
void flambeau_indexed_moe_mmq_q4_k_gate_up_turbo_q8_1(
    const flambeau_block_q4_K * __restrict__ gate_w,
    const flambeau_block_q4_K * __restrict__ up_w,
    const flambeau_block_q8_1_mmq * __restrict__ y_mmq,
    const int * __restrict__ expert_ids,
    const int * __restrict__ sorted_pair_idx_padded,
    const int * __restrict__ padded_offsets,
    float * __restrict__ gate_out,
    float * __restrict__ up_out,
    const int n_rows,          // inter (intermediate size)
    const int n_tokens,        // hidden activation row count (Y layout is per-token, not per-pair)
    const int top_k,           // pair = token * top_k + slot
    const int n_sb_per_row,    // n_sb per row (= hidden / QK_K) for gate_w / up_w (both same)
    const int n_experts
) {
    extern __shared__ int lds[];
    // LDS layout:
    //   tile_y          : MMQ_X * MMQ_TILE_Y_K_LDS ints              (576)
    //   tile_x_gate     : TILE_X_TOTAL ints                           (4880)
    //   tile_x_up       : TILE_X_TOTAL ints                           (4880)
    //   Total                                                         10336 ints = 41344 B
    int * tile_y    = lds;
    int * tile_x_gate = tile_y + MMQ_X * MMQ_TILE_Y_K_LDS;
    int * tile_x_up   = tile_x_gate + TILE_X_TOTAL;

    int   * xg_qs = tile_x_gate;
    __half2 * xg_dm = (__half2 *) (xg_qs + TXS_QS);
    int   * xg_sc = (int *) (xg_dm + TXS_DM);

    int   * xu_qs = tile_x_up;
    __half2 * xu_dm = (__half2 *) (xu_qs + TXS_QS);
    int   * xu_sc = (int *) (xu_dm + TXS_DM);

    const int tile_m = blockIdx.x * MMQ_Y;
    const int tile_n = blockIdx.y * MMQ_X;

    // Early-exit blocks past the padded total.
    __shared__ int padded_total_shared;
    if (threadIdx.x == 0 && threadIdx.y == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    // Per-block expert lookup (V2.6.a invariant: all MMQ_X slots same expert).
    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert     = expert_ids[first_pair];

    // Cache per-slot pair_idx (for output write) and token (for Y gather).
    __shared__ int slot_pair[MMQ_X];
    __shared__ int slot_token[MMQ_X];
    if (threadIdx.y == 0 && threadIdx.x < MMQ_X) {
        const int pair = sorted_pair_idx_padded[tile_n + threadIdx.x];
        slot_pair[threadIdx.x]  = pair;
        slot_token[threadIdx.x] = pair / top_k;  // token index (Y layout is per-token for gate/up)
    }
    __syncthreads();

    const flambeau_block_q4_K * gate_w_expert = gate_w + (size_t) expert * n_rows * n_sb_per_row;
    const flambeau_block_q4_K * up_w_expert   = up_w   + (size_t) expert * n_rows * n_sb_per_row;

    // DS4 activation: 128 elements per MMQ block. For a Q4_K super-block (256
    // elements) that's 2 MMQ blocks per kb0 iter.
    constexpr int sz_mmq_int = (int) (sizeof(flambeau_block_q8_1_mmq) / sizeof(int));  // 36
    const int * y_int = (const int *) y_mmq;

    // Accumulators: 2 × n_sum (one set for gate, one for up).
    constexpr int n_sum = (MMQ_X / NWARPS) * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE);  // 8
    float sum_gate[n_sum];
    float sum_up  [n_sum];
    #pragma unroll
    for (int s = 0; s < n_sum; ++s) { sum_gate[s] = 0.0f; sum_up[s] = 0.0f; }

    for (int kb0 = 0; kb0 < n_sb_per_row; ++kb0) {
        // Load gate X + up X for this super-block.
        load_x_q4_K_super_block(gate_w_expert, kb0, tile_m, n_rows, n_sb_per_row, xg_qs, xg_dm, xg_sc);
        load_x_q4_K_super_block(up_w_expert,   kb0, tile_m, n_rows, n_sb_per_row, xu_qs, xu_dm, xu_sc);

        // Y LDS load pass 1 (big_block = kb0 * 2, first half of super-block).
        // Indirect gather by TOKEN (gate+up activations are shared across the
        // top_k slots of a token — per-token, not per-pair).
        {
            const int big_block = kb0 * (QK_K / (4 * QK8_1_FLAMBEAU));  // = kb0 * 2
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_LDS; l0 += NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_LDS) {
                    const int c_in_tile = l / MMQ_TILE_Y_K_LDS;
                    const int intra     = l % MMQ_TILE_Y_K_LDS;
                    const int token     = slot_token[c_in_tile];
                    // Layout: y_int[(big_block * n_tokens + token) * sz + intra]
                    tile_y[l] = y_int[((size_t) big_block * n_tokens + token) * sz_mmq_int + intra];
                }
            }
        }

        __syncthreads();

        vec_dot_pass(xg_qs, xg_dm, xg_sc, tile_y, sum_gate, /*k00=*/ 0);
        vec_dot_pass(xu_qs, xu_dm, xu_sc, tile_y, sum_up,   /*k00=*/ 0);

        __syncthreads();

        // Y LDS load pass 2 (big_block = kb0 * 2 + 1).
        {
            const int big_block = kb0 * (QK_K / (4 * QK8_1_FLAMBEAU)) + 1;  // = kb0 * 2 + 1
            #pragma unroll
            for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K_LDS; l0 += NWARPS * WARP_SIZE) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                if (l < MMQ_X * MMQ_TILE_Y_K_LDS) {
                    const int c_in_tile = l / MMQ_TILE_Y_K_LDS;
                    const int intra     = l % MMQ_TILE_Y_K_LDS;
                    const int token     = slot_token[c_in_tile];
                    tile_y[l] = y_int[((size_t) big_block * n_tokens + token) * sz_mmq_int + intra];
                }
            }
        }

        __syncthreads();

        vec_dot_pass(xg_qs, xg_dm, xg_sc, tile_y, sum_gate, /*k00=*/ MMQ_TILE_NE_K);
        vec_dot_pass(xu_qs, xu_dm, xu_sc, tile_y, sum_up,   /*k00=*/ MMQ_TILE_NE_K);

        __syncthreads();
    }

    // Writeback — dual outputs (gate + up).
    #pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j_local = j0 + threadIdx.y;
        const int pair    = slot_pair[j_local];

        #pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i < MMQ_Y) {
                const int row = tile_m + i;
                if (row < n_rows) {
                    const size_t out_idx = (size_t) pair * n_rows + row;
                    const int s = j0 / NWARPS * ((MMQ_Y + WARP_SIZE - 1) / WARP_SIZE) + i0 / WARP_SIZE;
                    gate_out[out_idx] = sum_gate[s];
                    up_out  [out_idx] = sum_up  [s];
                }
            }
        }
    }
}
