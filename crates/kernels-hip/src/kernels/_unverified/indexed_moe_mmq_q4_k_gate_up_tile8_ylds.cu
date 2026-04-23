// indexed_moe_mmq_q4_k_gate_up_tile8_ylds — V2.24.b NULL. Kept for
// reference under the architectural-rule-10 "cfg(unverified)" policy.
//
// why: candle/turbo's Y-LDS staging pattern gave −41 % prefill on gfx906
// (689 → 405 tok/s at Qwen3.6-35B-A3B-UD-Q4_K_S Mesh<4> pp=512). The V2.6.b
// baseline's "redundant" Y re-reads are actually L1-hot broadcasts (64
// threads per block reading the same 8 Y int32s = one L1 line per sub).
// Adding DS_WRITE + DS_READ to that path strictly costs latency with no
// HBM savings. The candle 1.4× analysis referenced dense Q4_K MMQ (many
// cols/block, many reader waves/line) — doesn't apply to this MoE
// indexed kernel's shape (MMQ_X=8, single-wave block).
//
// Not registered in KERNEL_STEMS; not compiled-in unless build.rs is
// pointed at it. Left as a cautionary tale + a possible starting point
// for a future M-wide restructure.
//
// V2.24.b Y-LDS variant of V2.6.b's gate+up tile8 MMQ.
//
// Rationale: the V2.6.b kernel has all 64 threads in a wave64 redundantly
// re-reading the same Y bytes across 8 tile-cols × 8 sub-blocks per
// super-block iter. Per `ib`, each Q8_1 block (34 bytes) is read 64× from
// HBM/L1 — 139 KB of redundant Y traffic per super-block.
//
// Fix: cooperatively load all 64 Y blocks for this `ib` (8 slots × 8
// sub-blocks) into LDS once, then read DP4A operands from LDS (near-free).
// LDS budget per block: 2304 B (well under the 64 KB gfx906 cap).
//
// Inner math is byte-identical to V2.6.b — same Q4_K decode, same DP4A
// chain, same `sumi_y * d_y * dmin_sub` correction. Expected kernel-level
// savings: 64× reduction in Y-side HBM reads. Projected +5-15 %
// kernel time (Y is a minority of kernel wall time at MMQ_X=8; most wall
// time is Q4_K weight decode + DP4A ALU).
//
// Launch: identical to V2.6.b.
//   grid = (ceil(n_rows / 64), padded_total / 8, 1)
//   block = (64, 1, 1)
//
// Dispatch under a new impl_id so V2.6.b stays available as A/B baseline.

#include "block_quant.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef QK_K
#define QK_K 256
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#define MMQ_Y 64
#define TILE_N 8

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ __launch_bounds__(WARP_SIZE, 1)
void flambeau_indexed_moe_mmq_q4_k_gate_up_tile8_ylds_q8_1(
    const flambeau_block_q4_K* __restrict__ gate_w,
    const flambeau_block_q4_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    float*      __restrict__ gate_out,
    float*      __restrict__ up_out,
    const int n_rows,
    const int n_tokens,
    const int top_k,
    const int n_sb_per_row,
    const int n_experts
) {
    const int tile_m = blockIdx.x * WARP_SIZE;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    __shared__ int padded_total_shared;
    if (tid == 0) padded_total_shared = padded_offsets[n_experts];
    __syncthreads();
    if (tile_n >= padded_total_shared) return;

    const int row     = tile_m + tid;
    const bool row_ok = (row < n_rows);

    const int first_pair = sorted_pair_idx_padded[tile_n];
    const int expert = expert_ids[first_pair];

    // Per-slot (token, slot_idx) decode.
    int slot_token[TILE_N];
    int slot_out_idx[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int pair = sorted_pair_idx_padded[tile_n + c];
        const int t = pair / top_k;
        const int s = pair - t * top_k;
        slot_token[c] = t;
        slot_out_idx[c] = t * top_k + s;
    }

    constexpr int q8_per_super = QK_K / QK8_1;            // = 8

    // V2.24.b — Y-tile LDS. 64 Q8_1 blocks per super-block iter
    // (TILE_N=8 slots × 8 sub-blocks). Each block is 8 ints of qs + 1 f32 d.
    // Layout: y_qs_lds[slot][sub][int_idx], y_d_lds[slot][sub].
    __shared__ int   y_qs_lds[TILE_N * q8_per_super * 8];   // 512 ints = 2048 B
    __shared__ float y_d_lds [TILE_N * q8_per_super];       // 64 floats = 256 B

    // One load slot per thread: tid 0..63 covers all 64 (c, sub) pairs since
    // TILE_N * q8_per_super = 8 * 8 = 64 = WARP_SIZE.
    const int load_c   = tid / q8_per_super;
    const int load_sub = tid - load_c * q8_per_super;

    const int blocks_per_row_x = n_sb_per_row;

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        // --- Cooperative Y load: 64 threads × 1 (slot, sub) block each.
        {
            const flambeau_block_q8_1* by =
                &y[(size_t) slot_token[load_c] * (n_sb_per_row * q8_per_super) + ib * q8_per_super + load_sub];
            const int* y_packed = (const int*) by->qs;
            int*   qs_dst = &y_qs_lds[(load_c * q8_per_super + load_sub) * 8];
            float* d_dst  = &y_d_lds [ load_c * q8_per_super + load_sub ];
            #pragma unroll
            for (int j = 0; j < 8; ++j) qs_dst[j] = y_packed[j];
            *d_dst = (float) by->d;
        }
        // 1-wave block (WARP_SIZE=64, launch_bounds=1) — wave is inherently
        // synchronous on gfx906. A full __syncthreads here compiles to
        // s_barrier + lgkmcnt drain; skip it. The in-wave DS-write ordering
        // is sufficient to make the LDS writes visible to subsequent reads
        // inside the same wave.

        float g_d = 0.0f, g_dmin = 0.0f;
        float u_d = 0.0f, u_dmin = 0.0f;
        uint8_t g_sub_sc[8] = {0}, g_sub_m[8] = {0};
        uint8_t u_sub_sc[8] = {0}, u_sub_m[8] = {0};

        const flambeau_block_q4_K* gbx = nullptr;
        const flambeau_block_q4_K* ubx = nullptr;
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_sb_per_row + ib;
            gbx = &gate_w[w_row_off];
            ubx = &up_w[w_row_off];
            g_d    = (float) gbx->d;
            g_dmin = (float) gbx->dmin;
            u_d    = (float) ubx->d;
            u_dmin = (float) ubx->dmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                flambeau_q4k_scale_min(j, gbx->scales, &g_sub_sc[j], &g_sub_m[j]);
                flambeau_q4k_scale_min(j, ubx->scales, &u_sub_sc[j], &u_sub_m[j]);
            }
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int il   = sub >> 1;
            const int half = sub & 1;

            int g_v[8] = {0};
            int u_v[8] = {0};
            if (row_ok) {
                const int* g_ql_ptr = (const int*) (gbx->qs + 32 * il);
                const int* u_ql_ptr = (const int*) (ubx->qs + 32 * il);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gw = g_ql_ptr[j];
                    const int uw = u_ql_ptr[j];
                    g_v[j] = (half == 0) ? (gw & 0x0F0F0F0F) : ((gw >> 4) & 0x0F0F0F0F);
                    u_v[j] = (half == 0) ? (uw & 0x0F0F0F0F) : ((uw >> 4) & 0x0F0F0F0F);
                }
            }

            const float gd_sc = g_d    * (float) g_sub_sc[sub];
            const float gd_m  = g_dmin * (float) g_sub_m [sub];
            const float ud_sc = u_d    * (float) u_sub_sc[sub];
            const float ud_m  = u_dmin * (float) u_sub_m [sub];

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                // Read Y from LDS instead of HBM.
                const int*   y_packed = &y_qs_lds[(c * q8_per_super + sub) * 8];
                const float  d8       = y_d_lds [ c * q8_per_super + sub ];

                int sumi_gd = 0, sumi_ud = 0, sumi_y = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int y_j = y_packed[j];
                    sumi_gd = dp4a(g_v[j], y_j, sumi_gd);
                    sumi_ud = dp4a(u_v[j], y_j, sumi_ud);
                    sumi_y  = dp4a(0x01010101, y_j, sumi_y);
                }
                const float sumi_y_f = (float) sumi_y;
                sums_gate[c] += d8 * ((float) sumi_gd * gd_sc - sumi_y_f * gd_m);
                sums_up[c]   += d8 * ((float) sumi_ud * ud_sc - sumi_y_f * ud_m);
            }
        }

        // Ensure all threads are done reading y_*_lds before the next ib
        // overwrites it. Required because the load writes then-reads in the
        // same warp, but we have 64 threads reading each LDS slot so a
        // sync is needed before the next coop-load overwrites.
        // 1-wave block (WARP_SIZE=64, launch_bounds=1) — wave is inherently
        // synchronous on gfx906. A full __syncthreads here compiles to
        // s_barrier + lgkmcnt drain; skip it. The in-wave DS-write ordering
        // is sufficient to make the LDS writes visible to subsequent reads
        // inside the same wave.
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_out_idx[c] * n_rows + row;
        gate_out[out_idx] = sums_gate[c];
        up_out[out_idx]   = sums_up[c];
    }
}
