// mmq_q6_K_wave64 — wave64 MMQ for Q6_K × Q8_1 activation.
// candle has no Q6_K MMQ, so this kernel is written fresh using
// the same wave64 / MMQ_Y=64 / TILE_N=8 / DP4A structure as the Q5_K
// and Q4_K wave64 ports. The DP4A inner follows our existing
// `mmvq_q6_k_dp4a.cu` port of llama.cpp `vec_dot_q6_K_q8_1_impl_mmvq`,
// restructured to iterate serially within one thread (per row) rather
// than cooperatively across 64 lanes.
// Why: `mmq_q6_K_4warp` is an F32-tile placeholder. Without a
// first-class Q6_K MMQ, prefill on Qwen3.6-35B falls through to per-row
// Q6_K MMVQ at m ≥ 128 on `ffn_down_exps` (Q6_K in UD-Q4_K_S quant) —
// 12 k kernel launches per pp512, analogous to the Q5_K pre-gap.
// Q6_K layout (from block_quant.cuh):
// ql[128] = 128 bytes of low nibbles (4 bits per element × 256)
// qh[64] = 64 bytes of high bits (2 bits per element × 256)
// scales[16] = 16 signed int8 scales, one per 16 elements
// d = super-block fp16 scale
// Per-element reconstruction (matches llama.cpp vec_dot_q6_K_q8_1_impl_mmvq):
// q = (ql_nibble | (qh_bits << 4)) - 32 → signed int8 in [-32, 31]
// Sub-block decomposition (mirrors our `mmvq_q6_k.cu`'s h/q_idx/pos
// scheme, iterated serially inside one thread rather than cooperatively
// across lanes):
// sub ∈ [0, 8) — one per Q8_1 activation block
// h = sub / 4 — which 128-element half of the super-block
// q_idx = sub % 4 — 0..3, picks which ql-nibble + qh-bit shift
// For a sub-block, the 32 Q6_K weights cluster into 8 packed int32s, each
// holding 4 signed int8 Q6_K values (bytewise -32 shift applied). This
// packs as 4 × ql-bytes together with 4 × qh-bit-pairs shifted into the
// upper nibble.
// Args (8 scalar + 3 ptr — same signature as Q4_K / Q5_K wave64):
// vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst
// Correctness oracle: CPU dequant(weights) × Q8_1-roundtrip(act), via
// `crates/bench/src/sweep_mmq.rs` new `Q6KWave64` variant.

#include "block_quant.cuh"
#include "mmq_prefetch.cuh"
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
void flambeau_mmq_q6_K_wave64_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float*      __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst
) {
    const int tile_m = blockIdx.x * WARP_SIZE;
    const int tile_n = blockIdx.y * TILE_N;
    const int tid    = threadIdx.x;

    const int row     = tile_m + tid;
    const bool row_ok = (row < nrows_x);

    const flambeau_block_q6_K* x = (const flambeau_block_q6_K*) vx;
    const flambeau_block_q8_1* y = (const flambeau_block_q8_1*) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int q8_per_super = QK_K / QK8_1;  // = 8

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < blocks_per_row_x; ++ib) {
        float super_d = 0.0f;
        int8_t sc_buf[16] = {0};

        const flambeau_block_q6_K* bx = nullptr;
        if (row_ok) {
            bx = &x[(size_t) row * blocks_per_row_x + ib];
            super_d = (float) bx->d;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                sc_buf[j] = bx->scales[j];
            }
        }

        #pragma unroll
        for (int sub = 0; sub < q8_per_super; ++sub) {
            const int h     = sub >> 2;     // 0 or 1
            const int q_idx = sub & 3;      // 0..3
            const int qh_shift = 2 * q_idx;

            // Decode 32 Q6_K weights of this sub-block into 8 packed int32s.
            // Each int32 holds 4 UNSIGNED Q6_K values in [0, 63] — byte 7
            // always zero (upper 2 bits free). We DO NOT apply the -32
            // shift by byte-wise subtract because `(unsigned)x - const`
            // has a borrow-chain that corrupts bytes when any byte < 32.
            // Instead, compensate after DP4A via the identity
            // (raw - 32) · y = (raw · y) - 32 · sum_y
            // — the same pattern Q4_K / Q5_K use for their (d, m) bias.
            // ql layout for this sub-block (32 consecutive bytes):
            // base = 64h + (q_idx & 1) * 32
            // qh layout (32 consecutive bytes, shared shift-pattern across
            // q_idx=0..3 at the same `h`):
            // base = 32h
            int v[8] = {0};
            if (row_ok) {
                const int ql_base = 64 * h + ((q_idx & 1) ? 32 : 0);
                const int qh_base = 32 * h;
                const int* ql_words = (const int*) (bx->ql + ql_base);  // 8 ints
                const int* qh_words = (const int*) (bx->qh + qh_base);  // 8 ints

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int ql_word = ql_words[j];
                    const int qh_word = qh_words[j];

                    // 4 low or high nibbles as 4 positioned bytes.
                    const int nib4 = (q_idx < 2)
                        ? ((ql_word >> 0) & 0x0F0F0F0F)
                        : ((ql_word >> 4) & 0x0F0F0F0F);

                    // 2 high bits per byte, from qh bits [qh_shift, qh_shift+1]
                    // to result bits [4, 5] per byte. ((x >> s) << 4) & 0x30
                    // is bit-equivalent to (x << (4-s)) & 0x30 at the masked
                    // positions; borrow issues don't apply because the mask
                    // keeps only bits 4-5, which receive bits s..s+1 of the
                    // same byte with no cross-byte pollution.
                    const int hi4 = ((qh_word >> qh_shift) << 4) & 0x30303030;

                    v[j] = nib4 | hi4;  // unsigned, per-byte in [0, 63]
                }
            }

            // Scales apply per 16 elements. Sub-block has 32 elements = 2
            // scale positions. Within-super-block scale index base:
            // scale_idx = 8h + 2*q_idx + lsub (lsub ∈ {0, 1})
            const int sc_a = (int) sc_buf[8 * h + 2 * q_idx + 0];  // j ∈ 0..3
            const int sc_b = (int) sc_buf[8 * h + 2 * q_idx + 1];  // j ∈ 4..7

            #pragma unroll
            for (int c = 0; c < TILE_N; ++c) {
                const int col = tile_n + c;
                if (col >= ncols_y) break;

                const flambeau_block_q8_1* by =
                    &y[(size_t) col * blocks_per_col_y + ib * q8_per_super + sub];
                const float d8 = (float) by->d;

                const int* y_packed = (const int*) by->qs;

                // sumi_raw: dot(raw_weights, y). raw is unsigned [0,63]
                // which dp4a reads as signed int8 [0,63] — still correct.
                // sumi_y : dot(1,1,1,1, y) = Σ_i y_i, needed for the -32
                // bias correction.
                int sumi_a = 0, sumi_b = 0;
                int sumi_y_a = 0, sumi_y_b = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sumi_a = dp4a(v[j], y_packed[j], sumi_a);
                    sumi_y_a = dp4a(0x01010101, y_packed[j], sumi_y_a);
                }
                #pragma unroll
                for (int j = 4; j < 8; ++j) {
                    sumi_b = dp4a(v[j], y_packed[j], sumi_b);
                    sumi_y_b = dp4a(0x01010101, y_packed[j], sumi_y_b);
                }

                // (raw - 32) · y = raw · y - 32 · Σ y
                const int corrected_a = sumi_a - 32 * sumi_y_a;
                const int corrected_b = sumi_b - 32 * sumi_y_b;

                sums[c] += super_d * d8 *
                    ((float)(sc_a * corrected_a + sc_b * corrected_b));
            }
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const int col = tile_n + c;
        if (col < ncols_y && row < nrows_dst) {
            dst[(size_t) col * nrows_dst + row] = sums[c];
        }
    }
}
