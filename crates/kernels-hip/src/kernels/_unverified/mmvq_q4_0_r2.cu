// mmvq_q4_0_r2 — V2.28.d NULL. Kept for reference under arch-rule-10.
//
// why: the candle P29 r2 pattern (scalar F32 FMA per lane, half-warp reduce)
// strictly loses on Q4_0's flat-block DP4A path. Measured Qwen3.6-35B-A3B
// -Q4_0 Mesh<4> decode: 47.58 → 37.78 tok/s (−21 %) AND seed-9419 argmax
// shifts from last_id=17 to last_id=709 (F32 re-accumulation-order noise).
// The r2 pattern wins on K-quants because sub-block min adjustments block
// DP4A bias correction; Q4_0's `(q-8)·y = dp4a(q,y) - 8·s_y` identity
// already gives DP4A a clean win in the single-row kernel. Moving to
// _unverified; single-row DP4A MMVQ stays default for Q4_0 decode.
//
// Not registered in KERNEL_STEMS; build.rs's non-recursive enumerate skips
// this dir.
//
// mmvq_q4_0_r2 — Q4_0 MMVQ, multi-row r2 DPP-reduce (candle P29 pattern).
//
// Mirrors `mmvq_q4_k_r2.cu`'s structure but for Q4_0's flat 32-element block
// (no super-block). 64 threads = one wave64, splits into two 32-lane halves;
// each half owns one output row. `blockIdx.x` indexes the row *pair*.
//
// Activation `y[b]` is shared across the two rows, so per-iter Q8_1 traffic
// is halved vs two single-row launches. This is the primary decode lever
// for Qwen3.6-35B-A3B-Q4_0 where decode ran at 76 % of llama.cpp in V2.27.
//
// Per-block work: each 32-lane half-warp covers one Q4_0 block (32 elements)
// in parallel — 1 element per lane, no DP4A. Q4_0 reconstruction is
// `y_real = d · (q - 8)`; we compute `(q - 8) · y_q8_dequant` per-lane and
// sum across the 32-lane half with `gfx906_half_warp_reduce_sum`.

#include "block_quant.cuh"
#include "gfx906.cuh"

extern "C" __global__ void flambeau_mmvq_q4_0_r2_q8_1(
    const flambeau_block_q4_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,                 // total output rows (may be odd — boundary handled)
    const int n_blocks_per_row        // K / 32
) {
    const int row_pair = blockIdx.x;
    const int lane     = threadIdx.x;           // 0..63
    const int row_hi   = lane >> 5;             // 0 → row R+0, 1 → row R+1
    const int lane_lo  = lane & 31;             // 0..31 — position within block

    const int row = row_pair * 2 + row_hi;
    if (row >= n_rows) return;                  // boundary: odd-row count

    const int byte_idx = lane_lo & 15;          // 0..15 — which of the 16 nibble bytes
    const int is_high  = lane_lo >> 4;          // 0 (elements 0..15) or 1 (elements 16..31)

    const flambeau_block_q4_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks_per_row; ++b) {
        const flambeau_block_q4_0* bk = xrow + b;
        const float d = (float) bk->d;

        const int byte_v = (int) bk->qs[byte_idx];
        const int raw_q  = is_high ? (byte_v >> 4) : (byte_v & 0x0F);
        // Q4_0 reconstruction: y_real = d · (q - 8). Fold the −8 into the
        // per-element multiply; no aggregate bias-correction identity needed
        // because the half-warp reduce sums over independent elements already.
        const float x_val = d * (float) (raw_q - 8);

        const flambeau_block_q8_1* ya = y + b;
        const float d_y = (float) ya->d;
        const int   qi  = (int) ya->qs[lane_lo];
        const float y_val = d_y * (float) qi;

        acc += x_val * y_val;
    }

    acc = gfx906_half_warp_reduce_sum(acc);

    if (lane_lo == 0) {
        dst[row] = acc;
    }
}
