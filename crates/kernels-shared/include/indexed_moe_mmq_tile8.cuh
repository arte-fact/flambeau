#pragma once
// Generic register-tiled prefill MoE MMQ skeleton (tile8) for the simple block
// quants that ride `quant_traits<Q>` (Q8_0/Q4_0/Q4_1/Q5_0/Q5_1). MMQ_Y=64 weight
// rows (one per thread) × TILE_N=8 sorted slot-columns per block: each thread
// decodes its row's weight block to registers (`quant_traits<Q>::mmq_decode`,
// which already uses `load_int_b2`) and runs the per-block dot+scale
// (`mmq_partial`) against 8 sorted slot activations — NO LDS, NO cross-thread
// comms, so the block height is a logical tile, not a hardware warp (correct on
// wave64 AND wave32). Experts are pre-sorted + padded-to-8 (`sorted_pair_idx_padded`
// / `padded_offsets`) so all 8 slots in a block share one expert. `gate_up` fuses
// two weights sharing one per-token activation read; `down` is the single-weight,
// per-pair sibling. `OutT` is the destination dtype.
#include "quant_traits.cuh"

template<class Q, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_gate_up_tile8_row(
    const typename quant_traits<Q>::block_t* __restrict__ gate_w,
    const typename quant_traits<Q>::block_t* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ gate_out,
    OutT* __restrict__ up_out,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row, int n_experts)
{
    using T = quant_traits<Q>;
    constexpr int MMQ_Y  = 64;
    constexpr int TILE_N = 8;

    const int tile_m = blockIdx.x * MMQ_Y;
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

    float sums_gate[TILE_N];
    float sums_up[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) { sums_gate[c] = 0.0f; sums_up[c] = 0.0f; }

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        typename T::mmq_weight gw{};
        typename T::mmq_weight uw{};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            gw = T::mmq_decode(gate_w[w_row_off]);
            uw = T::mmq_decode(up_w[w_row_off]);
        }
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1& by =
                y[(size_t) slot_token[c] * n_blocks_per_row + ib];
            sums_gate[c] += T::mmq_partial(gw, by);
            sums_up[c]   += T::mmq_partial(uw, by);
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_out_idx[c] * n_rows + row;
        gate_out[out_idx] = sums_gate[c];
        up_out[out_idx]   = sums_up[c];
    }
    (void) n_tokens;
}

template<class Q, class OutT = float>
static __device__ __forceinline__ void indexed_moe_mmq_down_tile8_row(
    const typename quant_traits<Q>::block_t* __restrict__ w,
    const flambeau_block_q8_1* __restrict__ y,
    const int* __restrict__ expert_ids,
    const int* __restrict__ sorted_pair_idx_padded,
    const int* __restrict__ padded_offsets,
    OutT* __restrict__ dst,
    int n_rows, int n_tokens, int top_k, int n_blocks_per_row, int n_experts)
{
    using T = quant_traits<Q>;
    constexpr int MMQ_Y  = 64;
    constexpr int TILE_N = 8;

    const int tile_m = blockIdx.x * MMQ_Y;
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

    int slot_pair[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        slot_pair[c] = sorted_pair_idx_padded[tile_n + c];
    }

    float sums[TILE_N];
    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) sums[c] = 0.0f;

    for (int ib = 0; ib < n_blocks_per_row; ++ib) {
        typename T::mmq_weight wt{};
        if (row_ok) {
            const size_t w_row_off = ((size_t) expert * n_rows + row) * n_blocks_per_row + ib;
            wt = T::mmq_decode(w[w_row_off]);
        }
        #pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
            const flambeau_block_q8_1& by =
                y[(size_t) slot_pair[c] * n_blocks_per_row + ib];
            sums[c] += T::mmq_partial(wt, by);
        }
    }

    if (!row_ok) return;

    #pragma unroll
    for (int c = 0; c < TILE_N; ++c) {
        const size_t out_idx = (size_t) slot_pair[c] * n_rows + row;
        dst[out_idx] = sums[c];
    }
    (void) top_k; (void) n_tokens;
}
