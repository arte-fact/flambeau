// Device-position split-K decode attention chunk kernel, Q8_0 KV. CUDA-only
// (architectural rule 5 keeps the shared chunk body in attention_q8_kv.cuh
// backend-neutral; this thin wrapper reads `position` from a device buffer so a
// captured decode graph needs no per-token `n_tokens_kv` / `chunk_base` / K-V
// pointer rebind). Q8_0 fork of attention_decode_f16_splitk_dpos: identical
// regime derivation, only K/V are typed `flambeau_block_q8_0*` and the SlideBand
// byte offset uses the Q8 row stride passed from the host. The combine pass is
// position-free and reuses the shared attention_decode_q8_kv_splitk combine.
//
// The per-capture regime is BAKED as (window, ring_depth, slide_row_stride);
// only `position` varies within a bucket. On-device the wrapper reproduces the
// exact (eff n_tokens_kv, chunk_base, kernel window, cache pointer) arguments
// the host composite would pass in each regime, so replay is bit-identical to
// the non-graph path:
//   * slide_row_stride > 0 : SWA in-band slid — eff = window, chunk_base = 0,
//     kernel_window = 0, K/V base slid by (position+1-window)*slide_row_stride
//     bytes (the Q8 row stride in BYTES).
//   * ring_depth > 0       : SWA ring — eff = position+1, chunk_base =
//     eff-window, kernel_window = window, rows addressed mod ring_depth.
//   * otherwise            : global / pre-window — eff = position+1,
//     chunk_base = 0, kernel_window = 0, base pointers.
// Chunks whose token range is entirely >= eff run the empty body (running_max =
// -inf, running_sum = 0, out = 0); the combine's online-softmax weights them 0,
// so the grid n_chunks baked at capture stays valid for every position in the
// bucket.
#include "sm_80.cuh"
#include "block_quant.cuh"
#include "quant_traits.cuh"
#include "block_reduce.cuh"
#include "attention_q8_kv.cuh"

extern "C" __global__ void flambeau_attention_decode_q8_kv_splitk_dpos_chunk(
    const fb_fp16_t* __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    float* __restrict__ partials_m,
    float* __restrict__ partials_s,
    float* __restrict__ partials_o,
    const int n_heads_q,
    const int n_heads_kv,
    const int head_dim,
    const int n_chunks,
    const int chunk_size,
    const float scale,
    const int window,
    const int ring_depth,
    const int slide_row_stride,
    const int* __restrict__ pos_ptr
) {
    const int position = pos_ptr[0];

    int eff;
    int chunk_base;
    int kernel_window;
    const flambeau_block_q8_0* k_eff = k_cache;
    const flambeau_block_q8_0* v_eff = v_cache;

    if (slide_row_stride > 0) {
        // SWA in-band slid: the window sits at cache rows
        // [position+1-window, position+1); slide the base pointer there and
        // present the window as the full KV tail (no in-kernel SWA mask).
        eff = window;
        chunk_base = 0;
        kernel_window = 0;
        const long off_rows = (long) (position + 1 - window);
        const size_t off_bytes = (size_t) off_rows * (size_t) slide_row_stride;
        k_eff = (const flambeau_block_q8_0*) ((const char*) k_cache + off_bytes);
        v_eff = (const flambeau_block_q8_0*) ((const char*) v_cache + off_bytes);
    } else if (ring_depth > 0) {
        // SWA ring: chunk only the live window; rows wrap mod ring_depth.
        eff = position + 1;
        chunk_base = eff - window;
        if (chunk_base < 0) chunk_base = 0;
        kernel_window = window;
    } else {
        // Global / pre-window SWA: full causal tail from row 0.
        eff = position + 1;
        chunk_base = 0;
        kernel_window = 0;
    }

    attention_decode_q8_kv_splitk_chunk_body(
        q, k_eff, v_eff, partials_m, partials_s, partials_o,
        n_heads_q, n_heads_kv, head_dim, eff, n_chunks, chunk_size,
        scale, kernel_window, ring_depth, chunk_base);
}
