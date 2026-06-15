// mmvq_q5_k_dp4a — Q5_K weight × Q8_1 activation → {F32, F16} dst, DP4A.
//
// Ported from llama.cpp's `vec_dot_q5_K_q8_1_impl_vmmq` (vecdotq.cuh:561)
// adapted to flambeau's Q4_0 dp4a layout (256 threads/block, blocks_per_iter
// loop, warp + cross-warp shfl reduce).
//
// Why this exists: the prior Q5_K kernels (`mmvq_q5_k`, `mmvq_q5_k_r2`) do
// scalar per-element FP32 multiplies, slower per call than the
// dp4a-based path. On Qwen3.6-27B-Q4_0 PP4 the rocprofv3 trace showed
// `flambeau_mmvq_q5_k_r2_q8_1` at 268 µs/call × 2976 calls = 798 ms (24% of
// total kernel time), vs llama.cpp's generic `mul_mat_vec_q` at ~91 µs for
// equivalent Q5_K calls. This kernel uses gfx906's `v_dot4_i32_i8` SIMD
// dot product (via `__builtin_amdgcn_sdot4`) for the inner multiply-add.
//
// Q5_K block layout (256 elements, 8 sub-blocks of 32):
//   d, dmin       — FP16 super-block scale + min
//   scales[12]    — packed 6-bit (sc, m) pairs for sub-blocks 0..7
//   qh[32]        — high bit per element (bit `s` of qh[i] = high bit of
//                   element i in sub-block s)
//   qs[128]       — low 4 bits packed: qs[(s>>1)*32 + i] holds elements
//                   {2s, 2s+1} of position i (low nibble = even sub-block,
//                   high nibble = odd sub-block)
//
// Threading: 256 threads/block, `lane4 = tid & 3`, `block_idx = tid >> 2`.
// Each thread processes 1 sub-block per outer iter (`s ∈ 0..7`) by reading
// 4 packed bytes (= 4 elements × 2 nibbles where the high bit comes from
// qh). The dp4a inner loop sums 4 (5-bit × i8) products per call.

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_Q5_K_THREADS 256
#define MMVQ_Q5_K_WARPS (MMVQ_Q5_K_THREADS / WARP_SIZE)
// Each Q5_K super-block has 256 quants = 8 sub-blocks × 32 quants.
// Within a sub-block we want each thread to process 8 quants (= 2 int32 of
// packed bytes). With 4 threads per sub-block (one int32 of low nibbles
// each), 8 sub-blocks per super-block, that's 32 threads per super-block.
// `MMVQ_Q5_K_THREADS / 32 = 8` super-blocks processed per iter.
#define MMVQ_Q5_K_THREADS_PER_BLOCK 32
#define MMVQ_Q5_K_BLOCKS_PER_ITER (MMVQ_Q5_K_THREADS / MMVQ_Q5_K_THREADS_PER_BLOCK)

static __device__ __forceinline__ int flambeau_q5_k_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q5_k_dp4a_body(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    // Within the 32 threads working on one super-block: 4 threads per
    // sub-block × 8 sub-blocks. lane_sub = which sub-block this thread
    // works on; lane4 = which int32 of the 16-byte sub-block's low nibble
    // quartet (each sub-block has 32 low-nibbles = 16 packed bytes = 4 int32).
    const int sb_idx_in_iter = (tid >> 5);          // 0..7 — which super-block in this iter
    const int lane_in_sb     = tid & 31;            // 0..31 — index within the super-block-sized group
    const int sub            = lane_in_sb >> 2;     // 0..7 — sub-block within the super-block
    const int lane4          = lane_in_sb & 3;      // 0..3 — int32 within the sub-block

    const flambeau_block_q5_K* xrow = x + (size_t) row * n_superblocks_per_row;

    float acc = 0.0f;

    for (int sb = sb_idx_in_iter; sb < n_superblocks_per_row; sb += MMVQ_Q5_K_BLOCKS_PER_ITER) {
        const flambeau_block_q5_K* bk = xrow + sb;

        const float d    = (float) bk->d;     // super-block scale
        const float dmin = (float) bk->dmin;  // super-block min

        // Sub-block scale (sc) and min (m). 6-bit packed, decoded from
        // the 12-byte scales array.
        uint8_t sc_u8 = 0, m_u8 = 0;
        flambeau_q4k_scale_min(sub, bk->scales, &sc_u8, &m_u8);
        const float sc_f = (float) sc_u8;
        const float m_f  = (float) m_u8;

        // qs layout: pairs of sub-blocks share 32 bytes. Sub-block `sub`
        // takes either the low nibble (even sub) or high nibble (odd sub)
        // of qs[(sub>>1)*32 + i] for i ∈ [0, 32). Each thread covers 8
        // of the 32 elements via TWO dp4a calls (4 elements each). With
        // 4 threads per sub-block × 8 elems = full 32 elements covered.
        const uint8_t* qs_pair_base = &bk->qs[(sub >> 1) * 32];
        const int vl_lo = *((const int*) (qs_pair_base + 4 * lane4));
        const int vl_hi = *((const int*) (qs_pair_base + 16 + 4 * lane4));
        // qh: one high bit per element across 8 sub-blocks (bit `sub` of
        // qh[i] = high bit of element i in sub-block `sub`). Read 4 bytes
        // for the lo half (positions 4*lane4 .. +3) and 4 bytes for the
        // hi half (positions 16 + 4*lane4 .. +3).
        const int vh_lo = *((const int*) (&bk->qh[4 * lane4]));
        const int vh_hi = *((const int*) (&bk->qh[16 + 4 * lane4]));

        // Extract low nibble per byte: even sub → bits [0:3] of each
        // byte; odd sub → bits [4:7] of each byte.
        const int vlq_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
        const int vlq_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);
        // Extract high bit for this sub-block, shifted to bit position 4
        // so it adds 16 to the 4-bit nibble. mask 0x10101010 keeps only
        // bit 4 of each byte, masking out the int32-shift cross-byte
        // spillage. Pattern matches llama.cpp's vec_dot_q5_K_q8_1_impl_vmmq.
        const int vhq_lo = ((vh_lo >> sub) << 4) & 0x10101010;
        const int vhq_hi = ((vh_hi >> sub) << 4) & 0x10101010;
        // Combine 4-bit + 1-bit → 5-bit values (range [0, 31]).
        const int v_lo = vlq_lo | vhq_lo;
        const int v_hi = vlq_hi | vhq_hi;

        // Q8_1 activation for this sub-block: ya[sub] = 32 i8 quants.
        // Thread reads two int32s, one for each 4-element half:
        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = ((const int*) ya->qs)[lane4];
        const int u_hi = ((const int*) ya->qs)[lane4 + 4];
        const float d_y = (float) ya->d;
        const float s_y = (float) ya->s;  // d_y · sum(u_qs) pre-baked

        // Per-element contribution before reduce:
        //   x = d · sc · v - dmin · m       (v ∈ [0, 31])
        //   y = d_y · u                     (u ∈ [-128, 127])
        //   x · y = d · sc · d_y · (v·u) - dmin · m · d_y · u
        //
        // Sum over the 8 elements this thread covers (2 SIMD dp4a):
        //   d · sc · d_y · Σ(v·u) - dmin · m · d_y · Σu
        //
        // s_y already encodes d_y · sum(y_qs) for the FULL 32-element
        // sub-block; this thread covers 8/32 = quarter of the sub-block,
        // so the correction term is dmin · m · (s_y × 0.25). Warp-reduce
        // sums the 4 lanes' quarters into the full correction.
        int sumi_d = flambeau_q5_k_dp4a(v_lo, u_lo, 0);
        sumi_d = flambeau_q5_k_dp4a(v_hi, u_hi, sumi_d);
        acc += d * sc_f * d_y * (float) sumi_d - dmin * m_f * s_y * 0.25f;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_Q5_K_WARPS];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_Q5_K_WARPS) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q5_K_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ void flambeau_mmvq_q5_k_dp4a_q8_1(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q5_k_dp4a_body<float>(x, y, dst, n_rows, n_superblocks_per_row);
}

extern "C" __global__ void flambeau_mmvq_q5_k_dp4a_q8_1_f16(
    const flambeau_block_q5_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    mmvq_q5_k_dp4a_body<fb_fp16_t>(x, y, dst, n_rows, n_superblocks_per_row);
}
