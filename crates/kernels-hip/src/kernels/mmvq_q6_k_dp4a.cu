// mmvq_q6_k_dp4a — Q6_K MMVQ with DP4A inner dot product.
// Ports llama.cpp's `vec_dot_q6_K_q8_1_impl_mmvq` (vecdotq.cuh:624-644).
// Q6_K per-element math: raw_q (6-bit) = (4-bit ql nibble | (2-bit qh bits << 4)) - 32.
// DP4A packs 4 raw_q int8 values (-32..31) × 4 Q8_1 int8 values into one
// v_dot4_i32_i8 instruction per 4 elements.
// Indexing mirrors llama.cpp's `iqs`-based scheme so the bit-layout
// assumptions match exactly. QI6_K = QK_K/(4*QR6_K) = 32 iqs positions per
// super-block. Each iqs call processes 2×4 = 8 elements (QR6_K=2, inner dp4a
// per iteration on 4 elements).
// Thread layout:
// lane = threadIdx.x (0..63)
// iqs = lane & 31 — iqs position within super-block (0..31)
// super_hi = lane >> 5 — 0 or 1: which super-block in stride-2
// outer iter (fills all 64 lanes)
// Wave = 64 threads, 1 output row/block.
// Per lane per super-block:
// - Load vl (4 bytes of ql) and vh (4 bytes of qh, shifted) as int32s
// - Load u[0], u[1] (2 int32s from distinct Q8_1 blocks)
// - Two iterations (i=0,1): construct 4-element signed int8 vi, dp4a, scale
// - acc += d * Σ_i (d8[i] * dp4a(vi, u[i], 0) * sc[i])

#include "block_quant.cuh"
#include "gfx906.cuh"

#define Q6K_QR   2
#define Q6K_QI  32

static __device__ __forceinline__ int flambeau_dp4a_q6k(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q6_k_dp4a_q8_1(
    const flambeau_block_q6_K* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane     = threadIdx.x;              // 0..63
    const int iqs      = lane & 31;                // 0..31 — iqs position
    const int super_hi = lane >> 5;                // 0 or 1 — stride-2 super-block offset

    const flambeau_block_q6_K* xrow = x + (size_t) row * n_superblocks_per_row;
    const flambeau_block_q8_1* y_sb0 = y;          // super-block 0 base; y is already shifted externally

    // Pre-compute iqs-dependent offsets — constant across the super-block loop.
    // From llama.cpp vec_dot_q6_K_q8_1 (vecdotq.cuh:961-966):
    const int bq8_offset   = Q6K_QR * (iqs / (Q6K_QI / 2)) * 2
                           + (iqs % (Q6K_QI / 2)) / (Q6K_QI / 4);
    const int scale_offset = (Q6K_QI / 4) * (iqs / (Q6K_QI / 2))
                           + (iqs % (Q6K_QI / 2)) / (Q6K_QI / 8);
    const int vh_shift     = 2 * ((iqs % (Q6K_QI / 2)) / (Q6K_QI / 4));
    const int qh_int_idx   = (Q6K_QI / 4) * (iqs / (Q6K_QI / 2))
                           + iqs % (Q6K_QI / 4);

    float acc = 0.0f;

    for (int b_base = 0; b_base < n_superblocks_per_row; b_base += 2) {
        const int b = b_base + super_hi;
        if (b >= n_superblocks_per_row) continue;

        const flambeau_block_q6_K* bk = xrow + b;

        const int vl = ((const int*) bk->ql)[iqs];
        const int vh = ((const int*) bk->qh)[qh_int_idx] >> vh_shift;

        const int8_t* sc_ptr = bk->scales + scale_offset;

        // Two Q8_1 blocks per call (QR6_K=2), same iqs%QI8_1 int32 of each.
        const flambeau_block_q8_1* ya0 = y_sb0 + (b * 8 + bq8_offset + 0);
        const flambeau_block_q8_1* ya1 = y_sb0 + (b * 8 + bq8_offset + 2);
        const int u0 = ((const int*) ya0->qs)[iqs % 8];
        const int u1 = ((const int*) ya1->qs)[iqs % 8];
        const float d8_0 = (float) ya0->d;
        const float d8_1 = (float) ya1->d;

        // correctness fix: the original code did
        // `(unsigned)(vil | vih) - 0x20202020u`
        // as a 32-bit unsigned subtract to apply the Q6_K −32 bias byte-wise.
        // That is NOT a per-byte saturate subtract — when any source byte
        // is < 32 (~50 % of random weights), the borrow cascades into the
        // next byte and shifts its decoded value by −1. Errors up to 90 %
        // at m≥128, and 6 % even at the dispatched m=1, k=15360 shape.
        // Fix: use the DP4A identity
        // (raw − 32) · y = raw · y − 32 · Σ y
        // — store `raw = vil | vih` as unsigned [0, 63] (safe to read as
        // signed int8 because top 2 bits are zero), compute one extra
        // `dp4a(0x01010101, y, 0)` per term for the bias correction.
        // Same pattern mmq_q6_K_wave64 uses; HIP has no
        // per-byte saturate-subtract equivalent of CUDA's `__vsubss4`.
        float sumf = 0.0f;
        {
            const int sc  = (int) sc_ptr[0];
            const int vil = (vl >> 0) & 0x0F0F0F0F;
            const int vih = ((vh >> 0) << 4) & 0x30303030;
            const int raw = vil | vih;                       // [0, 63] per byte
            const int dot = flambeau_dp4a_q6k(raw, u0, 0);
            const int sumy = flambeau_dp4a_q6k(0x01010101, u0, 0);
            sumf += d8_0 * ((float) (dot - 32 * sumy) * (float) sc);
        }
        {
            const int sc  = (int) sc_ptr[4];
            const int vil = (vl >> 4) & 0x0F0F0F0F;
            const int vih = ((vh >> 4) << 4) & 0x30303030;
            const int raw = vil | vih;
            const int dot = flambeau_dp4a_q6k(raw, u1, 0);
            const int sumy = flambeau_dp4a_q6k(0x01010101, u1, 0);
            sumf += d8_1 * ((float) (dot - 32 * sumy) * (float) sc);
        }

        const float d = (float) bk->d;
        acc += d * sumf;
    }

    acc = gfx906_warp_reduce_sum(acc);

    if (lane == 0) {
        dst[row] = acc;
    }
}
