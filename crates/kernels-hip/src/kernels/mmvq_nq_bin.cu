// mmvq_nq_bin — NanoQuant two-stage binary GEMV prototype for gfx906 (MI50).
//
// Format per linear layer (arXiv:2602.06694, SamsungLabs/NanoQuant @ a9e0a430,
// Apache-2.0):  W ≈ diag(s1) · U · Vᵀ · diag(s2)
//   U ∈ {±1}^(dout×r), V ∈ {±1}^(din×r), s1 ∈ fp16[dout], s2 ∈ fp16[din].
//   U, V bit-packed row-major, 32 weights per u32 word, bit=1 → +1, bit=0 → −1
//   (paper: "−1→0, +1→1"; lsb_first packing in binary_packer).
//
// Decode GEMV (m = 1):  t = Vᵀ (s2 ⊙ x) ∈ f32[r]   then   y = s1 ⊙ (U t) ∈ f32[dout]
//
// Stage 1 (flambeau_nq_stage1_vt_x), v3 — LDS-staged tiles:
//   grid = (ceil(r/256), ceil(din/NQ_JTILE)), block = 256 threads (4 warps).
//   Load phase: the block stages a JTILE×256-bit tile of V into LDS fully
//   coalesced (32 B per 8 threads per row), plus the pre-scaled activations
//   x̃ = s2⊙x for its row range. Compute phase: warp w owns rank cols
//   [kc + w*64, kc + w*64 + 64); lane l takes bit l&31 of word (l>>5) of each
//   staged row — 1 LDS word load + 1 LDS x̃ load + 1 FMA per row per lane.
//   Partials are combined with atomicAdd into t (zeroed by flambeau_nq_zero_f32).
// Stage 2 (flambeau_nq_stage2_u_t), v3 — half-warp per output row:
//   block = 256 threads = 8 half-warps, one row each; 32 lanes split the rank
//   dim (k = lane, lane+32, …), reduce with gfx906_half_warp_reduce_sum, s1 on store.
//
// Weight traffic per token: (din + dout) × ceil(r/32) × 4 B + fp16 scales ≈ BPW·N/8.

#include <hip/hip_runtime.h>
#include "block_quant.cuh"   // fb_fp16_t
#include "gfx906.cuh"        // gfx906_half_warp_reduce_sum

#define NQ_STAGE1_THREADS 256
#define NQ_JTILE 512                    // V rows staged per block in LDS
#define NQ_STAGE1_WORDS 8               // 256 rank cols / 32 = 8 words per row
#define NQ_STAGE1_LDS_BYTES (NQ_JTILE * NQ_STAGE1_WORDS * 4 + NQ_JTILE * 4)  // V tile + x̃
#define NQ_STAGE2_THREADS 256           // 8 half-warps × 1 output row

extern "C" __global__ void flambeau_nq_zero_f32(float* __restrict__ t, const int n) {
    const int i = (blockIdx.x << 8) + threadIdx.x;
    if (i < n) t[i] = 0.0f;
}

__device__ __forceinline__ float nq_sign(unsigned int w, int bit) {
    return (float)((int)((w >> bit) & 1u) * 2 - 1);
}

extern "C" __global__ void flambeau_nq_stage1_vt_x(
    const unsigned int* __restrict__ vp,   // din × wpr packed V
    const fb_fp16_t* __restrict__ x,       // din activations (f16)
    const fb_fp16_t* __restrict__ s2,      // din input scales (f16)
    float* __restrict__ t,                 // r (f32), zeroed before launch
    const int din,
    const int r,
    const int wpr)                         // words per row = ceil(r/32)
{
    extern __shared__ unsigned char lds_raw[];
    unsigned int* v_tile = reinterpret_cast<unsigned int*>(lds_raw);              // JTILE × 8 words
    float* x_tile = reinterpret_cast<float*>(lds_raw + NQ_JTILE * NQ_STAGE1_WORDS * 4); // JTILE

    const int tid = threadIdx.x;
    const int j0 = blockIdx.y * NQ_JTILE;
    const int j1 = min(din, j0 + NQ_JTILE);
    const int rows = j1 - j0;
    const int kw0 = (blockIdx.x << 3);   // first word col of this block's 256 rank cols

    // Load phase: V tile (coalesced: 8 threads per row × 32 B) + x̃.
    for (int idx = tid; idx < rows * NQ_STAGE1_WORDS; idx += NQ_STAGE1_THREADS) {
        const int row = idx >> 3;
        const int w = idx & 7;
        v_tile[idx] = vp[(size_t)(j0 + row) * wpr + kw0 + w];
    }
    for (int idx = tid; idx < rows; idx += NQ_STAGE1_THREADS) {
        x_tile[idx] = (float)x[j0 + idx] * (float)s2[j0 + idx];
    }
    __syncthreads();

    // Compute phase.
    const int warp = tid >> 6;
    const int lane = tid & 63;
    const int k = (blockIdx.x << 8) + (warp << 6) + lane;
    if (k >= r) return;
    const int wloc = (warp << 1) + (lane >> 5);   // word within the 8-word row tile
    const int bit = lane & 31;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int row = 0;
    for (; row + 4 <= rows; row += 4) {
        const unsigned int* vrow = v_tile + row * NQ_STAGE1_WORDS + wloc;
        a0 += nq_sign(vrow[0], bit) * x_tile[row + 0];
        a1 += nq_sign(vrow[NQ_STAGE1_WORDS], bit) * x_tile[row + 1];
        a2 += nq_sign(vrow[2 * NQ_STAGE1_WORDS], bit) * x_tile[row + 2];
        a3 += nq_sign(vrow[3 * NQ_STAGE1_WORDS], bit) * x_tile[row + 3];
    }
    for (; row < rows; ++row) {
        a0 += nq_sign(v_tile[row * NQ_STAGE1_WORDS + wloc], bit) * x_tile[row];
    }
    atomicAdd(&t[k], (a0 + a1) + (a2 + a3));
}

extern "C" __global__ void flambeau_nq_stage2_u_t(
    const unsigned int* __restrict__ up,   // dout × wpr packed U
    const float* __restrict__ t,           // r (f32)
    const fb_fp16_t* __restrict__ s1,      // dout output scales (f16)
    float* __restrict__ y,                 // dout (f32)
    const int r,
    const int wpr,
    const int dout)
{
    const int hw = threadIdx.x >> 5;            // half-warp 0..7
    const int lane = threadIdx.x & 31;          // lane within half-warp
    const int i = blockIdx.x * 8 + hw;          // output row
    if (i >= dout) return;
    const unsigned int* urow = up + (size_t)i * wpr;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int k = lane;
    for (; k + 96 < r; k += 128) {
        a0 += nq_sign(urow[(k + 0) >> 5], (k + 0) & 31) * t[k + 0];
        a1 += nq_sign(urow[(k + 32) >> 5], (k + 32) & 31) * t[k + 32];
        a2 += nq_sign(urow[(k + 64) >> 5], (k + 64) & 31) * t[k + 64];
        a3 += nq_sign(urow[(k + 96) >> 5], (k + 96) & 31) * t[k + 96];
    }
    for (; k < r; k += 32) {
        a0 += nq_sign(urow[k >> 5], k & 31) * t[k];
    }
    float acc = (a0 + a1) + (a2 + a3);
    acc = gfx906_half_warp_reduce_sum(acc);
    if (lane == 0) y[i] = (float)s1[i] * acc;
}
