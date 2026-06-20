// Tensor-core (m16n8k16 f16) flash-attention-v2 prefill for sm_80+. CUDA-only
// counterpart of the backend-neutral scalar attention_flash_tile; gfx906 has no
// matrix core and keeps flash_tile. Same call contract: q/out [n_q, n_heads_q,
// D] f16, k/v cache [n_k(ring), n_heads_kv, D] f16, GQA h_kv = h_q/n_rep, causal
// + sliding-window mask, no logit softcap (flash_tile applies none).
//
// NW warps/block, each warp owns one 16-row Q M-tile (BR = NW*16 rows for one
// head). Q fragments and the O accumulator stay register-resident across the KV
// loop; K/V chunks (BC=16) are staged to shared cooperatively and reused across
// all BR rows. S=Q·Kᵀ and O=P·V both run on the f16 mma; the online softmax is
// reduced in shared. Caller routes here only when n_q % BR == 0 (no partial-tile
// OOB on the register Q load); the tail ubatch falls back to flash_tile.
#include "mma_f16.cuh"

namespace fm = flambeau_mma_f16;

template<int D, int NW>
static __device__ __forceinline__ void flash_attn_prefill_mma_impl(
    const __half* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc,
    __half* __restrict__ out, int n_q, int n_heads_q, int n_heads_kv, int n_k,
    int q_offset, float scale, int window_size, int ring_depth)
{
    constexpr int BR = NW * 16, BC = 16;
    const int q_tile = blockIdx.x, h_q = blockIdx.y, lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp * 32 + lane, nthreads = NW * 32;
    const int n_rep = n_heads_q / n_heads_kv;
    const int h_kv = n_rep > 1 ? h_q / n_rep : h_q;
    const int my_row0 = q_tile * BR + warp * 16;

    __shared__ __half Ks[BC * D];
    __shared__ __half Vs[BC * D];
    __shared__ float Ss[NW * 16 * BC];
    __shared__ __half Ps[NW * 16 * BC];
    __shared__ float m_i[NW * 16], l_i[NW * 16], al[NW * 16];
    float* mw = m_i + warp * 16; float* lw = l_i + warp * 16; float* aw = al + warp * 16;
    float* Sw = Ss + warp * 16 * BC; __half* Pw = Ps + warp * 16 * BC;

    fm::tile_a Qf[D / 16];
    #pragma unroll
    for (int kd = 0; kd < D / 16; ++kd)
        fm::load_a(Qf[kd], q + ((size_t)my_row0 * n_heads_q + h_q) * D + kd * 16, n_heads_q * D);
    float Of[D / 8][4];
    #pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) for (int l = 0; l < 4; ++l) Of[nt][l] = 0.f;
    for (int r = lane; r < 16; r += 32) { mw[r] = -__builtin_huge_valf(); lw[r] = 0.f; }

    const int first_q = q_offset + q_tile * BR;
    const int last_q  = q_offset + q_tile * BR + BR - 1;
    int block_limit = last_q + 1; if (block_limit > n_k) block_limit = n_k;
    int block_swa_min = 0;
    if (window_size > 0) { block_swa_min = first_q - window_size + 1; if (block_swa_min < 0) block_swa_min = 0; }

    for (int k_start = (block_swa_min / BC) * BC; k_start < block_limit; k_start += BC) {
        for (int idx = tid; idx < BC * D; idx += nthreads) {
            int j = idx / D, d = idx % D, row = k_start + j;
            __half kk = __float2half(0.f), vv = __float2half(0.f);
            if (row < block_limit) {
                int rp = ring_depth > 0 ? row % ring_depth : row;
                size_t off = ((size_t)rp * n_heads_kv + h_kv) * D + d;
                kk = kc[off]; vv = vc[off];
            }
            Ks[idx] = kk; Vs[idx] = vv;
        }
        __syncthreads();

        #pragma unroll
        for (int nt = 0; nt < BC / 8; ++nt) {
            fm::tile_c cs;
            #pragma unroll
            for (int kd = 0; kd < D / 16; ++kd) {
                fm::tile_b b; fm::load_b(b, Ks + (nt * 8) * D + kd * 16, D);
                fm::mma(cs, Qf[kd], b);
            }
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                Sw[fm::tile_c::get_i(l) * BC + nt * 8 + fm::tile_c::get_j(l)] = cs.x[l] * scale;
        }
        __syncwarp();

        if (lane < 16) {
            int r = lane, qg = q_offset + my_row0 + r;
            int lim = qg + 1; if (lim > n_k) lim = n_k;
            int smin = 0; if (window_size > 0) { smin = qg - window_size + 1; if (smin < 0) smin = 0; }
            float rmax = -__builtin_huge_valf();
            for (int cc = 0; cc < BC; ++cc) {
                int row = k_start + cc;
                bool masked = (row >= lim) || (row < smin) || (row >= block_limit);
                float s = masked ? -__builtin_huge_valf() : Sw[r * BC + cc];
                Sw[r * BC + cc] = s; rmax = fmaxf(rmax, s);
            }
            if (rmax == -__builtin_huge_valf()) { aw[r] = 1.f; for (int cc = 0; cc < BC; ++cc) Pw[r * BC + cc] = __float2half(0.f); }
            else {
                float m_new = fmaxf(mw[r], rmax), a = __expf(mw[r] - m_new), psum = 0.f;
                for (int cc = 0; cc < BC; ++cc) { float p = __expf(Sw[r * BC + cc] - m_new); Pw[r * BC + cc] = __float2half(p); psum += p; }
                lw[r] = a * lw[r] + psum; mw[r] = m_new; aw[r] = a;
            }
        }
        __syncwarp();

        fm::tile_a pa; fm::load_a(pa, Pw, BC);
        #pragma unroll
        for (int nt = 0; nt < D / 8; ++nt) {
            fm::tile_c co;
            fm::tile_b b; fm::load_b_kn(b, Vs + nt * 8, D);
            fm::mma(co, pa, b);
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                Of[nt][l] = aw[fm::tile_c::get_i(l)] * Of[nt][l] + co.x[l];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int r = fm::tile_c::get_i(l), d = nt * 8 + fm::tile_c::get_j(l);
            float inv = lw[r] > 0.f ? 1.f / lw[r] : 0.f;
            out[((size_t)(my_row0 + r) * n_heads_q + h_q) * D + d] = __float2half(Of[nt][l] * inv);
        }
    }
}

#define FLASH_MMA_ENTRY(NAME, D, NW)                                                              \
extern "C" __global__ __launch_bounds__(32 * NW)                                                 \
void NAME(const __half* __restrict__ q, const __half* __restrict__ kc,                            \
          const __half* __restrict__ vc, __half* __restrict__ out,                                \
          int n_q, int n_heads_q, int n_heads_kv, int n_k,                                        \
          int q_offset, float scale, int window_size, int ring_depth) {                           \
    flash_attn_prefill_mma_impl<D, NW>(q, kc, vc, out, n_q, n_heads_q, n_heads_kv, n_k,            \
                                       q_offset, scale, window_size, ring_depth);                 \
}

FLASH_MMA_ENTRY(flambeau_attention_prefill_mma_d128_f16, 128, 8)
FLASH_MMA_ENTRY(flambeau_attention_prefill_mma_d256_f16, 256, 8)
