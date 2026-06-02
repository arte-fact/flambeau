// mmvq_q5_k_gate_up_dp4a — fused gate+up Q5_K dense MMVQ with DP4A.
//
// `ffn_gate_shexp` and `ffn_up_shexp` share the same Q8_1 activation
// (post-rmsnorm hidden). Naive dispatch launches two `mmvq_q5_k_dp4a`
// kernels reading the activation twice from HBM and paying launch
// overhead twice. This fused kernel reads the activation ONCE per
// inner-loop block and computes BOTH dot products against gate_w and
// up_w with the same Q5_K decoding shared.
//
// Block / grid / threads mirror `mmvq_q5_k_dp4a`:
//   * 256 threads/block (= 8 super-blocks/iter × 32 threads/sb).
//   * grid_x = max(n_rows_gate, n_rows_up).
//   * Out-of-range row+output guarded (asymmetric fusion: e.g. attn_qkv
//     12288 + attn_gate 8192 stays correct on the per-output mask).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define MMVQ_Q5K_GU_THREADS 256
#define MMVQ_Q5K_GU_WARPS (MMVQ_Q5K_GU_THREADS / WARP_SIZE)
#define MMVQ_Q5K_GU_THREADS_PER_SB 32
#define MMVQ_Q5K_GU_BLOCKS_PER_ITER (MMVQ_Q5K_GU_THREADS / MMVQ_Q5K_GU_THREADS_PER_SB)

static __device__ __forceinline__ int flambeau_q5_k_gu_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmvq_q5_k_gate_up_dp4a_q8_1(
    const flambeau_block_q5_K* __restrict__ gate_w,
    const flambeau_block_q5_K* __restrict__ up_w,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int n_rows_gate,
    const int n_rows_up,
    const int n_superblocks_per_row
) {
    const int row = blockIdx.x;
    const bool do_gate = row < n_rows_gate;
    const bool do_up   = row < n_rows_up;
    if (!do_gate && !do_up) return;

    const int tid       = threadIdx.x;
    const int warp      = tid / WARP_SIZE;
    const int lane      = tid & (WARP_SIZE - 1);
    const int sb_idx_in_iter = (tid >> 5);
    const int lane_in_sb     = tid & 31;
    const int sub            = lane_in_sb >> 2;
    const int lane4          = lane_in_sb & 3;

    const flambeau_block_q5_K* g_row = gate_w + (size_t) row * n_superblocks_per_row;
    const flambeau_block_q5_K* u_row = up_w   + (size_t) row * n_superblocks_per_row;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int sb = sb_idx_in_iter;
         sb < n_superblocks_per_row;
         sb += MMVQ_Q5K_GU_BLOCKS_PER_ITER)
    {
        const flambeau_block_q5_K* gbk = g_row + sb;
        const flambeau_block_q5_K* ubk = u_row + sb;

        const flambeau_block_q8_1* ya = y + (size_t) sb * 8 + sub;
        const int u_lo = ((const int*) ya->qs)[lane4];
        const int u_hi = ((const int*) ya->qs)[lane4 + 4];
        const float d_y = (float) ya->d;
        const float s_y = (float) ya->s;

        if (do_gate) {
            const float d    = (float) gbk->d;
            const float dmin = (float) gbk->dmin;

            uint8_t sc_u8 = 0, m_u8 = 0;
            flambeau_q4k_scale_min(sub, gbk->scales, &sc_u8, &m_u8);
            const float sc_f = (float) sc_u8;
            const float m_f  = (float) m_u8;

            const uint8_t* qs_pair_base = &gbk->qs[(sub >> 1) * 32];
            const int vl_lo = *((const int*) (qs_pair_base + 4 * lane4));
            const int vl_hi = *((const int*) (qs_pair_base + 16 + 4 * lane4));
            const int vh_lo = *((const int*) (&gbk->qh[4 * lane4]));
            const int vh_hi = *((const int*) (&gbk->qh[16 + 4 * lane4]));

            const int vlq_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
            const int vlq_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);
            const int vhq_lo = ((vh_lo >> sub) << 4) & 0x10101010;
            const int vhq_hi = ((vh_hi >> sub) << 4) & 0x10101010;
            const int v_lo = vlq_lo | vhq_lo;
            const int v_hi = vlq_hi | vhq_hi;

            int sumi_d = flambeau_q5_k_gu_dp4a(v_lo, u_lo, 0);
            sumi_d = flambeau_q5_k_gu_dp4a(v_hi, u_hi, sumi_d);
            acc_g += d * sc_f * d_y * (float) sumi_d - dmin * m_f * s_y * 0.25f;
        }

        if (do_up) {
            const float d    = (float) ubk->d;
            const float dmin = (float) ubk->dmin;

            uint8_t sc_u8 = 0, m_u8 = 0;
            flambeau_q4k_scale_min(sub, ubk->scales, &sc_u8, &m_u8);
            const float sc_f = (float) sc_u8;
            const float m_f  = (float) m_u8;

            const uint8_t* qs_pair_base = &ubk->qs[(sub >> 1) * 32];
            const int vl_lo = *((const int*) (qs_pair_base + 4 * lane4));
            const int vl_hi = *((const int*) (qs_pair_base + 16 + 4 * lane4));
            const int vh_lo = *((const int*) (&ubk->qh[4 * lane4]));
            const int vh_hi = *((const int*) (&ubk->qh[16 + 4 * lane4]));

            const int vlq_lo = (sub & 1) ? ((vl_lo >> 4) & 0x0F0F0F0F) : (vl_lo & 0x0F0F0F0F);
            const int vlq_hi = (sub & 1) ? ((vl_hi >> 4) & 0x0F0F0F0F) : (vl_hi & 0x0F0F0F0F);
            const int vhq_lo = ((vh_lo >> sub) << 4) & 0x10101010;
            const int vhq_hi = ((vh_hi >> sub) << 4) & 0x10101010;
            const int v_lo = vlq_lo | vhq_lo;
            const int v_hi = vlq_hi | vhq_hi;

            int sumi_d = flambeau_q5_k_gu_dp4a(v_lo, u_lo, 0);
            sumi_d = flambeau_q5_k_gu_dp4a(v_hi, u_hi, sumi_d);
            acc_u += d * sc_f * d_y * (float) sumi_d - dmin * m_f * s_y * 0.25f;
        }
    }

    if (do_gate) acc_g = gfx906_warp_reduce_sum(acc_g);
    if (do_up)   acc_u = gfx906_warp_reduce_sum(acc_u);

    __shared__ float s_g[MMVQ_Q5K_GU_WARPS];
    __shared__ float s_u[MMVQ_Q5K_GU_WARPS];
    if (lane == 0) {
        s_g[warp] = acc_g;
        s_u[warp] = acc_u;
    }
    __syncthreads();

    if (warp == 0) {
        float g = (lane < MMVQ_Q5K_GU_WARPS) ? s_g[lane] : 0.0f;
        float u = (lane < MMVQ_Q5K_GU_WARPS) ? s_u[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_Q5K_GU_WARPS / 2; off > 0; off >>= 1) {
            g += __shfl_xor(g, off, WARP_SIZE);
            u += __shfl_xor(u, off, WARP_SIZE);
        }
        if (lane == 0) {
            if (do_gate) gate_out[row] = g;
            if (do_up)   up_out[row]   = u;
        }
    }
}
