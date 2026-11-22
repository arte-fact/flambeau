// Device-position decode attention, Q8_0 KV, one query row. CUDA-only.
//
// Block = 128 threads (4 warps) per (q_head, part). Q is quantized to Q8_0 in
// shared memory once. The KV tail [0, position+1) (position read from
// `pos_ptr[0]`) is walked in 128-token tiles strided over the gridDim.y parts;
// in each tile a warp owns 32 tokens. Q·K: a group of NB = D/32 lanes scores
// one token, each lane one Q8_0 block (8 dp4a), a log2(NB)-step shuffle sum,
// so a warp scores 32/NB tokens per step; the 32 tokens are a fully unrolled,
// masked loop (no early exit), which lets the loads of every step issue ahead
// of the reductions. Scores stay in registers; the warp updates its online
// softmax once per tile; V is accumulated with each lane owning D/32 output
// elements, the weight of each token broadcast from its group by shuffle. The
// four warps merge through shared memory at the end.
//
// gridDim.y == 1: the block writes the normalised F16 output. gridDim.y > 1:
// it writes the unnormalised split-K partial (m, s, o) at q_head * gridDim.y +
// part, the layout the shared attention_decode_q8_kv_splitk combine reads; a
// part with no token in the tail writes m = -inf, s = 0, o = 0 (before any
// load), which the combine weights 0. The grid therefore stays valid for every
// position.
//
// Global (full-context) layers only: no SWA window, ring or slide.
#include <cfloat>
#include "sm_80.cuh"
#include "block_quant.cuh"
#include "quant_traits.cuh"
#include "block_reduce.cuh"
#include "attention_q8_kv.cuh"

#define ATTN_VEC_NTHREADS 128
#define ATTN_VEC_NWARPS (ATTN_VEC_NTHREADS / 32)
#define ATTN_VEC_TILE ATTN_VEC_NTHREADS

template <int WIDTH>
static __device__ __forceinline__ float attn_vec_group_sum(float v) {
#pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off, WIDTH);
    }
    return v;
}

static __device__ __forceinline__ float attn_vec_warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off, 32));
    }
    return v;
}

template <int D>
static __device__ __forceinline__ void attention_decode_q8_kv_vec_dpos_body(
    const fb_fp16_t* __restrict__ q,
    const flambeau_block_q8_0* __restrict__ k_cache,
    const flambeau_block_q8_0* __restrict__ v_cache,
    fb_fp16_t* __restrict__ out,
    float* __restrict__ partials_m,
    float* __restrict__ partials_s,
    float* __restrict__ partials_o,
    const int n_heads_q,
    const int n_heads_kv,
    const float scale,
    const int* __restrict__ pos_ptr)
{
    // NB lanes score one token (lane b of the group: Q8_0 block b);
    // GROUPS tokens per step, STEPS steps per warp's 32 tokens.
    constexpr int NB = D / 32;
    constexpr int GROUPS = 32 / NB;
    constexpr int STEPS = 32 / GROUPS;
    // V: lane l owns the int32 quads j = l + 32*i (i < QPL) of a head row.
    constexpr int QPL = D / 128;
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int part = blockIdx.y;
    const int n_par = gridDim.y;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int group = lane / NB;
    const int blk = lane % NB;
    const int n_kv = pos_ptr[0] + 1;

    if (part * ATTN_VEC_TILE >= n_kv) {
        const int part_idx = q_head * n_par + part;
        for (int d = tid; d < D; d += ATTN_VEC_NTHREADS) {
            partials_o[(size_t) part_idx * D + d] = 0.0f;
        }
        if (tid == 0) {
            partials_m[part_idx] = -ATTN_Q8_INF;
            partials_s[part_idx] = 0.0f;
        }
        return;
    }

    __shared__ int8_t q_qs[D];
    __shared__ float q_d_block[NB];
    __shared__ float w_max[ATTN_VEC_NWARPS];
    __shared__ float w_sum[ATTN_VEC_NWARPS];
    __shared__ float w_acc[ATTN_VEC_NWARPS][D];

    if (tid < D / 4) {
        const int elem_base = tid * 4;
        attn_q8_quantize_q(q, (size_t) q_head * D, elem_base, (elem_base % 32) / 4, elem_base / 32, q_qs, q_d_block);
    }
    __syncthreads();

    int q_quad[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q_quad[i] = ((const int*) q_qs)[blk * 8 + i];
    }
    const float q_d = q_d_block[blk];

    float run_max = -FLT_MAX / 2.0f;
    float run_sum = 0.0f;
    float acc[QPL][4];
#pragma unroll
    for (int i = 0; i < QPL; ++i) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f; acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    for (int tile = part * ATTN_VEC_TILE; tile < n_kv; tile += n_par * ATTN_VEC_TILE) {
        const int t0 = tile + warp * 32;

        // Score of token t0 + s*GROUPS + group, held by every lane of the group.
        float score[STEPS];
#pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            const int t = t0 + s * GROUPS + group;
            float c = 0.0f;
            if (t < n_kv) {
                const flambeau_block_q8_0* kb = k_cache + ((size_t) t * n_heads_kv + kv_head) * NB + blk;
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sumi = dp4a(load_int_b2(kb->qs, i), q_quad[i], sumi);
                }
                c = (float) kb->d * q_d * (float) sumi;
            }
            c = attn_vec_group_sum<NB>(c);
            score[s] = (t < n_kv) ? c * scale : -ATTN_Q8_INF;
        }

        float tile_max = score[0];
#pragma unroll
        for (int s = 1; s < STEPS; ++s) tile_max = fmaxf(tile_max, score[s]);
        const float new_max = fmaxf(run_max, attn_vec_warp_max(tile_max));
        const float scale_old = __expf(run_max - new_max);
        float p_sum = 0.0f;
#pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            score[s] = __expf(score[s] - new_max);
            p_sum += score[s];
        }
        // Each token's weight is held by its NB group lanes: count it once.
        run_sum = run_sum * scale_old + (blk == 0 ? p_sum : 0.0f);
        run_max = new_max;
#pragma unroll
        for (int ii = 0; ii < QPL; ++ii) {
            acc[ii][0] *= scale_old; acc[ii][1] *= scale_old; acc[ii][2] *= scale_old; acc[ii][3] *= scale_old;
        }

#pragma unroll
        for (int s = 0; s < STEPS; ++s) {
#pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                const int t = t0 + s * GROUPS + g;
                const float pk = __shfl_sync(0xffffffffu, score[s], g * NB, 32);
                if (t < n_kv) {
                    const size_t row = ((size_t) t * n_heads_kv + kv_head) * NB;
#pragma unroll
                    for (int ii = 0; ii < QPL; ++ii) {
                        const int j = lane + 32 * ii;
                        const flambeau_block_q8_0* vb = v_cache + row + j / 8;
                        const float w = pk * (float) vb->d;
                        const int v_quad = load_int_b2(vb->qs, j % 8);
                        acc[ii][0] += w * (float) (int8_t) ((v_quad >> 0) & 0xFF);
                        acc[ii][1] += w * (float) (int8_t) ((v_quad >> 8) & 0xFF);
                        acc[ii][2] += w * (float) (int8_t) ((v_quad >> 16) & 0xFF);
                        acc[ii][3] += w * (float) (int8_t) ((v_quad >> 24) & 0xFF);
                    }
                }
            }
        }
    }

    float warp_sum = run_sum;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        warp_sum += __shfl_xor_sync(0xffffffffu, warp_sum, off, 32);
    }
    if (lane == 0) {
        w_max[warp] = run_max;
        w_sum[warp] = warp_sum;
    }
#pragma unroll
    for (int ii = 0; ii < QPL; ++ii) {
        const int e = 4 * (lane + 32 * ii);
        w_acc[warp][e + 0] = acc[ii][0];
        w_acc[warp][e + 1] = acc[ii][1];
        w_acc[warp][e + 2] = acc[ii][2];
        w_acc[warp][e + 3] = acc[ii][3];
    }
    __syncthreads();

    float m = w_max[0];
#pragma unroll
    for (int w = 1; w < ATTN_VEC_NWARPS; ++w) m = fmaxf(m, w_max[w]);
    float wgt[ATTN_VEC_NWARPS];
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < ATTN_VEC_NWARPS; ++w) {
        wgt[w] = __expf(w_max[w] - m);
        s += w_sum[w] * wgt[w];
    }

    if (n_par == 1) {
        const float norm = (s > 0.0f) ? 1.0f / s : 0.0f;
        for (int d = tid; d < D; d += ATTN_VEC_NTHREADS) {
            float o = 0.0f;
#pragma unroll
            for (int w = 0; w < ATTN_VEC_NWARPS; ++w) o += w_acc[w][d] * wgt[w];
            out[(size_t) q_head * D + d] = (fb_fp16_t) (o * norm);
        }
        return;
    }
    const int part_idx = q_head * n_par + part;
    for (int d = tid; d < D; d += ATTN_VEC_NTHREADS) {
        float o = 0.0f;
#pragma unroll
        for (int w = 0; w < ATTN_VEC_NWARPS; ++w) o += w_acc[w][d] * wgt[w];
        partials_o[(size_t) part_idx * D + d] = o;
    }
    if (tid == 0) {
        partials_m[part_idx] = (s > 0.0f) ? m : -ATTN_Q8_INF;
        partials_s[part_idx] = s;
    }
}

#define ATTN_VEC_ENTRY(D)                                                                    \
    extern "C" __global__ __launch_bounds__(ATTN_VEC_NTHREADS)                               \
    void flambeau_attention_decode_q8_kv_vec_dpos_d##D(                                      \
        const fb_fp16_t* __restrict__ q,                                                     \
        const flambeau_block_q8_0* __restrict__ k_cache,                                     \
        const flambeau_block_q8_0* __restrict__ v_cache,                                     \
        fb_fp16_t* __restrict__ out,                                                         \
        float* __restrict__ partials_m,                                                      \
        float* __restrict__ partials_s,                                                      \
        float* __restrict__ partials_o,                                                      \
        const int n_heads_q,                                                                 \
        const int n_heads_kv,                                                                \
        const float scale,                                                                   \
        const int* __restrict__ pos_ptr) {                                                   \
        attention_decode_q8_kv_vec_dpos_body<D>(q, k_cache, v_cache, out, partials_m,        \
            partials_s, partials_o, n_heads_q, n_heads_kv, scale, pos_ptr);                  \
    }

ATTN_VEC_ENTRY(128)
ATTN_VEC_ENTRY(256)
