// Device-position decode attention, Q8_0 KV, one query row. CUDA-only.
//
// Block = 128 threads (4 warps) per (q_head, part). Q is quantized to Q8_0 in
// shared memory once. The KV tail [0, position+1) (position read from
// `pos_ptr[0]`) is walked in 128-token tiles strided over the gridDim.y parts:
// in each tile a warp owns 32 tokens, the whole warp computes one token's Q·K
// (each lane D/128 dp4a quads, warp shuffle sum — no block barrier per token),
// lane i keeps token i's score, the warp updates its online softmax once per
// tile, then accumulates V with each lane owning D/32 output elements. The four
// warps merge through shared memory at the end.
//
// gridDim.y == 1: the block writes the normalised F16 output. gridDim.y > 1:
// it writes the unnormalised split-K partial (m, s, o) at q_head * gridDim.y +
// part, the layout the shared attention_decode_q8_kv_splitk combine reads; a
// part with no token in the tail writes m = -inf, s = 0, o = 0, which the
// combine weights 0. The grid therefore stays valid for every position.
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

static __device__ __forceinline__ float attn_vec_warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off, 32);
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
    // Lane l owns the int32 quads j = l + 32*i (i < QPL) of a head row: the
    // same quads for the Q·K dot and for the V elements it accumulates.
    constexpr int NB = D / 32;
    constexpr int QPL = D / 128;
    const int q_head = blockIdx.x;
    if (q_head >= n_heads_q) return;
    const int part = blockIdx.y;
    const int n_par = gridDim.y;
    const int kv_head = q_head / (n_heads_q / n_heads_kv);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_kv = pos_ptr[0] + 1;

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

    int q_quad[QPL];
    float q_d[QPL];
#pragma unroll
    for (int i = 0; i < QPL; ++i) {
        const int j = lane + 32 * i;
        q_quad[i] = ((const int*) q_qs)[j];
        q_d[i] = q_d_block[j / 8];
    }

    float run_max = -FLT_MAX / 2.0f;
    float run_sum = 0.0f;
    float acc[QPL][4];
#pragma unroll
    for (int i = 0; i < QPL; ++i) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f; acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    for (int tile = part * ATTN_VEC_TILE; tile < n_kv; tile += n_par * ATTN_VEC_TILE) {
        const int t0 = tile + warp * 32;

        float my_score = -ATTN_Q8_INF;
        for (int i = 0; i < 32; ++i) {
            const int t = t0 + i;
            if (t >= n_kv) break;
            const size_t row = ((size_t) t * n_heads_kv + kv_head) * NB;
            float c = 0.0f;
#pragma unroll
            for (int ii = 0; ii < QPL; ++ii) {
                const int j = lane + 32 * ii;
                const flambeau_block_q8_0* kb = k_cache + row + j / 8;
                const int k_quad = load_int_b2(kb->qs, j % 8);
                c += (float) kb->d * q_d[ii] * (float) dp4a(k_quad, q_quad[ii], 0);
            }
            c = attn_vec_warp_sum(c);
            if (lane == i) my_score = c * scale;
        }

        const float new_max = fmaxf(run_max, attn_vec_warp_max(my_score));
        const float scale_old = __expf(run_max - new_max);
        const float p = __expf(my_score - new_max);
        run_sum = run_sum * scale_old + p;
        run_max = new_max;
#pragma unroll
        for (int ii = 0; ii < QPL; ++ii) {
            acc[ii][0] *= scale_old; acc[ii][1] *= scale_old; acc[ii][2] *= scale_old; acc[ii][3] *= scale_old;
        }

        for (int i = 0; i < 32; ++i) {
            const int t = t0 + i;
            if (t >= n_kv) break;
            const float pk = __shfl_sync(0xffffffffu, p, i, 32);
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

    const float warp_sum = attn_vec_warp_sum(run_sum);
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
