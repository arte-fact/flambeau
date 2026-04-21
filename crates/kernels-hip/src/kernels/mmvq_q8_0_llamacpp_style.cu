// mmvq_q8_0_llamacpp_style — port of llama.cpp's `mul_mat_vec_q<Q8_0, 1, false, false>`
// specialised for our GCN/gfx906 decode case.
//
// Corrected config after reading `calc_nwarps` for MMVQ_PARAMETERS_GCN with
// ncols_dst=1: **nwarps = 2**, rows_per_cuda_block = 1, warp_size = 64.
// Threads/block = 2 × 64 = 128 (NOT 64 as in my initial port).
//
// Purpose: A/B microbench vs our `mmvq_q8_0_dp4a_vdr2` (256 threads / 4 warps).
// Isolates whether llama.cpp's 128-thread config beats our 256-thread config
// on MI50 for our specific decode shapes.
//
// Reference: /artefact/llama.cpp/ggml/src/ggml-cuda/mmvq.cu::calc_nwarps (line 309-322)
//            + mul_mat_vec_q body (lines 391-590). Inner dot:
//            vecdotq.cuh:243-255 (vec_dot_q8_0_q8_1_impl<float, 2>).

#include "block_quant.cuh"
#include "gfx906.cuh"

#define LLAMACPP_NWARPS 2
#define LLAMACPP_WARP_SIZE 64
#define LLAMACPP_THREADS (LLAMACPP_NWARPS * LLAMACPP_WARP_SIZE)  // 128

extern "C" __global__ __launch_bounds__(LLAMACPP_THREADS, 1)
void flambeau_mmvq_q8_0_llamacpp_style_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    constexpr int qi  = 8;         // QI8_0 = QK8_0 / 4 = 8 int32s per block
    constexpr int vdr = 2;         // VDR_Q8_0_Q8_1_MMVQ
    constexpr int warp_size = LLAMACPP_WARP_SIZE;
    constexpr int nwarps = LLAMACPP_NWARPS;

    const int tid  = threadIdx.x;          // 0..127
    const int warp = tid / warp_size;      // 0 or 1
    const int lane = tid & (warp_size - 1);
    const int row0 = blockIdx.x;
    if (row0 >= n_rows) return;

    // blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 2 * 64 / 8 = 32.
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const flambeau_block_q8_0* xrow = x + (size_t) row0 * n_blocks_per_row;

    float tmp = 0.0f;

    // Threads partition blocks: kbx = tid / (qi/vdr) = tid / 4 ∈ {0..31}.
    // Each thread's kqs = 2 * (tid % 4) ∈ {0, 2, 4, 6}.
    for (int kbx = tid / (qi / vdr); kbx < n_blocks_per_row; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi / vdr));
        const flambeau_block_q8_0* bx = xrow + kbx;
        const flambeau_block_q8_1* by = y + kbx;

        const int* qs_x = ((const int*) bx->qs) + kqs;
        const int* qs_y = ((const int*) by->qs) + kqs;

        int sumi = __builtin_amdgcn_sdot4(qs_x[0], qs_y[0], 0,    false);
        sumi     = __builtin_amdgcn_sdot4(qs_x[1], qs_y[1], sumi, false);

        const float d_x = (float) bx->d;
        const float d_y = (float) by->d;
        tmp += (d_x * d_y) * (float) sumi;
    }

    // In-warp reduction first.
    #pragma unroll
    for (int off = warp_size / 2; off > 0; off >>= 1) {
        tmp += __shfl_xor(tmp, off, warp_size);
    }

    // Inter-warp reduction via shared memory (nwarps=2 → 1 slot needed).
    __shared__ float tmp_shared[nwarps - 1][warp_size];
    if (warp > 0) {
        tmp_shared[warp - 1][lane] = tmp;
    }
    __syncthreads();
    if (warp > 0) return;

    // Warp 0 sums in warp 1's partial.
    tmp += tmp_shared[0][lane];

    if (lane == 0) {
        dst[row0] = tmp;
    }
}
