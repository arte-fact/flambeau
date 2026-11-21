// gdn_split_qkv_f32 — split packed silu_out[:, q|k|v] into contiguous q/k/v
// outputs. One thread per element. Pure copy.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BLOCK
#define BLOCK 256
#endif

extern "C" __global__ void flambeau_gdn_split_qkv_f32(
    const float* __restrict__ silu_out,
    float*       __restrict__ q_out,
    float*       __restrict__ k_out,
    float*       __restrict__ v_out,
    const int n_tokens,
    const int qk_size,
    const int v_size
) {
    const int conv_channels = 2 * qk_size + v_size;
    const size_t total = (size_t) n_tokens * (size_t) conv_channels;
    const size_t tid = (size_t) blockIdx.x * BLOCK + threadIdx.x;
    if (tid >= total) return;

    const int t = (int) (tid / (size_t) conv_channels);
    const int c = (int) (tid - (size_t) t * (size_t) conv_channels);
    const float v = silu_out[tid];

    if (c < qk_size) {
        q_out[(size_t) t * qk_size + c] = v;
    } else if (c < 2 * qk_size) {
        k_out[(size_t) t * qk_size + (c - qk_size)] = v;
    } else {
        v_out[(size_t) t * v_size + (c - 2 * qk_size)] = v;
    }
}
