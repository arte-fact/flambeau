// gdn_split_qkv_f32 — fused replacement for `gather_qkv_strided`'s 3×L
// DtoD memcpy loop ().
// Previous: for t in 0..n_tokens { 3 × hipMemcpyAsync(DeviceToDevice) }
// → 3 * L memcpy launches per GDN layer per prefill pass.
// → at L=512 × 20 GDN layers × 3 = 30,720 driver calls per prefill.
// This: single kernel launch per call site. 1 thread = 1 output element,
// 1D grid over the total output span (3 * qk_size per token for Q+K,
// plus v_size per token for V).
// Layout — source `silu_out[n_tokens, conv_channels]` where
// conv_channels = 2*qk_size + v_size
// and each row is packed as [q_section | k_section | v_section].
// We produce 3 contiguous outputs:
// q_out[n_tokens, qk_size] ← silu_out[:, 0 .. qk_size]
// k_out[n_tokens, qk_size] ← silu_out[:, qk_size .. 2*qk_size]
// v_out[n_tokens, v_size] ← silu_out[:, 2*qk_size .. 2*qk_size + v_size]
// This kernel is pure memory bandwidth — same HBM bytes move as the
// memcpy loop, but the launch overhead is O(1) per call site instead of
// O(3L). At L=512 that's a ~1500× reduction in driver calls per layer.

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef BLOCK
#define BLOCK 256
#endif

extern "C" __global__ void flambeau_gdn_split_qkv_f32(
    const float* __restrict__ silu_out,  // [n_tokens, conv_channels]
    float*       __restrict__ q_out,     // [n_tokens, qk_size]
    float*       __restrict__ k_out,     // [n_tokens, qk_size]
    float*       __restrict__ v_out,     // [n_tokens, v_size]
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
