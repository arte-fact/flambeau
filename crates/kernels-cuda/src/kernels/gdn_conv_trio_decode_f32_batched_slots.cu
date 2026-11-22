// gdn_conv_trio_decode_f32_batched_slots — fused single-token GDN conv trio
// (assemble + causal_conv1d + shift_history) across N decode slots, each with
// its own per-slot history buffer. Register-resident, no LDS.
// grid (ceil(conv_channels/256), N_slots, 1), block 256.

#include <cuda_runtime.h>

#ifndef CONV_TRIO_THREADS
#define CONV_TRIO_THREADS 256
#endif
#ifndef CONV_TRIO_KERNEL_MAX
#define CONV_TRIO_KERNEL_MAX 8
#endif

extern "C" __global__ void flambeau_gdn_conv_trio_decode_f32_batched_slots(
    float * const * __restrict__ slot_history_ptrs,
    const float * __restrict__ qkv_mixed,
    const float * __restrict__ weight,
    float * __restrict__ conv_out,
    int n_slots,
    int conv_channels,
    int conv_kernel
) {
    const int slot = blockIdx.y;
    const int c    = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_slots || c >= conv_channels) return;

    float * __restrict__ history = slot_history_ptrs[slot];
    const float * __restrict__ w_c = weight + (size_t) c * conv_kernel;
    const float new_val = qkv_mixed[(size_t) slot * conv_channels + c];

    const int hist_rows = conv_kernel - 1;
    float h_vals[CONV_TRIO_KERNEL_MAX - 1];
    #pragma unroll
    for (int k = 0; k < CONV_TRIO_KERNEL_MAX - 1; ++k) {
        if (k < hist_rows) {
            h_vals[k] = history[(size_t) k * conv_channels + c];
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < CONV_TRIO_KERNEL_MAX - 1; ++k) {
        if (k < hist_rows) {
            acc += h_vals[k] * w_c[k];
        }
    }
    acc += new_val * w_c[hist_rows];
    conv_out[(size_t) slot * conv_channels + c] = acc;

    #pragma unroll
    for (int k = 0; k < CONV_TRIO_KERNEL_MAX - 1; ++k) {
        if (k < hist_rows) {
            const float v = (k + 1 < hist_rows) ? h_vals[k + 1] : new_val;
            history[(size_t) k * conv_channels + c] = v;
        }
    }
}
