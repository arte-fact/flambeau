// gdn_conv_trio_decode_f32_batched_slots — fused single-token GDN conv
// trio (assemble_conv_input + causal_conv1d + shift_conv_history) across
// N decode slots, each owning its own `conv_history` buffer.
//
// Per layer per decode step the per-slot version costs 3 launches per
// slot (assemble DtoD memcpy + causal_conv1d kernel + shift DtoD memcpy)
// → 3N launches. This kernel does all three for every slot in one launch:
// each block handles a `(slot, channel_tile)` pair; threads load the
// slot's K-1 history rows + the slot's new qkv_mixed row from
// scratch.qkv_mixed_f32, compute the conv output for their channel, write
// to scratch.conv_out, and then shift the history (drop oldest row,
// append new row) — all in registers, no LDS.
//
// Weight layout matches `causal_conv1d_f32`: `[conv_channels, conv_kernel]`
// row-major (tap axis innermost).
//
// Grid: (ceil(conv_channels / THREADS), N_slots, 1). Block: 256 threads.

#include <hip/hip_runtime.h>

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
    // Load K-1 history values for this channel into registers.
    float h_vals[CONV_TRIO_KERNEL_MAX - 1];
    #pragma unroll
    for (int k = 0; k < CONV_TRIO_KERNEL_MAX - 1; ++k) {
        if (k < hist_rows) {
            h_vals[k] = history[(size_t) k * conv_channels + c];
        }
    }

    // Conv: y[c] = Σ_{k=0..K-1} w[c, k] · (k < K-1 ? hist[k] : new_val).
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < CONV_TRIO_KERNEL_MAX - 1; ++k) {
        if (k < hist_rows) {
            acc += h_vals[k] * w_c[k];
        }
    }
    acc += new_val * w_c[hist_rows];
    conv_out[(size_t) slot * conv_channels + c] = acc;

    // Shift history: new_history[k] = (k < K-2) ? hist[k+1] : new_val.
    #pragma unroll
    for (int k = 0; k < CONV_TRIO_KERNEL_MAX - 1; ++k) {
        if (k < hist_rows) {
            const float v =
                (k + 1 < hist_rows) ? h_vals[k + 1] : new_val;
            history[(size_t) k * conv_channels + c] = v;
        }
    }
}
