// gdn_assemble_conv_input_f32 — conv_input = [history (K-1 rows), current].
// One thread per F32 element. Pure copy.

#include <cuda_runtime.h>

extern "C" __global__ void flambeau_gdn_assemble_conv_input_f32(
    const float* __restrict__ history,
    const float* __restrict__ current,
    float*       __restrict__ conv_input,
    const int conv_channels,
    const int conv_kernel
) {
    const int n = conv_channels * conv_kernel;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int hist_elems = conv_channels * (conv_kernel - 1);
    if (idx < hist_elems) {
        conv_input[idx] = history[idx];
    } else {
        conv_input[idx] = current[idx - hist_elems];
    }
}
