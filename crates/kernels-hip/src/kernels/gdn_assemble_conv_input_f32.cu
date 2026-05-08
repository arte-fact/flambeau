// gdn_assemble_conv_input_f32 — 3.d.1 fused `conv_input = [history, current]`.
// Replaces two back-to-back device-to-device memcpys in `assemble_conv_input`
// (forward_gdn_decode) with a single pointwise kernel. At decode each GDN
// layer fires this pattern once per token — on 40-layer Qwen3.6-35B with
// ~30 GDN layers × 65 tokens / 4 ranks that's ~490 decode-time memcpy calls
// the driver no longer has to process.
// Contract:
// history[(conv_kernel - 1) * conv_channels] F32 — the last K-1 rows
// current[conv_channels] F32 — the newest row
// conv_input[conv_kernel * conv_channels] F32 — written:
// conv_input[0..K-1, :] = history[:]
// conv_input[K-1, :] = current[:]
// Launch: 1D, ceil(conv_kernel * conv_channels / 256) blocks × 256 threads.
// One thread per F32 element.

#include <hip/hip_runtime.h>

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
