// Device-position embedding-row gather, F16. CUDA-only. Copies one hidden-width
// row `embd[token_id * hidden + j]` into `dst[j]`, reading `token_id` from a
// device buffer (`ctx_ptr[1]` of the decode-graph `[position, token_id]`
// context) so a captured decode graph re-sources the embedding row per token
// with no memcpy-node rebind. Grid over `hidden`; used ONLY under decode-graph
// capture (the non-capture path keeps the plain DtoD memcpy).
#include "sm_80.cuh"
#include "block_quant.cuh"

extern "C" __global__ void flambeau_embed_gather_f16_dpos(
    const fb_fp16_t* __restrict__ embd,
    fb_fp16_t* __restrict__ dst,
    const int hidden,
    const int* __restrict__ ctx_ptr
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden) return;
    const int token_id = ctx_ptr[1];
    dst[j] = embd[(size_t) token_id * hidden + j];
}
