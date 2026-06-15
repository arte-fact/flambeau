#pragma once
// Backend-neutral compilation shim. Resolves the GPU runtime header and the
// fp16 storage type for both nvcc (CUDA) and hipcc (HIP) so algorithmic
// headers in kernels-shared carry zero backend intrinsics (architectural
// rule 5). fb_fp16_t is the 2-byte half storage type used for pointer-casts
// from mmap'd GGUF data; it must stay layout-compatible with ggml's
// ggml_fp16_t on both backends.

// Fixed-width integer types (uint8_t / int8_t / ...). HIP's hip_runtime.h
// drags these in transitively; nvcc's cuda_runtime.h does not — include
// explicitly so the shared block-quant layouts resolve on both backends.
#include <stdint.h>

#if defined(__CUDACC__)
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  typedef __half fb_fp16_t;
  // CUDA removed the non-sync warp shuffles (deprecated CUDA 9, gone by 13).
  // Shared algorithmic headers use the bare `__shfl_xor(var, mask, width)`
  // form; map it to the sync version with a full-warp mask (kernels launch
  // warp-aligned, so every lane participates).
  #ifndef __shfl_xor
    #define __shfl_xor(var, lane_mask, width) \
        __shfl_xor_sync(0xffffffffu, (var), (lane_mask), (width))
  #endif
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP_DEVICE_COMPILE__)
  #include <hip/hip_runtime.h>
  typedef _Float16 fb_fp16_t;
#else
  #error "backend_compat.cuh: unknown GPU backend (need __CUDACC__ or __HIP_PLATFORM_AMD__)"
#endif
