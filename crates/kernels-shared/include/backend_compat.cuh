#pragma once
// Backend-neutral compilation shim. Resolves the GPU runtime header and the
// fp16 storage type for both nvcc (CUDA) and hipcc (HIP) so algorithmic
// headers in kernels-shared carry zero backend intrinsics (architectural
// rule 5). fb_fp16_t is the 2-byte half storage type used for pointer-casts
// from mmap'd GGUF data; it must stay layout-compatible with ggml's
// ggml_fp16_t on both backends.
#if defined(__CUDACC__)
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  typedef __half fb_fp16_t;
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP_DEVICE_COMPILE__)
  #include <hip/hip_runtime.h>
  typedef _Float16 fb_fp16_t;
#else
  #error "backend_compat.cuh: unknown GPU backend (need __CUDACC__ or __HIP_PLATFORM_AMD__)"
#endif
