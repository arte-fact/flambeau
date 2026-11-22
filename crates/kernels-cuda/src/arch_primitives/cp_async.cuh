#pragma once
// cp.async global -> shared copies (sm_80+). NVIDIA-only (rule 5).

#include <cuda_runtime.h>

namespace flambeau_cp_async {

static __device__ __forceinline__ unsigned int to_shared(const void* p) {
    return (unsigned int) __cvta_generic_to_shared(p);
}

// 16-byte copy that bypasses L1; both addresses 16-byte aligned.
static __device__ __forceinline__ void cg_16(unsigned int dst_shared, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(dst_shared), "l"(src));
#else
    (void) dst_shared; (void) src;
#endif
}

// 16-byte copy reading only the first `src_bytes` (0..16) of `src`, the rest
// of the destination zero-filled; both addresses 16-byte aligned.
static __device__ __forceinline__ void cg_16_partial(unsigned int dst_shared, const void* src, int src_bytes) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" : : "r"(dst_shared), "l"(src), "r"(src_bytes));
#else
    (void) dst_shared; (void) src; (void) src_bytes;
#endif
}

static __device__ __forceinline__ void commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

// Wait until at most N committed groups are still in flight.
template <int N>
static __device__ __forceinline__ void wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" : : "n"(N));
#endif
}

static __device__ __forceinline__ void wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;");
#endif
}

}  // namespace flambeau_cp_async
