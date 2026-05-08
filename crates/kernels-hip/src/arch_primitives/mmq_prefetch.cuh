#pragma once
// gfx906 L2-prefetch helpers for MMQ kernels. Ported from
// /artefact/llamacpp-turbo/.../ggml-cuda/gfx906/matmul/mmq-prefetch.cuh
// ().
// Pattern: at the top of each K-iteration in a MMQ kernel, issue a
// handful of `global_load_dword` instructions pointed at the NEXT
// iteration's X / Y tile addresses. The loads don't wait — they just
// warm the L2 cache. When the real cooperative load runs one iteration
// later, it hits warm cachelines instead of stalling on HBM.
// The "consume" helper (`gfx906_prefetch_consume`) is a no-op
// `v_mov_b32 %0, %0` that the compiler can't dead-code-eliminate; without
// it the `global_load_dword` return value is unused and the compiler
// may elide the load entirely.
// All helpers are `no-op` on non-gfx906 targets — the kernel falls back
// to synchronous loads.

#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx906__)

/// Prefetch the NEXT iteration's Y-tile base address into L2.
/// Caller: expects a 2-D block with `threadIdx.y == 0` being the
/// "prefetch warp". Lanes 0..15 each issue one `global_load_dword` at
/// 64-byte-stride offsets to warm a 1 KB window.
/// `y_next_words` must already point at the FIRST int of the next
/// iteration's Y tile (caller computes stride from Y layout).
/// Returns a dummy int that the caller must pass to
/// `gfx906_prefetch_consume` to prevent compiler elimination.
static __device__ __forceinline__ int gfx906_prefetch_y_next(
    const int* __restrict__ y_next_words
) {
    if (threadIdx.y != 0 || threadIdx.x >= 16) return 0;
    const int* addr = y_next_words + (int) threadIdx.x * 16;
    int data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(data)
        : "v"(addr)
        : "memory"
    );
    return data;
}

/// Prefetch the NEXT iteration's X-tile rows into L2.
/// Caller: lanes 0..15 of warp 1 each prefetch one row of X. Returns a
/// dummy that must be consumed.
static __device__ __forceinline__ int gfx906_prefetch_x_next(
    const char* __restrict__ x_base,
    const int   offset_x_bytes_next,
    const int   stride_row_x_bytes
) {
    if (threadIdx.y != 1 || threadIdx.x >= 16) return 0;
    const char* row_ptr = x_base + offset_x_bytes_next
                        + (int) threadIdx.x * stride_row_x_bytes;
    int data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(data)
        : "v"((const int*) row_ptr)
        : "memory"
    );
    return data;
}

/// Keep the prefetched dword "live" across the compute loop so the
/// compiler doesn't remove the global_load. Inlined v_mov_b32 %0, %0.
static __device__ __forceinline__ void gfx906_prefetch_consume(int v) {
    asm volatile("v_mov_b32 %0, %0\n" : "+v"(v));
}

/// Single-warp prefetch (no 2D `threadIdx.y` gate). For 1D-block MMQ
/// kernels (Q5_K wave64, any kernel with `block = (N, 1, 1)`). Each
/// lane 0..15 issues one `global_load_dword` at `addr + lane * 16`.
static __device__ __forceinline__ int gfx906_prefetch_next_1d(
    const int* __restrict__ next_words
) {
    if (threadIdx.x >= 16) return 0;
    const int* addr = next_words + (int) threadIdx.x * 16;
    int data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(data)
        : "v"(addr)
        : "memory"
    );
    return data;
}

#else  // non-gfx906 fallback: no-op

static __device__ __forceinline__ int gfx906_prefetch_y_next(const int*) { return 0; }
static __device__ __forceinline__ int gfx906_prefetch_x_next(const char*, int, int) { return 0; }
static __device__ __forceinline__ void gfx906_prefetch_consume(int) {}
static __device__ __forceinline__ int gfx906_prefetch_next_1d(const int*) { return 0; }

#endif
