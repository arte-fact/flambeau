#pragma once
// F16 tensor-core (mma.sync m16n8k16) fragment primitives for sm_80+ flash
// attention. Architectural rule 5: mma.sync is NVIDIA-only, so this lives in
// kernels-cuda. Sibling of mma_int8.cuh (m16n8k32 s8); same fragment thread maps
// where they coincide. The accumulator is f32 (online-softmax precision), so its
// thread->element map is identical to mma_int8's tile_i16 C layout.
//
// mma form: D(f32 16x8) += A(f16 16x16, row-major) @ B(f16 16x8, col-major).
// A tile packs 8 f16 / thread as 4 .f16x2 regs; B packs 4 f16 / thread as 2
// .f16x2 regs; D packs 4 f32 / thread. get_i/get_j give (row, col) of the l-th
// element for the calling lane (threadIdx.x % 32).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flambeau_mma_f16 {

// A operand: M16 x K16 f16, row-major. 4 regs, each a .f16x2 packing two
// K-consecutive f16. reg l -> row = group + (l%2)*8, col base = tid*2 + (l/2)*8.
struct tile_a {
    static constexpr int ne = 4;
    unsigned x[ne] = {0, 0, 0, 0};
    static __device__ __forceinline__ int row(int l) { return (threadIdx.x % 32) / 4 + (l % 2) * 8; }
    static __device__ __forceinline__ int col(int l) { return ((threadIdx.x % 32) % 4) * 2 + (l / 2) * 8; }
};

// B operand: K16 x N8 f16, col-major. 2 regs, each a .f16x2 packing two
// K-consecutive f16 of the same N column. reg l -> col(N) = group,
// row(K) base = tid*2 + l*8.
struct tile_b {
    static constexpr int ne = 2;
    unsigned x[ne] = {0, 0};
    static __device__ __forceinline__ int col(int)    { return (threadIdx.x % 32) / 4; }
    static __device__ __forceinline__ int row(int l)  { return ((threadIdx.x % 32) % 4) * 2 + l * 8; }
};

// Accumulator / output: M16 x N8 f32. Same map as mma_int8::tile_i16.
struct tile_c {
    static constexpr int ne = 4;
    float x[ne] = {0.0f, 0.0f, 0.0f, 0.0f};
    static __device__ __forceinline__ int get_i(int l) { return ((l / 2) * 8) + (threadIdx.x % 32) / 4; }
    static __device__ __forceinline__ int get_j(int l) { return (((threadIdx.x % 32) % 4) * 2) + (l % 2); }
};

// Load A from a row-major [16][stride] f16 tile. Each reg is a contiguous
// half2 (cols col(l), col(l)+1 of row row(l)).
static __device__ __forceinline__ void load_a(tile_a& t, const __half* __restrict__ base, int stride) {
    #pragma unroll
    for (int l = 0; l < tile_a::ne; ++l)
        t.x[l] = *(const unsigned*) (base + tile_a::row(l) * stride + tile_a::col(l));
}

// Like load_a but for a partial M-tile: `row0` points at row 0 of the matrix,
// and each fragment's global row (start_row + row(l)) is clamped to `n_rows - 1`
// so an over-the-end tile reads in-bounds (its garbage output is dropped by a
// guarded store).
static __device__ __forceinline__ void load_a_bounded(
    tile_a& t, const __half* __restrict__ row0, int stride, int start_row, int n_rows) {
    #pragma unroll
    for (int l = 0; l < tile_a::ne; ++l) {
        int gr = start_row + tile_a::row(l);
        if (gr >= n_rows) gr = n_rows - 1;
        t.x[l] = *(const unsigned*) (row0 + (size_t) gr * stride + tile_a::col(l));
    }
}

// Load B from a row-major [N][stride] f16 tile (the K matrix in S=Q·Kᵀ stored
// [n][d]); the two K-consecutive f16 of a B reg are contiguous there.
static __device__ __forceinline__ void load_b(tile_b& t, const __half* __restrict__ base, int stride) {
    #pragma unroll
    for (int l = 0; l < tile_b::ne; ++l)
        t.x[l] = *(const unsigned*) (base + tile_b::col(l) * stride + tile_b::row(l));
}

// Load B from a row-major [K][stride] f16 tile (the V matrix in O=P·V stored
// [n][dv], K=n N=dv); the two K-consecutive f16 of a B reg are stride apart, so
// they are gathered and packed rather than read as one half2.
static __device__ __forceinline__ void load_b_kn(tile_b& t, const __half* __restrict__ base, int stride) {
    #pragma unroll
    for (int l = 0; l < tile_b::ne; ++l) {
        __half lo = base[(tile_b::row(l) + 0) * stride + tile_b::col(l)];
        __half hi = base[(tile_b::row(l) + 1) * stride + tile_b::col(l)];
        __half2 h2 = __halves2half2(lo, hi);
        t.x[l] = *reinterpret_cast<const unsigned*>(&h2);
    }
}

// D += A @ B over K=16. f16 x f16 -> f32 accumulate.
static __device__ __forceinline__ void mma(tile_c& d, const tile_a& a, const tile_b& b) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]), "r"(b.x[1]));
#else
    (void) d; (void) a; (void) b;
#endif
}

}  // namespace flambeau_mma_f16
