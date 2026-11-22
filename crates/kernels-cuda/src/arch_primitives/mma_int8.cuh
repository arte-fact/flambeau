#pragma once
// Int8 tensor-core (mma.sync) fragment primitives for sm_80+ (Ampere/Ada).
// Architectural rule 5: mma.sync is NVIDIA-only, so this lives in kernels-cuda,
// never in kernels-shared. gfx906 has no matrix core; the HIP daily target uses
// the dp4a MMQ instead. Fragment thread->element maps and the m16n8k32 op match
// the PTX ISA (and llama.cpp's ggml_cuda_mma, the correctness oracle).
//
// A is row-major M(16) x K(32) s8, B is col-major K(32) x N(8) s8, D/C is
// M(16) x N(8) s32. Tiles hold `ne` physical 32-bit lanes per thread; a 32-bit
// lane packs 4 s8 along K. get_i/get_j give the (row, physical-32bit-col) of the
// l-th element for the calling lane (threadIdx.x % 32).

#include <cuda_runtime.h>

namespace flambeau_mma {

// M16 x K32 s8 operand (also the M16 x N8 s32 accumulator — same layout).
struct tile_i16 {
    static constexpr int I = 16;
    static constexpr int ne = 4;
    int x[ne] = {0, 0, 0, 0};

    static __device__ __forceinline__ int get_i(int l) {
        return ((l / 2) * 8) + (threadIdx.x % 32) / 4;
    }
    static __device__ __forceinline__ int get_j(int l) {
        return (((threadIdx.x % 32) % 4) * 2) + (l % 2);
    }
};

// N8 x K32 s8 operand (B), col-major: I indexes N, J indexes K (32-bit packed).
struct tile_i8 {
    static constexpr int I = 8;
    static constexpr int ne = 2;
    int x[ne] = {0, 0};

    static __device__ __forceinline__ int get_i(int l) {
        return (threadIdx.x % 32) / 4;
    }
    static __device__ __forceinline__ int get_j(int l) {
        return (l * 4) + ((threadIdx.x % 32) % 4);
    }
};

// Direct indexed load from a row-major [I][stride] int32 tile in shared memory.
// Correct for the B (8x8) input and C output, whose fragment layout matches
// get_i/get_j. The A (16x8) *input* fragment differs — use load_ldmatrix.
template <typename Tile>
static __device__ __forceinline__ void load_generic(Tile& t, const int* __restrict__ base, int stride) {
    #pragma unroll
    for (int l = 0; l < Tile::ne; ++l) {
        t.x[l] = base[Tile::get_i(l) * stride + Tile::get_j(l)];
    }
}

// Load the A operand (16x32 s8) from a canonical row-major [16][stride] int32
// shared tile into the mma input fragment. `base` must be 16-byte aligned.
static __device__ __forceinline__ void load_ldmatrix(tile_i16& t, const int* __restrict__ base, int stride) {
#if __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x % 32;
    const int* xs = base + (lane % 16) * stride + (lane / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(t.x[0]), "=r"(t.x[1]), "=r"(t.x[2]), "=r"(t.x[3]) : "l"(xs));
#else
    (void) t; (void) base; (void) stride;
#endif
}

// D += A @ B over K=32. s8 x s8 -> s32, Ampere-native single instruction.
static __device__ __forceinline__ void mma(tile_i16& d, const tile_i16& a, const tile_i8& b) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d.x[0]), "+r"(d.x[1]), "+r"(d.x[2]), "+r"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]), "r"(b.x[1]));
#else
    (void) d; (void) a; (void) b;
#endif
}

}  // namespace flambeau_mma
