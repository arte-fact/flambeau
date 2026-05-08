// mmq_q8_0_4warp — 4-warp LDS-tiled MMQ for Q8_0 weights × Q8_1 activation.
// This is the first-class MMQ prefill kernel for Q8_0 on gfx906. It is a
// fresh port of llamacpp-turbo's `ggml-cuda/mmq.cuh` 4-warp pattern, with
// the following simplifications (all scheduled to land as follow-ups on
// top of this port):
// * no stream-K fixup — every output tile is computed by exactly one
// thread block (see turbo's `mul_mat_q_process_tile` for the
// fixup-enabled variant);
// * no L2 prefetch loop;
// * no software-pipelined asm load/store macros
// (`GFX906_LOAD_TILES_Q8_0_ASYNC` / `GFX906_STORE_TILES_Q8_0_LDS_*`);
// * uses the `__builtin_amdgcn_sdot4` intrinsic (gfx906 `v_dot4_i32_i8`)
// for the per-thread inner dot. The manual 4× int8 sign-extend + FMA
// version is kept as a comment in `dp4a()` below — same arithmetic,
// compiler sometimes lowers it, intrinsic guarantees the fast path.
// Tile shape:
// MMQ_Y = 32 output weight rows per block
// MMQ_X = 8 output batch columns per block
// MMQ_K = 32 K elements per iteration (= QK8_0)
// Thread layout:
// NWARPS = 4, WARP_SIZE = 64 → 256 threads/block
// Each thread computes exactly ONE output element `dst[batch, row]`.
// row_in_tile = warp * 8 + lane / 8 (0..31)
// col_in_tile = lane % 8 (0..7)
// Grid: (ceil(N / MMQ_Y), ceil(M / MMQ_X)).
// Per K-iter:
// 1. Load MMQ_Y × 32 Q8_0 quants into LDS `x_qs[256]` (one int per thread).
// 2. Load MMQ_Y scales into LDS `x_df[32]` (warp 0, lanes 0..31).
// 3. Load MMQ_X × 32 Q8_1 quants into LDS `y_qs[64]` (warp 0, all lanes).
// 4. Load MMQ_X scales into LDS `y_df[8]` (warp 0, lanes 0..7).
// 5. __syncthreads.
// 6. Each thread accumulates its output via manual 4× int8 dot over the
// 8 int32 qs slots of its (row, col) pair.
// 7. __syncthreads (before next iter's LDS writes).
// Correctness oracle: `mmq_q8_0_oracle` kernel (same impl_id family).

#include "block_quant.cuh"

#define MMQ_Y 32
#define MMQ_X 8
#define MMQ_K 32
#define NWARPS 4
#define WARP_SIZE 64
#define THREADS_PER_BLOCK (NWARPS * WARP_SIZE)

// 4-way packed int8 signed dot product using the gfx906 intrinsic.
// `__builtin_amdgcn_sdot4(a, b, c, clamp)` lowers directly to
// `v_dot4_i32_i8 vdst, a, b, c`. `clamp=false` keeps the wrap-on-overflow
// semantics (matches the manual version's int32 accumulation).
// Accumulator form keeps one VALU instruction per 4-element chunk and
// removes the 8 shifts + 4 sign-extends + 4 muls + 3 adds of the manual
// path. On gfx906 this is a 1/8-throughput MAD on SIMD; on GCN-era silicon
// it also gains 2× register-file bandwidth vs the manual sequence.
static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

extern "C" __global__ void flambeau_mmq_q8_0_4warp_q8_1(
    const flambeau_block_q8_0* __restrict__ x,   // [n_rows, n_blocks_per_row]
    const flambeau_block_q8_1* __restrict__ y,   // [n_batches, n_blocks_per_row]
    float* __restrict__ dst,                     // [n_batches, n_rows]
    const int n_rows,
    const int n_batches,
    const int n_blocks_per_row                   // K / 32
) {
    const int row_base   = blockIdx.x * MMQ_Y;
    const int batch_base = blockIdx.y * MMQ_X;

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;              // 0..3
    const int lane = tid % WARP_SIZE;              // 0..63

    const int row_in_tile = warp * 8 + lane / 8;   // 0..31
    const int col_in_tile = lane % 8;              // 0..7

    const int row   = row_base   + row_in_tile;
    const int batch = batch_base + col_in_tile;

    // LDS tiles for one K-iteration.
    __shared__ int   x_qs[MMQ_Y * 8];              // 32 rows × 8 int32 = 32 quant bytes × 32 rows
    __shared__ float x_df[MMQ_Y];                  // 32 scales
    __shared__ int   y_qs[MMQ_X * 8];              // 8 rows × 8 int32
    __shared__ float y_df[MMQ_X];                  // 8 scales (Q8_1 sum field unused in MMQ)

    float acc = 0.0f;

    for (int kb = 0; kb < n_blocks_per_row; ++kb) {
        // ---- Load X tile (32 rows × 8 ints) ----
        // tid ∈ [0, 256); maps 1:1 to the 256 ints. row = tid/8, qs_int = tid%8.
        {
            const int row_tile = tid / 8;            // 0..31
            const int qs_idx   = tid & 7;            // 0..7
            const int row_abs  = row_base + row_tile;
            if (row_abs < n_rows) {
                const flambeau_block_q8_0* bk = x + (size_t) row_abs * n_blocks_per_row + kb;
                // `qs_idx`-th int32 of the 32-byte `qs[]` array.
                const int* qs_as_int = reinterpret_cast<const int*>(bk->qs);
                x_qs[row_tile * 8 + qs_idx] = qs_as_int[qs_idx];
                if (qs_idx == 0) {
                    x_df[row_tile] = (float) bk->d;
                }
            } else {
                x_qs[row_tile * 8 + qs_idx] = 0;
                if (qs_idx == 0) {
                    x_df[row_tile] = 0.0f;
                }
            }
        }

        // ---- Load Y tile (8 rows × 8 ints) ----
        // Only warp 0 participates; tid ∈ [0, 64) loads the 64 ints, plus
        // lanes 0..7 load the scales.
        if (warp == 0) {
            const int bcol      = lane / 8;          // 0..7 → batch index in tile
            const int qs_idx    = lane & 7;          // 0..7
            const int batch_abs = batch_base + bcol;
            if (batch_abs < n_batches) {
                const flambeau_block_q8_1* by =
                    y + (size_t) batch_abs * n_blocks_per_row + kb;
                const int* yqs_as_int = reinterpret_cast<const int*>(by->qs);
                y_qs[bcol * 8 + qs_idx] = yqs_as_int[qs_idx];
                if (qs_idx == 0) {
                    y_df[bcol] = (float) by->d;
                }
            } else {
                y_qs[bcol * 8 + qs_idx] = 0;
                if (qs_idx == 0) {
                    y_df[bcol] = 0.0f;
                }
            }
        }

        __syncthreads();

        // ---- Compute this thread's (row_in_tile, col_in_tile) dot product ----
        if (row < n_rows && batch < n_batches) {
            int dot = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int xi = x_qs[row_in_tile * 8 + i];
                const int yi = y_qs[col_in_tile * 8 + i];
                dot = dp4a(xi, yi, dot);
            }
            const float scale = x_df[row_in_tile] * y_df[col_in_tile];
            acc += (float) dot * scale;
        }

        __syncthreads();
    }

    if (row < n_rows && batch < n_batches) {
        dst[(size_t) batch * n_rows + row] = acc;
    }
}
