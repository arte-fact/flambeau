// mmvq_q8_0_t128_vdr2 — Q8_0 single-row MMVQ combining t128 occupancy
// with the VDR=2 inner-loop unroll.
// C9-followup. C9's standalone t128 lost decisively to the cycle-1 vdr2
// default (-3.5 % on 27B-Q8_0 Qwen3.6, -19.2 % on Qwen3.5) because t128
// halves the threads/block while keeping VDR=1, throwing away the ILP
// win that vdr2 already amortizes via per-thread DP4A unrolling.
// This kernel keeps the vdr2 inner-loop pattern (2 dp4a per "pair slot",
// scheduler overlap on the 2nd dp4a's source fetch) but at 128 t/block
// = 2 wave64s/CU = 2 in-flight blocks/CU at the gfx906 occupancy ceiling.
// The expected win: vdr2's ILP * t128's latency-hiding > vdr2 alone.
// Thread layout (128 threads / block, 1 row / block):
// lane_in_grp = tid & 3 — 0..3: which int32-pair within a block
// block_idx = tid >> 2 — 0..31: 32 blocks/iter (vs 64 in 256t vdr2)
// Outer stride = 32 blocks/iter.
//
// Two output dtypes via templated __device__ body:
//   flambeau_mmvq_q8_0_t128_vdr2_q8_1      → F32 dst (legacy)
//   flambeau_mmvq_q8_0_t128_vdr2_q8_1_f16  → F16 dst (saturating)

#include "block_quant.cuh"
#include "gfx906.cuh"
#include "mmvq_store.cuh"

#define MMVQ_T128_VDR2_BLOCK_THREADS 128
#define MMVQ_T128_VDR2_WARPS_PER_BLOCK (MMVQ_T128_VDR2_BLOCK_THREADS / WARP_SIZE)
#define MMVQ_T128_VDR2_THREADS_PER_QBLK 4
#define MMVQ_T128_VDR2_BLOCKS_PER_ITER (MMVQ_T128_VDR2_BLOCK_THREADS / MMVQ_T128_VDR2_THREADS_PER_QBLK)

static __device__ __forceinline__ int flambeau_dp4a_t128_vdr2(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

template<typename OutT>
__device__ void mmvq_q8_0_t128_vdr2_q8_1_body(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    OutT* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid         = threadIdx.x;
    const int warp        = tid / WARP_SIZE;
    const int lane        = tid & (WARP_SIZE - 1);
    const int lane_in_grp = tid & 3;
    const int block_idx   = tid >> 2;

    const flambeau_block_q8_0* xrow = x + (size_t) row * n_blocks_per_row;

    float acc = 0.0f;
    for (int b = block_idx; b < n_blocks_per_row; b += MMVQ_T128_VDR2_BLOCKS_PER_ITER) {
        const flambeau_block_q8_0* bx = xrow + b;
        const flambeau_block_q8_1* by = y + b;

        // VDR=2: 2 consecutive int32s (8 Q8 bytes) per thread.
        const int xi0 = ((const int*) bx->qs)[lane_in_grp * 2 + 0];
        const int xi1 = ((const int*) bx->qs)[lane_in_grp * 2 + 1];
        const int yi0 = ((const int*) by->qs)[lane_in_grp * 2 + 0];
        const int yi1 = ((const int*) by->qs)[lane_in_grp * 2 + 1];

        int sumi = flambeau_dp4a_t128_vdr2(xi0, yi0, 0);
        sumi     = flambeau_dp4a_t128_vdr2(xi1, yi1, sumi);

        const float d = (float) bx->d * (float) by->d;
        acc += d * (float) sumi;
    }

    acc = gfx906_warp_reduce_sum(acc);

    __shared__ float s_warp[MMVQ_T128_VDR2_WARPS_PER_BLOCK];
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < MMVQ_T128_VDR2_WARPS_PER_BLOCK) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = MMVQ_T128_VDR2_WARPS_PER_BLOCK / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, WARP_SIZE);
        }
        if (lane == 0) {
            mmvq_store<OutT>(dst, row, v);
        }
    }
}

extern "C" __global__ __launch_bounds__(MMVQ_T128_VDR2_BLOCK_THREADS)
void flambeau_mmvq_q8_0_t128_vdr2_q8_1(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    float* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q8_0_t128_vdr2_q8_1_body<float>(x, y, dst, n_rows, n_blocks_per_row);
}

extern "C" __global__ __launch_bounds__(MMVQ_T128_VDR2_BLOCK_THREADS)
void flambeau_mmvq_q8_0_t128_vdr2_q8_1_f16(
    const flambeau_block_q8_0* __restrict__ x,
    const flambeau_block_q8_1* __restrict__ y,
    fb_fp16_t* __restrict__ dst,
    const int n_rows,
    const int n_blocks_per_row
) {
    mmvq_q8_0_t128_vdr2_q8_1_body<fb_fp16_t>(x, y, dst, n_rows, n_blocks_per_row);
}
