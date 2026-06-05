// p2p_allreduce_residual — kernel-launched BAR1 P2P AllReduce + residual add.
// Port of mi50grad's `kernel_p2p_allreduce.hip`.
// Each rank launches its own copy of this kernel on its own stream. The
// kernel reads peer GPUs' partial buffers directly via BAR1-mapped P2P
// pointers (authorised at session init by `hipDeviceEnablePeerAccess`,
// see `crates/backend-hip/src/cluster.rs::probe_and_enable_peer_access`)
// and folds the sum into the local hidden buffer.
// hidden[i] += partial_local[i] + Σ partial_peerK[i]
// Cross-rank synchronisation lives at the launch boundary (HIP events on
// the producer GEMV's stream are recorded then waited on by each rank's
// AR stream before launch). The kernel itself reads peer memory as it
// stood at launch time and contains no in-kernel barriers.
// Variants:
// *_residual_tp{2,4} — residual add path (this file)
// *_sum_tp{2,4} — pure AllReduce sum, writes back into partial_local
// Grid: ceil(n / (256 * 2)) blocks × 256 threads. Each thread processes
// 2 fp16 elements via half2 packing (4-byte vectorised load/store on local
// HBM; peer half2 reads go through BAR1 at PCIe bandwidth).
// FP32 accumulation is mandatory on gfx906 (no MFMA; F16 add precision is
// thin enough that 4-way sums in F16 introduce visible drift on hidden
// states with high dynamic range).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "block_quant.cuh"

// 256-thread block = 4 wave64s per block. Pointwise kernel with ~10 VGPRs
// per thread; gfx906 occupancy is bounded only by the launch grid.
#define P2P_AR_THREADS 256

// ------------------------------------------------------------------------
// TP=4 residual: hidden += partial_local + partial_peer{0,1,2}
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_allreduce_residual_tp4(
    fb_fp16_t* __restrict__ hidden,
    const fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const fb_fp16_t* __restrict__ partial_peer1,
    const fb_fp16_t* __restrict__ partial_peer2,
    const unsigned int n
) {
    const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    if (idx + 1u < n) {
        const __half2 h  = *reinterpret_cast<const __half2*>(hidden        + idx);
        const __half2 pl = *reinterpret_cast<const __half2*>(partial_local + idx);
        // BAR1-mapped peer reads — single dword128/256-style transactions
        // generate `flat_load_dword` against peer HBM via PCIe BAR1.
        const __half2 p0 = *reinterpret_cast<const __half2*>(partial_peer0 + idx);
        const __half2 p1 = *reinterpret_cast<const __half2*>(partial_peer1 + idx);
        const __half2 p2 = *reinterpret_cast<const __half2*>(partial_peer2 + idx);

        const float h_lo  = __half2float(__low2half(h));
        const float h_hi  = __half2float(__high2half(h));
        const float pl_lo = __half2float(__low2half(pl));
        const float pl_hi = __half2float(__high2half(pl));
        const float p0_lo = __half2float(__low2half(p0));
        const float p0_hi = __half2float(__high2half(p0));
        const float p1_lo = __half2float(__low2half(p1));
        const float p1_hi = __half2float(__high2half(p1));
        const float p2_lo = __half2float(__low2half(p2));
        const float p2_hi = __half2float(__high2half(p2));

        const float lo = h_lo + pl_lo + p0_lo + p1_lo + p2_lo;
        const float hi = h_hi + pl_hi + p0_hi + p1_hi + p2_hi;

        *reinterpret_cast<__half2*>(hidden + idx) = __halves2half2(
            __float2half(lo), __float2half(hi));
    } else if (idx < n) {
        // Tail: odd element at the end.
        const float h  = __half2float(hidden[idx]);
        const float pl = __half2float(partial_local[idx]);
        const float p0 = __half2float(partial_peer0[idx]);
        const float p1 = __half2float(partial_peer1[idx]);
        const float p2 = __half2float(partial_peer2[idx]);
        hidden[idx] = (fb_fp16_t) (h + pl + p0 + p1 + p2);
    }
}

// ------------------------------------------------------------------------
// TP=2 residual: hidden += partial_local + partial_peer0
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_allreduce_residual_tp2(
    fb_fp16_t* __restrict__ hidden,
    const fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const unsigned int n
) {
    const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    if (idx + 1u < n) {
        const __half2 h  = *reinterpret_cast<const __half2*>(hidden        + idx);
        const __half2 pl = *reinterpret_cast<const __half2*>(partial_local + idx);
        const __half2 p0 = *reinterpret_cast<const __half2*>(partial_peer0 + idx);

        const float lo = __half2float(__low2half(h))
                       + __half2float(__low2half(pl))
                       + __half2float(__low2half(p0));
        const float hi = __half2float(__high2half(h))
                       + __half2float(__high2half(pl))
                       + __half2float(__high2half(p0));

        *reinterpret_cast<__half2*>(hidden + idx) = __halves2half2(
            __float2half(lo), __float2half(hi));
    } else if (idx < n) {
        const float h  = __half2float(hidden[idx]);
        const float pl = __half2float(partial_local[idx]);
        const float p0 = __half2float(partial_peer0[idx]);
        hidden[idx] = (fb_fp16_t) (h + pl + p0);
    }
}

// ------------------------------------------------------------------------
// TP=4 sum (no residual): partial_local = partial_local + Σ partial_peerK
// Used by the LM-head and embedding paths where the input to AR is itself
// the value we want post-AR (no separate residual to fold in).
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_allreduce_sum_tp4(
    fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const fb_fp16_t* __restrict__ partial_peer1,
    const fb_fp16_t* __restrict__ partial_peer2,
    const unsigned int n
) {
    const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    if (idx + 1u < n) {
        const __half2 pl = *reinterpret_cast<const __half2*>(partial_local + idx);
        const __half2 p0 = *reinterpret_cast<const __half2*>(partial_peer0 + idx);
        const __half2 p1 = *reinterpret_cast<const __half2*>(partial_peer1 + idx);
        const __half2 p2 = *reinterpret_cast<const __half2*>(partial_peer2 + idx);

        const float lo = __half2float(__low2half(pl))
                       + __half2float(__low2half(p0))
                       + __half2float(__low2half(p1))
                       + __half2float(__low2half(p2));
        const float hi = __half2float(__high2half(pl))
                       + __half2float(__high2half(p0))
                       + __half2float(__high2half(p1))
                       + __half2float(__high2half(p2));

        *reinterpret_cast<__half2*>(partial_local + idx) = __halves2half2(
            __float2half(lo), __float2half(hi));
    } else if (idx < n) {
        const float pl = __half2float(partial_local[idx]);
        const float p0 = __half2float(partial_peer0[idx]);
        const float p1 = __half2float(partial_peer1[idx]);
        const float p2 = __half2float(partial_peer2[idx]);
        partial_local[idx] = (fb_fp16_t) (pl + p0 + p1 + p2);
    }
}

// ------------------------------------------------------------------------
// TP=2 sum (no residual): partial_local += partial_peer0
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_allreduce_sum_tp2(
    fb_fp16_t* __restrict__ partial_local,
    const fb_fp16_t* __restrict__ partial_peer0,
    const unsigned int n
) {
    const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    if (idx + 1u < n) {
        const __half2 pl = *reinterpret_cast<const __half2*>(partial_local + idx);
        const __half2 p0 = *reinterpret_cast<const __half2*>(partial_peer0 + idx);

        const float lo = __half2float(__low2half(pl))  + __half2float(__low2half(p0));
        const float hi = __half2float(__high2half(pl)) + __half2float(__high2half(p0));

        *reinterpret_cast<__half2*>(partial_local + idx) = __halves2half2(
            __float2half(lo), __float2half(hi));
    } else if (idx < n) {
        partial_local[idx] = (fb_fp16_t) (
            __half2float(partial_local[idx]) + __half2float(partial_peer0[idx]));
    }
}

// ------------------------------------------------------------------------
// TP=2 sum (F32): partial_local += partial_peer0
// F32 variant of the F16 kernel above. Used by the gemma4 attention
// output path (head_dim≥256 + sharp V spikes overflow F16 in the
// F32→F16 cast; F32 AR keeps the mmvq output bounded until the
// post-norm absorbs the spike). See
// `feedback_gemma4_attn_output_proj_f16_saturate`.
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_allreduce_sum_tp2_f32(
    float* __restrict__ partial_local,
    const float* __restrict__ partial_peer0,
    const unsigned int n
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        partial_local[idx] = partial_local[idx] + partial_peer0[idx];
    }
}

// ------------------------------------------------------------------------
// TP=4 sum (F32): partial_local += Σ partial_peer{0,1,2}
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_allreduce_sum_tp4_f32(
    float* __restrict__ partial_local,
    const float* __restrict__ partial_peer0,
    const float* __restrict__ partial_peer1,
    const float* __restrict__ partial_peer2,
    const unsigned int n
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        partial_local[idx] = partial_local[idx]
                           + partial_peer0[idx]
                           + partial_peer1[idx]
                           + partial_peer2[idx];
    }
}

// ------------------------------------------------------------------------
// L2 -> DRAM flush of a freshly-produced partial. The producer kernel's
// writes land in this device's L2; a peer's copy-engine read sources DRAM
// through the BAR1 aperture, so without a flush it reads a not-yet-evicted
// (timing-dependent) value. Re-store every word (volatile, so the value
// becomes THIS kernel's write — a separate kernel's threadfence cannot
// flush the producer kernel's writes), then a system-scope release fence
// pushes them past L2 to DRAM where the peer copy engine reads coherently.
// Far cheaper than a full Stream::synchronize. dtype-agnostic (32-bit
// words). See doc/DETERMINISM_INVESTIGATION.md.
// ------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(P2P_AR_THREADS)
void flambeau_p2p_l2_flush(
    unsigned int* __restrict__ buf,
    const unsigned int n_words
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_words) {
        volatile unsigned int* p = buf + idx;
        *p = *p;
    }
    __threadfence_system();
}
