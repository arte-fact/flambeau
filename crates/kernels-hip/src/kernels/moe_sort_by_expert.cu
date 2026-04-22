// moe_sort_by_expert — histogram + prefix-sum + scatter, for grouping
// (token, slot) pairs by expert_id.
//
// Input:
//   expert_ids[total]   int32 — total = n_tokens * top_k, value in [0, n_experts)
//
// Output:
//   counts[n_experts]             int32 — #pairs routed to each expert
//   offsets[n_experts + 1]        int32 — exclusive prefix-sum of counts; offsets[n_experts] = total
//   cursors[n_experts]            int32 — scatter cursor, init'd to offsets[e] by kernel 2
//   sorted_pair_idx[total]        int32 — input pair indices grouped by expert
//
// Usage (host side, 3 launches in order):
//   1. memset counts and cursors to 0, then launch `moe_sort_count`
//   2. launch `moe_sort_scan_offsets` (single 256-thread block)
//   3. launch `moe_sort_scatter`
//
// Complexity: O(total) × 2 + O(n_experts) scan. For pp=512 × top_k=8,
// total = 4096, n_experts = 256. All on-GPU, no host round-trip.

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef MAX_N_EXPERTS
#define MAX_N_EXPERTS 512
#endif

// ---------------------------------------------------------------------------
// Kernel 0: zero `counts` before the histogram. Single block of up to 512
// threads covers MAX_N_EXPERTS.
// ---------------------------------------------------------------------------
extern "C" __global__ void flambeau_moe_sort_zero_counts(
    int* __restrict__ counts,
    const int n_experts
) {
    const int tid = threadIdx.x;
    if (tid < n_experts) counts[tid] = 0;
}

// ---------------------------------------------------------------------------
// Kernel 1: histogram via atomic adds.
// grid = ((total + 255) / 256, 1, 1), block = (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void flambeau_moe_sort_count(
    const int* __restrict__ expert_ids,   // [total]
    int*       __restrict__ counts,       // [n_experts]  (must be zero-inited by caller)
    const int total
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = expert_ids[i];
    atomicAdd(&counts[e], 1);
}

// ---------------------------------------------------------------------------
// Kernel 2: exclusive prefix-sum on counts[] → offsets[], also initialise
// cursors[e] = offsets[e] for the scatter kernel.
//
// Single-block Blelloch scan on n_experts entries. n_experts is bounded
// by MAX_N_EXPERTS = 512 (so 1 block of 512 threads max).
// ---------------------------------------------------------------------------
extern "C" __global__ void flambeau_moe_sort_scan_offsets(
    const int* __restrict__ counts,       // [n_experts]
    int*       __restrict__ offsets,      // [n_experts + 1]
    int*       __restrict__ cursors,      // [n_experts]
    const int n_experts
) {
    __shared__ int s[MAX_N_EXPERTS + 1];
    const int tid = threadIdx.x;

    // Each thread loads one count (if in range) else 0.
    int v = (tid < n_experts) ? counts[tid] : 0;
    s[tid] = v;
    __syncthreads();

    // Blelloch inclusive scan (Hillis-Steele, simple and correct for n≤512).
    for (int offset = 1; offset < MAX_N_EXPERTS; offset <<= 1) {
        int t = 0;
        if (tid >= offset) t = s[tid - offset];
        __syncthreads();
        s[tid] += t;
        __syncthreads();
    }
    // s[tid] now holds INCLUSIVE prefix sum. Convert to EXCLUSIVE: shift right
    // by 1 and set s[0] = 0.
    int excl = (tid == 0) ? 0 : s[tid - 1];
    __syncthreads();
    s[tid] = excl;
    __syncthreads();

    if (tid < n_experts) {
        offsets[tid] = s[tid];
        cursors[tid] = s[tid];
    }
    // Last entry: total = sum of all counts.
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n_experts; ++i) total += counts[i];
        offsets[n_experts] = total;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: scatter — atomicAdd on cursors[e] to find the sorted position
// for each pair, then write pair index there.
// grid = ((total + 255) / 256, 1, 1), block = (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void flambeau_moe_sort_scatter(
    const int* __restrict__ expert_ids,       // [total]
    int*       __restrict__ cursors,          // [n_experts] — init'd to offsets[e]
    int*       __restrict__ sorted_pair_idx,  // [total]
    const int total
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int e = expert_ids[i];
    const int pos = atomicAdd(&cursors[e], 1);
    sorted_pair_idx[pos] = i;
}

// ---------------------------------------------------------------------------
// Kernel 4 (V2.6.a): scan counts → padded_offsets where
// padded_counts[e] = ceil(counts[e] / 8) * 8. Each expert's range is
// padded to a multiple of 8 for use by 8-slot-per-block MMQ kernels
// that need all 8 slots in a block to share the same expert.
// ---------------------------------------------------------------------------
extern "C" __global__ void flambeau_moe_sort_scan_padded_offsets(
    const int* __restrict__ counts,           // [n_experts]
    int*       __restrict__ padded_offsets,   // [n_experts + 1]
    const int n_experts
) {
    __shared__ int s[MAX_N_EXPERTS + 1];
    const int tid = threadIdx.x;

    // Load padded count: ceil(counts[e] / 8) * 8 = (counts[e] + 7) & ~7.
    int v = 0;
    if (tid < n_experts) {
        const int c = counts[tid];
        v = (c + 7) & ~7;
    }
    s[tid] = v;
    __syncthreads();

    // Hillis-Steele inclusive scan.
    for (int offset = 1; offset < MAX_N_EXPERTS; offset <<= 1) {
        int t = 0;
        if (tid >= offset) t = s[tid - offset];
        __syncthreads();
        s[tid] += t;
        __syncthreads();
    }
    // Convert inclusive → exclusive.
    int excl = (tid == 0) ? 0 : s[tid - 1];
    __syncthreads();
    s[tid] = excl;
    __syncthreads();

    if (tid < n_experts) padded_offsets[tid] = s[tid];
    if (tid == 0) {
        int total_padded = 0;
        for (int i = 0; i < n_experts; ++i) {
            total_padded += (counts[i] + 7) & ~7;
        }
        padded_offsets[n_experts] = total_padded;
    }
}

// ---------------------------------------------------------------------------
// Kernel 5 (V2.6.a): copy unpadded sorted_pair_idx → padded layout and
// fill each expert's tail padding with the last real entry. Using
// last-real (rather than -1 sentinel) keeps the 8-slot MMQ kernel branch-
// free: padding slots do the same matmul as the last real slot, just
// get their output written to a "don't care" buffer (or silently
// overwritten by the next real slot's write).
//
// grid = (ceil(max_padded_per_expert / 256), n_experts, 1)
// block = (256, 1, 1)
// Blocks whose chunk-offset >= padded_count[e] early-exit.
// ---------------------------------------------------------------------------
extern "C" __global__ void flambeau_moe_sort_pad_copy(
    const int* __restrict__ sorted_pair_idx,        // [total] — unpadded input
    const int* __restrict__ offsets,                // [n_experts + 1] — unpadded
    const int* __restrict__ counts,                 // [n_experts]
    const int* __restrict__ padded_offsets,         // [n_experts + 1]
    int*       __restrict__ sorted_pair_idx_padded, // [total_padded]
    const int n_experts
) {
    const int e = blockIdx.y;
    if (e >= n_experts) return;
    const int padded_count = padded_offsets[e + 1] - padded_offsets[e];
    const int chunk_start = blockIdx.x * 256;
    const int pad_idx = chunk_start + threadIdx.x;
    if (pad_idx >= padded_count) return;

    const int real_count = counts[e];
    int pair;
    if (pad_idx < real_count) {
        pair = sorted_pair_idx[offsets[e] + pad_idx];
    } else if (real_count > 0) {
        // Repeat the last real entry; same expert → same weight tile,
        // padding slot's output is a duplicate that can be ignored.
        pair = sorted_pair_idx[offsets[e] + real_count - 1];
    } else {
        // Expert had zero pairs. padded_count should be 0 too, but guard anyway.
        pair = -1;
    }
    sorted_pair_idx_padded[padded_offsets[e] + pad_idx] = pair;
}
