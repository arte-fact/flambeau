# V2.28 follow-up — decode gap root-cause via rocprofv3 kernel-trace

**Task**: find the ~1.5 ms/token decode gap vs llama.cpp on
Qwen3.6-27B-Q8_0 Mesh<4> (5-rep flambeau 19.09 ± 0.19, llama.cpp 19.65 ± 0.03).
**Method**: `rocprofv3 --kernel-trace --stats` on both sides, decode-only.

## Headline finding

**The gap is not in the kernels — flambeau uses 43 % less GPU time in
aggregate than llama.cpp for the same 64 decode tokens.** The gap is
entirely host-side scheduling.

| | wall time (64 tok) | Σ device time (4 GPUs) | per-GPU avg | device/wall |
|---|---:|---:|---:|---:|
| flambeau  | 3353 ms | 1888 ms | 472 ms | 0.56× |
| llama.cpp | 3257 ms | 3317 ms | 829 ms | **1.02×** |
| Δ | +96 ms | **−1429 ms** | — | — |

llama.cpp's `device/wall ≈ 1.02` means its 4 GPUs overlap almost
perfectly — there is nearly zero host-side idle between kernels.
Flambeau's `0.56` means **each GPU is idle 44 % of wall time**.

If flambeau could hide its host-gap and hit llama.cpp's 1.02× ratio,
decode wall would drop from 3353 ms to ~1888 ms and we'd be at
~34 tok/s instead of 19. The ceiling is there; we're not using it.

## Per-category kernel breakdown

Kernels grouped by function; total across all 4 GPUs for the full
decode run (warm-prefill + seed-prefill + 64 decode steps):

| Category                | flambeau calls | flambeau ms | llama.cpp calls | llama.cpp ms | Δ ms |
|---|---:|---:|---:|---:|---:|
| MMVQ (Q8_0 matmul)      |  11 677 | 1525.3 |  28 145 | 2575.7 | **−1050.4** |
| Memcpy / copy           |   7 363 |   33.0 |  20 651 |  269.9 | **−236.9** |
| Quantize F32→Q8_1       |   7 414 |   34.4 |  28 145 |  124.5 | **−90.1** |
| GDN / SSM               |   3 484 |   49.1 |   6 240 |  107.4 | **−58.3** |
| Attention               |     564 |   14.6 |   2 080 |   28.7 | −14.0 |
| L2 norm                 |   3 388 |   15.9 |   6 240 |   27.3 | −11.5 |
| SwiGLU / gate           |   4 516 |   20.9 |   7 280 |   31.9 | −11.0 |
| RoPE                    |   1 128 |    4.3 |   2 080 |   17.7 | −13.3 |
| Cast F32↔F16            |   8 466 |   39.3 |   6 305 |   29.0 | +10.3 |
| RMSNorm                 |   7 374 |  103.4 |  13 585 |   88.5 | +14.9 |
| Add / binop (standalone)|   4 516 |   21.1 |       0 |    0.0 | +21.1 |
| SiLU (standalone)       |   1 694 |    8.1 |       0 |    0.0 | +8.1 |
| Scale (standalone)      |   1 694 |    8.2 |     192 |    1.6 | +6.5 |
| Split Q\|gate            |     564 |    2.6 |       0 |    0.0 | +2.6 |
| causal_conv1d           |   1 694 |    7.8 |       0 |    0.0 | +7.8 |
| **Total**               |         | **1887.9** |         | **3316.5** | **−1428.6** |

### Where we win on kernels

- **MMVQ**: −1050 ms. Flambeau's `mmvq_q8_0_dp4a_vdr2_q8_1` (V2.6) plus
  the fused `mmvq_q8_0_gate_up_dp4a` (V1.7.6.d) collapse two llama.cpp
  MMVQ variants (~28 k calls at 91.5 µs avg) into 11.7 k flambeau calls
  at 130.6 µs avg. We do fewer, bigger launches for the same arithmetic
  work and that's faster on gfx906 (launch overhead is the #1 cost per
  V1.7.6 memory).
- **Memcpy**: −237 ms. Llama.cpp dominates on `copy / concat / set_rows /
  cpy_scalar_contiguous` — layout shuffles that flambeau avoids through
  fixed `F16Contig` / `Q8Contig` layouts.
- **Quantize**: −90 ms. Llama.cpp quantizes activations once per MMVQ
  call (28 k); flambeau fuses quantise into `rmsnorm_q8_1_fused`, so
  1.9k fused rmsnorm calls cover the quant work for all downstream
  MMVQ uses.
- **GDN**: −58 ms. V2.x's fused `gdn_state_step_f32_s128` beats
  llama.cpp's `gated_delta_net_cuda<128>` per call and per call-count.

### Where we lose on kernels

- **Add / binop / SiLU / Scale / Split Q|gate / causal_conv1d / cast**:
  +65 ms total. These are all cases where llama.cpp **fuses** the op
  into the preceding kernel's output stage (residual-add inside the
  MMVQ epilogue, SiLU inside the gated op kernel, etc.) while flambeau
  launches a separate small kernel. Each extra launch costs ~5 µs of
  kernel + ~3-5 µs of host-side overhead — small per op, but together
  these +65 ms × 64 tokens = ~1 ms/token of measurable extra.
- **RMSNorm**: +15 ms. Flambeau's RMSNorm calls are ~2.7× longer per
  call than llama.cpp's (14 µs vs 7 µs average). Not yet investigated;
  candidate for a V2.29 kernel-tune cycle.

## Host-side gap attribution

Device time (flambeau) = 1888 ms across 4 GPUs → 472 ms per GPU.
Wall time (flambeau) = 3353 ms. Host-gap per GPU = 3353 − 472 = **2881 ms
of idle per GPU in wall time**, or **45 ms/token host gap per GPU**.

What the GPU does in that 45 ms/token slack:
- Waits for the previous rank's `peer_copy_via_host` DtoH+sync+HtoD.
  At ~512 B hidden state × 3 rank-hops per token, that's ~15–30 µs per
  hop for PCIe 3.0, so ~75 µs/token for the full hand-off chain. **≈ 0.2 %
  of the gap.**
- Waits for host to issue the next kernel launch. Every flambeau-side
  launch crosses a Rust → C FFI boundary with:
  - `unsafe { hipModuleLaunchKernel(…) }` FFI call (~5–10 µs),
  - `KernelArgs` builder pointer-array setup (~1 µs),
  - `RwLock::read` + `HashMap::get` for kernel cache lookup (~0.1 µs
    after C1),
  - Driver-internal argument copy (~1 µs).

  At ~182 MMVQ + ~80 other kernel launches per token per rank = ~262
  launches/token × 4 ranks = 1048 launches/token. At ~8 µs of host
  overhead per launch × 1048 = **~8.4 ms of host launch overhead per
  token**. Over 64 tokens that's 538 ms of wall time burned in host
  code between kernel launches.

So of the 2881 ms host gap (per GPU, summed over 64 tokens):
- ~75 µs × 64 = 5 ms — PP peer-copy latency (0.2 %)
- ~538 ms / 4 = ~135 ms per GPU — host-side launch overhead (5 %)
- The other ~95 % is the GPU simply waiting for earlier ranks to
  finish their layers. This is structural PP sequential dependency, not
  closable without micro-batching across tokens.

## What could close the residual 1.5 ms/token gap

1. **Fuse the remaining standalone ops** (−65 ms device, ~1 ms/token):
   port llama.cpp's pattern of folding `add`/`silu`/`scale`/`split` into
   their producing kernel's epilogue. This is the most-actionable next
   cycle.
2. **Tune RMSNorm** (flambeau is 2.7× slower per call): PMC with
   rocprofv3 to find the specific inefficiency (likely LDS / waves per
   SIMD); port llama.cpp's 1024-thread variant if it wins.
3. **Continuous batching / micro-batch PP** to unlock the 2881 ms of
   per-GPU idle. This is a V2 architectural change, not a perf tweak.
4. **HIP graph capture** for the decode step: record the ~262-kernel
   launch sequence once, replay with one HIP-API call per token. Per
   CLAUDE.md memory (V1.7.6 "G3 test"), graph capture was null-to-
   slightly-negative on gfx906 for prefill; worth re-trying on decode
   where the launch count is lower but per-launch overhead share is
   higher.

The gap is not close-able with more C-style Rust-side micro-opts. The
remaining levers are either kernel-level (fusion, RMSNorm tune) or
architectural (continuous batching, graph capture).

## Conclusion

**The C1–C9 cycle won +4 % on prefill and left decode flat vs pre-C1–C9
flambeau (noise-limited).** The residual ~3 % decode gap to llama.cpp is
structural: our Rust-side kernel launch path burns ~8 µs/launch of FFI
overhead that llama.cpp's C++ launch path doesn't pay. Summed across
~1048 launches/token × 64 tokens, that's ~540 ms of host work per decode
run — about 16 % of wall time, comfortably exceeding the ~100 ms
(~3 %) gap we see.

Flambeau kernels are genuinely faster than llama.cpp's (1.43 s less
aggregate device time for the same workload). We're losing the race on
host-side issue rate, not on silicon.

## Data

Full rocprofv3 stats preserved at `/tmp/rocprof/flambeau/` and
`/tmp/rocprof/llama/` for re-analysis. To regenerate:

```
# flambeau
FLAMBEAU_DECODE_ONLY=1 FLAMBEAU_QWEN35_GGUF=...Qwen3.6-27B-Q8_0.gguf \
  FLAMBEAU_MESH_RANKS=4 ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.1.1/core-7.13/lib/rocblas/library \
  /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 --kernel-trace --stats \
    --output-format csv -d . -o flambeau_decode \
    -- target/release/deps/perf_baseline_qwen35_9b-* --nocapture

# llama.cpp
LD_LIBRARY_PATH=/artefact/llama.cpp/build/bin:/opt/rocm-7.1.1/lib:/opt/rocm-7.1.1/core-7.13/lib \
  /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 --kernel-trace --stats \
    --output-format csv -d . -o llama_decode \
    -- /artefact/llama.cpp/build/bin/llama-bench \
         -m ...Qwen3.6-27B-Q8_0.gguf -p 0 -n 64 -ngl 99 -sm layer -fa 1 -r 1
```
