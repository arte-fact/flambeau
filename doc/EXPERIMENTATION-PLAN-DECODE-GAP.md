# Decode-Gap Closure — Experimentation Plan

<!-- Drafted 2026-04-23. Source evidence:
     - certs/perf/v2_28_decode_gap_rocprofv3.md  (the rocprofv3 breakdown)
     - certs/perf/v2_28_head_to_head_qwen36_27b_q8_0.md  (5-rep bench)
     - /artefact/candle/candle-hip-kernels/  (HIP graph experiment)
     - /home/sandbox/.claude/projects/-artefact-candle/memory/  (G2/G3 findings)
     - /artefact/llama.cpp/ggml/src/ggml-cuda/  (upstream fusion surface)
     - /artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/  (gfx906-specific ports) -->

## Context

Post-C1–C9, on Qwen3.6-27B-Q8_0 at Mesh<4>, 5 reps each side:

| | flambeau | llama.cpp | Δ |
|---|---:|---:|---:|
| pp=512 | 141.36 ± 0.16 | 135.70 ± 3.68 | **+4.2 %** |
| tg=64  |  19.09 ± 0.19 |  19.65 ± 0.03 | −2.9 % |

The prefill win is secured. This plan targets the remaining decode gap.

**The gap is host-side, not kernels.** rocprofv3 on both sides showed
flambeau uses **1.43 s less aggregate GPU time** for the same 64 decode
tokens (1888 ms vs 3317 ms across 4 GPUs). llama.cpp's device/wall ratio
is 1.02 (near-perfect 4-way GPU overlap); flambeau's is 0.56 (each GPU
idle 45 ms/token). ~540 ms / decode run of flambeau's wall time is spent
in Rust-FFI kernel-launch code; the rest is structural PP sequential
wait. See the rocprofv3 cert for the full attribution.

This plan enumerates every lever identified, with evidence for each, a
reference implementation (where one exists), and a bounded experiment
template. Land the highest-ROI levers first; discard the null results
with measurements, not speculation.

---

## Lever inventory

Each lever scored on:
- **Estimated wall gain** (ms/token, best case). Translated back to tok/s
  to see if it closes a measurable fraction of the gap.
- **Cost** (sessions). One = "one coding session". Multi = a feature.
- **Risk to parity** (L/M/H). L = algorithmically bit-exact; M =
  re-orders floating-point but within cert tolerance; H = semantic
  change, needs a full parity reverify.
- **Prior-art signal** (+/−/?). + = candle or llama.cpp has it and it
  wins. − = prior art measured it losing on gfx906. ? = open question.

| # | Lever | Est gain | Cost | Parity risk | Prior art |
|---|---|---:|---:|:---:|:---:|
| **L1** | Fuse standalone `add_f16` residual into MMVQ output | 0.3 ms/tok | 1 | M | + |
| **L2** | Fuse `silu / scale / split` into producing kernel | 0.3 ms/tok | 1 | M | + |
| **L3** | Port RMSNorm to 1024-thread block | 0.2 ms/tok | 0.5 | L | + |
| **L4** | HIP graph capture (bounded N=1 layer) | 0.0–0.5 ms/tok | 1 | L | − / +? |
| **L5** | Manual-replay (G2) launch path | 0.1 ms/tok | 1 | L | ± |
| **L6** | Pre-resolve every `KERNEL_STEMS` at load (finish C1) | 0.05 ms/tok | 0.3 | L | + (candle) |
| **L7** | Batched-launch FFI call (N launches → 1 FFI hop) | 1.0 ms/tok | 2–3 | L | — |
| **L8** | Tune `peer_copy_via_host` — one-shot DtoH→HtoD | 0.05 ms/tok | 0.5 | L | — |
| **L9** | Micro-batch PP decode (1F1B across 2+ requests) | 1.5+ ms/tok | 3+ | M | + |
| **L10** | Persistent kernels (kernel stays resident, feeds from queue) | unknown | 4+ | H | turbo |

Est gain column: how much of the 1.5 ms/token gap could plausibly be
reclaimed if the lever lands at its upper bound. Numbers stack
sub-additively at best; do not sum across the column.

---

## Tier 1 — actionable this cycle (L1–L6)

### L1. Fuse standalone `add_f16` residual into MMVQ output epilogue

**Evidence.** rocprofv3 (`v2_28_decode_gap_rocprofv3.md`):
`flambeau_add_f16` fires 4 516 times × 4.68 µs = 21.1 ms device-time on
64 tokens = 0.33 ms/tok. llama.cpp has **zero** standalone binary-add
calls — the residual is fused into the preceding MMVQ's output stage
via its `ggml_cuda_mm_fusion_args_device::x_bias`.

**Reference impl.**
`/artefact/llama.cpp/ggml/src/ggml-cuda/common.cuh`:

```cpp
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias     = nullptr;
    const void * gate       = nullptr;
    const void * gate_bias  = nullptr;
    ggml_glu_op  glu_op;
};
```

Used in `ggml/src/ggml-cuda/mmvq.cu` around line 392 and 668: the kernel
takes a nullable bias pointer and folds `sumf += bias[row]` into the
final store. When `x_bias == nullptr`, the compiler elides the add via
constant-folding on a template parameter.

**Flambeau port sketch.** Add an optional `const __half* residual` arg
to `flambeau_mmvq_q8_0_dp4a_vdr2_q8_1` (and the gate_up variant). At the
write-back:

```cuda
__half sum_f16 = (__half) sumf;
if (residual != nullptr) {
    sum_f16 = (__half)((float) sum_f16 + (float) residual[row]);
}
dst[row] = sum_f16;
```

Template the `bool fused_residual` so the branch compiles out when the
caller passes `nullptr`.

**Experiment.**
1. Add the cert row for `mmvq_q8_0_dp4a_vdr2_with_residual` (new kernel
   or a templated variant).
2. Sweep correctness vs existing `mmvq_q8_0_dp4a_vdr2` then `add_f16`
   at 5 shapes.
3. Wire `forward_layer_decode` to pass `residual` instead of calling
   `add_f16`.
4. 5-rep tg=64 bench on Qwen3.6-27B-Q8_0 Mesh<4>. Expected delta:
   +~0.3 ms/tok ≈ +0.1 tok/s.
5. Parity bit-exact check on Qwen3.6-35B-A3B-UD-Q4_K_S; if F16 add order
   trips the 8-token greedy match, record the new expected sequence and
   verify it still matches llama.cpp's (llama.cpp does the same fusion).

**Risk.** F16 add order changes; parity may shift by 1 LSB and cascade.
If the 8-token match breaks, log the first-divergence-step and check
whether llama.cpp's 8-token output matches our new output (it should).

### L2. Fuse `silu / scale / split_q_gate` into producing kernel

**Evidence.** 1 694 `silu_f32` + 1 694 `scale_f32` + 564 `split_q_gate_f16`
calls per decode run = 18.9 ms device-time = 0.30 ms/tok. llama.cpp fuses
every one into either the `unary_gated_op_kernel` or the attention-
decode output stage.

**Reference impl.** `/artefact/llama.cpp/ggml/src/ggml-cuda/unary.cu`
line 255 (`unary_gated_op_kernel<op>`) — takes both operands, applies
`op(g) * x` in one kernel. We already have `swiglu_f32` and
`sigmoid_mul_f16` that do this; the work is to **move callers** that
currently do two launches (e.g. `silu_f32` followed by a `mul`) onto
the fused variant. And to **inline `scale_f32`** into the attention-
decode output kernel.

**Flambeau port sketch.** `scale_f32` is a single multiplicative
broadcast — fold its scalar into `attention_decode_f16`'s final store
via an extra `output_scale` arg. `split_q_gate_f16` writes two separate
buffers from one QKV matmul output; inline the split into the MMVQ
epilogue so the split buffers are the kernel's direct outputs.

**Experiment.**
1. Split into three sub-PRs; each independently parity-tested.
2. Per sub-PR: 5-rep tg=64 bench; accept if +≥ 0.03 tok/s and parity
   bit-exact.
3. If cumulative gain < 0.05 tok/s, bank the code-quality win (fewer
   kernels, cleaner call graph) and move on.

**Risk.** Medium. Each fusion changes FP32 add order somewhere. Same
parity-verify discipline as L1.

### L3. Port RMSNorm to 1024-thread block

**Evidence.** `flambeau_rmsnorm_f16` average call 14 µs over 3 514
calls = 49.4 ms device. llama.cpp `rms_norm_f32<1024, ...>` average call
7.8 µs. **2.7× faster per call** on comparable shape.

Our `crates/kernels-hip/src/kernels/rmsnorm_f16.cu:20` uses 256 threads.
At Qwen3.6-27B-Q8_0 `hidden = 5120`, each thread processes 20 elements.
llama.cpp's 1024-thread block has each thread doing 5 elements — more
parallelism per row, fewer serial sweeps for the cross-warp reduction.

**Reference impl.** `/artefact/llama.cpp/ggml/src/ggml-cuda/norm.cu`
contains both `rms_norm_f32<256>` and `rms_norm_f32<1024>` variants.
The dispatch chooses based on `ncols`:

```cpp
if (ncols < 1024) {
    // 256-thread variant
} else {
    const dim3 block_dims(1024, 1, 1);
    rms_norm_f32<1024, ...><<<blocks_num, block_dims, ...>>>(...);
}
```

**Flambeau port sketch.** Template `RMSNORM_THREADS` on the existing
kernel; add a 1024-variant; dispatch at call site based on `k` (hidden
dim). The per-thread `K / THREADS` ratio is unit-size-agnostic, so the
existing loop body is shape-correct for 1024.

**Experiment.**
1. Land `flambeau_rmsnorm_f16_1024` as a new variant with its own cert
   row (correctness sweep at 5 shapes).
2. Dispatch switches to 1024-thread for `k ≥ 2048`.
3. Expected per-kernel time 14 → ~7 µs ⇒ 49.4 ms → 24.7 ms device-time
   saved = 0.4 ms/tok. Wall-time impact is smaller because RMSNorm
   already overlaps with the subsequent MMVQ launch on the same stream.

**Risk.** Low. Algorithmically identical; just a larger block.

### L4. HIP graph capture — bounded experiment (one layer)

**Evidence — conflicting.**
- **candle** (same rig, ROCm 7.1.1, gfx906): measured 511-node graph at
  **8.9 ms / hipGraphLaunch** vs 5.1 ms for 511 individual launches.
  Marked "NOT a viable perf lever on gfx906 MI50". Source:
  `/home/sandbox/.claude/projects/-artefact-candle/memory/project_g3_hipGraphLaunch_slow_gfx906.md`.
- **llamacpp-turbo** (gfx906 branch): ships `-DGGML_HIP_GRAPHS=ON` and
  claims **+8–10 % gen speed** on Qwen3-Coder-Next. Source:
  `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/NEXT_OPTIMIZATIONS.md`.

The discrepancy likely comes from graph size: candle tested a
511-node-per-token graph; turbo's pipeline-parallel setup likely has
**far fewer, larger graph nodes** per token (one graph per pipeline
segment). Flambeau decode is ~262 kernels/token — larger than turbo's
likely config, smaller than candle's.

**Experiment — bounded scope.**
1. Capture a HIP graph for **one layer's decode forward** (17 ops).
   Measure the single-graph replay overhead in isolation.
2. If per-graph launch overhead < 17 × `hipModuleLaunchKernel` (~5 µs
   each = 85 µs), scale to full-decode graph and remeasure.
3. If graph-launch overhead on a 17-node graph already exceeds the
   raw-launch baseline, **abandon** — it matches candle's finding and
   there's no smaller unit of value to capture.

**Success threshold.** +≥ 0.05 tok/s wall. Else null-result cert and
skip.

**Risk.** Low. HIP graphs don't change arithmetic; parity is
structurally unchanged.

### L5. Manual-replay launch path (candle's "G2")

**Evidence.** Candle's G2 was 5.9 ms/tok vs 5.1 ms raw on TinyLlama — a
0.8 ms/tok **loss** because candle's Rust-side bookkeeping is already
fast and the G2 replay loop adds counter-advance overhead. On flambeau
(which has more Rust-side setup per launch: `KernelArgs::push` × N,
`RwLock::read` for kernel cache), the break-even shifts.

G2 pattern: pre-build every `KernelArgs` for the decode loop once at
session init; at decode time, just update the 2–4 pointers that change
per step (token id, position) and replay the launches in a tight loop
with no allocations.

**Experiment.**
1. Prototype for one decode layer: pre-build `KernelArgs` × 17 at
   `forward_layer_decode` construction.
2. Expose `KernelArgs::set(idx, &T)` (already exists — see
   `module.rs:170`) as the per-step mutator.
3. Bench 5-rep tg=64.

**Risk.** Low. Same kernels, same args, same arithmetic.

**Already partially in place.** C1 kernel-cache + existing
`KernelArgs::set` are the building blocks. L5 is "wire these into the
forward loop".

### L6. Pre-resolve every `KERNEL_STEMS` entry at `OpsRegistry::new`

**Evidence.** C1 lazily caches on first use. The warmup-prefill hits
prefill-only kernels (MMQ, rmsnorm_quant_q8_1_mmq); decode kernels
(MMVQ variants, rmsnorm, sigmoid_mul) first-touch cold on the first
real decode step. Cost per cold-path lookup: `CString::new` (~200 ns) +
`hipModuleGetFunction` driver call (~1–5 µs) + `RwLock::write` (~100 ns).

The authoritative kernel list is already enumerated:
`crates/ops/src/hip/mod.rs::KERNEL_STEMS`.

**Flambeau port sketch.** Add a `pub fn resolve_all(&self) ->
DeviceResult<()>` on `HipModule` that iterates a caller-supplied
`&[&'static str]` and pre-populates the cache. Call it in
`OpsRegistry::new` after every module is loaded.

**Experiment.**
1. First-token latency before/after. This lever shifts a one-time
   ~5 ms cost from first-decode to load time.
2. tg=64 steady-state: no measurable change expected (first-token is 1
   of 64, so the steady-state averaging already buries the cost).

**Success threshold.** −5 ms first-token latency.

**Risk.** Nil.

---

## Tier 2 — bigger surgery (L7–L10)

### L7. Batched-launch FFI call

**Premise.** Every `unsafe { kernel.launch(...) }` crosses the Rust→C
FFI boundary at ~8 µs wall. 262 launches/token × 4 ranks × 8 µs =
~8.4 ms/token Rust host-gap. If we could batch N launches into a single
FFI call, the per-launch FFI overhead drops by a factor of N.

HIP doesn't expose a native batch-launch API. The closest primitive is
HIP graph record/replay (L4), which we already evaluated as risky on
gfx906.

An alternative — a **custom C shim** that accepts `[(hipFunction_t,
LaunchCfg, arg_array)*]` in one call and issues the loop C-side. The
individual `hipModuleLaunchKernel` calls aren't faster, but the
Rust→C boundary transit is taken once per N instead of N times.

**Reference impl.** None in candle, llama.cpp, or llamacpp-turbo.
Greenfield.

**Experiment.**
1. Prototype the C shim with a fixed N=16.
2. Bench launch-overhead micro-benchmark first: does `launch_batch16`
   come in under `16 × launch` wall time? If not, the idea is dead
   before we try integrating.
3. Only if the micro-bench wins, integrate into `forward_one_token_pp`
   and re-bench tg=64.

**Success threshold.** +≥ 0.3 tok/s (non-trivial, given 2–3 sessions of
work).

**Risk.** Low arithmetically; medium architecturally (FFI lifetimes for
the batched-arg arrays).

### L8. One-shot DtoH→HtoD peer-copy

**Premise.** Current `peer_copy_via_host` does:
`memcpy_async(DtoH)` → `stream_sync` → `memcpy_async(HtoD)`. The
stream_sync is a CPU wait. On PCIe 3.0 x16 at ~6.7 GB/s, the DtoH of a
~4 KB hidden vector is ~0.6 µs device + PCIe; the sync adds ~5–10 µs.

**Evidence — weak.** Per the rocprofv3 cert, PP peer-copy accounts for
~5 ms out of 3353 ms wall (0.15 %). Even optimised to zero, this is
noise.

**Why keep in the plan.** Listed for completeness — one-shot
`hipMemcpyPeerAsync` exists but candle already tried it and found
gfx906 PCIe-only rigs wedge the stream state. Our host-bounce is the
*correct* path.

**Experiment.** Attempt per-layer event chaining so the HtoD on rank N+1
can start as soon as DtoH on rank N completes, without a CPU sync
between them. Use `hipEventRecord` on src stream, `hipStreamWaitEvent`
on dst stream.

**Success threshold.** +≥ 0.05 tok/s. Otherwise null.

**Risk.** Low — same bytes moving, just fewer CPU syncs.

### L9. Micro-batch PP decode (1F1B across 2+ requests)

**Premise.** Currently each decode token runs rank 0 → rank 1 → rank 2 →
rank 3 in strict sequence. For token N+1 to start, token N's rank 3
output must arrive. **Each GPU is idle 3/4 of the decode pipeline per
token** — see rocprofv3 device/wall = 0.56.

If we decode two requests in parallel, rank 0 can work on request B's
token N while rank 1–3 are still finishing request A's token N. This
is the standard pipeline-parallel 1F1B (one-forward-one-backward)
schedule applied to decode.

**Reference impl.** vLLM-gfx906 supports continuous batching; see their
scheduler for pattern. Llama.cpp server has `-np N` parallel slots for
the same reason.

**Experiment.** Prototype two sessions interleaved. Measure aggregate
throughput (Σ requests/s across both sessions) vs doubling one
session. Expected: near-doubling on decode throughput, zero effect on
first-token latency.

**Risk.** Medium — V2 architectural work. Affects server state,
session lifecycle, KV-cache per-request. Not a perf-tweak PR.

**Success threshold.** ≥ 1.8× aggregate decode throughput at batch 2.

### L10. Persistent kernels

**Premise.** llamacpp-turbo's `NEXT_OPTIMIZATIONS.md` cites "persistent
kernels (keep kernel running, feed work via queue)" as a Medium-Impact
lever for kernel-launch-overhead reduction. The idea: one kernel
occupies the GPU, reads commands from a device-memory queue, executes
them, loops. Zero launches after startup.

**Reference impl.** None landed in llama.cpp or candle. MegaBlocks and
DeepSeek's `deep_gemm` use similar patterns for MoE dispatch. Research
territory, not a port.

**Experiment.** Not in this cycle. Listed for completeness; revisit
only if L1–L9 exhaust and the gap remains.

**Risk.** High — architectural rewrite of the kernel-launch path.

---

## Not-in-plan (decided-skip, with rationale)

- **HIP peer-to-peer direct memcpy.** Candle tested; `hipMemcpyPeerAsync`
  wedges the HIP stream on PCIe-only rigs. Our host-bounce is the
  architectural choice per CLAUDE.md. Don't revisit.
- **Mesh<N>·row_split (TP) decode path.** Per V1 CLAUDE.md:
  `llama.cpp split_mode=row is 3–8× slower than layer on non-P2P
  topologies`. TP for decode is a non-starter on PCIe.
- **Replace `RwLock` in kernel cache with `parking_lot::RwLock`.** Adds
  a dep for ~10 ns/launch × 262 launches/tok = 2.6 µs/tok wall. Below
  measurable threshold.
- **LTO `"thin"` → `"fat"`.** Per `RUST-PERF-CORRECTIONS.md` and CLAUDE.md
  measurement discipline, link time jumps from ~10 s to multi-minute;
  sub-1 % wall-clock win on a HIP-bound critical path. Not worth it.

---

## Execution template (use for every lever above)

Per CLAUDE.md's measurement rules:

1. **Rebuild from clean.** `cargo clean -p <crate> && cargo build
   --release --features hip`. Stale binaries cost sessions.
2. **Correctness first.** If the lever touches a kernel, run
   `cargo run -p bench -- sweep --impl <id>` at ≥ 5 shapes before any
   perf measurement.
3. **5-rep bench** on Qwen3.6-27B-Q8_0 Mesh<4> pp=512 tg=64.
   Accept if mean ± σ distribution doesn't overlap the pre-lever
   distribution. Single-rep data lies — see
   `certs/perf/v2_28_head_to_head_qwen36_27b_q8_0.md` for the mea culpa.
4. **Parity bit-exact** on Qwen3.6-35B-A3B-UD-Q4_K_S (the
   `expect_pass=true` cert). If parity shifts, document the new
   reference tokens and verify they still match llama.cpp (which does
   the same fusion for L1/L2).
5. **Cert the result**, positive or null, in `certs/perf/`. Null
   results are first-class — don't hide them.

**Order of attack.** L6 (trivial), then L3 (low-risk kernel port), then
L1+L2 (correlated fusion work), then L5 (bounded), then L4 (risky given
candle's finding). L7–L10 are research tier, don't start without
finishing 1–6.

## Budget gate

If the sum of Tier-1 gains doesn't close the gap past 99 % of llama.cpp
on this model, Tier 2 (L7 + L9) is where the remaining work is.
Continuous batching (L9) is the single biggest lever — it's the only
thing that changes the device/wall ratio from 0.56 toward 1.0. But it's
also a V2 architectural commitment, not a perf-tweak cycle.

Honest expectation after Tier 1: close ~0.5–1.0 ms/tok of the 1.5 ms/tok
gap. Reach ≥ 98 % of llama.cpp at steady state. Tier 2 is a separate
project.
