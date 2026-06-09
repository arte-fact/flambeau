# CUDA port plan — fresh audit (supersedes prior draft)

**Status:** Draft v1. Re-audited against `main` on 2026-06-09; replaces the
earlier draft, which assumed a "purely additive, mirror two crates" port.
That premise is false today: the stack **above** the kernels grew pervasive
HIP coupling. This plan is grounded in a file-level audit of the current tree.

**Plan-only — no code lands until each phase is signed off.**

## What the audit actually found

Counted off `main`:

| Surface | Current state |
| --- | --- |
| Compiled kernels | **268** `.cu` files / 268 `extern "C" __global__` entry points, flat under `crates/kernels-hip/src/kernels/` |
| Arch primitives | `arch_primitives/gfx906.cuh` (DPP/SFU/`__builtin_amdgcn_sdot4`/inline-asm reductions) + `mmq_prefetch.cuh` (`global_load_dword` L2 prefetch). **135** kernel files call `gfx906_*`; **84** use bare `__shfl_xor` |
| Wave width | `-DWARP_SIZE=64` build define; wave64 reduction trees (`gfx906_{,half_,quarter_,eighth_}warp_reduce_sum`) |
| Shared headers | `kernels-shared/include/{block_quant,iq_grid,mmvq_store}.cuh` — **4 architectural-rule-5 leaks** (`<hip/hip_runtime.h>`, `_Float16`) in `block_quant.cuh` + `iq_grid.cuh` |
| Backend Rust | `crates/backend-hip/` — **7,397 LOC**, 14 files, **42** HIP FFI symbols in `sys.rs` (driver-API level: `hipModuleLoadData`/`hipModuleLaunchKernel`) |
| Dispatch | `dispatch/hip/gfx906.toml` — 768 lines, **83** active rows; compiled into Rust `KernelDescriptor` tables in `backend-hip/src/impls.rs`, validated by the `dispatch_toml_roundtrip` + `cert_check` tests |
| Certs | `certs/hip/gfx906/` — **123** JSON files (`{schema_version, impl_id, backend, arch, op, dtype_*, results[], pass, pmc}`) |
| TP collectives | `backend-hip/{bar_p2p,rccl,rccl_sys,cluster}.rs` — TP AllReduce is hand-rolled **BAR1 P2P kernels** (`flambeau_p2p_allreduce_*`), *not* RCCL |

### The decisive finding: coupling lives above the kernels

`crates/core` is clean. `Device` and `Stream`
(`core/src/device.rs:82,97`) are genuine backend-neutral traits; `op.rs`
defines `KernelImpl<O: Op, D: Device>` generically. A CUDA backend can
implement these without touching `core`.

But everything between the kernels and the HTTP API hard-names HIP types in
**production** code, not just tests:

| Crate | Coupling | Evidence |
| --- | --- | --- |
| `forward` | **Severe.** `ForwardEngine::new`, `StageHooks`/`TopologyHooks`, `ArCallback`, all loaders, GDN composites, AR coordinators name `&HipDevice`/`&HipStream`/`HipEvent`/`BarP2pAllReduce`/`HipCluster` | `engine.rs` (32 refs), `runtime/ar.rs:16`, `loader/*.rs`, `core/hooks.rs` |
| `model-ops` | Moderate. `BackendCtx { device: &HipDevice, stream: &HipStream }` threaded through every op | `delta_net.rs:336`, `ops/*.rs` |
| `models/{qwen35-v2,qwen35moe-v2,gemma4-v2}` | High. Every `arch.rs`/`loader.rs` load/dispose/shard fn takes `&HipDevice` | `*/src/{arch,loader}.rs` |
| `server-core` | Moderate. `SessionContext::cluster(&self) -> &HipCluster` on the per-arch contract | `traits.rs:8,29` |
| `server` | High. Constructs `Arc<HipCluster>::new(...)` directly; cluster stored as concrete type | `serve.rs:266`, `serve_common.rs:172`, `routes.rs:63` |
| `ops` | Contained. All HIP behind `#[cfg(feature="hip")]` in `ops/src/hip/`; the `Ops` trait (`ops_trait.rs`) is *already* "backend-portable" but leaks `ScalarSlot` and is hip-gated; `sig.rs` re-exports `HipStream` |
| `cli` | Low. Hard-codes `"hip:"` device-string prefix; only `hip_{sweep,serve}` features |
| `bench` | All sweep/harness files name `HipDevice` directly — relevant because **certifying a CUDA kernel goes through this harness** |

**This already strains architectural rules 12 and 13** (shared/server-side
surfaces naming `Hip*` types; per-arch contracts returning concrete backend
handles). Per CLAUDE.md, the right response is to surface the prerequisite
refactor rather than layer a second-backend workaround on top. That is the
spine of this plan.

## Strategy (locked with the user 2026-06-09)

1. **Live CUDA hardware available** (RTX 3090 / sm_86). Every CUDA phase
   carries a real sweep + cert gate, same discipline as gfx906 — not
   build-only.
2. **Decouple the upper layers first.** Generify `forward` / `model-ops` /
   model crates / `server` over a backend-neutral trait surface *before* the
   CUDA backend carries real weight. HIP stays the sole impl throughout the
   decoupling, so any regression shows up immediately on the daily gfx906
   target.
3. **gfx906 stays green on every commit.** The decoupling is a behavior-
   preserving refactor; the HIP sweep + forward parity snapshots are the
   guardrail.

Consequence: two interleaved tracks. **Track A** lifts the missing seams and
genericizes the stack (HIP-only, no new behavior). **Track B** builds
`kernels-cuda` + `backend-cuda` and certifies kernels in isolation via
`bench`. They share a critical path: end-to-end CUDA inference (B7) needs the
generic forward/model/server stack (A2–A3); CUDA NCCL collectives (B6) need
the collectives trait (A1). Kernel porting + single-GPU cert (B0–B5) can run
largely in parallel with Track A.

## The seam: what a "backend" must provide

`core::Device`/`Stream` cover alloc / memcpy / stream / sync. `forward` and
`server` need **more** than that, and those extras are exactly what is HIP-
named today. The decoupling introduces backend-neutral traits for each, with
the existing HIP types as the first impl (re-home, do not rewrite):

| Seam | Needed by | HIP impl today | CUDA impl |
| --- | --- | --- | --- |
| `Event` (record / stream-wait / elapsed) | PP + Hybrid cross-stream ordering | `HipEvent` | `CudaEvent` (`cuEvent*`) |
| `bind()` (context bind before ops) | every multi-GPU op | `backend_hip::bind` | `cuCtxSetCurrent` |
| peer copy (`memcpy_peer_async`) | PP host-bounce / stage hand-off | `HipDevice::memcpy_peer_*` | `cuMemcpyPeerAsync` |
| `Cluster` (peer matrix, bounces, aux streams) | server, PP/TP orchestration | `HipCluster` | `CudaCluster` |
| **`Collectives` / AllReduce** | TP forward (`tp_allreduce_*`) | **`BarP2pAllReduce` (BAR1 kernels)** | **NCCL** (`ncclAllReduce`) + separate fused residual/rmsnorm kernel |
| `GraphExec` + `ScalarSlot`/`MemcpySlot` | progressive-dispatch replay | `HipGraphExec` (shadow-param slot map) | `CudaGraphExec` (`cuGraph*` — same surface, more mature) |
| `Ops` registry (the fat op surface) | model-ops, forward | `HipOps` / `OpsRegistry` | `CudaOps` (mirrors `Ops` 1:1) |

**The AllReduce seam is the one place the two backends genuinely diverge.**
HIP folds AR + residual + RMSNorm into hand-rolled BAR1 P2P kernels; CUDA will
do `ncclAllReduce` then a separate fused kernel (or an NCCL + epilogue). The
`Collectives` trait must therefore be defined at the *semantic* level
(`ar_sum`, `ar_residual`, `ar_residual_rmsnorm`), not as "call the same
kernel" — two real impls behind one contract, exactly the rule-3 shape.

Recommended shape (refine in A1): a single `Backend` bundle trait grouping the
associated types (`Device`, `Stream`, `Event`, `Cluster`, `Collectives`,
`GraphExec`, `Ops`) so `forward` threads one generic parameter `<B: Backend>`
rather than seven. Generic names only (rule 12) — `Backend`, `Cluster`,
`Collectives`, never `HipBackend` on the shared trait. `ScalarSlot` /
`MemcpySlot` / `MoeShape` move from `backend-hip` to `core` (or a neutral
`flambeau-runtime`) so the `Ops` trait stops leaking backend types.

## Primitives mapping (kernel-side)

The HIP→CUDA equivalence contract `kernels-cuda/src/arch_primitives/sm_*.cuh`
must implement, same function names so kernel bodies and shared headers port
unchanged:

### Wave64 → warp32 (the structural diff)

gfx906 is wave64; all CUDA archs are warp32. Two consequences:

1. `__shfl_xor(x, off, 64)` → `__shfl_xor_sync(0xffffffffu, x, off, 32)`. A
   macro shim in `arch_primitives/sm_80.cuh` lets the 84 bare-`__shfl_xor`
   kernels compile unchanged.
2. Reduction trees: gfx906 half/quarter/eighth-warp reduces map to wave64 lane
   groups {32,16,8,4}; on warp32 these become {16,8,4,2}, and any reduction
   that today crosses the 32→64 lane boundary (full-warp `gfx906_warp_reduce_sum`)
   needs a **block-level** second stage through shared memory. The
   `gfx906_warp_reduce_sum` family is the seam: ship `sm80_warp_reduce_sum` +
   `cuda_block_reduce_sum<BLOCK>` with the same external contract (every lane
   in the group leaves with the group sum). **Wave64 reductions do not survive
   the port** — the multi-row MMVQ layouts (`nw1_r2` etc.) become genuinely
   different launches (`w1_r2_sm86`), and the cert grid is where the
   discrepancy is supposed to surface.

### Cross-lane / SFU / dp4a

| gfx906 | CUDA sm_80+ |
| --- | --- |
| `gfx906_dpp_add_xor{1,2}` / `ror8` / `shuffle_xor4` / `swizzle_xor16` (DPP / `ds_swizzle`) | `__shfl_xor_sync(mask, x, n, 32)` + add (loses DPP-fused FMA; +1 VALU/stage — accept for first port, structural perf later) |
| `__builtin_amdgcn_sdot4(a,b,c,false)` | `__dp4a(a,b,c)` |
| `gfx906_rcp` (`v_rcp_f32` asm) | `__frcp_rn` |
| `gfx906_exp2` / `gfx906_fast_exp` (`v_exp_f32` asm) | `__exp2f` (same `exp2(x·log2e)` identity) |
| `mmq_prefetch.cuh` `global_load_dword` L2 prefetch | sm_80 `cp.async` (first port: plain `__shared__`, no prefetch — perf null, correctness first) |

### Quant byte layouts + the rule-5 fix

`block_q*` layouts are byte-identical (both mmap the same GGUF). The only
port-side change is sanitising the 4 leaks in `block_quant.cuh` + `iq_grid.cuh`:

```c
#if defined(__CUDACC__)
  #include <cuda_fp16.h>
  typedef __half fb_fp16_t;
#elif defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  typedef _Float16 fb_fp16_t;
#else
  #error "block_quant.cuh: unknown backend"
#endif
```

This is the single HIP-affecting kernel edit; it pays for itself by letting
`nvcc` compile the shared headers, and HIP keeps resolving to `_Float16`.

## Driver API + build (backend-cuda)

`backend-cuda` uses the **driver API** (`cu*`), matching `backend-hip`'s
driver-level posture (`hipModuleLoadData` ↔ `cuModuleLoadData`), explicit
`CUcontext` per device. The 42-symbol `sys.rs` maps one-to-one (memory /
stream / event / module / peer / graph); `cuMemcpyAsync` direction is encoded
in the function name, mapped in the Rust shim. `kernels-cuda/build.rs` mirrors
the HIP `build.rs`: walk `src/kernels/*.cu`, `nvcc --cubin -arch=sm_86 -O3
-std=c++17 -DWARP_SIZE=32 -I arch_primitives -I ../kernels-shared/include`,
emit `cubin.rs` (`CATALOGUE: &[(&str,&[u8])]`), respect `CUDA_SKIP_BUILD=1`.
Collectives link `libnccl` behind an `nccl` feature; **no BAR1 P2P port** —
NCCL handles PCIe/NVLink topology.

## Phasing

Each phase: green build, green sweep (on live hardware once kernels land),
committed certs, gfx906 regression-clean. `cargo build --features hip` and the
HIP sweep + forward parity snapshots stay green on **every** Track-A commit.

### Track A — backend-neutral seam (HIP-only, no new behavior)

- **A0 — Sanitise shared headers.** Fix the 4 rule-5 leaks in
  `block_quant.cuh` + `iq_grid.cuh`. Verify HIP builds + sweep green. Cheap;
  unblocks `nvcc` on shared headers.
- **A1 — Lift the seams.** Define the `Event` / `Cluster` / `Collectives` /
  `GraphExec` / `Ops`-registry traits (+ `Backend` bundle) in `core` /
  `runtime`. Implement each for the **existing** HIP types (re-home, not
  rewrite). Move `ScalarSlot` / `MemcpySlot` / `MoeShape` to a neutral crate so
  `Ops` stops leaking `backend-hip`. Net behavior change: zero. Gate:
  `dispatch_toml_roundtrip` + `cert_check` + HIP unit tests green.
- **A2 — Genericize `forward`.** Rewrite `ForwardEngine`, `StageHooks` /
  `TopologyHooks`, `ArCallback`, the AR coordinators (HIP `BarP2pAllReduce`
  moves behind the `Collectives` trait), GDN composites, and all loaders over
  `<B: Backend>`. **The highest-risk phase** — do it with HIP as the only impl
  so the full forward parity-snapshot suite (`parity_snapshot_*`, `synth_*`) is
  the regression net. Gate: every `forward` test green on gfx906.
- **A3 — Genericize models + server.** `model-ops::BackendCtx<B>`; the three
  model crates' `arch`/`loader` over `<B: Backend>`;
  `server-core::SessionContext::cluster -> &dyn Cluster` (or generic);
  `server` cluster construction via a backend selector; `cli` `--backend` /
  `cuda:`-prefix parsing + `cuda_{sweep,serve}` features. Gate: gfx906
  end-to-end serve smoke still green.

End of Track A: the entire stack above kernels is backend-generic, HIP is the
sole impl, gfx906 fully green. CUDA now has a socket to plug into.

### Track B — CUDA backend + kernels (certified in isolation)

- **B0 — Scaffold.** Empty `backend-cuda` + `kernels-cuda` crates,
  `dispatch/cuda/sm_86.toml` (header + zero rows), `certs/cuda/sm_86/`,
  workspace `members` + `cuda` feature parallel to `hip`, `nvcc` `build.rs`
  (honors `CUDA_SKIP_BUILD`). `cargo build` clean with no toolchain.
- **B1 — One kernel, two backends.** `backend-cuda` driver-API core
  (`sys`/`device`/`module`/`stream`/`event` + `bind`) implementing the core
  `Device`/`Stream`/`Event` traits + the bench harness genericized enough to
  drive it. Port `add_f32`. **First live sm_86 cert.**
- **B2 — Primitives + first MMVQ.** `arch_primitives/sm_80.cuh` + `sm_86.cuh`
  (full `gfx906_*`→`sm80_*` with wave32 reduction-tree corrections, `__shfl_xor`
  shim, `__dp4a`, `__exp2f`/`__frcp`). Port `mmvq_q8_0` + `quantize_q8_1`;
  `CudaOps` registry begins. Gate: logit-cosine ≥ 0.9999 vs CPU reference on a
  single matmul shape.
- **B3 — Full quant coverage.** The rest of MMVQ + MMQ to match the 83-row
  parity set (Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q8_0 + IQ + F16). Per-impl certs; no
  "MMVQ certifies MMQ" shortcut (rule 2).
- **B4 — Attention + norms + KV cache.** `rmsnorm`/`rope`/`softmax`/`silu`/
  `swiglu`/`cast` + `attention_decode_f16{,_splitk}` + prefill (flash-tile
  first port without `cp.async`). KV-cache layout types are pure data — adding
  `CudaDevice` as a `Device` makes them work unchanged.
- **B5 — MoE family.** `indexed_moe_mmvq/mmq_*`, `moe_combine*`,
  `moe_sort_by_expert`, `topk`, `sampler_*`, `shared_expert_scale`. Largest set;
  expect a cert-grid-driven tile-N sub-sweep where Ampere diverges from gfx906.
- **B6 — Multi-GPU + collectives.** `backend-cuda` `Cluster` + NCCL
  (`nccl`/`nccl_sys`) implementing the A1 `Collectives` trait — no BAR1. CUDA
  Graphs impl of `GraphExec`. Gate: 2-GPU AllReduce cert green.
- **B7 — End-to-end model parity.** Smallest full-stack model through the now-
  generic forward/server: **qwen35moe-v2 Q4_K** (already has a glue crate).
  `flambeau infer --backend cuda` logit-cosine ≥ 0.9999 vs llama.cpp CUDA over
  64 tokens. No perf gate — bar is correctness + portability.
- **B8 — sm_80 / sm_89 sweep.** Recompile under `-arch=sm_80` / `sm_89`; per-
  arch cert grids; dispatch-table inheritance carries rows with no kernel-
  algorithmic change (rule-5 holds across the port). sm_90 (TMA/wgmma) is out
  of scope.

## Verification posture (every commit on the branch)

- `cargo build` (no features) green.
- `cargo build --features hip` green; HIP sweep regression-clean; **forward
  parity snapshots green** (the Track-A guardrail).
- `cargo build --features cuda` green; once a CUDA cert exists, sweeping it
  stays green on live sm_86.
- `dispatch_toml_roundtrip` + `cert_check` green for `hip/gfx906.toml` and (once
  seeded) `cuda/sm_86.toml`.

## Out of scope (file V2.x tickets, do not slip)

- cuBLAS / cuBLASLt — no call site in `backend-hip` today; CUDA owes no parity.
- BAR1 P2P on CUDA — NCCL covers it.
- Hopper TMA / wgmma (sm_90) — separate arch header + MMQ rewrite.
- Tensor-core MMA — no F16/BF16 matmul kernel in V1 uses it.
- Triton — kernels stay hand-written CUDA C++ for the cert + PMC narrative.

## Decisions to lock before B1

1. `Backend` bundle trait vs. seven free generic parameters threaded through
   `forward` — recommend the bundle (one `<B: Backend>` param). Confirm before
   A1 lands, since A2/A3 inherit the choice.
2. Where the neutral `ScalarSlot`/`MemcpySlot`/`MoeShape` types live — `core`
   vs. `runtime`. Recommend `core` (already the home of `Device`/`Op`).
3. NCCL version pin + multi-GPU CUDA rig topology (NVLink pair vs PCIe-only) —
   determines the B6 AllReduce baseline.
