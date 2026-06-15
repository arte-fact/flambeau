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

---

## A2 progress + A1 Collectives design (build-on-Mesh) — 2026-06-12

### Done
- **A2.1 — loaders genericized over the `Device` seam** (`c8e7982`, branch
  `feature/cuda-decouple`). `crates/forward/src/loader/*.rs`: `device:
  &HipDevice` → `&impl Device`, HipDevice import dropped (8 files, 28 params).
  Static dispatch → identical code; build + clippy clean (hip_serve); forward
  parity/synth suite green on gfx906 (snapshot_restore 5/5, parity_snapshot_*,
  synth dense/gdn/moe/gemma). Leaf-most, AR-free slice.

### A2.4 prerequisite — the AR seam (`Collectives`)
A2.4 (genericize `ForwardEngine` / `StageHooks` / `TopologyHooks` /
`ArCallback`) is blocked: the engine's AR contract (`TopologyHooks` in
`crates/forward/src/core/hooks.rs`) is typed on `&HipDevice`/`&HipStream` and
routes through `BarArCoordinator` (`crates/forward/src/runtime/ar.rs`) →
`BarP2pAllReduce` (`crates/backend-hip/src/bar_p2p.rs`, the BAR1 kernels). A1
lifted Device/Stream/Event/Cluster/Ops but **not** a Collectives seam.

**Audit finding:** there are TWO parallel collective abstractions today —
1. `runtime::collective` (`AllReduce`/`AllGather`/`AllToAll`/`Broadcast` over
   `runtime::mesh::Mesh`, byte-buffer `&mut [u8]` + `CollectiveCfg`) with only
   a **CPU host-bounce reference impl** (`RefMesh`). Grep confirms **no
   production consumer** — it is currently an oracle/cert target, not wired to
   the real path.
2. The real device AR — `BarP2pAllReduce` (backend-hip) + `BarArCoordinator`
   (forward) + the fused ops `ar_sum_f32` / `ar_sum_f16` / `ar_residual_f16` /
   `ar_residual_rmsnorm_f16` / `ar_postattn_residual_rmsnorm_f32_to_f16`. The
   fused ops collapse AllReduce + residual-add + RMSNorm into one BAR1 kernel;
   they CANNOT be expressed as a plain byte-buffer `AllReduce`.

**Decision (user, 2026-06-12): build the seam ON the `Mesh`/`AllReduce`
framework** — unify the two abstractions rather than add a third parallel
trait. This is a multi-session architectural effort; sliced as:

- **C1 — device collective surface on `Mesh`.** Extend the `collective`
  framework with a device-pointer AllReduce op (`DevicePtr` + `&Stream`, sum,
  F32/F16), parallel to the byte-buffer one. Implement it for the HIP cluster
  (wrapping `BarP2pAllReduce`/`BarArCoordinator`, preserving the deterministic
  copy-engine DtoD path — see `doc/DETERMINISM_INVESTIGATION.md`). Keep the CPU
  `RefMesh` byte-buffer impl as the oracle. Gate: existing collective ref
  tests + a HIP device AR cert.
- **C2 — fused-epilogue ops as a layer.** Express `ar_residual_f16` /
  `ar_residual_rmsnorm_f16` / `ar_postattn_…` at the **semantic** level on the
  seam (so HIP fuses via BAR1 and CUDA does NCCL-allreduce + a separate
  epilogue kernel — rule-3 two-impls-one-contract). `supports_*` capability
  (n_ranks ∈ {2,4}, ranks == 2) moves onto the impl.
- **C3 — rewire forward.** `TopologyHooks`/`TpHooks`/`HybridHooks` delegate to
  the Mesh-based seam; genericize the signatures + `ArCallback` over the
  Device seam. Unblocks A2.4. **Highest-risk** — AR has a long subtle-bug
  history (BAR1 incoherence, determinism, write-target).
- **C4 — validate.** CPU-ref oracle parity (kept) + `gdn_tp_mode` + a live
  pp2tp2 smoke (gemma4 + qwen3.6) for byte-identical decode vs pre-refactor.

After C1–C3 the engine names no `Hip*` AR type and A2.4 (CoreState/ctx/hooks
genericization) proceeds; then A2.2/A2.3 (ScratchPool, composites) finish A2.

---

## A2.2/A2.4 scope finding — the engine genericization is one connected cascade (2026-06-12)

C1–C3 landed the AR seam (`DeviceAllReduce`/`FusedAllReduce`) and routed the
forward engine's fused AR through it (live pp2tp2 byte-identical gate green,
`76b5e55`). The remaining A2 (ScratchPool → CoreState → composites →
ForwardCtx → hooks genericization) is **not** a series of isolated mechanical
edits like A2.1 (loaders).

Tried A2.2 in isolation — `device: &HipDevice` → `&impl Device` on the 7
`ScratchPool` methods (scratch.rs). It builds the 7 sites fine but **breaks 4
model-ops callees**: `delta_net.rs:{430,485,544}` + `moe_experts.rs:352`
(`alloc_prefill_scratch`) — ScratchPool passes its device into those model-ops
scratch allocators, which take a concrete `&HipDevice`. So genericizing
ScratchPool requires genericizing the model-ops scratch-alloc fns first, and
those likely reach further (HipOps construction, etc.).

**Consequence — order the engine genericization bottom-up, as one connected
refactor that compiles at each step:**
1. model-ops scratch-alloc fns (`delta_net` GDN scratch, `moe_experts`
   prefill scratch) over `&impl Device`.
2. `ScratchPool` methods (scratch.rs, 7 sites) over `&impl Device`.
3. `CoreState<'a>` → carry the backend seam (`device: &B::Device`,
   `stream: &B::Stream`, `reg: &B::Registry`) — the `Backend` bundle, not just
   `Device`, since `reg: &OpsRegistry` and ops construction (`HipOps::new`)
   are backend-typed.
4. The `ForwardCtx` impls (SingleDevice/Tp/Pp/Hybrid) + `TopologyHooks` /
   `StageHooks` / `ArCallback` signatures over `<B: Backend>` — `TpHooks`/
   `HybridHooks` swap `Arc<BarArCoordinator>` for a generic `FusedAllReduce`
   handle (the C1–C3 seam makes this a type swap, not a logic change).
5. The composites (`standard_attn`, `dense_ffn`, `moe_ffn`, `gdn`, …) inherit
   `CoreState<B>` — mostly free once 1–4 land.

Type-only (no behavior change), but a **wide, single-session-too-big** sweep:
the crate does not compile between steps 1 and 4. Do it in a dedicated session,
bottom-up, with the forward parity/synth suite + a pp2tp2 byte-identical gate
as the net. A2.1 (loaders) + C1–C3 (AR seam) are the isolated pieces already
banked; the rest is this one connected cascade.

### Empirical depth of step 1 (the model-ops alloc layer) — 2026-06-12

Attempted the bottom of the cascade (genericize `alloc_zeroed` /
`alloc_zeroed_tracked` + the 4 scratch-alloc fns over `&impl Device`). It does
NOT stop there: the scratch-alloc fns use the **entire `RawAllocTracker` API**
(`alloc_i32`, `alloc_f32`, `alloc_zeroed_tracked`, `track`, …) plus the free
helpers in `driver_utils.rs` — every one `&HipDevice`-typed — so widening the
scratch fns surfaced **64 compile errors** across model-ops. Step 1 is
therefore a genericization of the **whole `driver_utils` driver/alloc layer**
(`RawAllocTracker` methods + `alloc_zeroed`/`upload_f16_ones`/`embed_token_host`
/…), per the model-ops CLAUDE.md rule-9 form `<D: Device>(device: &D,
stream: &D::Stream)` — careful per-fn (device+stream must be paired, not blanket
sed), ~64 sites, before ScratchPool even starts.

So the realistic cascade size, revised: **step 1 ≈ a full model-ops alloc-layer
pass (~64 sites)**, then ScratchPool, then the `CoreState`/`Backend`-bundle top.
Each step compiles only once its whole layer is converted — bottom-up, but each
layer is itself a non-trivial mechanical sweep. This is a dedicated multi-hour
session with full context budget, not an end-of-session continuation. Banked so
far: A2.1 (loaders) + C1–C3 (AR seam, live-gated). The alloc-layer→engine
cascade remains, now correctly sized.

---

## C4 — peer-copy / stage-handoff seam (the last Track-A blocker) — 2026-06-12

A2.4 finalized the forward **execution** layer (composites + `CoreState<B>` +
`TopologyHooks<B>`, commits `8bfc8bc`→`9d822e1`, pp2tp2 byte-identical). The
engine (`ForwardEngine` / `StageHooks`) deliberately stayed `HipBackend`-pinned
because `ForwardEngine<B>` is **blocked on a missing peer-copy seam** — exactly
analogous to how A2.4's AR genericization was blocked on the AR seam until
C1–C3. C4 builds that seam, mirroring C1–C3's shape.

### Audit (file-grounded, branch `feature/cuda-decouple`)

The PP/Hybrid stage handoff bottoms out in HIP P2P with no seam:
- **`Device`-seam gap.** The peer-copy primitives are HIP-**inherent** methods,
  NOT on the `core::Device` trait: `HipDevice::bind` (`backend-hip/device.rs:1029`),
  `memcpy_peer_async` (`:1210`), `memcpy_peer_in_async` (`:1251`). They resolve
  today only because `B = HipBackend`. These are precisely the `bind()` + "peer
  copy" seam rows the plan already lists (§"The seam", lines 84–85) — the only
  A1 seams never lifted.
- **`PeerSlot`** (`forward/src/runtime/ar.rs:823`) holds
  `send_done: Mutex<Option<HipEvent>>` + `consumer_device: Option<Arc<HipDevice>>`;
  constructors `new_peer_edge`/`new_peer_edge_prealloc` (`:869`,`:888`) call
  `HipDevice::new`.
- **`StageHooks`** (`engine.rs:263`) methods take `&mut CoreState<'_>` (HipBackend);
  `PpStage`/`HybStage` (`:301`,`:471`) borrow `&PeerSlot`, and the bodies call
  `HipEvent::new`, `device.memcpy_peer_async`, `device.bind`.
- The `Cluster` seam (A1, `runtime/cluster.rs`) already covers the peer matrix
  (`peer_access_full()`); peer-access *authorization* stays in `HipCluster::new`
  (the construction boundary). C4 does **not** touch `Cluster`.

### Slices (bottom-up, green per slice; mirrors C1–C3)

- **C4.1 — extend the `Device` seam.** Add `bind(&self)`,
  `memcpy_peer_async(&self, stream, dst, dst_dev_id, src, bytes)`, and
  `memcpy_peer_in_async(...)` to `core::Device`; implement on `HipDevice` by
  delegating to the existing inherent methods (re-home, not rewrite — the A1
  pattern). Net behavior: zero. The AR DtoD path (`ar.rs:577`
  `device.memcpy_peer_in_async`) and `PpStage::peer_send` keep calling the same
  code, now via the trait. Gate: build/clippy `--features hip` + HIP unit tests
  + forward parity/synth.
- **C4.2 — genericize `PeerSlot<D: Device = HipDevice>`.**
  `send_done: Mutex<Option<D::Event>>`, `consumer_device: Option<Arc<D>>`
  (`PeerDeviceBuffer` is pure `DevicePtr`+`bytes`, unchanged). The constructors
  stop calling `HipDevice::new`: take a caller-supplied `Arc<D>` (orchestrate.rs
  already owns the cluster's devices) — pushes device construction to the
  orchestration boundary (rule 12). Default `= HipDevice` keeps callers green
  (de-atomization trick). Gate: build + parity.
- **C4.3 — genericize `StageHooks<B: Backend = HipBackend>` + `PpStage`/`HybStage`
  over `B`.** Methods take `&mut CoreState<'_, B>`; the stage structs hold
  `&'a PeerSlot<B::Device>`; bodies swap `HipEvent::new(id)` → `core.device.new_event()`
  and use the C4.1 trait methods. `SoloStage` impls `StageHooks<B>` (peer_* are
  bails — trivially generic). **AR/handoff-adjacent → live pp2tp2 gate** (pp2tp2
  is PP×TP, so HybStage peer-copy + TP AR both fire): one `.probe/c3_gate.py`
  run, qwen3.6 `cb529e9c…` / gemma4 `805a6178…` (`PORT=18080`).
- **C4.4 — `ForwardEngine<'a, B, H: TopologyHooks<B>, S: StageHooks<B>>` falls
  out.** `core: CoreState<'a, B>`; `build`/the 4 `new` impls take
  `&B::Device`/`&B::Stream`/`&B::Registry`; `ForwardCtx for ForwardEngine` is
  mechanical (composites already infer `B`). `workers.rs` pins `B = HipBackend`
  at per-rank construction; `orchestrate.rs` constructs `HipCluster` and feeds
  `Arc<HipDevice>` into the now-generic `new_peer_edge`. Gate: full forward
  suite + final pp2tp2 byte-identical.

### After C4

`ForwardEngine<B>` is generic; the only `Hip*` left in `flambeau-forward` is the
deliberate **construction/selection boundary** — `workers.rs` (`HipDevice::new`),
`orchestrate.rs` (`HipCluster::new`), and `runtime/ar.rs` (the HIP AR+P2P impl
behind the `DeviceAllReduce`/`FusedAllReduce`/`Device` seams). That completes A2.
Remaining for end-to-end CUDA: **A3** (models + server + cli generic) then
**Track B** (B0–B8: the `backend-cuda`/`kernels-cuda` port + certs). C4 + the
A1 seams + C1–C3 are the runtime prerequisites B6/B7 depend on.

Optional tidy (not required): `PeerSlot`/`new_peer_*` are now generic forward
types living beside the HIP-concrete `BarArCoordinator` in `runtime/ar.rs`;
could move to a `runtime/peer.rs`. Cosmetic — defer unless ar.rs churns.

### C4 — DONE (2026-06-12). A2 engine genericization complete.

All four slices landed green + pp2tp2 byte-identical (`805a6178…` / `cb529e9c…`):
- **C4.1** (`ae564b0`): `bind`/`memcpy_peer_async`/`memcpy_peer_in_async` lifted
  to `core::Device`; HipDevice delegates to its inherent methods.
- **C4.2** (`5b4b12b`): `PeerSlot<D: Device = HipDevice>`; edge constructors take
  a caller-supplied `Arc<D>` (HipDevice::new moved to `orchestrate.rs`).
- **C4.3** (`dcd582e`): `StageHooks<B>` + `PpStage<'a, B>`/`HybStage<'a, B>`;
  handoff event via `core.device.new_event()`; peer-copy/bind through the seam.
- **C4.4** (`d52db7c`): `ForwardEngine<'a, B, H: TopologyHooks<B>, S: StageHooks<B>>`
  generic; the gemma per-layer-embd builder now routes `dense_gemv` through
  `Ops` (last engine trait-bypass closed).

`flambeau-forward` now names `Hip*` ONLY at the construction/selection boundary
(`engine.rs` 4 HIP `new` ctors + `*Engine` aliases, `workers.rs`
`HipDevice::new`/`OpsRegistry::new`, `orchestrate.rs` `HipCluster::new`) +
`runtime/ar.rs` (the HIP AR+P2P impl behind the `DeviceAllReduce`/`FusedAllReduce`
seams) + the concrete `TpHooks`/`HybridHooks` `TopologyHooks` impls (rule-12
arch-specific impls wrapping `BarArCoordinator`). Every one of these is the
deliberate HIP backend-selection point, not leakage.

**Track A status:** A0 ✅, A1 ✅, A2 ✅ (loaders A2.1 + AR seam C1–C3 + engine
cascade A2.4 + peer-copy seam C4). **A3 remains** (model crates' arch/loader +
server cluster + cli `--backend`/`cuda:` — see §"Track A" A3). Then Track B
(B0–B8: `backend-cuda` + `kernels-cuda` + certs). The runtime prerequisites
B6/B7 depend on (Device/Event/Cluster/Collectives/peer-copy seams) are all in
place; a CUDA `Backend` impl can now instantiate `ForwardEngine<CudaBackend>`
once its kernels + cluster land.

### A3 — scope (file-grounded audit, 2026-06-12)

Coupling is far lighter than the original §"coupling" table guessed. Counts of
`Hip*`/`OpsRegistry` refs: qwen35-v2 / qwen35moe-v2 / gemma4-v2 = **8 each**
(all `&HipDevice` in `arch.rs`+`loader.rs` load/dispose/shard fns); server-core
= **2** (`SessionContext::cluster -> &HipCluster`); server = **21** (cluster
construction + `state.cluster: Arc<HipCluster>` + a few `&HipCluster` reads);
cli = **2** (`hip:` device-string prefix). The `Arch::forward<C: ForwardCtx>`
methods are already backend-neutral.

Slices:
- **A3a — model loaders + `Arch` over the Device seam.** `Arch::load`/`dispose`
  (`forward/runtime/mod.rs:77,135`) `&HipDevice` → `&impl Device`, plus the 3
  model crates' `loader.rs` free fns + `dispose` method + `arch.rs` impls.
  **Atomic** across the trait + 3 impls (a method-level `&impl Device` can't use
  the `=HipBackend`-default trick), but pure A2.1-style widening — `workers.rs`
  call sites pass `&HipDevice`, which coerces. Gate: build + forward
  parity/synth + a gfx906 serve smoke.
- **A3b — `server-core::SessionContext::cluster` over the Cluster seam.**
  `cluster(&self) -> &HipCluster` (`server-core/traits.rs:29`) →
  `&impl Cluster` / generic, + the impl (`server/routes.rs:269`). Lets the
  per-arch contract stop naming `HipCluster`.
- **A3c — server cluster construction + `state.cluster` (DEFERRED — design
  fork).** The server is a *runtime* backend selector (`--backend cuda` is a
  startup flag), so it can't be purely compile-time-generic — it needs runtime
  dispatch (`enum {Hip,Cuda}` / `Box<dyn Cluster>`) OR a per-backend
  monomorphized entry point chosen by a startup `match`. That design is built
  WITH the CUDA backend (Track B), not speculatively — there is no second
  backend to select today. Until then `server` naming `HipCluster` is the
  correct selection boundary (rule 12), exactly like `workers.rs`/`orchestrate.rs`.
- **A3d — cli `--backend` / `cuda:` prefix + `cuda_{sweep,serve}` features.**
  Depends on A3c; deferred with it.

So the useful-now A3 = **A3a + A3b** (decouple the model + per-arch-contract
surfaces); A3c/A3d are deferred to Track B by the same "selection boundary"
principle that kept the engine/workers HIP-concrete.
