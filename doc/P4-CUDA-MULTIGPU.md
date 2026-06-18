# P4 — CUDA multi-GPU runtime (NCCL collectives + CUDA Graphs)

**Status:** scoped 2026-06-18 (post-P3). The work-list tracker for roadmap phase
**P4**; see `PORTABILITY-ROADMAP.md` for the ledger and `CUDA-PORT-PLAN.md` §B6
for the original grouping. P4 is **multi-GPU only** — it does not gate the P5
single-GPU decode-parity milestone (which is unblocked now that P3 is done).

## What P4 is

The CUDA side of the multi-GPU runtime: a `CudaBackend` that completes the
`Backend` bundle (`crates/backend/src/lib.rs`) with a `CudaCluster`, an NCCL
implementation of the `DeviceAllReduce` / `FusedAllReduce` collective seams
(`crates/runtime/src/collective.rs`), and a `CudaGraphExec` mirroring
`HipGraphExec`. The 3 HIP-only `p2p_allreduce_*` BAR1 kernels are **not** ported
— NCCL handles PCIe/NVLink topology, and the fused epilogues become NCCL-allreduce
+ a separate epilogue kernel (CUDA-PORT-PLAN §B6, "no BAR1 P2P port").

## Multi-GPU is arch-agnostic at the seam — but a cluster is homogeneous

The `Backend` / `Cluster` / `Collectives` seams are generic traits, so the
**generic multi-GPU forward logic** (TP/PP orchestration, the AR coordinator,
the engine) is backend-neutral and runs over whichever backend's cluster. A
single cluster, however, is **homogeneous**: the `Backend` bundle binds ONE
`Device` and ONE `Cluster` type, and the device collective libraries do not
interoperate (NCCL is NVIDIA-only, RCCL AMD-only; there is no cross-vendor GPU
all-reduce). **You cannot mix an MI50 and a 3090 in one TP/PP group.**

What this means for the current rig (**2× MI50 gfx906 + 1× RTX 3090 sm_86**):

- ✅ The arch-agnostic multi-GPU *architecture* is already validated — on the
  **2× MI50** (the HIP path: BAR1 / RCCL collectives + `cluster_peer_access` /
  `bar_p2p_smoke` / `rccl_collectives` / pp2tp2 forward parity). That is the
  generic engine/cluster/collective code; it is backend-neutral and green.
- ⚠️ The **CUDA NCCL collective impl** (`CudaCluster` + NCCL `DeviceAllReduce` /
  `FusedAllReduce`) is the only piece that needs **2 NVIDIA GPUs** to validate
  real peer P2P / NVLink — and the rig has only 1.

So P4 is reasonable to **build now** (the generic side is proven on AMD; the CUDA
layer is contained), with partial validation that DOES fit 1 NVIDIA GPU:
   - NCCL 2-ranks-on-1-GPU (two communicators on device 0): exercises
     `ncclCommInitAll` + `ncclAllReduce` API + reduction math, NOT real P2P.
   - `RefMesh` (CPU host-bounce) oracle parity: the math reference (already exists).
   - `CudaGraphExec` compile + graph-capture smoke (single GPU — no 2-GPU need).
The real **2-GPU NCCL P2P gate** (B6's "2-GPU AllReduce cert") + the multi-GPU
`ForwardEngine<CudaBackend>` smoke (P4.5) defer to when a 2nd NVIDIA card lands.

**Other prereq: NCCL is not installed** (`ldconfig -p | grep nccl` empty, no
`libnccl.so`). Need `libnccl` + headers (CUDA 13.x-matched) before `nccl_sys`
FFI links. Pin the version (decision below).

## Current seam state (what exists, what P4 adds)

| seam | trait / type | HIP today | CUDA (P4) |
| --- | --- | --- | --- |
| Backend bundle | `Backend` (`backend/src/lib.rs`) | `HipBackend` ✅ | **`CudaBackend`** (new) — shares Device/Stream/Event with P1-P3; needs `Registry`/`Ops` (also a P5 prereq) |
| rank set | `Cluster` (`runtime/cluster.rs`: `ranks`/`device`/`peer_access_full`) | `HipCluster` ✅ | **`CudaCluster`** (new) — device list + `ncclComm` per rank |
| core AR | `DeviceAllReduce::all_reduce_sum` (`runtime/collective.rs`) | BAR1 (`forward/runtime/ar.rs` + `backend-hip/bar_p2p.rs`) | **NCCL** `ncclAllReduce(Sum)` F32/F16 |
| fused AR | `FusedAllReduce` (`ar_sum_f16` / `ar_residual_f16` / `ar_residual_rmsnorm_f16` / `ar_postattn_residual_rmsnorm_f32_to_f16`, each `supports_*`) | BAR1 fused kernels | **NCCL allreduce + epilogue kernel** — the epilogues are the P3-ported `add_f16` / `rmsnorm_f16_add_residual` / `rmsnorm_f32_to_f16_add_residual` (already cross-backend!) |
| graph replay | `HipGraphExec` (concrete, `backend-hip/device.rs`; shadow-param slot map) | ✅ | **`CudaGraphExec`** (`cuGraph*` — same surface, more mature; B6 notes the seam may need lifting to a trait first) |

**Key leverage from P3:** the `FusedAllReduce` epilogues on CUDA are NCCL +
existing kernels. `add_f16` (residual fan-in), `rmsnorm_f16_add_residual`,
`rmsnorm_f32_to_f16_add_residual` are all already on CUDA sm_86 (P3 Wave 1b/2).
So `FusedAllReduce` for CUDA = `ncclAllReduce` then dispatch one of those — no new
kernel authoring, just composition.

## Phasing (B6 sliced)

- **P4.0 — Prereqs.** Install + pin NCCL; decide the rig path (2nd NVIDIA GPU vs
  1-GPU partial validation). Add `nccl`/`nccl_sys` (or hand FFI) behind an `nccl`
  cargo feature on `backend-cuda`.
- **P4.1 — `CudaBackend` + `CudaCluster` (no collectives).** Complete the
  `Backend` bundle for CUDA (Device/Stream/Event already exist from P1-P3; add the
  CUDA `Registry`/`Ops` op-launch layer — shared with P5). `CudaCluster`: device
  list + peer-access matrix (`cudaDeviceCanAccessPeer`). Gate: `ranks()` /
  `device(r)` smoke on 1 GPU.
- **P4.2 — NCCL `DeviceAllReduce`.** `ncclCommInitAll` in `CudaCluster::new`;
  `all_reduce_sum` → `ncclAllReduce(ncclSum, F32/F16)` on the rank's stream. Gate:
  2-GPU (or 2-rank-1-GPU) AllReduce vs the `RefMesh` CPU oracle.
- **P4.3 — `FusedAllReduce` (NCCL + epilogue).** Implement the 4 fused ops as
  `ncclAllReduce` + the P3 add/rmsnorm kernels; `supports_*` mirror the HIP rank
  gates (TP=2 residual, TP∈{2,4} sum/post-attn). Gate: fused-AR cert vs oracle.
- **P4.4 — `CudaGraphExec`.** `cuGraph*` capture/instantiate/launch + the
  shadow-param slot map (`ScalarSlot`/`MemcpySlot`) mirroring `HipGraphExec`. Lift
  `GraphExec` to a trait if not already. Gate: graph-capture smoke (single GPU —
  this one does NOT need 2 GPUs).
- **P4.5 — multi-GPU forward smoke.** `ForwardEngine<CudaBackend>` pp2/tp2 on
  2 NVIDIA GPUs (gated on hardware). Byte/logit parity vs single-GPU CUDA decode.

## Notes / decisions to lock

1. **Rig** — get a 2nd NVIDIA GPU, or accept 1-GPU partial validation (API +
   math, not topology) for P4.0-P4.4 and defer P4.2's real 2-GPU gate + P4.5 to
   when hardware lands. (Decision needed from the user.) **2nd-card arch floor:
   sm_75 (Turing).** CUDA 13.3 dropped Maxwell/Pascal/Volta (min sm_75), and
   every quantized kernel needs `dp4a` (sm_61+) — so a Maxwell card (e.g. Quadro
   M5000, sm_52) will NOT work. Best fit is a 2nd **sm_86** card (matches the
   3090 → identical cert grid, real PCIe-P2P pair); any sm_75/sm_89/sm_80 works
   with its own cert grid (B8 machinery).
2. **NCCL version pin** + bindings: `nccl_sys` crate vs hand FFI (the project hand-
   writes FFI elsewhere — `cu*`/`hip*` — so hand FFI is consistent; `nccl_sys`
   saves typing). NVLink-pair vs PCIe-only topology determines the AllReduce
   baseline (CUDA-PORT-PLAN "Decisions to lock #3").
3. **`GraphExec` lift** — it is HIP-concrete (`HipGraphExec`) today, consumed via
   `ops_trait`. Confirm whether forward needs it as a generic `Backend` assoc type
   or whether `CudaGraphExec` can mirror the concrete surface (P4.4 / P5 overlap).
4. **`CudaBackend` overlaps P5.** The Registry/Ops op-launch layer is needed by
   both P4 (to dispatch the fused epilogues) and P5 (e2e decode). Build it once;
   P4 and P5 share it. This argues for doing the `CudaBackend` Registry/Ops slice
   first regardless of which phase formally "owns" it.

## Out of scope (per CUDA-PORT-PLAN)

- BAR1 P2P on CUDA (NCCL covers it). The 3 `p2p_allreduce_*` HIP kernels stay
  HIP-only.
- cuBLAS / cuBLASLt, Hopper TMA/wgmma (sm_90), tensor-core MMA, Triton.
