# P5 — CUDA single-GPU end-to-end decode parity

**Status:** scoped 2026-06-18 (post-P3, parallel to P4). Tracker for roadmap
phase **P5**; `CUDA-PORT-PLAN.md` §B7. Gate: **`flambeau infer --backend cuda`
on qwen35moe-v2 Q4_K → logit-cosine ≥ 0.9999 vs llama.cpp CUDA over 64 tokens.**
No perf gate — the bar is correctness + portability on one sm_86 GPU (the RTX
3090 we have). This is the phase that proves the whole CUDA port runs a real
model.

## The one missing piece: `CudaBackend` (Registry + Ops)

Everything *around* the op-launch layer is already vendor-clean and ready:

| layer | state |
| --- | --- |
| `ForwardEngine<B>`, `StageHooks`/topology, models, server, loaders | ✅ generic over the `Backend` bundle (A0–A3 decoupling; `engine.rs` names `B::Registry`) |
| `Device` / `Stream` / `Event` / `Cluster` (CUDA) | ✅ `CudaDevice`/`CudaStream`/`CudaEvent`/`CudaCluster` (P1–P4) |
| CUDA kernels (`.cubin`) | ✅ all of P2 (mmvq/mmq) + P3 (attention/norm/rope/sampler/gdn/…) compiled for sm_86 |
| `dispatch/cuda/sm_86.toml` | 🟡 seeded with the autotuned `qmatmul` rows; needs the rest of the decode-path ops filled in |
| **`CudaBackend: Backend`** = CUDA **`Registry`** (load cubins) + **`CudaOps: Ops`** (145-method launch layer) | ❌ **not started — this is P5** |

The `Backend` bundle (`crates/backend/src/lib.rs`) already declares
`type Registry` + `type Ops<'a>: Ops` + `fn ops(reg, stream)`; `HipBackend`
fills them with `OpsRegistry` + `HipOps`. **P5 is: fill those two slots for
CUDA.** Once `CudaBackend` exists, `ForwardEngine<CudaBackend>` is the existing
generic engine — no forward/model/server changes.

## Central decision: genericize the launch glue, do not copy it

The HIP op-launch glue is large — `crates/ops/src/hip/` is ~450 KB across
`qmatmul.rs` (98 KB), `moe.rs` (136 KB), `attention.rs` (49 KB), `ops_impl.rs`
(48 KB), norm/cast/pe/rope/sampling/recurrent. Each launch body is the same
shape: `reg.expect_module(stem) → module.kernel(name) → push args → launch with
grid/block`. The **only** HIP-specific things are the module/kernel/args/launch
*types* (`HipModule`/`HipKernel`/`KernelArgs`); the dispatch arithmetic (shape →
variant, grid/block) is backend-identical.

**Recommendation: genericize `OpsRegistry`/`OpCtx` + the launch bodies over a
backend launch trait, so one body serves both backends** — not a parallel
`cuda/` copy. Rationale:
- **Proven**: `crates/bench/src/backend.rs` `BenchBackend` already abstracts
  `Module`/`Kernel`/`Args` + `image`/`load`/`kernel`/`launch` and runs the
  *same* sweep bodies on HIP **and** CUDA across ~90 kernels (all of P2/P3). P5
  extends that exact proven pattern from the sweep harness to the runtime layer.
- **Rule 14**: copying 450 KB of launch glue is the canonical "don't duplicate
  100-LOC blocks across backends" violation; it doubles maintenance for every
  future kernel/arch forever.
- **Consistency**: A0–A3 already genericized Device/Stream/Event/Cluster + the
  whole forward/model/server stack. The op-launch layer is the last HIP-concrete
  island; genericizing it finishes the decoupling rather than forking it.

Trade-off: a larger upfront refactor (touching the big `hip/` files) vs. faster
first-decode-but-permanent-duplication. The refactor is the right multi-session
investment; the HIP forward-parity snapshots are the guardrail that it stays
behaviour-preserving.

## Phasing

- ✅ **P5.0 — Lift the launch primitives.** Done (`65a8e1c`/`dc5ee2d`/`1d28e52`):
  `core::KernelBackend` + `LaunchArgs` is the launch seam (module load / entry
  resolve / launch + `bind`), impl'd by `HipKernels` (backend-hip) and
  `CudaKernels` (backend-cuda). `OpsRegistry` is now `Registry<HipKernels>` (a
  type alias over the ungated `ops::registry`), so every existing body + all
  `OpsRegistry::new` sites compile unchanged via `expect_module → &B::Module`.
  `cast.rs` is the first body migrated to the generic form (`B::kernel`/
  `B::Args`/`B::launch`), with `ops_impl` inferring `B=HipKernels` at the call
  site. `BenchBackend` now *extends* `KernelBackend` (parallel trait removed).
  Guardrail green: `ops_smoke` 5/5, `parity_snapshot_dense`/`gdn` +
  `snapshot_restore` bit-identical on gfx906; `add_f16` sweep certs both
  backends.
- 🔄 **P5.1 — Genericize the launch bodies, domain by domain.** Bodies move to
  `<B: KernelBackend>(reg: &Registry<B>, stream, …)` with `B::kernel`/`B::Args`/
  `B::launch`; `ops_impl` callers pass `self.reg, self.stream`. Done via a
  scripted mechanical pass (`/tmp/migrate_ops.py`) + per-file import fixes, each
  batch gated on the gfx906 parity snapshots.
  - ✅ **Batch 1** (`237116d`): `cast`, `softcap`, `softmax`, `conv`, `pe`
    (rope ×3), `sampling`, `router` (dense_gemv ×4).
  - ✅ **Batch 2** (`b78bf6e`): `norm` (14), `mlp` (12), `recurrent` (7 GDN).
  - Guardrail green throughout: `ops_smoke` 5/5, `synth_dense/gdn/gemma_mixed` +
    `parity_snapshot_dense/gdn/attn_gated` + `snapshot_restore` bit-identical.
  - ⏸️ **Remaining (the big three):**
    - **`attention.rs` — blocked.** Uses 6 `KernelArgs::push_slot` graph-slot
      bindings (`ScalarSlot`); the generic `LaunchArgs` has no slot surface yet.
      Needs `LaunchArgs` extended with a `push_slot` (or the slot path lifted) —
      the same progressive-dispatch slot support the P4.4 graph slot-map needs.
      Do that seam extension first, then migrate.
    - **`qmatmul.rs`** (24 op fns + 10 helper fns) and **`moe.rs`** (74 op fns +
      2 helpers) — migratable mechanically, but the script needs two additions
      first: `KernelArgs::with_capacity(n)` → `<B::Args<'_>>::default()`, and the
      private helper fns that take `reg`/module must also gain `<B>` (the `pub fn`
      regex misses them). Large but low-risk once those are handled.
- **P5.2 — `CudaBackend` + CUDA `Registry`/`CudaOps`.** Implement `Backend` for
  `CudaBackend` (Device/Stream/Event/Cluster already done); `Registry<CudaBackend>`
  loads `flambeau_kernels_cuda::cubin(stem)`; `CudaOps` is the generic `Ops` over
  `CudaBackend`. Gate: a CUDA op smoke (e.g. `rmsnorm_f16` / `attention_decode_f16`
  vs the F64 oracle) through the runtime path, not just the sweep.
- **P5.3 — Fill `dispatch/cuda/sm_86.toml` for the decode path.** The
  qwen35moe-v2 Q4_K decode ops (Q4_K mmvq/mmq, attention_decode_f16, rmsnorm,
  rope, swiglu, moe sort/combine/topk, sampler). Each row needs a green
  `certs/cuda/sm_86/<impl>.json` (rule 2). `flambeau autotune --backend cuda`
  seeds the mmvq rows; the rest from the P3 certs.
- **P5.4 — The parity gate.** `flambeau infer --backend cuda --model
  qwen35moe-v2-Q4_K`: single token first (logit-cosine vs llama.cpp CUDA), then
  64 tokens ≥ 0.9999. File the diff; no perf claim.

## Validation posture

- **HIP forward-parity snapshots are the guardrail** through P5.0–P5.1 — every
  genericization step must leave the gfx906 forward bit-identical (the same
  snapshot suite that protected A0–A3). A red snapshot means the refactor
  changed behaviour; stop and fix before proceeding.
- **CUDA correctness** rides the existing sweep certs (per-op) → a runtime op
  smoke (P5.2) → the end-to-end logit-cosine (P5.4).

## What P5 unblocks

- **P4.3 `FusedAllReduce`** — needs `CudaOps` to dispatch the `add_f16` /
  `rmsnorm_*_add_residual` epilogues after `ncclAllReduce`. Once P5.2 lands the
  CUDA op-launch layer, P4.3 is `ncclAllReduce` + an existing op call.
- **P4.4 graph slot-map** — the progressive-dispatch param rebind needs the CUDA
  kernel-node param surface, which the CUDA `Ops` launch path establishes.
- **B8** — sm_80 / sm_89 are then recompiles + per-arch cert grids over the same
  generic Ops (no launch-glue changes).

## Risks / notes

- The big-file genericization (`moe.rs`, `qmatmul.rs`) is where subtle
  HIP-isms may hide (wave64 assumptions in *host-side* launch arithmetic, e.g.
  block dims hard-coded to 64). Audit each for the `WARP_SIZE` host analog as the
  bodies go generic — the kernel side is already arch-seamed from P3.
- `KernelArgs` vs a CUDA args type: the launch trait must abstract arg-pushing
  (the `LaunchArgs` trait in `bench/backend.rs` already does this — reuse it).
- qwen35moe-v2 Q4_K is the smallest full MoE; it exercises mmvq/mmq + attention
  + moe routing + sampler — a broad first end-to-end, but the right one (it has
  a glue crate already).
