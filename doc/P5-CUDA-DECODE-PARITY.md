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
- ✅ **P5.1 — COMPLETE. All 13 op-launch domains are generic over `KernelBackend`.**
  The op layer (`crates/ops/src/hip/*`) names no concrete `HipModule`/`KernelArgs`;
  HIP forward + the full ops suite (incl. the 14-test attention parity) stay
  bit-identical. Two seams were built along the way: `dispatch_qmatmul`
  (per-backend table) and `LaunchArgs::push_slot` (graph-capture slot tagging,
  default = plain push for backends without a graph slot-map). Bodies move to
  `<B: KernelBackend>(reg: &Registry<B>, stream, …)` with `B::kernel`/`B::Args`/
  `B::launch`; `ops_impl` callers pass `self.reg, self.stream`. Done via a
  scripted mechanical pass (`/tmp/migrate_ops.py`) + per-file import fixes, each
  batch gated on the gfx906 parity snapshots.
  - ✅ **Batch 1** (`237116d`): `cast`, `softcap`, `softmax`, `conv`, `pe`
    (rope ×3), `sampling`, `router` (dense_gemv ×4).
  - ✅ **Batch 2** (`b78bf6e`): `norm` (14), `mlp` (12), `recurrent` (7 GDN).
  - ✅ **Batch 3** (`5de5a58`): `moe` (74 fns — the biggest file). Pure launch
    glue (hardcoded/parameterised stems, no dispatch-table lookup); migrated
    mechanically (named-kernel launches + `with_capacity` handled).
  - Guardrail green throughout: `ops_smoke` 5/5, `synth_dense/gdn/gemma_mixed/moe`
    + `parity_snapshot_dense/gdn/attn_gated` + `snapshot_restore` bit-identical.
  - ✅ **Batch 4 — dispatch seam + `qmatmul`** (`8364d06`): `dispatch_qmatmul`
    is now a `KernelBackend` method over `core::first_match` (lifted from
    backend-hip); `HipKernels` → `QMATMUL_GFX906`, `CudaKernels` → new
    `QMATMUL_SM86` (`backend-cuda::impls`, generated from `dispatch/cuda/sm_86.toml`,
    20 MMVQ+dp4a rows). `qmatmul.rs` migrated (`B::dispatch_qmatmul` + the 7
    `*_launch` helpers genericized). Parity bit-identical.
  - ✅ **Batch 5 — slot seam + `attention`** (`ebf424b`): `LaunchArgs::push_slot`
    (default ignores the slot; HIP records the tag for `HipGraphExec`). `attention.rs`
    migrated; `push_scalar_maybe_slot` genericized over `<A: LaunchArgs>`. Full
    attention parity (14 tests) + graph_slot + forward snapshots bit-identical.
  - **P5.3 head start:** the CUDA `QMATMUL_SM86` table + `dispatch_qmatmul` path
    already exist; P5.3 is then the remaining decode-path dispatch rows.

- ✅ **P5.2 — COMPLETE (structural).** `CudaBackend: Backend` exists; `CudaOps`
  / `HipOps` are now both the *one* generic `OpsImpl<'a, B: KernelBackend>`
  (type aliases over `OpsImpl<CudaKernels>` / `OpsImpl<HipKernels>`) — the
  13 op-launch domains moved out of the HIP-gated module into the always-compiled
  `crates/ops/src/launch/`, so a new kernel or arch never re-copies the
  145-method forwarding layer (rule 14). `CudaOpsRegistry = Registry<CudaKernels>`.
  `ForwardEngine<CudaBackend>` is the existing generic engine — no forward / model
  / server changes. Commit `78765d30`. Guardrail green on gfx906: full ops suite
  (incl. the 14-test attention parity) + forward `parity_snapshot_*` /
  `snapshot_restore` / `synth_*` all bit-identical; ops / backend / bench / cli
  build clean on both `hip` and `cuda`.
  - **Runtime smoke still gated on catalogue completeness** (see P5.3): `Registry::new`
    *eagerly* loads every `KERNEL_STEMS` entry and fails fast on the first stem
    with no compiled `.cubin`, so a CUDA registry cannot be constructed until the
    decode-path stems all compile (or `KERNEL_STEMS` is trimmed / made lazy).

### Next: P5.3 — close the CUDA `.cubin` catalogue gap (measured)

The CUDA toolchain + RTX 3090 are reachable (`/usr/local/cuda-13.3/bin/nvcc`,
driver 610.43.02). A full `flambeau-kernels-cuda` build compiles **203 / 254**
`KERNEL_STEMS`. The **53** unbuilt stems split:
- **18 IQ-quant** (`mmvq_iq*`, `indexed_moe_mmvq_iq*`, `mmq_iq*`) — **out of
  CLAUDE.md quant scope** (no IQ quants). HIP-inherited cruft in `KERNEL_STEMS`;
  the cleanup is to **prune them from `KERNEL_STEMS`** (helps HIP too — it stops
  loading dead modules), not to port them.
- **35 non-IQ** — static `.cu` kernels present in `kernels-hip/src/kernels/` but
  never ported to `kernels-cuda/src/kernels/`. These are **not** manifest-driven
  (the `instantiations.toml` variants, e.g. `mmvq_q4_k_batched`, already build on
  both); they are the hand-written base kernels. The decode-critical subset is the
  **K-quant + Q8_0 MMVQ family** — `mmvq_q4_k`, `mmvq_q4_k_r2`, `mmvq_q5_k`,
  `mmvq_q6_k`, `mmvq_q2_k`, `mmvq_q3_k` (+ `_r2`/`_r4`/`_dp4a` variants),
  `indexed_moe_mmvq_q4_k{,_r2,_gate_up_r4_dp4a,_r4_sorted_dp4a}`,
  `indexed_moe_mmq_q4_k`, `mmvq_q8_0{,_gate_up_dp4a,_llamacpp_style}` — exactly
  what **qwen35moe-v2 Q4_K** decode dispatches.

**The runtime-blocking subset is now pinned to exactly the kernels the decode
path dispatches** (traced through `dispatch/cuda/sm_86.toml` + `moe_experts.rs`):

- ✅ **Keystone — `Registry::new` is load-present** (`19d9e9d7`). It loads the
  stems this backend's catalogue contains and skips the rest (drift moves to a
  loud launch-time `expect_module` error). `Registry<CudaKernels>::new` now
  constructs on the RTX 3090 — `cuda_registry_smoke` (skips 52, dense decode
  spine `mmvq_q4_k_r2_dp4a` / `attention_decode_f16` / `rmsnorm_f16` / `swiglu`
  loaded). HIP compiles all 254 → skips nothing → bit-identical.
- **Dense path: already satisfied.** Q4_K attention/router/lm_head dispatch
  `qmatmul_q4_k_mmvq_dp4a` → `mmvq_q4_k_r2_dp4a` (present). Attention / norm /
  rope / swiglu / topk / moe_combine / sampler all present.
- **MoE expert path: needs exactly 2 kernels ported.** qwen35moe-v2 Q4_K
  experts dispatch:
  - gate+up → `indexed_moe_mmvq_q4_k_gate_up_r4_dp4a`
  - down    → `indexed_moe_mmvq_q4_k_r2`

  Both are static `kernels-hip/src/kernels/*.cu` with **inline gfx906/wave64
  bodies** (no shared `.cuh` yet; include `gfx906.cuh`). The rule-5-correct port
  is the path `mmvq_q4_k_r2_dp4a` already took: extract the body into a
  backend-neutral `kernels-shared/include/<stem>.cuh` over the arch header
  (DPP/wave64 reduce + `dp4a` → the sm_80 `warp_reduce_sum` + `__dp4a`), add a
  manifest entry so both backends generate it, then a green
  `certs/cuda/sm_86/<impl>.json` (rule 2). The other ~33 missing non-IQ stems
  (legacy single-row oracles, non-dispatched variants) are **not** on the decode
  path; the 18 IQ stems are out of scope (prune from `KERNEL_STEMS`).

Tracing the dispatch narrowed the decode-blocking set to **one** kernel: the down
launch fn picks `indexed_moe_mmvq_q4_k_r2_dp4a` / `_r4_dp4a` by token count (both
present), so only the fused gate+up — hard-coded to
`indexed_moe_mmvq_q4_k_gate_up_r4_dp4a` — was missing.

- ✅ **Kernel ported** (`<this commit>`): backend-neutral
  `indexed_moe_mmvq_q4_k_gate_up_r4_dp4a_row` added to the shared header (the
  sorted variant minus the `sorted_pair_idx` indirection; reuses the
  CUDA-certified `flambeau_q4k_subblock_dot` + `mmvq_reduce_store_group<16>`) +
  a CUDA static wrapper `.cu`. HIP keeps its hand `.cu` untouched (the new
  template only compiles when instantiated, which HIP cells don't → bit-identical;
  kernels-hip rebuilds clean, synth/ops_smoke green). Registry smoke: skipped
  52→51, the stem loads. **The full qwen35moe-v2 Q4_K decode dispatch set now
  builds + loads on sm_86.**
- ✅ **Kernel certified both backends.** Extended the gate+up MoE sweep with the
  non-sorted r4 variant (`MoeGateUpDtype::Q4KR4` → `Q4_K_gate_up_r4_moe_mmvq`);
  the runner already handled the non-sorted arg layout. Certs green:
  `certs/cuda/sm_86/indexed_moe_mmvq_q4_k_gate_up_r4.json` (max_rel_err ~3.8e-3 vs
  3e-2 tol) + the gfx906 cert (closing a pre-existing HIP gap — the dispatched r4
  kernel had no cert of its own). Rule 2 satisfied.

### P5.4 — end-to-end CUDA decode (core proof ✅)

`flambeau infer` was removed (#221); inference is `serve` (HIP-concrete server)
or a direct `ForwardEngine<B>` harness. The whole stack below the server is
already backend-generic — `model-ops` takes `&impl Ops`, the GGUF loader takes
`&impl Device`, `ForwardEngine::build` is generic — so the engine runs on
`CudaBackend` with only construction + crate features to wire.

- ✅ **First end-to-end CUDA decode** (`3d7374d4`). A synthetic dense block
  (embed → q8_0 MMVQ attention + rope + attention_decode → q8_0 SwiGLU FFN →
  q8_0 lm_head), single token, run through `ForwardEngine<CudaBackend>` on the
  RTX 3090, reproduces the gfx906 snapshot logits to ~1e-6 (argmax exact, head16
  within 5e-3). The HIP reference is the oracle → the whole CUDA forward path is
  validated, not just per-op. `parity_snapshot_dense_cuda` (build `--features
  hip,cuda`). Wiring: `ForwardEngine::new_solo` (generic ctor), `NoopHooks:
  TopologyHooks<B>` genericized, `forward`/`model-ops` `cuda` features, the
  `qmatmul_q8_0_mmvq_dp4a` recipe alias, generic test `DeviceAllocs<D>`. HIP
  forward parity unaffected.
- ✅ **Per-dtype qmatmul recipe resolution** (`d52181e6`). Every in-scope quant
  dtype's committed sm_86 decode dispatch impl_id resolves to a present stem;
  `cuda_registry_smoke` asserts all 11. K-quants/IQ already matched; Q8_0/Q4_1
  aliased; Q4_0/Q5_0/Q5_1 got arms → their base mmvq stems.
- ✅ **MoE-composite proof — Q4_K experts bit-identical** (`3972ca74`). The full
  composite (router → topk → fused indexed-MoE gate+up → down → weighted combine)
  with Q4_K experts through `ForwardEngine` on both backends: CUDA matches HIP
  **exactly** (max_abs=0.0, logit-cosine=1.0000000 — K-quant dp4a is
  integer-exact). Drives the kernel certified this cycle end-to-end.
- ✅ **Real model — gemma-4-12B Q4_K runs correctly on CUDA** (`f0287d7c` loaded
  it; the HIP↔CUDA cross-check found a bug, fixed in `c05ca260`). The 6.7 GB Q4_K
  GGUF loads and a 7-token prefill runs through `ForwardEngine<CudaBackend>` →
  **logit-cosine 0.9969 vs the gfx906 reference, argmax matches exactly,
  deterministic across runs**, initcheck clean. Per-layer cosine is ~1e-6 from 1.0
  (layer 0 = 0.9999975); the residual ~0.997 over 48 layers is legitimate
  wave64↔wave32 F16 reduction drift, not a defect. Test:
  `gemma4-v2/tests/real_gguf_cuda_parity.rs` (cross-check, threshold 0.995).

  **The bug the cross-check caught** (and why it mattered): the synthetic decode
  tests all passed, but the first real model — a *prefill* through gemma's op set
  — was wrong + non-deterministic (cosine 0.68, argmax drift). `compute-sanitizer
  --tool initcheck` pinned it to `attention_prefill_flash_tile_d256_br8_f16`
  reading uninitialized `__global__` memory: the host launch **hard-coded
  `block.x = 64`** (wave64) while the kernel indexes by `threadIdx.x < WARP_SIZE`
  (32 on sm_8x), so lanes 32–63 ran out-of-range and raced the cooperative K/V
  load. Fixed by adding `KernelBackend::WARP_SIZE` (64/32) and launching the
  flash-tile at `block:(B::WARP_SIZE, br)`. The single-token decode synthetic
  tests never hit the prefill kernel, which is why this hid until a real model.

**Status: P5.4 core complete — the CUDA decode path runs real models correctly**
(deterministic, argmax-correct, ~0.997 cross-backend cosine reflecting only F16
wave-width drift). Remaining for the *literal* doc gate: logit-cosine vs llama.cpp
on the same backend + a multi-token decode loop (the cross-backend wave64↔wave32
drift means HIP↔CUDA won't hit 0.9999, but same-backend-vs-llama.cpp can).
The same wave64 launch hard-code in `launch/recurrent.rs` (GDN/qwen3-next state
step) was fixed alongside (`12548bbe`, `block:(B::WARP_SIZE, …)`); the GDN kernel
was already warp-width-neutral, so it's a pure launch fix (HIP unchanged, the
qwen3-next CUDA *runtime* path still needs its own validation when it lands).
attention's CUDA graph-slot rebind stays deferred (P4.4) — not needed here.

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
