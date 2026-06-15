# Portability roadmap — one program for source, backend, and arch portability

**Status:** Draft v0. Unifies three previously-separate efforts into one program
and one sequencing. Supersedes the *phasing* of `doc/CUDA-PORT-PLAN.md` and
`doc/KERNEL-TEMPLATING.md` (which remain the detailed sub-plans for their
mechanics); a future `doc/AUTOTUNE.md` details the harness. Plan-only.

## Thesis: three axes of portability of one artifact

The kernel is the artifact. The three efforts each make it portable along a
different axis:

| effort | axis | one-liner |
| --- | --- | --- |
| **Kernel templating** | **source** portability | one parameterized source → N variants, not N hand-written files |
| **CUDA port** | **backend** portability | HIP + CUDA built from one backend-neutral source |
| **Autotune** | **arch** portability | per-arch dispatch chosen by measurement, not by hand |

They share **one mechanism** — kernels that are backend-neutral (intrinsics
behind the `arch_primitives` seam) *and* parameterized (variants are template
instantiations) — and **one safety net**: the cert grid. Treated as one program
they collapse into a single per-family loop; treated separately they each pay a
tax the others would have removed.

## Why not sequential (the waste argument)

- **Port-then-template**: hand-author ~196 CUDA `.cu` mirroring the HIP scatter,
  then refactor them into templates → most of that authoring is thrown away.
- **Template-all-then-port**: a tree-wide refactor validated on *one* arch (gfx906)
  before any nvcc compile → the "backend-neutral" claim is unproven until late,
  and the first CUDA compile may invalidate the decomposition wholesale.
- **The unit that avoids both**: the **kernel family, as a vertical slice** —
  template it, prove it on HIP (bit-identical to the hand-written oracle), compile
  the *same source* for CUDA, cert it on sm_86, then let autotune pick the dispatch
  on both archs. Neutrality is proven per family the moment nvcc compiles it; no
  kernel is authored twice.

## Phase 0 — Foundations (DONE)

The runtime + backend + primitive scaffolding the program stands on:

- **Decouple (CUDA-PORT-PLAN Track A) ✅** — backend-neutral seams
  (`Device`/`Stream`/`Event`/`Cluster`/`Collectives`/`Ops`), `ForwardEngine<B>`
  generic, models/server backend-clean. Validated gfx906 (parity + pp2tp2).
- **CUDA backend driver core (B0/B1) ✅** — `cu*` FFI, `CudaDevice`/`Stream`/
  `Event`/`Module`/`Kernel`, nvcc→cubin build. `add_f32` runs on the RTX 3090.
- **CUDA arch primitives (B2 foundation) ✅** — `arch_primitives/sm_80.cuh`
  (warp32 reductions, `dp4a`, SFU) + `backend_compat.cuh` (`__shfl_xor` shim,
  `stdint`, fp16). Compiles on sm_86.
- **Shared-header rule-5 neutrality ✅** — `kernels-shared` carries no backend
  intrinsics; both compilers resolve it via `backend_compat`.

What remains is the **kernel program** (below) plus the CUDA **multi-GPU runtime**
(NCCL/cluster/graph) and **end-to-end parity**.

## The two kernel streams

**Stream M — matmul families (the templating+autotune lever).** `mmvq`, `mmq`,
`indexed_moe_mmvq`, `indexed_moe_mmq`: 196 files / ~24.6k LOC, a sparse variant
scatter. These get templated, instantiated per backend, and autotuned. This is
where source+backend+arch portability all land at once.

**Stream F — fixed kernels (straight port).** ~70 single-variant kernels:
attention (11), norms/rope/softmax (9), elementwise (cast/scale/silu/swiglu/gelu/
sigmoid), `moe_combine`/`moe_sort`, gdn (8), sampler, `kv_append*`, `quantize*`,
`dense_gemv`, `l2_norm`, `topk`. Mostly nothing to choose between → port to CUDA
(backend-neutral where clean, seamed where not), one dispatch row each, cert. **No
autotune needed.** (`quantize_q8_1` is the critical early one — every MMVQ/MMQ
activation depends on it. Attention carries a light dtype/KV-layout templating
sub-track, far smaller than the matmul one.)

**Shared infra (built once, in P1, reused by both streams):**
- **Instantiation codegen** — a manifest (`kernels/instantiations.toml`) →
  generated `extern "C" __global__` wrapper `.cu`, driven by a build helper shared
  by `kernels-hip/build.rs` and `kernels-cuda/build.rs`.
- **Autotune harness** — genericize `bench` over the backend (today it hard-names
  `HipDevice`), add the rank-and-emit loop that times certified candidates per
  shape bucket and writes `dispatch/<backend>/<arch>.toml`.

## The per-family vertical slice (the unit of work)

For each family (or family×dtype slice), in this order — HIP first because it has
the strongest oracle:

1. **Templatize** — decompose into `quant_traits<Q>` + skeleton + `Epilogue`/
   `Loader` policies (per `KERNEL-TEMPLATING.md`), in `kernels-shared`.
2. **HIP instantiate + bit-match** — generate the HIP wrappers; assert each certs
   **bit-identical** to the hand-written `.cu` on gfx906; delete the hand `.cu`.
   (Strongest gate — a proven oracle exists.)
3. **CUDA instantiate + cert** — compile the *same* templated source for sm_86;
   cert vs the CPU dequant reference (logit-cosine ≥ 0.9999) on the 3090.
4. **Autotune both** — run the harness on gfx906 (must reproduce the hand-tuned
   rows — a check on the refactor) and on sm_86 (emit the real `cuda/sm_86.toml`
   rows). Commit both tables + certs.
5. **Done when** both dispatch tables are autotuned-green and the hand `.cu` are
   gone.

## Phases

Each phase: green sweep both rigs, gfx906 regression-clean, dispatch tables +
certs committed.

- **P1 — The spine (pilot: `quantize_q8_1` + `mmvq` Q8_0).** Build *all* reusable
  machinery on the smallest real slice: port `quantize_q8_1` (Stream F, needed
  now); stand up `quant_traits` + the `mmvq` skeleton + `Plain` epilogue + the
  manifest/codegen for both backends; genericize `bench` + build the autotune
  rank-and-emit. Run the full vertical slice on `mmvq` Q8_0 (10 variants).
  **Gate: one family-dtype proven end-to-end across template + both backends +
  autotune.** This de-risks the entire program before scaling.
- **P2 — Stream M rollout.** Repeat the slice per family:
  - **P2a** `mmvq` all dtypes + `ROWS`/`DP4A`/`THREADS` configs.
  - **P2b** `mmvq` fusion (`gate_up`/`batched`/`row_tile` loaders+epilogues).
  - **P2c** `mmq` (two tiling skeletons sharing `quant_traits` + Q8_1 quantize).
  - **P2d** `indexed_moe_mmvq` + `indexed_moe_mmq` (expert gather/bucket `Loader`,
    sorted-scatter `Epilogue`).
  - Outcome: 196 hand `.cu` → ~5 templates + ~21 traits + manifests; both dispatch
    tables autotuned.
- **P3 — Stream F port (parallel to P2).** Port the ~70 fixed kernels to CUDA —
  attention (decode/prefill/splitk; cp.async on CUDA vs prefetch on HIP, seamed),
  norms/rope/softmax, elementwise, moe_combine/sort, gdn, sampler, kv_append,
  dense_gemv, topk, l2_norm. Cert each on both rigs; one dispatch row each.
- **P4 — CUDA multi-GPU runtime (parallel; needs 2× CUDA).** `backend-cuda`
  `Cluster` + NCCL implementing the `Collectives` seam; CUDA Graphs implementing
  `GraphExec`. (CUDA-PORT-PLAN B6.) Independent of the kernel streams once B1 is
  proven; gates only multi-GPU.
- **P5 — End-to-end model parity (CUDA, single-GPU).** (B7.) Smallest full model
  (qwen35moe-v2 Q4_K) through `ForwardEngine<CudaBackend>` on the 3090;
  logit-cosine ≥ 0.9999 vs llama.cpp CUDA over 64 tokens. Requires the Q4_K MoE +
  dense paths from P2, the attention/norm/sampler kernels from P3, and the
  single-GPU runtime. The integration milestone.
- **P6 — Arch breadth + autotune-as-CI.** (B8 + autotune generalization.)
  Recompile the templates under sm_80/sm_89 and the HIP breadth (gfx1031 wave32,
  gfx908/90a/942); run autotune per arch to emit each table — **adding an arch is
  now "run autotune," not hand-tuning.** Wire autotune into CI as the dispatch-table
  generator (or a regen-and-diff ratchet).

## Critical path & parallelism

```
P0 (done) ─► P1 (spine) ─┬─► P2 (matmul templates) ──┐
                         └─► P3 (fixed kernels port) ─┼─► P5 (end-to-end CUDA parity) ─► P6 (arch breadth)
              P4 (NCCL/cluster/graph) ───────────────┘   (single-GPU; P4 only for multi-GPU)
```

- **Critical path to first CUDA decode (P5)**: P1 → enough of P2 (Q4_K MoE + dense)
  + enough of P3 (attention/norms/rope/sampler) → P5. P4 is *not* on the single-GPU
  path.
- **P2 and P3 run in parallel** — disjoint kernel sets, both rigs.
- **P4 runs in parallel** from after P1.

## Rigs (what runs where — no mixed-GPU box needed)

- **gfx906 box** — HIP cert + the **bit-identity oracle** (step 2) + HIP autotune.
- **sm_86 box (this one: RTX 3090, nvcc 13.3, llama.cpp)** — CUDA cert vs CPU ref +
  CUDA autotune + the llama.cpp parity oracle (P5).
- **Dev box: both *compilers* (hipcc + nvcc), no second GPU** — compile-check every
  `kernels-shared` template edit against both backends before it reaches either rig
  (catches the portability breaks already seen: missing `stdint`, no `__exp2f`
  intrinsic, the bare `__shfl_xor` unsupported on sm_70+). This box has nvcc;
  adding compile-only hipcc closes the gap.
- **2× CUDA** — only for P4 (NCCL). **2–4× gfx906** — only for the HIP BAR1 path.
  Validation is per-backend against the shared CPU reference, so the two backends
  are never co-resident; the existing two rigs suffice.

## Decisions to lock before P1

1. **Vertical-slice interleave** (this doc's spine) vs port-all-then-template vs
   template-all-then-port — recommend the slice.
2. **Templated bodies live in `kernels-shared/include`** (both backends instantiate)
   — requires the `arch_primitives` seam to cover every backend call the matmul
   bodies make (reduce, dp4a, prefetch/cp.async).
3. **Build the autotune harness in P1, not later** — it's the only thing that
   produces a trustworthy per-arch dispatch table; hand-curating sm_86 rows would
   re-introduce the very tax this program removes.
4. **Pilot = `quantize_q8_1` + `mmvq` Q8_0** — smallest slice that exercises the
   whole loop (a fixed kernel + a real variant family + both backends + autotune).
5. **HIP-first within each slice** — the hand-written kernel is the bit-identity
   oracle; CUDA leans on the (looser) CPU-ref cert, so prove HIP first.

## Risks & mitigations

- **Templated codegen ≠ hand-tuned register pressure** (VGPR/occupancy shift).
  Mitigation: per-instantiation cert+PMC snapshot; `#[cfg(unverified)]` escape
  hatch for any cell the template provably loses, with a one-line diagnosis (rule 10).
- **Shared-header backend-portability breaks.** Mitigation: both compilers on the
  dev box (decision-rig above); the break is a compile error caught pre-rig.
- **Autotune measurement noise.** Mitigation: locked clocks, warmup, median-of-N,
  and switch-incumbent-only-if-beats-by-margin hysteresis; keep buckets coarse
  (anti-overfit, per the dispatch design).
- **Over-abstraction.** Stop at ~5 skeletons + policies; do not unify the MMQ
  tiling strategies. Validate the decomposition on `mmvq` (most-varied) before
  rolling (rule 14).
- **Scope drift / multi-session reality.** P1 is the de-risking gate; if context
  fills, stop at a green family slice and hand off — never a half-migrated family.

## Sub-plan references

- `doc/CUDA-PORT-PLAN.md` — backend/decouple/runtime detail (Track A done; B-phases
  fold into P1–P6 here).
- `doc/KERNEL-TEMPLATING.md` — the template decomposition + codegen mechanics
  (K-phases fold into P1–P2 here).
- `doc/AUTOTUNE.md` — the harness: candidate registry, rank-and-emit, measurement
  protocol, CI integration, and the crowdsourced tuning-DB upload design.
