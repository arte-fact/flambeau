# Kernel templating plan — collapse the variant scatter to parameterized families

**Status:** Draft v0. Plan-only. Targets `ARCHITECTURE.md` principle 6 (*"Single
source of truth per kernel family … parameterized by traits/templates — not
hand-duplicated"*), which the current kernel tree has drifted from. Tightly
coupled to `doc/CUDA-PORT-PLAN.md` (same lever) and a future `doc/AUTOTUNE.md`
(the consumer of a dense candidate matrix).

## The problem

`crates/kernels-hip/src/kernels/` holds **268 hand-written `.cu` files**; the
four matmul families account for **196 files / ~24.6k LOC (77%)**:

| family | files | `.cu` LOC |
| --- | --- | --- |
| `mmvq_*` (decode GEMV) | 81 | 9,165 |
| `mmq_*` (prefill GEMM) | 30 | 4,680 |
| `indexed_moe_mmvq_*` | 44 | 4,506 |
| `indexed_moe_mmq_*` | 41 | 6,272 |

These are not duplicates — they are the **cross-product of orthogonal axes**
(dtype × reduction-layout × dp4a × thread-tiling × fusion × batching), but only
a **sparse, demand-driven ~20% of cells** were ever hand-written (Q4_K MoE has 11
variants; Q5_0 has 1). Counting the Rust that exists to launch/sweep/describe
them (`ops/src/hip/{qmatmul,moe}.rs`, the `KernelDescriptor` tables, the
`bench/src/sweep_*` files), the variant machinery is **~20k LOC, ~13% of the
whole tree**.

Three costs follow:
1. **Per-arch tuning is hand-work.** gfx906's sparse fill pattern is wrong for
   sm_86 (warp32) / gfx1031 (wave32) / gfx908+ (matrix cores) — each arch wants
   a *different* winning variant per cell, and there's no mechanism to fill the
   matrix without writing more `.cu`.
2. **The CUDA port would double the duplication.** Mirroring 196 hand-written
   `.cu` into `kernels-cuda/` (as `CUDA-PORT-PLAN.md` currently assumes) is
   ~24k more LOC authored twice.
3. **The autotuner has nothing to choose from** in the thin cells — a dense,
   per-arch candidate pool requires the cells to be *generated*, not stamped.

## The goal

Replace the 196 hand-written matmul `.cu` with **~4–5 templated kernel families +
~21 dtype traits specializations + a generated instantiation list**. Each
physical variant becomes a `(dtype, config, epilogue)` instantiation, not a
file. The algorithmic core is **backend-neutral** (lives in `kernels-shared`,
uses the `arch_primitives` seam), so HIP and CUDA **share one kernel source** —
the CUDA port becomes "instantiate the templates with the sm_8x primitives,"
not "author 196 kernels again."

## The decomposition

The grid axes map to composable policies:

| grid axis | becomes | where it lives |
| --- | --- | --- |
| dtype (`q4_0…iq4_xs`) | `quant_traits<Q>` (block layout + `vec_dot`) | `kernels-shared` (block structs already in `block_quant.cuh` / `iq_grid.cuh`) |
| `r2/r4/r8` rows-per-block | `int ROWS` | template param |
| `dp4a` / `vdr2` | `bool DP4A`, `int VDR` | template param + traits sub-path |
| `t128` / `warpcoop` | `int THREADS` | template param |
| `gate_up` / `down` / `sorted` | `class Loader` + `class Epilogue` | policy structs |
| `batched` (n_slots) | slot loop / `int N_SLOTS` | template param |
| wave64↔warp32 reduce | `group_reduce_sum<N_LANES>` | **already** seamed in `arch_primitives` |

### Traits — the only per-dtype code

```c
// kernels-shared/include/quant_traits.cuh  (backend-neutral)
template<class Q> struct quant_traits {
    using block_t = block_q4_K;            // from block_quant.cuh
    static constexpr int QK  = 256;        // block size
    static constexpr int QI  = QK / 4;     // int32s per block
    // The inner product against a Q8_1 activation block. Two code paths
    // (int8 dp4a / float FMA) selected by DP4A; `dp4a()` is the arch seam.
    template<bool DP4A>
    __device__ static float vec_dot(const block_t&, const block_q8_1&, int j);
};
```

~21 specializations (one per dtype) replace the per-dtype unpack logic currently
copy-pasted across ~196 files. The IQ block's perfect uniformity in the variant
grid is evidence this unpack logic is already templatable-by-hand.

### Skeleton — one kernel, every other axis a parameter

```c
// kernels-shared/include/mmvq.cuh  (backend-neutral; zero backend intrinsics)
template<class Q, int ROWS, int THREADS, bool DP4A, class Epilogue>
__device__ void mmvq_impl(const void* w, const void* act, float* y,
                          int k, int n, ExtraArgs ex) {
    // 1. cooperatively stage the Q8_1 activation strip into LDS
    // 2. for r in 0..ROWS:  acc[r] += quant_traits<Q>::vec_dot<DP4A>(...)
    // 3. acc = group_reduce_sum<THREADS / ROWS>(acc);   // arch_primitives seam
    // 4. Epilogue::store(acc, y, ex);   // Plain | SiluGateUp | SortedScatter
}
```

### Epilogue / Loader policies (the structural axes)

```c
struct Plain        { __device__ static void store(...); };        // y[row] = acc
struct SiluGateUp   { /* loads gate+up rows, stores silu(gate)*up */ };
struct SortedScatter{ /* MoE expert-sorted output scatter */ };
struct KvF16Dst     { /* cast + store into the F16 KV tensor */ };
```

`gate_up` is a `Loader`+`Epilogue` pair (it reads two weight tensors), not a
flag — but it still composes; the inner `vec_dot` loop is shared.

### Entry points stay name-addressable

Rust launches kernels by string name (`module.kernel("flambeau_…")`), and the
cert/dispatch model keys on `impl_id`. So a **generated wrapper file** emits one
thin `extern "C" __global__` per instantiation, named by `impl_id`:

```c
// GENERATED by build.rs from the instantiation manifest — do not edit.
extern "C" __global__ void flambeau_qmatmul_q4_K_mmvq_r2_dp4a(...) {
    mmvq_impl<q4_K, /*ROWS=*/2, /*THREADS=*/64, /*DP4A=*/true, Plain>(...);
}
```

So the 196 hand `.cu` → **~5 template headers + ~21 traits + a handful of
generated wrapper `.cu`**. Binary size is unchanged (same kernel count); source
LOC drops ~10–20×.

## The instantiation manifest

A reviewed artifact (TOML, alongside `dispatch/`) lists the cells to build per
backend/arch:

```toml
# kernels/instantiations.toml
[[mmvq]]
dtype = "q4_K"; rows = 2; threads = 64; dp4a = true; epilogue = "Plain"
[[mmvq]]
dtype = "q4_K"; rows = 4; threads = 64; dp4a = true; epilogue = "SiluGateUp"
```

- `build.rs` reads the manifest → generates the wrapper `.cu` → compiles them
  (same per-kernel `.hsaco`/`.cubin` flow as today).
- **Adding a variant = one manifest line.** Filling the matrix for a new arch =
  regenerate the manifest (eventually by the autotuner).
- Compile-time / binary-size are bounded by the manifest length — you instantiate
  exactly the cells dispatch/autotune needs on that arch, no template explosion.

## What templatizes, and what stays separate (honest edges)

- **Fully collapses**: the entire `mmvq` family and the dtype-unpack inside `mmq`
  — ~120 of the 196 files become instantiations of one skeleton.
- **Two–three skeletons, not one, for MMQ**: `wave64` single-warp vs
  `4warp_lds`/`turbo` 4-warp stream-K LDS tiling are *different algorithms*.
  Templating shares the dtype `vec_dot` + the activation Q8_1 quantize, but keeps
  separate tiling skeletons. Net: 196 → **~4–5** templated kernels, not 1.
- **Arch-specific perf primitives stay seamed, not templated**: gfx906 DPP-fused
  reduce vs CUDA `__shfl_xor_sync`; gfx906 `global_load_dword` prefetch vs sm_80
  `cp.async`. These live in `arch_primitives/{gfx906,sm_80}.cuh` behind common
  names (already the design) — the templated body calls them, never `#ifdef`s.
- **Attention is out of scope for v1** of this plan — it has its own
  flash-tile/online-softmax templating story; tackle after the matmul families
  prove the pattern.

## Cert + dispatch integration (no model change)

- Each instantiation keeps a stable `impl_id` and its **own cert + PMC snapshot**
  — principle 2 ("no kernel ships without a cert") holds at instantiation
  granularity. The cert grid is unchanged in shape.
- The `dispatch/<backend>/<arch>.toml` format is unchanged; rows point at
  instantiation `impl_id`s exactly as today.
- **Migration safety net**: for each hand-written kernel, build the equivalent
  instantiation and assert it is **bit-for-bit identical** to the hand-written
  output across the existing sweep grid *before* deleting the `.cu`. The current
  cert harness is precisely this regression net.

## Relationship to the CUDA port (the multiplier)

Because the templated body is backend-neutral (rule 5 — it carries no backend
intrinsics, only `arch_primitives` calls), **the same source compiles on HIP and
CUDA**. This rewrites `CUDA-PORT-PLAN.md`'s economics:

- Its B2–B5 currently assume "author each kernel family twice." With templating,
  the CUDA port of a family = add the `sm_80` `arch_primitives` (done in B2) +
  list the instantiations for sm_86 in the manifest. **No second hand-port.**
- The "dominated on gfx906" variants are no longer dead weight to carry or skip —
  they are *manifest lines*, regenerated per arch by sweeping.

Recommended order: do the templating refactor **interleaved with** the CUDA port,
not before it — the port is what forces the backend-neutral decomposition, and
templating means the port writes each kernel once.

## Phasing

Each phase: green sweep, every migrated instantiation bit-matches its
hand-written predecessor on the cert grid, gfx906 dispatch unchanged.

- **K0 — Decomposition + codegen spine.** `quant_traits.cuh` skeleton, the
  `mmvq.cuh` template, `Plain` epilogue, `instantiations.toml`, and the `build.rs`
  manifest→wrapper codegen. No kernel deleted yet. Gate: a generated
  `flambeau_qmatmul_q8_0_mmvq_dp4a` instantiation certs bit-identical to the
  hand-written `mmvq_q8_0_dp4a.cu`.
- **K1 — One dtype end-to-end.** Migrate all `mmvq` Q8_0 variants (10) to
  instantiations; cert each bit-matches; delete the 10 hand `.cu`. Proves the
  pattern across ROWS / DP4A / THREADS / `t128`.
- **K2 — `mmvq` across all dtypes.** Roll the traits + manifest to every dtype +
  the reduction/thread configs. Delete the migrated `.cu`. (~50–60 files → ~21
  traits + manifest.)
- **K3 — Fusion axis.** `SiluGateUp` / batched / row-tile loaders+epilogues for
  `mmvq`; migrate the `gate_up_*` / `*_batched` variants.
- **K4 — `mmq` family.** Two tiling skeletons (`wave64`, `turbo`/`4warp`) sharing
  `quant_traits` + the Q8_1 quantize; migrate.
- **K5 — MoE families.** `indexed_moe_mmvq` / `indexed_moe_mmq` — expert
  gather/bucketing as a `Loader`, sorted output as a `SortedScatter` epilogue.
- **K6 — Regenerate dispatch + wire autotune.** Regenerate `gfx906.toml` from the
  instantiations (must reproduce the hand-tuned table — a correctness check on
  the refactor) and hand the dense candidate matrix to `bench autotune`.

## Risks & mitigations

- **Compile-time / binary blow-up.** Bounded by the manifest — instantiate only
  dispatched/autotuned cells per arch. Same kernel count as today, so binary size
  is flat; per-TU template compile is slower but parallelized by `build.rs`.
- **A templated kernel doesn't match the hand-tuned one's codegen** (register
  allocation, VGPR pressure shifting occupancy). Mitigation: the cert+PMC snapshot
  per instantiation catches regressions; keep a hand-written `#[cfg(unverified)]`
  escape hatch for any cell where the template provably loses (rare; document the
  diagnosis per rule 10).
- **Over-abstraction.** Stop at ~5 skeletons + policies; do **not** try to unify
  the MMQ tiling strategies into one kernel — they're different algorithms.
  Validate the abstraction on `mmvq` (the most-varied family) before rolling.
- **Cert-grid drift.** The migration's whole safety contract is bit-identity vs
  the hand-written kernel; any cell that can't be made bit-identical is a finding
  to investigate, not a tolerance to relax.

## Non-goals

- Attention / GDN / elementwise templating (separate, later).
- Tensor-core / wgmma MMA paths (those are arch-specific kernels, not template
  cells).
- A runtime JIT / Triton path — instantiation is build-time codegen, hand-written
  CUDA/HIP C++, preserving the per-impl cert + PMC narrative.

## Decisions to lock before K0

1. Where the templated headers live — recommend `kernels-shared/include/`
   (backend-neutral, both backends instantiate). Confirm the `arch_primitives`
   seam covers every backend-specific call the matmul bodies make (reduce, dp4a,
   prefetch) so the bodies stay `#ifdef`-free.
2. Manifest format + home — recommend `kernels/instantiations.toml`, one section
   per family, generated wrappers under `OUT_DIR`.
3. Sequencing vs the CUDA port — recommend interleave (templating lands per
   family as that family is CUDA-ported), so each kernel is authored once.
