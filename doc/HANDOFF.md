# Handoff — resume on the combined MI50 + 3090 rig

Transient working note. Start here when the dev rig (2× MI50 gfx906 + 1× RTX 3090
sm_86) is built and `feature/cuda-decouple` is pulled in.

> **P1 IS CLOSED (2026-06-16, tip `5e3ed3f`).** mmvq Q8_0 is proven end-to-end:
> templated kernel → both backends → bit-match (gfx906) + CPU-ref cert (sm_86) →
> generic bench → **autotune rank-and-emit** (`dispatch/cuda/sm_86.toml` emitted,
> cert-check green both arches). **Next = P2** (mmvq all dtypes → fusion → mmq →
> MoE), now mechanical: add manifest cells + run the autotune loop per family.
> This doc can be deleted or repurposed as the P2 handoff.

## Where we are

- **Program:** `doc/PORTABILITY-ROADMAP.md` (one program: templating + CUDA port +
  autotune as per-family vertical slices). Sub-plans: `CUDA-PORT-PLAN.md`,
  `KERNEL-TEMPLATING.md`, `AUTOTUNE.md`.
### Done since the rig came up (2026-06-16)

- **NVIDIA driver is up.** Was nouveau-bound + a kernel-module mismatch; the user
  set it up on the host (the Claude sandbox can't load modules — see memory
  `reference-dev-rig-nvidia-sandbox`). Now driver 610.43.02 / CUDA 13.3, nvcc 13.3
  at `/usr/local/cuda`. Both backends green on one box: `cargo test -p
  flambeau-backend-cuda` (3090) + gfx906 qmatmul sweep (2× MI50).
- **P1 remaining items 1–3 DONE** (commit `e900ba3`): neutral primitive aliases in
  `gfx906.cuh`; `mmvq_q8_0_dp4a.cu` is now the `mmvq_dp4a_row<Q8_0,256>`
  instantiation, proven **bit-identical** to the retired hand-written kernel on the
  gfx906 grid; scalar parity + qmatmul sweep green.
- **`quantize_q8_1` CUDA port DONE** (commit `36b7a18`): extracted to a shared
  `kernels-shared/include/quantize_q8_1.cuh` device fn; HIP kernel + new CUDA
  kernel are thin wrappers (same `flambeau_quantize_row_q8_1` entry). CUDA
  bit-exact vs CPU ref on the 3090. The CUDA activation path is real.

- **Instantiation manifest + codegen DONE** (commit `49ee597`): a manifest cell
  generates the wrapper `.cu` for both backends; the `mmvq_q8_0_dp4a` wrappers are
  now generated, not hand-written.

- **Bench genericization DONE** (commits `968f6a3` + `76e17bc`): `LaunchCfg` hoisted
  to `core`; new bench-local `BenchBackend` trait (`crates/bench/src/backend.rs`,
  impls `Hip`/`Cuda`) abstracts module-load + launch + static attrs; `harness.rs` +
  `sweep_mmvq.rs` are generic over it; bench `cuda` feature + cli `cuda_sweep` feature
  + `sweep --arch sm_86` routing (`sweep_cuda`). New `Dtype::Q8_0Dp4a` (arch-neutral
  impl_id `qmatmul_q8_0_mmvq_dp4a`). **First-ever CUDA cert emitted** —
  `sweep --arch sm_86 --op qmatmul --dtype Q8_0_dp4a` pass=true/15 on the 3090;
  gfx906 dp4a sweep also green; gfx906 refactor check holds (single-row cert results
  byte-identical, only PMC/timestamp churn). Build the combined binary with
  `--features "hip_sweep cuda_sweep"`. (BenchBackend → `core::Device` promotion still
  deferred to P5.)

**Next P1 item = the autotune rank-and-emit loop** (the last P1 gate): on top of the
generic harness, time the certified mmvq Q8_0 candidate variants per shape bucket and
write `dispatch/<backend>/<arch>.toml` rows. Run on gfx906 (must REPRODUCE the
hand-tuned `dispatch/hip/gfx906.toml` rows — a real correctness check) and sm_86 (emit
`dispatch/cuda/sm_86.toml`, currently an empty placeholder). CUDA has one mmvq
candidate today (dp4a → trivial rank); HIP has ~10. See `AUTOTUNE.md`. After that P1
closes and P2 (mmvq all dtypes → fusion → mmq → MoE) is mechanical via manifest cells.

- **Branch `feature/cuda-decouple`** (pushed, tip ~`76e17bc`) has it all:
  - Decoupling A0–A3 — `ForwardEngine<B>`, the backend seams, models/server
    backend-clean. **Validated on gfx906** before this work.
  - CUDA backend B0/B1 + B2-foundation — `cu*` driver core, `add_f32`,
    `arch_primitives/sm_80.cuh`. **Validated on the 3090.**
  - **P1 pilot — templated `mmvq_q8_0`** (`quant_traits.cuh` + `mmvq.cuh` +
    `kernels-cuda/.../mmvq_q8_0_dp4a.cu`). **Validated on the 3090** (vs CPU ref,
    max rel err 1.7e-7).
  - Comment cleanup (426→0) + the `scripts/lint/comment_lint.py` linter.
- **`main`** untouched (decoupling not landed there yet, by choice).

## First: verify the rig

```
nvidia-smi            # 3090 present, CUDA 13.x driver
rocm-smi              # 2× MI50 present
nvcc --version        # 13.x
/opt/rocm*/bin/hipcc --version   # ROCm 7.1.1 pin
```
- IOMMU/BAR grub tweaks (`scripts/setup/grub_*`, tuned for MI50 BAR1 P2P) must
  tolerate the 3090; scope toolkits with `ROCR_VISIBLE_DEVICES` (ROCm) /
  `CUDA_VISIBLE_DEVICES` (CUDA) if anything cross-talks.
- Build gate: `export RUSTFLAGS="-C link-arg=-fuse-ld=gold"` (or install mold and
  drop the override). HIP-feature Rust needs hipcc; CUDA needs nvcc.
- **Re-add the comment-lint hook** — `.claude/settings.json` is gitignored, so it
  won't pull in. Recreate it (a `PostToolUse` `Write|Edit|MultiEdit` hook running
  `python3 $CLAUDE_PROJECT_DIR/scripts/lint/comment_lint.py`; see
  `scripts/lint/README.md`).

## Sanity checks (both backends, one box now)

```
# CUDA half still green on the 3090:
cargo test -p flambeau-backend-cuda            # add_f32_smoke + mmvq_q8_0_cuda

# HIP daily driver still green (FIRST hip build is a FULL recompile — the
# comment cleanup edited .cu comments and build.rs now hashes headers; behaviour
# unchanged, certs should stay green):
cargo run -p bench -- sweep --arch gfx906      # confirm gfx906 certs green
```

## P1 remaining — the HIP half that needed this rig  — ✅ DONE (commit e900ba3)

The three items below are complete; kept for context. The HIP twin is the
canonical `mmvq_q8_0_dp4a`, bit-identical to the old hand-written kernel.

1. **Neutral primitive aliases in `kernels-hip/src/arch_primitives/gfx906.cuh`** so
   the shared `quant_traits.cuh` / `mmvq.cuh` compile under hipcc (CUDA's
   `sm_80.cuh` already exposes these neutral names; HIP exposes `gfx906_*`). Add:
   ```c
   static __device__ __forceinline__ float warp_reduce_sum(float x)        { return gfx906_warp_reduce_sum(x); }
   static __device__ __forceinline__ float half_warp_reduce_sum(float x)   { return gfx906_half_warp_reduce_sum(x); }
   static __device__ __forceinline__ float quarter_warp_reduce_sum(float x){ return gfx906_quarter_warp_reduce_sum(x); }
   static __device__ __forceinline__ float eighth_warp_reduce_sum(float x) { return gfx906_eighth_warp_reduce_sum(x); }
   static __device__ __forceinline__ int   dp4a(int a, int b, int c)       { return gfx906_dp4a(a, b, c); }
   ```
   (`__shfl_xor` is native on hipcc — no shim needed on HIP.) `mmvq_q8_0` only uses
   `warp_reduce_sum` + `dp4a`; the rest are for the next families.
2. **HIP instantiation** — a `kernels-hip/src/kernels/mmvq_q8_0_dp4a.cu` that calls
   `mmvq_dp4a_row<Q8_0, 256>` (include order: `block_quant.cuh`, `gfx906.cuh`,
   `quant_traits.cuh`, `mmvq.cuh`). It must keep the entry-point name
   `flambeau_mmvq_q8_0_dp4a_q8_1`.
3. **Bit-match** — cert the templated instantiation produces output **bit-identical**
   to the hand-written `mmvq_q8_0_dp4a` on the gfx906 sweep grid (the migration
   safety net). `load_int_b2` reconstructs the same int32 bits the direct read
   gave, and `dp4a` is exact, so it *should* be bit-identical (only fp accumulation
   order could differ — verify it doesn't on this shape). If identical → delete the
   hand-written `.cu`. If not → diagnose before deleting.

## Then continue P1 (now trivial to do both-backends on one box)

- ~~Port `quantize_q8_1` (CUDA) so the activation path is real.~~ ✅ DONE (commit
  `36b7a18`) — now a shared `kernels-shared/include/quantize_q8_1.cuh` device fn
  with thin HIP + CUDA wrappers; CUDA bit-exact vs CPU ref on the 3090.
- ~~Instantiation manifest + codegen~~ ✅ DONE (commit `49ee597`):
  `crates/kernels-shared/instantiations.toml` declares cells (stem/entry/includes/
  params/body); both `build.rs` generate the wrapper `.cu` into `OUT_DIR/generated/`
  (arch header injected: `gfx906.cuh` HIP / `sm_80.cuh` CUDA) and compile them. The
  hand-written `mmvq_q8_0_dp4a` wrappers are gone. **P2 = add cells to the manifest,
  not pairs of files.** The codegen helper lives in `flambeau-kernels-shared`
  (`codegen` module, next to the manifest); both `build.rs` use it as a
  build-dependency — one source of truth, a third backend reuses it for free.
- **Backend-generic `bench` + `bench autotune`** (rank-and-emit): genericize bench
  over `core::Device` (it still hard-names `HipDevice`), then the rank/emit loop.
  Run on gfx906 (must reproduce the hand-tuned rows) and sm_86 (emit
  `dispatch/cuda/sm_86.toml`). See `AUTOTUNE.md`.

After that the loop is proven end-to-end; P2 (mmvq all dtypes → fusion → mmq → MoE)
is mechanical.

## Gotchas (don't relearn — see also memory `cuda-port-decoupling`)

- **Misaligned load**: Q8_0's offset-2 `qs` faults a 4-byte read on NVIDIA
  (`CUDA_ERROR_MISALIGNED_ADDRESS`); `load_int_b2` (2× `uint16`) is the fix and is
  bit-identical on AMD. Recurs for any odd-offset quant.
- **build.rs header cache**: both `kernels-{hip,cuda}/build.rs` now fold header
  contents into the per-kernel cache key, so header edits recompile (they didn't
  before).
- **Naming**: CUDA code uses neutral primitive names (`warp_reduce_sum`, `dp4a`) —
  no `gfx906_*` in CUDA. Comments stay minimal (the lint hook enforces it).
- **Entry-point name is the dispatch contract**: identical on both backends.
- **Oracles**: CUDA correctness vs the CPU dequant reference (`flambeau-quant`) /
  the `mmvq_q8_0_cuda` pattern; HIP via bit-match + the gfx906 certs; end-to-end
  (P5) vs llama.cpp at `/home/artefact/Workspace/llama.cpp`.

## Rig fallback

2× MI50 validates 2-GPU TP/PP (BAR1 pairs). For 4-GPU `forward` scaling, pop the
two MI50s back in temporarily (reversible) — see `PORTABILITY-ROADMAP.md` §"Dev rig".
