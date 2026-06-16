# Handoff — resume on the combined MI50 + 3090 rig

Transient working note. Start here when the dev rig (2× MI50 gfx906 + 1× RTX 3090
sm_86) is built and `feature/cuda-decouple` is pulled in.

> **CURRENT STATUS (2026-06-16): P2a mmvq DONE; P2b (mmvq fusion) STARTED.**
> All 20 GGUF weight quants are templated + certed on both backends
> (`cuda/sm_86` = 20 mmvq rows; cert-check cuda 20 / hip 82). **P2a is done**:
> all dtypes + the Q8_0 ROWS/DP4A/THREADS configs (`t128` `ad68761`, `vdr2`
> `29258e1`, `r2`/`r4` `685bc7c`). Roadmap note: ROWS/DP4A/THREADS configs ARE
> P2a (not P2b) — `r2`/`r4` were NULL for Q8_0 on both backends (dp4a/t128_vdr2
> win; Q8_0 is dense/cheap-decode so 256-thread single-row MLP beats the
> 64-thread wave64-DPP layout), kept as certified bench-reference candidates; the
> generic `mmvq_dp4a_nrow_row` + `mmvq_reduce_store_group<GROUP>` are the reusable
> infra for the heavy-decode K-quant r2/r4 (Q4KR2/Q6KR4 DO win as hand kernels).
>
> **P2b = mmvq FUSION (`gate_up`/`batched`/`row_tile`).** **All dense `gate_up`
> DONE** — Q8_0 (`dcac582`) + Q4_0 (`eb4e1c7`) + Q4_1 (`fa6e752`) + Q5_K — each
> its own neutral body reusing the shared two-output reduce
> `mmvq_reduce_store_gate_up` (Q8_0 via the generic VDR skeleton; Q4_0/Q4_1
> `mmvq_q4_{0,1}_gate_up.cuh` 4-lane nibble, `(q-8)` vs `+m·s_y`; Q5_K
> `mmvq_q5_k_gate_up.cuh` super-block 5-bit, 256t/32-per-sb, scales+dmin). All
> certed cross-backend (`--dtype <dt>_gate_up`; max_rel_err 2e-3 / 2.6e-4 /
> 1.8e-4 / 1.3e-2, HIP↔CUDA equal), same stem/entry → ops/runtime untouched, hand
> `.cu` retired, bench-reference registered (`GateUpDtype` + `run_gate_up_sweep`
> in `sweep_mmvq.rs`).
>
> **`batched` (N>1 columns) STARTED — Q8_0 done.** Generic
> `mmvq_dp4a_batched_row<Q,THREADS,N,OutT>` (one weight row × N slot-major Q8_1
> columns → slot-major out; the weight load is slot-invariant so LICM hoists it
> out of the unrolled N-loop = read-once amortization, the mirror of gate_up) +
> N-way reduce `mmvq_reduce_store_batched<THREADS,N>`. n2/n3/n4 entries, same
> stem/entry (`mmvq_q8_0_batched`) → ops untouched, hand `.cu` retired. New
> harness path `BatchedDtype`/`run_batched_sweep` (sweeps m×k×n_slots = 45
> shapes; `--dtype Q8_0_batched`); cert both backends err 2.3e-3, HIP↔CUDA equal,
> bench-reference registered. The CUDA port also fixes the hand kernel's
> NVIDIA-unsafe `((int*)qs)[i]` (now `load_int_b2` via `partial`). **NEXT
> batched:** Q4_0/Q4_1 (reuse the skeleton via their `quant_traits::partial`),
> then K-quant batched (Q4_K/Q5_K-r2/Q6_K — own super-block bodies). Then
> `row_tile`. Then **P2c mmq** (4-warp LDS-tiled prefill lever) + **P2d**
> indexed_moe.
>
> *(first gate_up slice writeup)*
> **Q8_0 `gate_up`** (dense FFN decode kernel) templated onto a neutral skeleton
> `mmvq_gate_up_dp4a_vdr_row<Q,THREADS,VDR,OutT>` (`mmvq_gate_up.cuh`) + new
> two-output reduce `mmvq_reduce_store_gate_up<THREADS>` + a shared `load_q8_1_
> ints<VDR>` / `quant_traits<Q8_0>::partial_vdr_y` split so the fused caller
> reads the Q8_1 activation ONCE and reuses it across gate+up (the bandwidth win)
> — the single-row `partial_vdr` routes through it and stays bit-identical. Same
> stem/entry (`mmvq_q8_0_gate_up_t128_vdr2`) → ops launcher + runtime untouched;
> hand `.cu` retired. **First formal cert for this kernel** (it had no test/cert
> before) on BOTH backends via a new sweep gate_up path (`run_gate_up_sweep` /
> `GateUpDtype` in `sweep_mmvq.rs`, `--dtype Q8_0_gate_up`); max_rel_err ~2e-3,
> HIP↔CUDA numerically equal. Registered as a bench-reference (launched directly,
> not via a dispatch row). cert-check green (hip 82 / cuda 20).
> **NEXT P2b:** other dense `gate_up` dtypes (Q4_0/Q4_1/Q5_K hand kernels exist),
> then `batched` (N>1 columns) + `row_tile` (separate skeletons — different op
> signature). Then **P2c mmq** (4-warp LDS-tiled prefill lever) + **P2d**
> indexed_moe. Build kernels with `RAYON_NUM_THREADS=8` (thermal). Detailed phase
> log + per-slice recipe + rig gotchas below. (Historical detail follows.)
>
> ---
>
> *(history)* **P1 CLOSED. P2a was UNDERWAY at tip `10d210a`.** Q4_0 was the first
> P2a dtype slice, proven end-to-end exactly like the Q8_0 pilot: templated
> `quant_traits<Q4_0>` (nibble unpack + (q-8) bias correction) on the now
> `OutT`-generic `mmvq_dp4a_row` skeleton → HIP **bit-match** (cert max_rel_err
> byte-identical to the retired hand `mmvq_q4_0.cu`) → CUDA cert (pass/15 on the
> 3090) → `dispatch/cuda/sm_86.toml` Q4_0 row. cert-check green both
> (cuda/sm_86 **2 rows**, hip/gfx906 82). Two enabling refactors landed with it
> (reused by every later dtype): **multi-entry codegen** (a `[[kernel]]` cell now
> carries N `[[kernel.entry]]` blocks → one module holds both the F32 and
> F16-dst entries) and **emit_dispatch merge-by-dtype** (autotune `--emit` now
> preserves other dtypes' rows instead of rewriting the whole table).
>
> **The per-dtype loop is now fully mechanical** — for the next Stream-M dtype:
> 1. add a `quant_traits<Qx>` specialization (`kernels-shared/include/quant_traits.cuh`);
> 2. add a `[[kernel]]` cell (F32 + F16 entries) to `instantiations.toml`;
> 3. move the hand `mmvq_qx*.cu` out of `kernels-hip/src/kernels/`, build HIP,
>    re-cert via its sweep, confirm bit-match (cert max_rel_err unchanged), `git rm`;
> 4. add `Dtype::Qx` to `sweep_mmvq.rs` (name/ggml/impl_id/stem/entry/bytes/threads
>    + `tame_scales` arm) and the cli `sweep_cuda`/`autotune` match arms;
> 5. build CUDA, `sweep --arch sm_86 --dtype Qx`, `autotune --arch sm_86 --emit`,
>    `cert-check` both.
>
> **All four legacy quants done** (Q4_0 `17d2f8d`/`10d210a`; Q4_1/Q5_0/Q5_1
> `3bd4e94`/`ffe53ce`). `dispatch/cuda/sm_86.toml` now has 5 mmvq rows; hip
> dispatch unchanged (HIP bit-matched via the bespoke sweeps). Shared
> `expand_bits4` handles the Q5 5th-bit plane; `load_int_b2` is the CUDA-safe
> reader for every odd-offset field (Q5_0's `qh` at block-offset 2 faults a
> 4-byte read on NVIDIA). Bit-match held for all (Q5_0/Q5_1 cert max_rel_err
> byte-identical despite the templated kernel using fewer VGPRs — 23 vs 29).
>
> **NEXT = K-quants (Q2_K…Q6_K, Q8_K) — a genuine multi-session sub-phase, NOT
> the mechanical legacy loop.** Assessed 2026-06-16 by reading
> `mmvq_q6_k_dp4a.cu` (the cleanest K-quant). Four structural facts the legacy
> slices did not have — read these before starting:
>
> 1. **Wave-size leak (rule 5) — the blocker.** The K-quant dp4a kernels use a
>    64-thread/super-block layout (`lane=threadIdx.x`, `iqs=lane&31`,
>    `super_hi=lane>>5`) and finish with `gfx906_warp_reduce_sum(acc)` +
>    `if(lane==0) store`. That assumes a **64-wide** warp — on CUDA wave32 it
>    sums only the first 32 lanes → silently wrong. Every K-quant CUDA port
>    needs the reduce made warp-size-agnostic: `WARPS = 64/WARP_SIZE`,
>    warp_reduce then an inter-warp LDS rendezvous (exactly the tail of the flat
>    `mmvq_dp4a_row`). On HIP WARPS=1 → degenerates to the current code →
>    bit-match preserved. **Recommended first step: extract that reduce tail
>    into a shared `mmvq_reduce_store<THREADS, OutT>(acc, dst, row)` helper
>    (refactor the flat skeleton to use it too — 2nd site, justified), then each
>    K-quant body just calls it.**
> 2. **No single cross-K-quant template (rule 14).** Each K-quant has a distinct
>    `iqs`/scale-offset decode (Q2_K/Q3_K 2–3-bit + min; Q4_K/Q5_K 6-bit packed
>    scales + dmin; Q6_K int8 scales + the (raw−32)·y = raw·y − 32·Σy bias trick
>    with `dp4a(0x01010101,u,0)`). Do NOT abstract a shared super-block skeleton
>    from one example. Port each as its **own** backend-neutral shared `.cuh`
>    (body moved out of the hand `.cu`, intrinsics → neutral `dp4a`/
>    `warp_reduce_sum`, misaligned `((int*)ql)[i]`/`((int*)qh)[i]` →
>    `load_int_b2`) + a manifest cell (F32+F16 entries). Only after ≥2–3 are
>    ported and the shared shape is visible, consider a skeleton.
> 3. **Misalignment.** Super-block fields sit at non-mul-4 block offsets
>    (block_q6_K is 210 B; `ql`@0, `qh`@128, `d`@208) → 4-byte reads fault on
>    NVIDIA for odd super-blocks. Use `load_int_b2` for every `int`-cast load,
>    same as the legacy quants. fp16 `d` reads are 2-byte and fine.
> 4. **impl_id naming debt — batch it.** The K-quant mmvq dp4a impl_ids are
>    legacy `_gfx906`-suffixed (e.g. `qmatmul_q6_K_mmvq_dp4a_gfx906`, 6 refs:
>    hip dispatch + `backend-hip/src/impls/qmatmul.rs` registry + an `impls.rs`
>    test assert + an `ops` recipe arm + the `sweep_mmvq` enum + cert file). A
>    `cuda/sm_86` row must NOT carry `_gfx906`. **Do a single dedicated rename
>    pass** neutralizing all K-quant mmvq dp4a ids (`_gfx906` → drop, `q6_K` →
>    `q6_k`) across those refs + `git mv` the certs, cert-check HIP green, commit
>    — THEN port, reusing the now-neutral `Dtype` variants for both backends.
>    Don't add parallel neutral `Dtype` variants (the Q4_1Dp4a trick doesn't
>    scale to 6 K-quants).
>
> **Q6_K DONE** (`c10cf1f` HIP + `cb5745c` CUDA) — the K-quant pilot. Validated
> the whole approach: shared `mmvq_reduce_store<THREADS,OutT>` (warp-size
> agnostic — bit-match on gfx906 wave64 WARPS=1, AND CUDA wave32 cert green = the
> inter-warp LDS path sums all 64 threads); neutral `mmvq_q6_k.cuh` body;
> `load_int_b2` for misaligned `ql`/`qh`; impl_id neutralized
> (`qmatmul_q6_k_mmvq_dp4a`). cuda/sm_86 now 6 rows. CUDA max_rel_err matches HIP
> to ~1e-6 (integer-exact dp4a). Also fixed a P1 orphan-cert test failure
> (`every_cert_on_disk_is_registered`, red since `76e17bc`).
>
> **Q2_K / Q3_K / Q5_K / Q6_K ALL DONE** — every single-row dp4a K-quant is
> templated on both backends (`c10cf1f`+`cb5745c` Q6_K, `d93f16d` Q5_K,
> `2c156cb` Q3_K, `4ac1ed7` Q2_K). All reuse `mmvq_reduce_store<256>` (Q6_K uses
> `<64>`); each its own neutral `mmvq_q{2,3,5,6}_k.cuh`; `load_int_b2` for the
> misaligned super-block loads; impl_ids neutralized. Q3_K/Q2_K weren't
> dispatched on HIP / had no prior cert → HIP correctness certs registered as
> bench-references in `direct_call.rs`. **cuda/sm_86 = 9 mmvq rows**; cert-check
> green (cuda 9, hip 82). Also fixed the P1 orphan-cert test + removed a stray
> cert copy.
>
> **ALL K-quants DONE** — Q4_K (`a6cb930`) + Q8_K (`cf4efaa`) landed.
> - **Q4_K** (production decode kernel, `r2`/2-row) bit-matched **byte-identical**:
>   the `__shfl_xor(v, off, 32)` butterfly in *ascending* offset order
>   (1,2,4,8,16) exactly reproduces gfx906's DPP `half_warp_reduce_sum` add
>   order — no ULP shift. New shared `mmvq_reduce_store_2row<OutT>` in
>   `mmvq_reduce.cuh` (width-32 = sub-warp on wave64, full warp on wave32 since
>   the 2 rows are 2 warps; no LDS). CUDA pass validated wave32.
> - **Q8_K** (simplest K-quant) reused `mmvq_reduce_store<256>`, byte-identical.
>
> **+ IQ4 family DONE** — IQ4_NL (`e4526c3`) + IQ4_XS (`334d8bd`). Needed a new
> neutral **`byte_perm`** primitive (codebook LUT permute) in both
> arch_primitives headers: HIP = `__builtin_amdgcn_perm`; CUDA =
> `__byte_perm(b,a,packed)` with the byte-per-output selector repacked to
> nibble-per-output + source operands swapped (valid for selectors 0..7, which
> is all the LUTs use). HIP byte-identical, CUDA cert validated the translation.
> IQ4_XS reuses `iq4_nl_table_4` + the shared neutral `flambeau_iq4_xs_scale`.
>
> **cuda/sm_86 = 13 mmvq rows** (Q2/Q3/Q4/Q5/Q6/Q8_K + Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
> + IQ4_NL/IQ4_XS); cert-check green (cuda 13, hip 82). Every standard GGUF
> weight quant **and** the IQ4 family is now templated on both backends.
>
> **+ ALL IQ DONE — the entire mmvq weight-quant family is templated on both
> backends.** IQ2_XXS (`134e225`/`6949ff1`, first grid-codebook), IQ3_XXS/IQ3_S
> (`a5e2bf1`), IQ2_XS/IQ2_S (`325846a`), IQ1_S/IQ1_M (`699f0c3`). The grid kernels
> all reuse the IQ2_XXS sign helpers (`iq2xxs_apply_signs`/`unpack_ksigns`) +
> `mmvq_reduce_store<256>` + `load_int_b2` + the already-neutral `iq_grid.cuh`
> (proven on nvcc). IQ1 uses a direct grid + `delta·Σu` bias (no sign LUT).
> All HIP bit-matched byte-identical; all CUDA certs pass/15 (IQ1 reproduced
> identical after a rig restart — see the rig note below).
>
> **`cuda/sm_86` = 20 mmvq rows — every GGUF weight quant** (Q4_0/Q4_1/Q5_0/Q5_1/
> Q8_0 · Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K · IQ4_NL/XS · IQ3_XXS/S · IQ2_XXS/XS/S ·
> IQ1_S/M); cert-check green (cuda 20, hip 82). **P2a mmvq is DONE.**
>
> Rig notes (memory `reference-dev-rig-nvidia-sandbox`): (1) `cuInit 999` =
> sandbox missing `/dev/nvidia-uvm` (host modprobe + sandbox restart). (2) The
> box hard-reset under all-core compile load — CPU-thermal (cooler swapped +
> 3090↔MI50 PSU rails rebalanced; CPU now 64–66°C vs 85°C). **Build kernels with
> `RAYON_NUM_THREADS=8`** (build.rs rayon pool runs `hipcc` on all 32 cores
> otherwise; `cargo -j` doesn't reach it). (3) A cuda-feature binary built while
> uvm was absent segfaults at startup forever — force a relink (`touch
> crates/cli/src/main.rs`) after the GPU returns.
>
> **P2b STARTED (`ad68761`)** — config variants + the autotune fix that makes
> multi-candidate selection actually work. Q8_0 `t128` templated as the trivial
> `mmvq_dp4a_row<Q8_0,128>` (skeleton already has THREADS); first real
> multi-candidate CUDA autotune (dp4a-256 vs t128-128). **Key fix: the runtime
> dispatch is m-only** (`KernelDescriptor.m_range`, NO k-range) — a config
> variant that wins at one k and loses at another can't be two rows (that's the
> "two rows match the same shape" build error). The autotune now picks the per-m
> winner by **best total time across the k-grid**, not per-(m,k). Candidate sets
> are **arch-aware** (`sm_86` = CUDA-ported subset). Mechanics for the rest:
> - **`vdr2` DONE (`29258e1`)** — added a **VDR template param**:
>   `quant_traits<Q8_0>::partial_vdr<VDR>` + `mmvq_dp4a_vdr_row<Q,THREADS,VDR,OutT>`
>   (`IPB/VDR` threads/block, VDR dp4a chained → one scale multiply; VDR=1 ==
>   `mmvq_dp4a_row`). Both Q8_0 vdr2 kernels templated (bit-identical); t128_vdr2
>   is the production decode kernel via the runtime alias (gate_up parity test
>   verifies it). **Second emit fix:** non-monotonic winners in m
>   (dp4a@1, t128_vdr2@8, dp4a@16) made the impl-grouped emit produce OVERLAPPING
>   rows; `emit_dispatch` now builds **disjoint contiguous m-runs**. Q8_0 sm_86
>   variants are within noise (~0.3%) on the 3090 → dp4a wins (single row).
> - **`r2`/`r4` DONE — NULL for Q8_0.** Generic `mmvq_dp4a_nrow_row<Q,THREADS,
>   ROWS,OutT>` (one block = ROWS output rows, `THREADS/ROWS` lanes/row, shares
>   the Q8_1 activation reads) + width-templated `mmvq_reduce_store_group<GROUP>`
>   (generalises the Q4_K `_2row`; 32 for r2, 16 for r4). Certed both backends
>   (`r2`/`r4` max_rel_err ~1.6e-3, HIP↔CUDA bit-identical). Autotune null on
>   BOTH: dp4a/t128_vdr2 win at every m (Q8_0 dense + cheap-decode → 256-thread
>   MLP beats the 64-thread wave64-DPP layout). Kept as bench-reference /
>   autotune candidates; no dispatch row earned (tables unchanged). The skeleton
>   is the reusable template path for the heavy-decode K-quant r2/r4 (Q4KR2 /
>   Q6KR4 hand kernels DO win — porting them onto the nrow skeleton is the
>   consolidation follow-up). `gate_up`/`batched`/`row_tile` change the op
>   signature (2 outputs / N columns) — separate skeletons, later.
> - Because dispatch is m-only, a config variant only earns a dispatch row if it
>   wins the **per-m aggregate**; otherwise it stays a certified bench-reference
>   candidate (like t128/t128_vdr2 on the 3090). Two emit bugs are now fixed
>   (per-m k-collapse + disjoint m-runs) — the autotune emit handles
>   multi-candidate config variants correctly.
> - Then **P2c mmq** (4-warp LDS-tiled — the big prefill lever) and **P2d**
>   indexed_moe. See `PORTABILITY-ROADMAP.md`.
>
> Per-slice pattern: own neutral `.cuh` (intrinsics → neutral `dp4a`/reduce,
> misaligned loads → `load_int_b2`), manifest cell (F32+F16), neutralize the
> `_gfx906` impl_id (full ref set: dispatch + backend registry + any `impls.rs`
> test assert + `ops` recipe + sweep enum + cli PMC list + cert `git mv`), HIP
> bit-match (cert max_rel_err unchanged; non-dispatched/uncerted dtypes →
> register the HIP cert as a bench-reference in `direct_call.rs`), CUDA cert +
> `autotune --emit`, `cert-check` both. **Rebuild the cli before each sweep** — a
> stale binary re-emits the pre-rename impl_id cert (hit this on Q8_K).

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
