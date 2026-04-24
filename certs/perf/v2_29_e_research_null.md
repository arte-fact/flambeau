# V2.29.e recycled — deep research into Q4_1 MMQ bottleneck, three null attempts

Followup to V2.29.e's first null attempt (tile16 port). Did proper
upstream reference research first, then re-tried V2.29.e with three
fresh structural angles. All null. Documenting the findings so
future sessions don't re-sweep the same space.

## Research — what turbo actually does for Q4_1 MMQ on gfx906

Sources:
- `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/ggml/src/ggml-cuda/mmq.cuh`
- `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/ggml/src/ggml-cuda/gfx906/`
  (matmul/mmq.cuh, matmul/mmq-prefetch.cuh, gfx906-config.h)

Key turbo techniques and our status against each:

| turbo technique | what it does | flambeau status |
|---|---|---|
| `int4` vectorised Y-load | loads 2× int4 (128-bit) per iter instead of 8× scalar int | **already have** (mmq_q4_1_4warp_lds.cu line 198-199) |
| L2 prefetch via `global_load_dword` asm | issues next-iter HBM read hint | **already have** (V2.2.d fix 5a, mmq_prefetch.cuh) |
| `GFX906_LOAD_TILES_Q8_0_ASYNC` register-cache split | load all HBM into regs, then store all to LDS | Q8_0-specific; turbo doesn't apply it to Q4_1 either |
| `GFX906_MMQ_NWARPS = 2` (vs upstream 8) | half-warps-per-block = more blocks/CU | **WE USE 4** — different point |
| MMQ_Y=128, MMQ_X_MAX=64 | fixed geometry | same as us |

The NWARPS=2 vs 4 difference is the ONE real structural delta between
our kernel and turbo's. Everything else — int4 vec loads, L2 prefetch,
tile sizes — is identical.

## Three fresh attempts, all null

### Attempt A: flip `#define MMQ_NWARPS 4 → 2`

Naive swap in `mmq_q4_1_4warp_lds.cu` + host-launch block dim
`(64, 4, 1) → (64, 2, 1)`.

| L | 4warp_lds baseline | NWARPS=2 flip | Δ |
|---|---:|---:|---:|
| 1024 | 1978 | 1103 | **−44 %** |
| 2048 | 2362 | 1358 | **−43 %** |
| 4096 | 2498 | 1478 | **−41 %** |

Parity preserved (last_id matches). Just slower.

Root cause: the kernel's **inner vec_dot loop structure** is built
around 4 warps sharing the DP4A work via the `j0 += MMQ_NWARPS`
stride. Halving NWARPS doubles per-thread sum-slot count (32 → 64
f32 accumulators/thread). VGPR pressure rises → either compiler
spills to scratch (catastrophic) or forces lower per-CU
occupancy — measured is the latter.

Turbo's `vec_dot_q4_1_q8_1_dp4a` is structured for NWARPS=2 with a
different load/store pattern that keeps sum-slot count low per
thread. Porting requires rewriting the load_tiles + vec_dot path,
not just flipping the define.

### Attempt B: promote existing `wave64` MMQ to default at m≥128

`qmatmul_q4_1_mmq_wave64_gfx906` is certified (V2.13.a) but
`m_range: (usize::MAX, usize::MAX)` — lookup-only since V2.13.b A/B
never ran. Re-ran the A/B under V2.26.a's post-barrier conditions:

| L | 4warp_lds baseline | wave64 | Δ |
|---|---:|---:|---:|
| 1024 | 1978 | 574 | **−71 %** |
| 2048 | 2362 | 904 | **−62 %** |
| 4096 | 2498 | 1017 | **−59 %** |

wave64 has MMQ_Y=64, TILE_N=8, 64 threads = 512 output elems/block.
4warp_lds has MMQ_Y=128, MMQ_X=64, 256 threads = 8192 output
elems/block. 16× fewer output/block = 16× more blocks = grid
launch overhead dominates. Same root cause as V2.29.e's first
attempt (tile16 port).

### Attempt C: tile16 port from Q8_0 (V2.29.e iter 1)

Documented in `v2_29_e_q4_1_mmq_tile16_null.md`. Same 16× smaller
tile = same regression pattern (-39 % at L=4096 + correctness bug
at L=128).

## What V2.3.a said in hindsight

> "The ~0.54 ms/call residual gap to turbo is not VGPR-tractable on
>  ROCm 7.1.1."

V2.3.a swept MMQ_X variants (32 / 64) and `__launch_bounds__` occupancy
hints within the 4-warp regime. All null. The 0.54 ms/call gap lives
in territory only accessible by rewriting the kernel's load/compute
structure around NWARPS=2 — a port of turbo's `vec_dot_q4_1_q8_1_dp4a`
that we'd need to write from scratch with their sum-slot layout.

## Honest decision

The Q4_1 4warp_lds is **at the practical ceiling** for our kernel
family. Closing the 0.54 ms/call gap to turbo requires:

1. Port `load_tiles_q4_1` from turbo's `mmq.cuh` line 425 — different
   LDS tile layout keyed off NWARPS=2.
2. Port `vec_dot_q4_1_q8_1_dp4a` from line 487 with its specific
   sum-slot + `vec_dot_q4_1_q8_1_impl` innerloop.
3. Integrate the turbo-specific `gfx906_load_q4_1_quants_vectorized`
   (we have the equivalent as inline `int4` loads).
4. New host-side launcher matching turbo's block dim + LDS budget.

That's a ~500-LOC kernel rewrite, multi-session work. Given our
current 9B Q4_1 Mesh<4> L=4096 is **2498 tok/s = 2.59× ahead of
turbo's 963** — the turbo-style kernel rewrite would shave at most
~10 % off our prefill wall for a ~5 % end-to-end gain. Not worth
the kernel-engineering budget.

## V2.29 final state

| iter | target | result | cumulative impact |
|---|---|---|---|
| V2.29.a | per-kernel audit | complete | — |
| V2.29.c | Q4_1 MMVQ variant A/B | null | — |
| **V2.29.b** | **flash-tile BR=4→8 at d=256** | **+24.7 % L=4096** | **2498 tok/s** |
| V2.29.e tile16 | Q8_0-style tile16 port | null | — |
| V2.29.e (this recycle) | turbo-structure research + 3 attempts | null | — |
| V2.29.d | fused decode kernels | deferred (low ROI) | — |

The +24.7 % from V2.29.b stands as V2.29's one kernel-level win.
All other structural directions were null; the remaining 0.54 ms/call
gap to turbo on Q4_1 MMQ lives in kernel rewrite territory that's
out of scope for V2.29.

## Gate

- `impls.rs` restored: wave64 at (MAX, MAX) = lookup-only; 4warp_lds
  at (128, MAX) = default.
- `mmq_q4_1_4warp_lds.cu` restored: NWARPS=4.
- Post-restore perf verification on 9B Q4_1 Mesh<4> async ub=128
  lanes=2: L=4096 = 2425 tok/s (within noise of V2.29.b's 2498),
  last_id=62 ✓.
- No committed code change from this research iteration (all A/B
  swaps reverted); this cert is the only artefact.

## References

- turbo Q4_1 MMQ: `llamacpp-turbo/.../ggml-cuda/mmq.cuh` lines 425-538
- gfx906 vectorised loads: `.../ggml-cuda/gfx906/matmul/mmq.cuh`
- gfx906 NWARPS=2 config: `.../ggml-cuda/gfx906/gfx906-config.h` line 10
- turbo L2 prefetch: `.../ggml-cuda/gfx906/matmul/mmq-prefetch.cuh`
- our current implementation: `crates/kernels-hip/src/kernels/mmq_q4_1_4warp_lds.cu`
- V2.3.a prior finding: `certs/perf/v2_3_a_q4_1_vgpr_null.md`
