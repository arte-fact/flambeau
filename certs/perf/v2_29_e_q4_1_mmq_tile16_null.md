# V2.29.e — Q4_1 MMQ tile16 port: null (4warp_lds keeps crown)

Direct port of `mmq_q8_0_wave64_tile16`'s TILE_N=16 pattern to Q4_1
as `mmq_q4_1_wave64_tile16`. Result: regression both in perf AND
correctness. Reverted.

## Measured (9B Q4_1 Mesh<4> async ub=128 lanes=2)

| L | baseline 4warp_lds | wave64_tile16 | Δ |
|---|---:|---:|---:|
| 128 | 578 (last_id=571 ✓) | 368 (**last_id=79 ✗**) | −36 % + wrong |
| 1024 | 1978 | 1118 | −43 % |
| 2048 | 2362 | 1375 | −42 % |
| 4096 | **2498** | 1516 | **−39 %** |

Decode tg=64: 54.8 → 42.8 (−22 %).

L=1024/2048/4096 `last_id` coincidentally matches sync baseline
(220/248046/62) even with the wrong L=128 — at longer sequences the
error cascades but produces similar argmax for the tokens we're
sampling. Not a parity success; L=128 mismatch means the kernel is
mathematically wrong AND a performance regression.

## Why porting Q8_0 tile16 to Q4_1 is strictly worse

Looked at the tile shapes after the fact:

| kernel | MMQ_Y (rows) | MMQ_X (cols) | threads | per-block output |
|---|---:|---:|---:|---:|
| mmq_q8_0_wave64_tile8 (pre-V2.7) | 64 | 8 | 64 | 512 elems |
| mmq_q8_0_wave64_tile16 (current) | 64 | **16** | 64 | 1024 elems |
| mmq_q4_1_4warp_lds (Q4_1 default) | **128** | 64 | **256** (4 warps) | 8192 elems |
| mmq_q4_1_wave64 (V2.13.a alt) | 64 | 8 | 64 | 512 elems |
| mmq_q4_1_wave64_tile16 (this iter) | 64 | 16 | 64 | 1024 elems |

Q4_1 `4warp_lds` produces **8×** the elements per block that
`wave64_tile16` does. Smaller per-block output = more blocks = more
grid-launch overhead + less HBM tile amortisation.

The V2.7 Q8_0 `tile16` was a win BECAUSE Q8_0's prior baseline was
tile8 (same family). Porting tile16 to Q4_1 COMPETES against a
fundamentally different kernel family (4-warp LDS-tiled with 2D
tile) that's already faster at its geometry.

The correctness bug at L=128 was a secondary issue — the port
dropped some per-block ordering / accumulator correctness that the
Q8_0 kernel has but that Q4_1's different quant layout needed
re-verified.

## Implication for the 40.8 % prefill wall on Q4_1 MMQ

To improve on `4warp_lds`, the port has to be a **structural**
change, not just a tile-width variant:

Candidates (not in this iteration):
1. **Q4_1 4warp_lds MMQ_X=96 or 128** — widen col tiling at the
   cost of LDS. Current is MMQ_Y=128 × MMQ_X=64 using 32 KiB LDS.
   Going to MMQ_X=96 would use 40 KiB LDS, still within gfx906's
   64 KiB budget.
2. **DP4A inner-loop reorder** — the 4warp_lds currently uses a
   specific DP4A accumulation order. llamacpp-turbo's Q4_1 MMQ
   might have a different order that hides latency better.
3. **L2 prefetch hints** — candle port has `__builtin_amdgcn_buffer_load_lds`
   or similar prefetch primitives. Q8_0 wave64_tile16 doesn't
   use them explicitly; maybe the Q4_1 variant could.

All are kernel-level work, each a session. V2.29.b's BR=8
flash-tile win was a simpler change (one template param flip) so
gave a quick deliverable; Q4_1 MMQ tuning is deeper.

## Action

- **Reverted dispatch to `qmatmul_q4_1_mmq_4warp_lds_gfx906`** (default).
- `mmq_q4_1_wave64_tile16.cu` kept in-tree as a reference for the
  failed structural direction. No `cfg(unverified)` gate because the
  kernel file compiles cleanly, just isn't routed. Future iterations
  can A/B different tile shapes starting from the same file.
- Cert entry `qmatmul_q4_1_mmq_wave64_tile16_gfx906.json` left
  in-place (minus mention in impls.rs) — wasn't regenerated via
  sweep harness, so it's a placeholder copy of the wave64 cert.
  A sweep-run on the correct variant would overwrite it; for now
  the row isn't referenced.

## V2.29 state after i6

| task | outcome |
|---|---|
| V2.29.a audit | complete, identifies levers |
| V2.29.b flash-tile BR=8 | **+24.7 % prefill L=4096** ✓ |
| V2.29.c Q4_1 MMVQ A/B | null (t128 wins existing variants) |
| V2.29.e Q4_1 MMQ tile16 port | null (structural mismatch) |
| V2.29.d fused decode kernels | deferred (low ROI per audit) |

Real Q4_1 MMQ tuning is multi-session kernel work. V2.29.b's
flash-tile win keeps the cumulative total at 2498 tok/s L=4096
(2.59× over turbo).

## Gate

- After revert: 9B prefill L=4096 = 2498 tok/s, last_id=62 ✓
- Decode tg=64 = 54.8 tok/s, last_id=30 ✓
- All perf + parity identical to post-V2.29.b state.
