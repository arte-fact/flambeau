# V1-BENCH-#112 — Q4_K turbo MMQ promotion (35B-A3B-UD-Q4_K_S)

GGUF: `/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`
Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1.

Promoted `qmatmul_q4_K_mmq_turbo_gfx906` (V2.14.b llamacpp-turbo 4-warp
LDS-tiled Q4_K MMQ, MMQ_Y=128 / MMQ_X=16 / 4×wave64 / DS4 Q8_1 / 22528 B
LDS) from lookup-only (`m_range = MAX..MAX`) to default at `m >= 128`.
The wave64 row (`qmatmul_q4_K_mmq_wave64_gfx906`) now owns m=32..127.

Wiring: `Recipe::from_impl_id` adds an `MmqLdsX64` entry for the turbo
kernel; `mmq_lds_x64_launch` was generalised so per-stem
`block_elems_w` + `shared_bytes` come from `mmq_lds_x64_params(stem)`
instead of the previous Q4_1-hardcoded `QK4_1=32 / 7584*4 B`.

Cert-check post-promotion: **50 rows, 0 failures**.
Q4_K turbo sweep: 7/7 shapes pass.

## Results — pp4

| L      | Before (V1 bench)¹ | After rep 1 | After rep 2 | Δ vs before |
|--------|-------------------:|------------:|------------:|------------:|
| pp128  |              495.9 |       493.4 |           — |       −0.5% |
| pp512  |              605.7 |       629.3 |       627.4 |       +3.7% |
| pp2048 |              606.4 |       595.7 |       616.2 |        ~0%  |
| tg64   |               62.1 |        62.3 |        56.8 |       noise² |

¹ `qwen36_35b_a3b_ud_q4_k_s_summary.md` (pre-promotion baseline).
² tg64 rep 2 noise (decode unchanged — kernel only at m≥128).

## Why the win is small

Q4_K wave64 was already a DP4A kernel (1 warp / 64 rows × 8 cols, V2.3.b.2
candle port). Turbo's edge over wave64 here is the DS4 activation layout
plus double-buffered Y LDS — modest on this attention-projection shape.
Compare V1-BENCH-C1 (Q4_0) where wave64 was a single-warp baseline and
4warp_lds delivered 1.43× on 27B-Q4_0 prefill.

Net: minor win (+3.7% pp512), no regression, lookup-only row resolved.

## Files touched

- `crates/ops/src/hip/qmatmul.rs` — `mmq_lds_x64_params(stem)` lookup +
  recipe entry for `qmatmul_q4_K_mmq_turbo_gfx906`.
- `crates/backend-hip/src/impls.rs` — turbo m_range
  `(MAX,MAX)` → `(128,MAX)`, wave64 m_range `(128,MAX)` → `(32,127)`;
  static dispatch ordering: turbo BEFORE wave64.
- `dispatch/hip/gfx906.toml` — TOML rows mirror the impls.rs change.

## Closes

- #112 #105a — kernel was already authored as `mmq_q4_K_turbo` in V2.14.b;
  task collapsed to wiring + dispatch promotion.
- #113 #105b — cert sweep green, dispatch promoted, bench cert above.
