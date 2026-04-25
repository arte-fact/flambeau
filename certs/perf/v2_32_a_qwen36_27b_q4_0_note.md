# V2.32.a-addendum — Qwen3.6-27B-Q4_0 measured, Qwen3.5-27B-Q4_1 still the winner

## Results (100 W/GPU, Mesh<4>, async u_lanes=2 ub=128)

| L | Qwen3.5-27B-Q4_1 | **Qwen3.6-27B-Q4_0** | Δ          |
|---|-----------------:|---------------------:|-----------:|
| 128   | 177 | 80  | **−55 %**  |
| 512   | 315 | 124 | **−61 %**  |
| 1024  | 463 | 177 | **−62 %**  |
| 2048  | 561 | 219 | −61 %      |
| 4096  | 559 | 247 | −56 %      |
| 8192  | 566 | 254 | −55 %      |
| decode tg=64 | **21.1** | **19.8** | −6 %  |

## Root cause

Qwen3.6-27B-Q4_0 routes Q4_0 MMQ through `qmatmul_q4_0_mmq_wave64_gfx906`
(V2.28.a 64-thread wave64). Qwen3.5-27B-Q4_1 routes Q4_1 MMQ through
`mmq_q4_1_4warp_lds` (V2.13 128-thread × 128×64 LDS-tiled kernel) —
the gfx906 local optimum.

**Decode**: both use MMVQ at M=1. Close (within 6 %). Q4_1 slightly
edges because `mmvq_q4_1_t128` (128 threads) is slightly better tuned
than `mmvq_q4_0` (256 threads).

**Prefill**: wave64 vs 4warp_lds is a 2×-5× kernel-level gap. No Q4_0
4warp_lds variant exists on flambeau (potential V2.31.h follow-up —
~200 LOC new kernel).

## Implication for "focus on Qwen3.6-27B"

If Qwen3.6-27B is the target, **we either**:

1. Port `mmq_q4_1_4warp_lds` to Q4_0 (V2.31.h, not yet done). Would
   lift Qwen3.6-27B-Q4_0 prefill to parity with Qwen3.5-27B-Q4_1
   (~550 tok/s peak) and decode to ~21 tok/s. Still same spec-decode
   ceiling as Qwen3.5-27B-Q4_1.

2. Or pick Qwen3.6-27B-Q4_1 / Q5_K_M / Q4_K_M instead. None are in
   `/artefact/models`. Would require requantising from F16 source.

3. Or accept the 6 % decode penalty of Qwen3.6-27B-Q4_0 (19.8 tok/s)
   and ship V2.33 spec-decode on it anyway — gains are multiplicative.

Recommendation: **build V2.33 on Qwen3.6-27B-Q4_0** (spec decode on
top of 19.8 tok/s baseline → projected ~28-30 tok/s general text).
If V2.31.h ports Q4_0 to 4warp_lds later, decode stays flat (decode
is MMVQ, not MMQ), but prefill jumps which helps spec-verify cost.

## Regeneration

```bash
TEST=$(ls -t target/release/deps/perf_baseline_qwen35_9b-* | grep -v '\.d$' | head -1)
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 FLAMBEAU_MESH_RANKS=4 \
  FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.6-27B-Q4_0.gguf \
  $TEST perf_baseline_qwen35_9b --nocapture
```
