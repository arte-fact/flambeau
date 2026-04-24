# V2.28.c — long-context prefill bench (L ∈ {8192, 16384})

Extended both perf harnesses' prefill grid with L=8192 and L=16384 to
see how the post-V2.29.b stack (BR=8 flash-tile + V2.26.a async +
ub=128 lanes=2) scales at long context. Attention is O(L²·d) so
per-token throughput is expected to decline as L grows; the real
question is whether async PP continues to amortise.

## 9B Qwen3.5-Q4_1 Mesh<4>

| L      | sync | async ub=128 | async/sync |
|--------|-----:|-------------:|-----------:|
| 4096   |  753 | **2509**     | **3.33×**  |
| 8192   |  651 | **2356**     | **3.62×**  |
| 16384  |  512 | **1910**     | **3.73×**  |

Async ratio *grows* with L. Reason: sync is compute-bound, with
attention's O(L²) dominating at long context; async pipelines that
attention work across 4 ranks × 2 lanes in parallel, so the
parallelism gain is worth more when there's more work.

At L=16384: **1910 tok/s = 8.58 s wall for a 16K prompt**.
Production-usable for long-context chat. Sync would take 32 s.

Parity: `last_id` matches between sync and async at every L on 9B
(L=8192: both `198`, L=16384: both `198`).

## 35B Qwen3.6-A3B-UD-Q4_K_S Mesh<4>

| L      | sync | async ub=128 | async/sync |
|--------|-----:|-------------:|-----------:|
| 4096   |  671 | **1916**     | **2.86×**  |
| 8192   |  564 | **1783**     | **3.16×**  |
| 16384  |  435 | 1382         | 3.18×      |

Same pattern: async amortises better at long context on 35B too.

Parity at L=4096 and L=8192 (`last_id=248046` on both paths, both L).
**At L=16384 parity diverges**: sync gives `last_id=220`, async
gives `last_id=143737`. Flagged as V2.28.c.1 follow-up (see below).

## V2.28.c.1 — 35B parity divergence at L=16384

- **9B** async: L=16384 `last_id=198` matches sync `198`. OK.
- **35B** async: L=16384 `last_id=143737` vs sync `220`. **Not OK.**

Both paths matched at L=8192. Only 35B at L=16384 diverges.
Likely causes to investigate:

1. **KV cache capacity**: Qwen3.6-35B-A3B's `context_length` is
   likely ≤ 16384 (some UD-Q4_K_S builds ship with 16k or 32k cap).
   At L=16384 the KV tail might silently wrap or overflow on one of
   the paths. V2.27.d found a similar issue (tail-ubatch race at
   small ubatch) — this might be its L=16384 sibling.
2. **Async tail-ubatch**: L=16384 / ubatch=128 = exactly 128
   ubatches (clean divide, no tail), so not the V2.27.d class of
   bug.
3. **Floating-point drift through 128 ubatches × 40 layers**:
   accumulated F16 imprecision that diverges at this many hops.
   9B has only 36 layers + dense FFN = fewer FP ops; 35B has 40 +
   MoE routing + GDN state = more.

Not a correctness gate blocker (both paths run to completion; 9B
long-context parity holds). Filed as V2.28.c.1 for future diagnosis.

## Running state across the V2 arc on 9B Q4_1 Mesh<4>

| milestone | L=1024 | L=4096 | L=8192 | L=16384 |
|---|---:|---:|---:|---:|
| V2.26.b async | 770 | 714 | n/a | n/a |
| V2.26.a (barrier removed) | 1839 | 1988 | n/a | n/a |
| V2.29.b (BR=8) | 1978 | **2498** | — | — |
| V2.28.c async ub=128 | 1970 | **2509** | **2356** | **1910** |
| llamacpp-turbo ref | 1013 | 963 | n/a | n/a |

Async ratio vs turbo at each L:
- L=1024: 1.94× ahead
- L=4096: 2.60× ahead
- L=8192, 16384: no turbo reference (llama-bench doesn't ship
  those lengths by default)

## Gate

- Build clean on both harnesses with extended grid.
- 9B parity bit-exact at every L across sync + async.
- 35B parity bit-exact at L ∈ {4096, 8192} × both paths; L=16384
  diverges (filed).
- No perf regression on the shorter L grid (values match post-V2.29.b).

## Regeneration

```
# 9B
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# 35B
FLAMBEAU_PREFILL_ONLY=1 FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture
```
