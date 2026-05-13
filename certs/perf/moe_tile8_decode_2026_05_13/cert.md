# MoE tile8 decode threshold — topology-aware gate

Date: 2026-05-13
Model: Qwen3.6-35B-A3B-Q4_0 / pp2 (hip:0,2) / inflight=4 / batched-decode
Lever: lower the tile8 MoE-MMQ gate from the prefill-only `n_tokens >= 32`
to a decode-aware `total_pairs >= 8 = TILE_N` so batched-decode at small
N hits tile8 instead of indexed-MMVQ row-by-row.

## Design

At batched-decode `n_tokens = N` (concurrent slot count, typically
2–4). With `top_k = 4`, `total_pairs = N * top_k = 8…16` —
exactly tile8's sweet spot. The tile8 padded-sort pipeline (#266c
ancestor) already supports this contract; the only change is which
`if` branch fires.

`blocks/moe_experts.rs::forward_prefill` (non-TP path used by pp2)
and `qwen3-moe/src/forward/moe_tp.rs::forward_moe_ffn_prefill_tp`
(TP path used by tp/pp+tp) both had the same `>= 32` gate. Both
patched in this commit.

**Topology-aware**: the TP path under `tp_world >= 2` keeps the
original prefill-only threshold. tile8 vs indexed-MMVQ has different
overlap behaviour against the per-layer BarP2pAllReduce on residual,
and the per-MMVQ launches happen to hide AR latency better at small
N — measured -10% on tp2 / N=4 when tile8 fires under TP.
`tp_world == 1` (pure PP path through moe_tp.rs, or pp-only via
moe_experts.rs) gets the new lower threshold.

`FLAMBEAU_MOE_TILE8_DECODE=0` disables the decode path back to
indexed-MMVQ.

## A/B on pp2

```
| Threshold variant     | N=2 conc | N=4 conc | conc/seq (N=4) |
|-----------------------+----------+----------+----------------|
| tile8 OFF (default-was) | 53.0    | 64.0     | 1.16           |
| tile8 ON (this commit)  | 52.0    | 72.6     | 1.24           |
| Δ                       | -1.0    | +8.6     | +0.08          |
```

(median of 3–4 runs each, same warm process within each run group.)

- **pp2 N=4 conc: 64.0 → 72.6 (+13.4%)** — clear win.
- pp2 N=2 conc: 53.0 → 52.0 (-1.9%, within run-to-run noise; total_pairs=8 is exactly at the threshold so the padded-overhead barely beats indexed-MMVQ).
- Scheduler aggregation variance: one of three N=4 runs missed
  aggregation and stayed at 52 t/s. When the scheduler actually
  aggregates to N=4, the win is reproducible (2 of 3 runs at 72-73).

## tp2 sanity (gate correctly skips)

```
| Variant            | N=2 conc | N=4 conc |
|--------------------+----------+----------|
| tp2 (before this)  | 54.85    | 54.4     |
| tp2 (this commit)  | 53.8     | 54.2     |
```

tp2 unchanged within noise — the topology-aware gate correctly
holds tp_world>=2 to the legacy prefill-only threshold so the
measured -10% tp2 regression observed when forcing tile8 on TP
doesn't fire here.

## llama.cpp gap

```
| Stack    | Mode  | N=4 conc |
| flambeau | pp2   | 72.6     |   ← post this commit
| flambeau | pp2   | 68.65    |   ← pre this commit (from 3-way cert)
| llama.cpp| layer | 104–108  |
```

flambeau pp2 closes ~16% of the gap to llama.cpp (was 35%, now
~30% slower). Remaining gap likely sits on:
- Continuous batching at the scheduler level (llama.cpp issues all
  4 slots' work into one PP iteration; flambeau still serializes
  some scheduler aggregation steps).
- Per-layer hand-off overhead in flambeau's PP runtime (peer_copy).

## Closes

The "MoE tile8 decode wiring" lever from the
`certs/perf/tp2_3way_2026_05_13/cert.md` next-step section.
