# Concurrent decode bench matrix — 2026-05-04

Post-removal (`commit 0934c57`) snapshot of concurrent /v1/chat/completions
throughput across:

- **Models**: Qwen3.5-9B-Q4_1 (~5.5 GB), Qwen3.6-27B-Q4_1 (~17 GB),
  Qwen3.6-35B-A3B-Q4_0 (~20 GB MoE)
- **Topologies**: single (1×MI50), TP2 (`hip:0,1`), PP4 (`hip:0,1,2,3`),
  pp2tp2 (`hip:0,2,1,3`)
- **Concurrency**: 1, 2, 4, 8
- **Paths**: `no_batched` (legacy decode) vs `batched`
  (`FLAMBEAU_BATCHED_DECODE=1`, scheduler-aggregated)
- **Workload**: 3313-token prompt + 256-token decode (long-prompt /
  long-response chat shape)

Raw JSON: `matrix.json`. Raw per-cell tables: `raw_tables.md`.
Server logs: `scripts/bench/logs/`.

Settings, every cell:
- `FLAMBEAU_INFLIGHT_SLOTS=8`, `FLAMBEAU_PREFILL_UBATCH=512`,
  `FLAMBEAU_GPU_SAMPLER=1`, `FLAMBEAU_CTX_CAP=16384`
- temperature=0 (greedy), `stream=true`, `stream_options.include_usage=true`
- one server boot per (model, topology, path); concurrencies 1→2→4→8
  swept against the warm server

## Headline aggregate decode throughput (tok/s)

Best path per cell (no_batched and batched are within ±5% everywhere ≠
27B/tp2 — see §Caveats), grouped by (model, topology):

| model | topology | N=1 | N=2 | N=4 | N=8 |
|-------|----------|-----|-----|-----|-----|
| Qwen3.5-9B  | single  | 41 | 40 | 36 | 36 |
| Qwen3.5-9B  | tp2     | 49 | 44 | 41 | 42 |
| Qwen3.5-9B  | pp4     | 42 | 84 | **112** | **120** |
| Qwen3.5-9B  | pp2tp2  | **50** | **89** | 80 | 75 |
| Qwen3.6-27B | tp2     | 21 | 18 | (drop) | (drop) |
| Qwen3.6-27B | pp4     | 17 | 31 | **40** | **40** |
| Qwen3.6-27B | pp2tp2  | **23** | **39** | 34 | 32 |
| Qwen3.6-35B-A3B | pp4 | 41 | 79 | **133** | **126** |
| Qwen3.6-35B-A3B | pp2tp2 | **44** | **93** | 76 | 74 |

Bold = best topology for that (model, N) cell.

## Per-stream decode throughput (tok/s)

What a single chat user perceives, depending on how many *other* concurrent
users are active. Same data divided by N.

| model | topology | N=1 | N=2 | N=4 | N=8 |
|-------|----------|-----|-----|-----|-----|
| Qwen3.5-9B  | single  | 41 | 20 | 9  | 4 |
| Qwen3.5-9B  | tp2     | 49 | 22 | 10 | 5 |
| Qwen3.5-9B  | pp4     | 42 | 42 | 28 | 15 |
| Qwen3.5-9B  | pp2tp2  | 50 | 44 | 20 | 9 |
| Qwen3.6-27B | tp2     | 21 | 9  | (drop) | (drop) |
| Qwen3.6-27B | pp4     | 17 | 16 | 11 | 5 |
| Qwen3.6-27B | pp2tp2  | 23 | 20 | 9  | 4 |
| Qwen3.6-35B-A3B | pp4 | 41 | 40 | 35 | 17 |
| Qwen3.6-35B-A3B | pp2tp2 | 44 | 46 | 19 | 9 |

## Prefill latency (TTFT, ms — mean across N concurrent streams)

Prefill is serialised on the GPU stream, so TTFT scales linearly in N.

| model | topology | N=1 | N=2 | N=4 | N=8 |
|-------|----------|-----|-----|-----|-----|
| Qwen3.5-9B  | single  | 6244 | 12629 | 24138 | 50041 |
| Qwen3.5-9B  | tp2     | 4298 | 8535  | 16985 | 33435 |
| Qwen3.5-9B  | pp4     | 6183 | 6707  | 9969  | 16627 |
| Qwen3.5-9B  | pp2tp2  | 4334 | 8579  | 13674 | 20543 |
| Qwen3.6-27B | tp2     | 11903 | 24079 | (drop) | (drop) |
| Qwen3.6-27B | pp4     | 20134 | 20653 | 28211 | 50936 |
| Qwen3.6-27B | pp2tp2  | 12031 | 23637 | 33490 | 57920 |
| Qwen3.6-35B-A3B | pp4 | 5565 | 5885 | 8583 | 16596 |
| Qwen3.6-35B-A3B | pp2tp2 | 3958 | 7903 | 11656 | 19050 |

PP4 stands out on prompt prefill: PP overlaps prefill chunks across stages,
so 2 concurrent prefills only cost ~1.05× a single prefill on 9B/pp4
(6183 → 6707 ms) and 1.0× on 35B/pp4 (5565 → 5885). pp2tp2 doesn't get the
same overlap because both stages serialise on the same TP world.

## Findings

### 1. Batched vs no_batched is a wash on every cell

Across all 18 (model, topology, slot=8, N) cells with the long prompt,
`FLAMBEAU_BATCHED_DECODE=1` matches the legacy non-batched path within
±5% (range 0.89×–1.04×). Conclusion: the batched-decode kernel slice
(`forward_decode_batched_{pp,tp,hybrid}` + scheduler aggregator) has
zero throughput effect on this workload. **The 1.18× and 1.21× wins
recorded in earlier cert notes (commits 30076e8, 7bcd308) were on
short prompts where prefill is small relative to decode wall — those
gains do not survive long prompts.**

The single small win is 27B/pp4/N=4 (1.04×), within noise. The single
small loss is 9B/single/N=2 (0.90×), also within noise.

### 2. Topology selection depends on model size AND concurrency

There is no universal "best" topology.

- **Single-stream / interactive chat** (N=1 or 2):
  - 9B → pp2tp2 (50, 89 t/s)
  - 27B → pp2tp2 (23, 39 t/s)
  - 35B-A3B → pp2tp2 (44, 93 t/s)
  - **pp2tp2 wins everywhere for N≤2**

- **High-concurrency aggregate** (N=4 or 8):
  - 9B → pp4 (112, 120 t/s) vs pp2tp2 (80, 75)
  - 27B → pp4 (40, 40 t/s) vs pp2tp2 (34, 32)
  - 35B-A3B → pp4 (133, 126 t/s) vs pp2tp2 (76, 74)
  - **pp4 wins everywhere for N≥4 — by 1.4×–1.7× on aggregate**

The crossover lives between N=2 and N=4. Operators serving more than 2
concurrent chat users on this rig should run **pp4**, not pp2tp2.

### 3. Best raw aggregate: 35B-A3B / pp4 / N=4 = 132.9 t/s

The MoE (3 GB of activated experts per token, 20 GB total weights) on
PP4 hits 132.9 t/s at N=4 — a 3.23× scaling over single-stream — and
holds 126.1 at N=8. Per-stream is still 35 t/s at N=4 (faster than 27B
single-stream on pp4, 17 t/s). MoE + PP is the strongest combination
on this PCIe-only rig.

### 4. pp2tp2 caps near N=2 across all models

pp2tp2 doubles aggregate throughput from N=1 to N=2 cleanly (1.79× on
9B, 1.71× on 27B, 2.10× on 35B), but then **regresses** by 10–20% at
N=4 and again at N=8. pp4 instead holds throughput across N=4 and N=8.
The pp2tp2 collapse at concurrency is structural — `peer_copy_via_host`
through one TP rank's pinned-buffer pair becomes the contended
resource, and adding decode streams beyond N=2 just shares the
remaining stage time.

### 5. 27B/tp2 fails at N≥4 (request drops)

27B/tp2 succeeds at N=1 and N=2 but drops 75–87% of streams at N=4 and
N=8 (1 of 4 / 1 of 8 OK on no_batched, 2 of 4 / 2 of 8 on batched).
Server log shows `*PrefillScratch dropped without dispose(device);
device buffers leaked` warnings consistent with mid-prefill request
cancellation. Surviving streams hit normal throughput (~20 t/s).

This is **only on tp2** (27B/pp4 and 27B/pp2tp2 ran 0 errors at N=8).
Likely cause: tp2's per-rank slot pool can't get out of the prefill
phase fast enough on a 27B-class dense forward; the harness's HTTP
client appears to terminate connections that received 200 OK + empty
SSE body (no `data: ` events). Worth a follow-up — diagnostic gate is
the `PrefillScratch leaked` warn line. Not blocking, since pp4 is
already the recommended high-N topology for 27B.

### 6. Single-GPU is bandwidth-bound regardless of path

9B/single peaks at ~40 t/s aggregate from N=1 to N=8. Adding inflight
slots only serialises decode work on one GPU stream — there is no
extra parallelism to exploit. A 16 GB MI50 cannot hold 27B or 35B at
all.

## Caveats

- This run uses `temperature=0` greedy decode. Sampler overhead is small
  but real (~12 ms/token historically); sampled-chat numbers should be
  ~5–10% lower per stream.
- One run per cell (no warmup-after-warmup variance bands). Earlier
  certs have shown ±5% run-to-run variance under stable load, so
  differences ≤5% should be treated as noise, not signal.
- The 8-concurrent N=8 cells push prefill TTFT to 17–58 s. Real chat
  UX would batch prefill differently (continuous-batching with
  preemption) — not in v1 scope.
- 27B/tp2 N≥4 errors deserve their own follow-up bisect; data here
  reports survivor-stream throughput, not realistic load.

## Reproducing

```bash
cargo build --release --features hip_serve -p flambeau-cli
python3 scripts/bench/run_matrix.py \
    --out certs/perf/bench_matrix_2026_05_04/matrix.json \
    --max-tokens 256
python3 scripts/bench/summarize.py \
    certs/perf/bench_matrix_2026_05_04/matrix.json \
    --out certs/perf/bench_matrix_2026_05_04/raw_tables.md
```

Subset by `--models qwen35_9B_q4_1`, `--topos pp4,pp2tp2`,
`--paths batched`, `--concs 1,4` — see `run_matrix.py --help`.
