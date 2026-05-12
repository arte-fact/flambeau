# feature/batched-mmvq-decode vs main — perf cert #2 (2026-05-12, post-T1)

Re-bench after the tier-1 K-quant kernel work landed
(Q2_K / Q3_K / Q8_K MMVQ + MMQ + MoE variants, +14 tile8 MoE MMQ certs,
the Q3_K alignment bug fix, the generic tile8 cert harness, and the
`// 1.b —` / Q3_K helper cleanups). Branch is now 38 commits ahead of
main; previous cert (`cert.md`) was at commit `17d442d`.

## Setup

- Rig: 4× MI50 / gfx906, ROCm 7.1.1.
- Topology: `pp2tp2` (`hip:0,2,1,3` = pp_size=2, tp_size=2 — user's
  primary prod topology).
- Concurrency: N ∈ {1, 2}. Inflight slots = max(N) = 2.
- Paths: `batched` (`FLAMBEAU_BATCHED_DECODE=1`) and `no_batched`.
- `max_tokens = 256`, prompt `'x' * 1024` (script default).
- main was patched with `af189c0` (qwen35 arch registration) so it can
  load 27B-Q4_0 / 27B-Q4_1; reverted after bench.
- Runs: `certs/perf/branch_vs_main/HEAD2.json`, `MAIN2.json`,
  `HEAD2.log`, `MAIN2.log`, `diff2.txt`.

## Pre-flight

- `cert-check hip/gfx906`: 63 dispatch rows + 36 DirectCallKernel certs,
  0 failures.
- Tier-1 sweep harness green on all 9 new kernels (Q2_K / Q3_K / Q8_K
  MMVQ + MMQ wave64 + Q2_K / Q3_K MoE indexed-MMVQ + Q2_K / Q3_K MoE
  MMQ tile8). Tier-2 tile8 harness (T2.3b) green on all 14 existing
  tile8 kernels (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0/Q5_0 × down + gate_up).
- **Caveat**: the new K-quant kernels do not fire on any model in this
  bench — Qwen3.6-27B-Q4_0/Q4_1 + Qwen3.6-35B-A3B-Q4_0 + UD-Q4_K_S all
  use Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / Q8_0 dispatch rows that were
  already in place on `main`. This bench is a **regression check** for
  the tier-1 work, not a measurement of it. A Q2_K/Q3_K/Q8_K model
  bench needs a different model (e.g. 122B-A10B-UD-Q3_K_XL).

## Results

### Prefill — per-token rate (normalised away from the tokenizer change)

| model | path | N | main tok | head tok | main pp t/s | head pp t/s | ratio |
|---|---|---|---|---|---|---|---|
| 27B-Q4_0          | no_batched | 1 | 3313 | 3003 | 290.6 | 297.9 | 1.025× |
| 27B-Q4_0          | no_batched | 2 | 3313 | 3003 | 163.9 | 164.6 | 1.004× |
| 27B-Q4_0          | batched    | 1 | 3313 | 3003 | 288.7 | 295.6 | 1.024× |
| 27B-Q4_0          | batched    | 2 | 3313 | 3003 | 162.6 | 163.2 | 1.003× |
| 27B-Q4_1          | no_batched | 1 | 3313 | 3003 | 280.4 | 287.0 | 1.024× |
| 27B-Q4_1          | no_batched | 2 | 3313 | 3003 | 159.2 | 159.7 | 1.003× |
| 27B-Q4_1          | batched    | 1 | 3313 | 3003 | 279.7 | 285.7 | 1.022× |
| 27B-Q4_1          | batched    | 2 | 3313 | 3003 | 159.0 | 159.1 | 1.001× |
| 35B-A3B-Q4_0      | no_batched | 1 | 3313 | 3003 | 834.9 | 859.6 | 1.030× |
| 35B-A3B-Q4_0      | no_batched | 2 | 3313 | 3003 | 468.8 | 478.0 | 1.020× |
| 35B-A3B-Q4_0      | batched    | 1 | 3313 | 3003 | 836.9 | 858.7 | 1.026× |
| 35B-A3B-Q4_0      | batched    | 2 | 3313 | 3003 | 464.8 | 474.5 | 1.021× |
| 35B-A3B-UD-Q4_K_S | no_batched | 1 | 3313 | 3003 | 648.1 | 660.0 | 1.018× |
| 35B-A3B-UD-Q4_K_S | no_batched | 2 | 3313 | 3003 | 370.0 | 374.2 | 1.011× |
| 35B-A3B-UD-Q4_K_S | batched    | 1 | 3313 | 3003 | 646.4 | 659.9 | 1.021× |
| 35B-A3B-UD-Q4_K_S | batched    | 2 | 3313 | 3003 | 366.6 | 375.2 | 1.024× |

**Per-token prefill: 1.00–1.03× across all cells.** Same story as
`cert.md`: wall-clock prefill_ms drops 10–14 % on HEAD, but ~9 % is
the tokenizer change (`c190ab4` from earlier in the branch) producing
3003 vs 3313 tokens for the same prompt text. Real prefill rate is
flat-to-marginal-win.

### Decode — aggregate (server view) and per-stream (user view)

| model | path | N | main agg | head agg | agg Δ | main /stream | head /stream | str Δ |
|---|---|---|---|---|---|---|---|---|
| 27B-Q4_0          | no_batched | 1 | 23.24 | 23.10 | 0.99× | 23.24 | 23.10 | 0.99× |
| 27B-Q4_0          | no_batched | 2 | 32.17 | 30.83 | **0.96×** | 19.57 | 19.26 | 0.98× |
| 27B-Q4_0          | batched    | 1 | 23.20 | 23.07 | 0.99× | 23.20 | 23.07 | 0.99× |
| 27B-Q4_0          | batched    | 2 | 31.85 | 31.06 | 0.98× | 19.49 | 19.42 | 1.00× |
| 27B-Q4_1          | no_batched | 1 | 23.30 | 23.20 | 1.00× | 23.30 | 23.20 | 1.00× |
| 27B-Q4_1          | no_batched | 2 | 32.71 | 31.70 | **0.97×** | 19.75 | 19.63 | 0.99× |
| 27B-Q4_1          | batched    | 1 | 23.33 | 23.21 | 0.99× | 23.33 | 23.21 | 0.99× |
| 27B-Q4_1          | batched    | 2 | 32.37 | 31.83 | 0.98× | 19.67 | 19.49 | 0.99× |
| 35B-A3B-Q4_0      | no_batched | 1 | 45.01 | 44.68 | 0.99× | 45.01 | 44.68 | 0.99× |
| 35B-A3B-Q4_0      | no_batched | 2 | 65.45 | **67.41** | **1.03×** | 39.16 | 39.58 | 1.01× |
| 35B-A3B-Q4_0      | batched    | 1 | 44.47 | 44.50 | 1.00× | 44.47 | 44.50 | 1.00× |
| 35B-A3B-Q4_0      | batched    | 2 | 64.55 | **66.97** | **1.04×** | 38.74 | 39.42 | 1.02× |
| 35B-A3B-UD-Q4_K_S | no_batched | 1 | 46.99 | 46.92 | 1.00× | 46.99 | 46.92 | 1.00× |
| 35B-A3B-UD-Q4_K_S | no_batched | 2 | 61.10 | **63.34** | **1.04×** | 38.94 | 39.68 | 1.02× |
| 35B-A3B-UD-Q4_K_S | batched    | 1 | 46.34 | 46.34 | 1.00× | 46.34 | 46.34 | 1.00× |
| 35B-A3B-UD-Q4_K_S | batched    | 2 | 59.51 | **63.20** | **1.06×** | 38.32 | 39.82 | 1.04× |

**Mixed by arch:**
- **MoE wins at N=2** — 35B-A3B-Q4_0 +3-4 % cum, UD-Q4_K_S +4-6 % cum.
  Per-stream wins are smaller (+1-4 %); the rest is host-side overlap
  (scheduler aggregating two pending decodes into one tp+pp forward).
  These were the wins from the earlier per-N batched MMVQ commits
  (`31ba1a2` Q8_0, `3955d58` Q4_K, `d41cef7` Q6_K) firing on the MoE
  expert weights.
- **Dense 27B regresses at N=2, no_batched** — 27B-Q4_0 -4.1 % cum,
  27B-Q4_1 -3.1 % cum. Per-stream is flat (-1 to -2 %), so the
  aggregate hit is scheduler-level, not kernel-level. Both are dense
  Q4_0/Q4_1; the batched path (which is identical to the previous bench
  at this commit range for these dtypes) is closer to flat. Likely
  candidate: 27B's slot-2 forward shares state on the per-rank stream
  in a way that's more sensitive to whatever drift came in with the
  Q3_K alignment fix / helper dedup. Per-stream behaviour is fine, so
  this is a 2-slot scheduling effect, not a correctness or kernel-
  speed change.
- **N=1 is flat across the board** (0.99×–1.00× across all 16 cells) —
  no per-call kernel regression on the production decode path.

## Headline

- **Prefill**: per-token rate flat-to-+3 %. Wall-clock 1.10-1.14×
  driven by the tokenizer fix from earlier in the branch, not by the
  K-quant work.
- **Decode N=1**: flat. No regression in the production single-stream
  case.
- **Decode N=2**: MoE +3-6 % wins (continuing the per-N batched MMVQ
  story from earlier commits); dense 27B sees -3-4 % aggregate
  regression on the no_batched path only; per-stream flat.
- **None of the tier-1 K-quant kernels fire on these models.** A
  Q2_K / Q3_K / Q8_K-weighted model would be needed to measure their
  impact — out of scope for this regression check.

## Reproducibility

Branch:
```
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/run_matrix.py \
  --out certs/perf/branch_vs_main/HEAD2.json \
  --models qwen36_27B_q4_0,qwen36_27B_q4_1,qwen36_35B_a3b_q4_0,qwen36_35B_a3b_ud_q4_k_s \
  --topos pp2tp2 --paths batched,no_batched --concs 1,2
```

main (with `af189c0` cherry-picked + `scripts/bench/run_matrix.py`
from branch for the model list):
```
git checkout main && git cherry-pick af189c0
git checkout feature/batched-mmvq-decode -- scripts/bench/run_matrix.py
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/run_matrix.py --out certs/perf/branch_vs_main/MAIN2.json [same args]
git reset --hard HEAD~1 && git checkout -- scripts/bench/run_matrix.py
git checkout feature/batched-mmvq-decode
```

Diff:
```
python3 scripts/bench/diff_matrix.py \
  --baseline certs/perf/branch_vs_main/MAIN2.json \
  --candidate certs/perf/branch_vs_main/HEAD2.json
```
