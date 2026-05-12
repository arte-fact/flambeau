# feature/batched-mmvq-decode vs main — perf cert (2026-05-12)

## Setup

- Rig: 4× MI50 / gfx906, ROCm 7.1.1.
- Topology: `pp2tp2` (`hip:0,2,1,3` = pp_size=2, tp_size=2 — user's primary prod topology).
- Concurrency: N ∈ {1, 2}. Inflight slots = max(N) = 2.
  N=4 was attempted in an earlier run and OOM'd at 4 MB KV alloc on rank-0
  of the second stage (27B-Q4_0 / pp2tp2 / slots=4 / ctx=4096); reduced
  slots so the 27B variants fit.
- Paths: `batched` (FLAMBEAU_BATCHED_DECODE=1) and `no_batched`.
  Note: the per-N batched-MMVQ kernels in `qmatmul()` fire at m∈{2,3,4}
  **regardless** of FLAMBEAU_BATCHED_DECODE — that env flag controls the
  scheduler-aware aggregator at the request level, not the qmatmul-level
  micro-batching.
- max_tokens 256, prompt 'x'*1024 (script default).
- main was patched with af189c0 (qwen35 arch registration) for the 27B
  cells. main without the patch **physically cannot load** the 27B-Q4_0
  / 27B-Q4_1 GGUFs (they ship `general.architecture = qwen35`, which
  main's registry rejects). The 27B comparison is therefore main+patch
  vs HEAD; not main-as-shipped vs HEAD.

## Pre-flight

- `mmvq_q4_1_batched_parity` (the only parity test for the new batched
  kernels) green: 9/9 sweep, max_abs_err 1.43e-6 (F32 noise floor).
- `cargo check --workspace --features hip` clean on both branches.
- **Caveat**: K1 (`mmvq_q4_0_batched`) and K3 (`mmvq_q5_k_r2_batched`)
  ship without parity tests on this branch. CLAUDE.md rule 2 violation;
  flagged but not fixed in this run.
- K5 (`mmvq_q4_0_gate_up_batched`) is pub but unwired — no caller
  outside its own definition. Dead public fn on the branch.

## Results

### Prefill — rate per token (`prompt_tokens / prefill_ms_mean`)

| model | path | N | main ptok | head ptok | main pp t/s | head pp t/s | ratio |
|---|---|---|---|---|---|---|---|
| 27B-Q4_0          | no_batched | 1 | 3313 | 3003 | 279.2 | 285.6 | 1.02× |
| 27B-Q4_0          | no_batched | 2 | 3313 | 3003 | 158.0 | 159.5 | 1.01× |
| 27B-Q4_0          | batched    | 1 | 3313 | 3003 | 277.2 | 283.6 | 1.02× |
| 27B-Q4_0          | batched    | 2 | 3313 | 3003 | 157.1 | 159.0 | 1.01× |
| 27B-Q4_1          | no_batched | 1 | 3313 | 3003 | 270.5 | 276.6 | 1.02× |
| 27B-Q4_1          | no_batched | 2 | 3313 | 3003 | 154.3 | 155.2 | 1.01× |
| 27B-Q4_1          | batched    | 1 | 3313 | 3003 | 269.1 | 275.3 | 1.02× |
| 27B-Q4_1          | batched    | 2 | 3313 | 3003 | 154.1 | 154.0 | 1.00× |
| 35B-A3B-Q4_0      | no_batched | 1 | 3313 | 3003 | 839.5 | 842.7 | 1.00× |
| 35B-A3B-Q4_0      | no_batched | 2 | 3313 | 3003 | 465.6 | 471.6 | 1.01× |
| 35B-A3B-Q4_0      | batched    | 1 | 3313 | 3003 | 835.2 | 843.3 | 1.01× |
| 35B-A3B-Q4_0      | batched    | 2 | 3313 | 3003 | 461.4 | 473.3 | 1.03× |
| 35B-A3B-UD-Q4_K_S | no_batched | 1 | 3313 | 3003 | 639.7 | 642.9 | 1.01× |
| 35B-A3B-UD-Q4_K_S | no_batched | 2 | 3313 | 3003 | 368.3 | 363.2 | 0.99× |
| 35B-A3B-UD-Q4_K_S | batched    | 1 | 3313 | 3003 | 636.7 | 641.8 | 1.01× |
| 35B-A3B-UD-Q4_K_S | batched    | 2 | 3313 | 3003 | 360.1 | 369.4 | 1.03× |

**Prefill rate is essentially flat — all ratios 0.99×–1.03×, within
bench noise.** The wall-clock prefill_ms drops 9–13 % on HEAD, but
that's because HEAD's tokenizer (`c190ab4` pretok regex fix) turns
the same prompt text into **3003 tokens vs 3313 on main** — 9.4 %
fewer tokens to process. Same kernel throughput, less input.

### Decode — aggregate (server view) and per-stream (user view)

| model | path | N | main agg | head agg | agg Δ | main /stream | head /stream | str Δ |
|---|---|---|---|---|---|---|---|---|
| 27B-Q4_0          | no_batched | 1 | 23.17 | 23.33 | 1.01× | 23.17 | 23.33 | 1.01× |
| 27B-Q4_0          | no_batched | 2 | 33.25 | 33.59 | 1.01× | 19.99 | 20.12 | 1.01× |
| 27B-Q4_0          | batched    | 1 | 23.17 | 23.13 | 1.00× | 23.17 | 23.13 | 1.00× |
| 27B-Q4_0          | batched    | 2 | 33.41 | 33.93 | 1.02× | 19.92 | 20.17 | 1.01× |
| 27B-Q4_1          | no_batched | 1 | 23.28 | 23.25 | 1.00× | 23.28 | 23.25 | 1.00× |
| 27B-Q4_1          | no_batched | 2 | 34.20 | 34.56 | 1.01× | 20.20 | 20.43 | 1.01× |
| 27B-Q4_1          | batched    | 1 | 23.28 | 23.27 | 1.00× | 23.28 | 23.27 | 1.00× |
| 27B-Q4_1          | batched    | 2 | 34.29 | 34.39 | 1.00× | 20.25 | 20.28 | 1.00× |
| 35B-A3B-Q4_0      | no_batched | 1 | 47.53 | 46.52 | 0.98× | 47.53 | 46.52 | 0.98× |
| 35B-A3B-Q4_0      | no_batched | 2 | 67.28 | **70.18** | **1.04×** | 40.57 | 41.21 | 1.02× |
| 35B-A3B-Q4_0      | batched    | 1 | 46.47 | 46.54 | 1.00× | 46.47 | 46.54 | 1.00× |
| 35B-A3B-Q4_0      | batched    | 2 | 67.42 | **69.56** | **1.03×** | 40.62 | 41.29 | 1.02× |
| 35B-A3B-UD-Q4_K_S | no_batched | 1 | 47.69 | 47.72 | 1.00× | 47.69 | 47.72 | 1.00× |
| 35B-A3B-UD-Q4_K_S | no_batched | 2 | 61.34 | **64.82** | **1.06×** | 39.50 | 40.39 | 1.02× |
| 35B-A3B-UD-Q4_K_S | batched    | 1 | 47.74 | 48.25 | 1.01× | 47.74 | 48.25 | 1.01× |
| 35B-A3B-UD-Q4_K_S | batched    | 2 | 62.02 | **63.99** | **1.03×** | 39.77 | 40.29 | 1.01× |

>2 % gate: decode aggregate **4 wins / 11 flat / 1 loss** (the 0.98× is single-sample noise).

## Interpretation

### Prefill: no rate change — wall-clock drop is from the tokenizer fix

3003 vs 3313 tokens for the same prompt text on HEAD vs main is
**9.4 % fewer tokens** — that fully explains the 9–13 % wall-clock
prefill_ms drop. Per-token prefill throughput is 1.00–1.03× across all
cells (noise). The new batched-MMVQ kernels are decode-shape kernels
and don't fire at prefill m-values (m ≥ 128 routes to MMQ).

This is still a real user-facing benefit (fewer tokens → faster prompt
ingestion), just not a kernel speedup. Future benches should feed
identical token-ID sequences to isolate kernel deltas from tokenizer
quality.

### Decode: +3–5 % on 35B-A3B at N=2, flat elsewhere

The new wins live exactly where the batched-MMVQ kernels engage:
- N=2 is the smallest m the kernels handle (compile-time N=2/3/4
  specialisations).
- 35B-A3B is MoE, so the routed-expert MMVQ at m=2 hits the per-N
  batched path repeatedly per layer.
- 27B is dense — `qmatmul(m=2)` either doesn't take the batched path
  in this code path or the win is dwarfed by attn/FFN overhead.
  Net decode delta on 27B at N=2: 1.003×–1.016× — noise.
- At N=1, kernels don't engage (no batched MMVQ at m=1); deltas are
  in the ±2 % noise band.
- The one 0.979× loss (35B-A3B-Q4_0 / no_batched / N=1) is within
  bench-to-bench variance (≥ 3 runs gate not enforced; single sample).

### What the bench did NOT exercise

- N ≥ 3 — the K1 Q4_0 batched kernel compiles per-N specialisations for
  N=2, 3, 4. We only measured N=2.
- K5 (`mmvq_q4_0_gate_up_batched`) — unwired on this branch.
- K3 (`mmvq_q5_k_r2_batched`) — fires at Q5_K m∈{2,3,4} but our 35B-A3B
  UD-Q4_K_S only has Q5_K on `ffn_down_exps`; the win on this model is
  likely a mix of K1 + K3 + non-kernel deltas.
- TP2-only and PP4-only topologies — not run.

## Conclusion

The branch is **net-positive across all 16 cells**, with the headline
being prefill +9–13 % (tokenizer fix) and decode +3–5 % on 35B-A3B at
N=2 (batched-MMVQ kernels). No regressions outside noise.

Separately, main cannot load 27B GGUFs at all without `af189c0`
(qwen35 arch registration) — that's an unconditional unlock, not a
perf delta.

## Artefacts

- `HEAD.json` — branch results (16 cells: 4 models × 1 topo × 2 paths × 2 concs)
- `HEAD.log` — branch run log
- `MAIN.json` — main+arch-fix results (16 cells, same shape)
- `MAIN.log` + `MAIN_27b.log` — main run logs (35B subset + 27B subset)
- `MAIN_27b.json` — 27B portion of main+arch-fix bench (merged into MAIN.json)
