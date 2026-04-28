# Qwen3-Coder-Next-80B (qwen3next) — flambeau V1 final perf cert

**Status: shipping. flambeau is the only stack running this model on
the 4× MI50 rig** (llama.cpp + llamacpp-turbo crash on 4-rank
qwen3next per upstream issue ggml-org/llama.cpp#19518/#19893; the
user's working invocation uses `llama-server` with turbo3 KV +
`split_mode=layer`, which sidesteps the upstream peer-DMA bug the
same way flambeau's `peer_copy_via_host` PP path does — but that
config is not directly comparable to flambeau bench numbers).

This cert aggregates the CN-80B (#127–#134) work — TP loader gaps
closed, topology baselines measured, four perf iterations executed
and documented honestly.

## Headline (Coder-Next-Q4_0.gguf, 4× MI50, 100 W cap, ROCm 7.1.1)

| topology  | load s | pp128  | pp512  | pp2048 | tg64  |
|-----------|-------:|-------:|-------:|-------:|------:|
| pp4 (V1 cert, pre-CN-80B work) | 83 |  ~440 |  538.0 | 560.0 | 40.0 |
| pp4 (post-iter-4)              | 81 | 436.5 | **568.0** | **598.0** | 41.2 |
| pp2tp2 (first measurement, post-CN-80B-1) | 102 | 164.9 | 179.9 | 181.9 | **46.2** |

- **pp4 prefill +5.7 % to +7.6 %** vs the pre-CN-80B baseline.
  Decode flat (within rep variance).
- **pp2tp2 wins decode** by +12 % vs pp4 (46.2 vs 41.2 tg64) at the
  cost of −67 % prefill — same shape as 35B-A3B. Best-topology
  recommendation: pp4 for batch / prefill-heavy, pp2tp2 for chat /
  decode-heavy.
- tp2 OOMs (80 GB / 2 ranks won't fit 16 GB cards). tp4 inaccessible
  per the 2↔3 BAR1 fault.

## Per-task summary (CN-80B-1 → CN-80B-7)

| # | Task | Result |
|---|------|--------|
| CN-80B-1 #127 | TP loader gaps (MXFP4 + ssm_ba split) | **Shipped.** Coder-Next now loads on tp2 / pp2tp2 (was pp4-only). |
| CN-80B-2 #128 | pp2tp2 forward smoke for qwen3next | **Pass.** Forward ran on first try post-CN-80B-1, no qwen3next-specific TP gaps. |
| CN-80B-3 #129 | Topology baseline bench | **Captured.** Numbers above. |
| CN-80B-4 #130 | Iter 1 — profile + lever + A/B | **Inconclusive (rocprofv3 4-rank SIGABRT).** No code change. Bonus: `FLAMBEAU_VARIANT=baseline` A/B confirmed DP4A is +27 % decode lever (default kept). |
| CN-80B-5 #131 | Iter 2 — HipEvent profiler + section breakdown | **Infrastructure shipped.** New `flambeau_backend_hip::profile` + intra-`forward_one_token_pp` markers. First profile: layer chain owns 93 % of decode wall. |
| CN-80B-6 #132 | Iter 3 — F16 router weight | **+5.5 to +7.6 % prefill, decode flat.** Only material perf shift across the iters. New `dense_gemv_f16_f16{,_batched}` kernels + load-time F32→F16 conversion + dispatch. F16 parity bit-exact. |
| CN-80B-7 #133 | Iter 4 — intra-GDN profile + Q4_0 qkv+gate fuse | **Null perf, kept on code-quality.** Closes a documented dispatch gap; future Q4_0 GDN models inherit fuse. Intra-GDN markers in-tree for future iters. |

## Detailed iter trail

### Iter 3 (the win)

Hot-spot: router `dense_gemv_f32_f16` runs in every layer (1 536
calls in the bench), F32 weight = 4 MiB HBM read per layer per token.
Lever: switch router weight to F16 at load — halves HBM read.
Quality preserved (F16 parity bit-exact: 35B-A3B prefill L=1 → 11 /
L=2 → 271 vs llama.cpp). Win scales with prefill L: pp2048 +7.6 %.
Decode (L=1) flat — per-call HBM saving too small to amortise.

### Iter 4 (the kept-but-null)

Profile #1: `gdn_ssm_out` at 30.5 % of GDN wall — Q5_K MMVQ on
`[2048, 4096]`, already on the optimal `mmvq_q5_k_r2` variant; per-
call wall is ~12× HBM ceiling, dominated by Q5_K dequant compute.
No quick lever.
Profile #2: `gdn_proj_qkv_gate` at 18.5 % — Coder-Next is Q4_0, but
`forward_gdn_decode`'s fuse-into-`mmvq_*_gate_up` check only fired
for Q8_0. Wired up the Q4_0 path. Null perf (Q4_0 weight HBM
dominates; the 2 KiB activation save per call is microscopic) but
correctness gain — `mmvq_q4_0_gate_up`'s own docstring names this as
the canonical target.

## Path to V2 (deferred — the real headroom)

`gdn_ssm_out` (Q5_K MMVQ) is 30 % of GDN ≈ 12 % of total decode wall
and is the obvious next target. Levers (V2 work):

1. **Q5_K MMVQ kernel rewrite** — LDS scale caching, DPP-merge across
   rows, possibly a multi-row variant tuned for the Coder-Next
   `[2048, 4096]` shape. Estimate: 1–2 sessions.
2. **Q5_K → Q8_0 conversion at load for `ssm_out`** — same trick as
   the iter-3 router. Q8_0 has ~50 % more HBM read (34 vs 23 byte/
   block) but DP4A-friendly per-byte unpack vs Q5_K's bit-fiddle.
   May trade compute for HBM in the right direction. Estimate:
   1 session including a dequant→Q8_0 host conversion.
3. **`attention_decode_q8_kv_splitk`** for Q8 KV (not Coder-Next-
   specific but the path-to-positive on the Q8 KV bench from
   V1-BENCH-#118).

Not blocked: TP loader F16 router conversion (currently only the PP
loader does the conversion — pp2tp2 still uses F32 router). Easy
follow-up if pp2tp2 perf wants the same iter-3 lift.

## What's *not* in this cert

- **No llama.cpp comparison.** Per `project_qwen3next_coder_next_80b`
  memory: stock llama.cpp + llamacpp-turbo both crash loading this
  GGUF on 4× MI50 in our flambeau bench harness. The user's
  `llama-monitor` invocation works (different binary path + turbo3
  KV + `split_mode=layer` env), but it's not directly comparable to
  flambeau's bench numbers without harness work; recorded as a
  follow-up rather than mis-stated as parity.
- **No quality cert.** F16 parity preserved bit-exact for 35B-A3B
  (the canonical V1.7.4 reference); a Coder-Next-specific delta-ppl
  cert is a V2 follow-up if/when wikitext-2 ppl infra lands.

## Files / commits

- 159845d  CN-80B-1+2: TP loader MXFP4 + ssm_ba split
- 86f5d04  CN-80B-4 (iter 1) — rocprofv3 4-rank crash, iter pivots
- ce2e828  iter-1 supplementary: FLAMBEAU_VARIANT=baseline A/B
- 1737e1e  CN-80B-5 (iter 2) — HipEvent profiler + breakdown
- c32608d  CN-80B-6 (iter 3) — F16 router weight (+5–8 % prefill)
- c4fb588  CN-80B-7 (iter 4) — intra-GDN profile + Q4_0 qkv+gate fuse

Plus the topology baseline cert at
`certs/perf/v1_bench_matrix/qwen3_coder_next_80b.json`.

## Closes

CN-80B-8 #134 — aggregate cert.
The Coder-Next-80B "full support + 4 iter" arc is **complete**.
