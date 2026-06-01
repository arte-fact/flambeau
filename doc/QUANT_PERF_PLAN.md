# Quant × perf × VRAM-fit plan — Qwen3.6-27B reference model

**Goal.** Make every flambeau-supported quant a first-class production
target. Today Q4_0 is the hot path everyone measures; Q4_K_M is the
canonical K-quant in the field; UD-2.0 XL series is what Unsloth
recommends. The rest are silently slower or untested at the end-to-end
level. This plan turns the dispatch matrix's quant coverage (broad)
into measured perf evidence (sparse → comprehensive) and ships a
per-VRAM-budget recommendation table users can act on.

**Reference model.** `unsloth/Qwen3.6-27B-GGUF` — dense, mature
implementation in flambeau, MTP available, the most-released quant
catalogue any modern model offers (26 variants). Pick once, drive
the whole sweep.

**Target hardware.** gfx906 (MI50). Three configurations matter:
- 1× MI50 32 GB → which quants fit at usable ctx?
- 2× MI50 (tp2) → Q5/Q6/Q8 budget opens up
- 4× MI50 pp2tp2 → reference highest-throughput config

CUDA dev card (sm_86, RTX 3090, 24 GB) is the secondary target. CUDA
kernels inherit from the same `Op` trait + dispatch table — perf
results may differ but quant coverage should be identical.

---

## Surface area

### Quant axis

Three families with different perf characteristics:

| Family   | Members on disk      | Members to fetch        | Dispatch rows (gfx906) |
|----------|----------------------|-------------------------|-----------------------:|
| Legacy   | Q4_0, Q4_1, Q8_0     | —                       | 4 / 2 / 4              |
| K-quant  | Q4_K_M (mtp)         | Q3_K_{S,M}, Q4_K_S, Q5_K_{S,M}, Q6_K | 2..6     |
| UD-2.0   | UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL, UD-Q8_K_XL | UD-Q2_K_XL, UD-Q5_K_XL, UD-IQ2_M, UD-IQ3_XXS | inherits K-quant rows |
| IQ       | —                    | IQ4_NL, IQ4_XS, IQ2_M, IQ3_XXS | 2 each (mmvq + mmq) |

UD-2.0 XL variants ship mixed precision per tensor (attn at higher,
ffn at lower). They use the same kernel set as the underlying
K-quants but in different ratios — perf may diverge from plain
K-quant of the same name.

### KV-cache axis

- F16 (baseline, reference quality)
- Q8 (~2× HBM saving on KV reads, ~0.5 % perplexity cost — cert
  exists for Qwen3.6-35B-A3B-UD-Q4_K_S)

### Context axis

512 (decode-dominated baseline), 4 k (typical chat), 16 k (long
context). Above 16 k starts hitting VRAM ceilings on 1× MI50 for
some quants — that's the point.

### Topology axis

- 1× MI50 (where it fits)
- 2× MI50 tp (devices 0,2)
- 4× MI50 pp2tp2 (devices 0,2,1,3) — canonical reference

---

## Gap budget per quant (predictive)

Per-token decode wall ≈ weights bytes / achieved-HBM-bandwidth.
Qwen3.6-27B weight sizes:

| Quant       | GB    | 1×MI50 ceiling @ 700 GB/s | pp2tp2 ceiling @ 4× |
|-------------|------:|--------------------------:|--------------------:|
| Q3_K_M      | 12.7  | 55 tps                    | 220 tps             |
| UD-Q3_K_XL  | 13.5  | 52 tps                    | 207 tps             |
| Q4_K_S      | 14.8  | 47 tps                    | 189 tps             |
| **Q4_0**    | 14.7  | 48 tps                    | 190 tps             |
| Q4_K_M      | 15.7  | 45 tps                    | 178 tps             |
| UD-Q4_K_XL  | 16.4  | 43 tps                    | 171 tps             |
| Q4_1        | 16.1  | 43 tps                    | 174 tps             |
| Q5_K_M      | 18.2  | 38 tps                    | 154 tps             |
| UD-Q5_K_XL  | 18.7  | 37 tps                    | 150 tps             |
| Q6_K        | 21.0  | 33 tps                    | 133 tps             |
| UD-Q6_K_XL  | 23.9  | 29 tps                    | 117 tps             |
| Q8_0        | 26.6  | 26 tps                    | 105 tps             |
| UD-Q8_K_XL  | 32.9  | 21 tps                    | 85 tps              |

These are upper-bound estimates (700 GB/s effective HBM, no
attention/sampler overhead, no inter-GPU comms). Measured Q4_0
pp2tp2 = 33.4 tps achieves 504 GB/s aggregate ≈ 0.18 of theoretical.
Real per-quant ceilings should track the same fraction.

**Q4_0 is the canary.** Achieves the highest per-byte rate of any
quant on the dispatch table. Every other quant should approach the
same fraction of its bandwidth ceiling, modulo the dequant-path cost.

---

## Phase 1 — Bench matrix (sweep, no kernel work)

End-to-end decode tps for the full quant × KV × topo product:

```
quants:  Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0
         + UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
kv:      f16, q8
ctx:     512, 4k
topo:    pp2tp2 + 1×MI50 (where it fits)
```

That's 9 quants × 2 KV × 2 ctx × 2 topo = 72 cells. Single-stream
greedy decode, 64 tg, 3-rep median. Bench script:
`scripts/bench/quant_perf_matrix.py`. Cert at
`certs/perf/quant_matrix_qwen36_27b_2026_06_XX.md`.

**Output.** Two tables:
1. tps per (quant, KV, ctx, topo) — raw measurement
2. % of per-quant bandwidth ceiling — normalised; flags outliers

Outliers are quants achieving < 0.7× the Q4_0 ceiling fraction.

## Phase 2 — Fetch missing quants

Q3_K_S/M, Q4_K_S, Q5_K_S, Q6_K (non-UD), IQ4_NL, IQ4_XS, UD-Q2_K_XL,
UD-Q5_K_XL, UD-IQ2_M, UD-IQ3_XXS. Extend Phase 1 matrix.

## Phase 3 — Per-outlier kernel work

For each quant flagged in Phase 1:
- Profile decode kernel with rocprofv3.
- Compare PMC against Q4_K (the best-tuned K-quant) — VGPR, waves/EU,
  MemBusy/Stall, VALUBusy.
- If the gap is the dequant arithmetic (LUT lookups for IQ-quants,
  Q5_K six-bit sub-block decode), port the multi-row DPP pattern
  that Q4_K uses (`mmvq_q*_K_nw1_r{2,4}` shape).
- If the gap is bandwidth-bound (Q6_K, Q8_0), look at MMQ-turbo /
  4-warp LDS-tiled at the m-shape that decode hits.
- Q5_K and Q6_K have only 2 dispatch rows each on gfx906 today —
  prime candidates for the multi-row treatment Q4_K got in V1.3.

## Phase 4 — KV-quant pairings

Today `--kv q8` is wired but only validated end-to-end on a few
models. For each quant in Phase 1, verify:
- correctness (Paris-smoke + delta-perplexity vs F16 KV)
- perf delta (long-ctx win when global layer scan dominates)

The gemma4 work landed a KV-quant tradeoff for head_dim=512 globals
(`feedback_q8_kv_head_dim_512_two_wave`) — Qwen3.6-27B head_dim=128
doesn't have the same shape, but the crossover-ctx analysis pattern
transfers.

## Phase 5 — VRAM fit decision table

Per (target VRAM, target ctx) cell, pick:
- best quant by quality-preserving-throughput
- KV layout
- topology

Output: a markdown table users can read in 10 seconds:
"Have 16 GB, want 32 k ctx? Use UD-Q3_K_XL + Q8 KV."

---

## What's NOT in this plan

- Adding new GGUF block types. Flambeau already supports every
  modern GGUF dtype.
- Quality work (perplexity sweeps, instruction-tuning eval).
  Trust Unsloth's quality cert; this plan measures *perf at given
  quant*.
- Quant porting on CUDA — same kernel work falls out from the same
  dispatch table; CUDA-specific tuning is a separate plan.
- gfx1031 portability — covered by the existing arch-canary process
  per rule 5; not part of this plan's primary axis.

---

## Order of operations

1. Phase 1 with quants already on disk (Q4_0, Q4_K_M, Q8_0,
   UD-{Q3,Q4,Q6,Q8}_K_XL) — covers the 5 most-used cells with zero
   download bandwidth. **First slice.**
2. Phase 2 download of the missing 10 — single rsync from HF, then
   extend the matrix.
3. Phase 3 picks the worst outlier and does kernel work. One quant
   per session.
4. Phase 4 KV-quant pairings layered on top.
5. Phase 5 decision table closes the plan.

Total surface: estimated 5–8 sessions depending on how many quants
need kernel work in Phase 3. Phase 1 alone is one session (~2 hours
of bench wall + summary).

## Acceptance criteria

- Every dispatch-table row for Qwen3.6-27B has an end-to-end perf
  number on at least one (KV, ctx, topo) triple.
- No quant is silently regressed vs its bandwidth ceiling by
  > 30 % without a `cfg(unverified)`-style explanation.
- Decision table at `doc/QUANT_FIT_TABLE.md` exists and is current.
