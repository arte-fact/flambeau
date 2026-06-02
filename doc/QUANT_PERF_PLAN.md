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

## Phase 1 — Bench matrix (sweep, no kernel work) — ✅ DONE 2026-06-01

Disk-resident quants only (Q4_K_M-mtp excluded — MTP head not loaded
by qwen35 arch; replace with non-MTP Q4_K_M in Phase 2).

```
quants:  Q4_0, Q4_1, Q8_0
         + UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
kv:      f16, q8
ctx:     512, 4k
topo:    pp2tp2
```

28 cells, all green. Cert at
`certs/perf/quant_matrix_qwen36_27b_2026_06_01.{json,md}`. Bench
script: `scripts/bench/quant_perf_matrix.py`. 1×MI50 deferred (these
are 16 GB cards, not 32 GB — most quants don't fit single-GPU at
ctx_cap ≥ 4 k).

**Outliers (< 0.7× Q4_0 fraction):**

| Quant       | ctx 4k F16 tps | vs Q4_0 | % HBM ceiling |
|-------------|---------------:|--------:|--------------:|
| UD-Q3_K_XL  |           9.54 |   0.29× |          4 %  |
| UD-Q4_K_XL  |          13.72 |   0.42× |          8 %  |

Q8 KV pattern (uniform 0.80-0.93× of F16 across all quants at ctx
4 k) is consistent with the gemma4-31B-Q4_0 finding — Q8 KV's
per-call floor dominates below ctx ≈ 14 k. Use F16 KV for this
ctx range; Q8 KV pays off above.

## Phase 2 — Fetch missing quants (deferred)

Q3_K_S/M, Q4_K_S (no MTP), Q5_K_S/M, Q6_K (non-UD), IQ4_NL, IQ4_XS,
UD-Q2_K_XL, UD-Q5_K_XL, UD-IQ2_M, UD-IQ3_XXS. ~120 GB download.
**Deferred** — Phase 3 already has clear direction from rocprofv3
traces; the missing quants would all hit the same dp4a gap that
Phase 3 closes. Re-run the matrix after Phase 3 lands.

## Phase 3 — Dense K-quant + IQ-quant dp4a port — IN PROGRESS

### Diagnosis (✅ DONE 2026-06-02)

rocprofv3 kernel-trace harness:
`scripts/profile/trace_ud_q{3,4}_k_xl.sh`. Single-rank pp2 trace
under `--kernel-trace`, 1 prefill + 3 decode warm steps, dump CSV.
Findings filed at `[[dense-q4-k-r2-scalar-fp32]]`:

| Trace            | Top kernel(s) (% wall)                              |
|------------------|-----------------------------------------------------|
| UD-Q4_K_XL pp2   | mmvq_q4_k_r2 72.6 % @ 384 us/call                   |
| UD-Q3_K_XL pp2   | mmvq_q3_k_r2 51.7 % + mmvq_q4_k_r2 26.6 % + mmvq_iq4_xs_r2 13.9 % = **95.2 %** scalar-FP32 r2 |

Reference: `mmvq_q5_k_dp4a` 100 us/call, `mmvq_q6_k_dp4a` 60 us/call
on the same model on the same GPU. Q5_K and Q6_K got the dp4a
treatment in V1.6 / V2.3.d.1; the rest of the family didn't.

**Family-wide gap.** Every dense `mmvq_<quant>_r2_q8_1` except Q5_K
and Q6_K does per-element scalar FP32 multiplies. MoE has dp4a
variants for all of them (`indexed_moe_mmvq_<quant>_r2_dp4a`) —
copy-paste-edit templates.

### Full dp4a coverage inventory

**Already shipped:**

| Quant   | Status                                                          |
|---------|-----------------------------------------------------------------|
| Q4_0    | runtime intercept (`mmvq_q4_0_dp4a` via Recipe::from_impl_id)   |
| Q4_1    | runtime intercept                                               |
| Q5_K    | dense `mmvq_q5_k_dp4a.cu` (V1.6)                                |
| Q6_K    | dense `mmvq_q6_k_dp4a.cu` (V2.3.d.1)                            |
| Q8_0    | runtime intercept (`mmvq_q8_0_dp4a_vdr2`)                       |

**Missing (port targets, 11 dense kernels):**

| Slice | Quant    | Template source                                       | UD wall share              | Difficulty | Affected models                       |
|-------|----------|-------------------------------------------------------|----------------------------|------------|---------------------------------------|
| 3a    | **Q4_K** | MoE `indexed_moe_mmvq_q4_k_r2_dp4a.cu` ✓              | 72.6 % (Q4_XL), 26.6 % (Q3_XL) | easy (copy MoE) | **most-used quant**: Q4_K_M/S, UD-Q4_K_XL, every Q4_K base in qwen/llama/mistral |
| 3b    | **Q3_K** | from scratch — no MoE template                        | 51.7 % (Q3_XL)             | medium (3-bit + 6 subscales) | Q3_K_M/S, UD-Q3_K_XL, UD-Q2_K_XL (Q2 contains Q3 sub-blocks) |
| 3c    | IQ4_XS   | from scratch — port llama.cpp `vec_dot_iq4_xs_q8_1`   | 13.9 % (Q3_XL)             | medium (LUT pre-apply) | Unsloth IQ4_XS (every model), some UD-Q4_K_XL share |
| 3d    | IQ4_NL   | from scratch — same shape as IQ4_XS minus sub-block scale | n/a in Phase 1         | medium      | Unsloth IQ4_NL (every model)          |
| 3e    | Q2_K     | from scratch — 2-bit blocks + 4-bit min scales        | n/a in Phase 1             | medium      | UD-Q2_K_XL, Q2_K (very-low-VRAM)      |
| 3f    | IQ3_S    | from scratch — 8-elem signed codebook lookup          | 3.0 % (Q3_XL)              | hard (codebook in constant mem) | Unsloth IQ3_S                         |
| 3g    | IQ3_XXS  | from scratch — 8-elem unsigned codebook + sign bits   | n/a in Phase 1             | hard        | UD-IQ3_XXS, Unsloth IQ3_XXS           |
| 3h    | IQ2_S    | from scratch — 8-elem 2-bit codebook                  | n/a in Phase 1             | hard        | Unsloth IQ2_S                         |
| 3i    | IQ2_XS   | from scratch — 8-elem 2-bit codebook                  | n/a in Phase 1             | hard        | Unsloth IQ2_XS                        |
| 3j    | IQ2_XXS  | from scratch — 8-elem 2-bit codebook + signs          | n/a in Phase 1             | hard        | UD-IQ2_XXS, UD-IQ2_M                  |
| 3k    | IQ1_S    | from scratch — 1-bit + 3-bit scale                    | n/a in Phase 1             | hard        | Unsloth IQ1_S (extreme low-VRAM)      |
| 3l    | IQ1_M    | from scratch — IQ1_S variant                          | n/a in Phase 1             | hard        | Unsloth IQ1_M                         |

**Out of scope:** Q5_0, Q5_1 — legacy format, almost no model in the
wild ships these as the headline quant. Leave on scalar until a
real consumer surfaces.

### Difficulty rubric

- **easy** (copy MoE template): 1-2 hours per slice. Strip expert
  index, identical dp4a + 1-FMA-per-superblock body. Only Q4_K.
- **medium** (port llama.cpp `vec_dot_<quant>_q8_1_impl`): 3-5 hours
  per slice. Read llama.cpp reference (`ggml-cuda/vecdotq.cuh`),
  match the per-block dequant pattern, adapt to flambeau's 256-thread
  / wave64 launch shape, sweep cert. Q3_K, IQ4_*, Q2_K.
- **hard** (codebook + signs in constant memory): full session per
  slice. IQ-quants 2/3/1 use 8-element signed codebooks loaded into
  constant memory at module init, plus per-block sign bits packed
  with the quant. The dp4a path requires codebook-aware Q→int8
  pre-expansion in LDS, then standard dp4a. Pattern shared across
  IQ3_S, IQ3_XXS, IQ2_*, IQ1_* — pay the cost once on IQ3_S, then
  the rest inherit the helper.

### Per-slice steps

1. Mirror or write `crates/kernels-hip/src/kernels/mmvq_<quant>_r2_dp4a.cu`.
2. Register the impl_id in `crates/backend-hip/src/impls.rs`.
3. Swap the gfx906.toml dispatch row for the new impl.
4. Run `flambeau sweep --arch gfx906 --impl <new>` against the
   existing 15-shape grid.
5. Re-bench the affected quant on Qwen3.6-27B pp2tp2 to
   measure end-to-end lift.
6. Commit + cert in one go (rule: "commit when code actually ships").

### Acceptance per slice

- Sweep green (matches the kernel's existing cert dtype).
- End-to-end re-bench on Qwen3.6-27B pp2tp2 hits **≥ 1.5×** the
  pre-port number. Below that, leave the kernel behind
  `cfg(unverified)` with a one-line diagnosis.
- No regression on any *other* quant in the Phase 1 matrix.

### Q5_K / Q6_K MMQ-turbo upgrade (optional 3h)

Q5_K and Q6_K dp4a are already shipped but MMQ-turbo is not. At
prefill m ≥ 128 the turbo kernel beats dp4a. UD-Q6_K_XL prefill TTFT
is ~13 s at ctx 4 k — turbo port plausibly halves that. Defer until
3a-3c land and prefill becomes a measurable share of the wall.

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

1. ~~Phase 1~~ ✅ DONE — 28-cell pp2tp2 matrix shipped 2026-06-01.
2. ~~Phase 3 diagnosis~~ ✅ DONE — rocprofv3 traces identify a
   family-wide dp4a gap across every K-quant + IQ-quant except
   Q5_K, Q6_K, Q8_0.
3. **Phase 3a: Q4_K dp4a port** — easy (MoE template). Lifts
   UD-Q4_K_XL 2.2× + UD-Q3_K_XL 1.4× + every Q4_K_M base model.
4. **Phase 3b: Q3_K dp4a port** — medium (no template, port from
   llama.cpp). Composed with 3a, lifts UD-Q3_K_XL ~3×, UD-Q2_K_XL.
5. **Phase 3c-e (medium): IQ4_XS, IQ4_NL, Q2_K.** Each 1 session.
6. **Phase 3f-l (hard): IQ3_S, IQ3_XXS, IQ2_S/XS/XXS, IQ1_S/M.** Pay
   the codebook-in-constant-memory cost once on IQ3_S; the rest are
   ~half a session each inheriting the helper.
7. Phase 2: download missing K-quant variants once 3a-3b land —
   they'll inherit the dp4a kernels and won't need a separate
   diagnosis pass.
8. Phase 4: KV-quant pairings re-validated after Phase 3a-3b (decode
   wall composition changes once K-quant kernels stop dominating).
9. Phase 5: decision table — final cert, plan close.

Total remaining surface: ~12-15 sessions if all 11 quant ports land
(realistic: 3a-3e are the productive core, 3f-l are the IQ tail
landed only as user demand surfaces). Phase 4 and 5 are 1 session
each.

## Acceptance criteria

- Every dispatch-table row for Qwen3.6-27B has an end-to-end perf
  number on at least one (KV, ctx, topo) triple.
- No quant is silently regressed vs its bandwidth ceiling by
  > 30 % without a `cfg(unverified)`-style explanation.
- Decision table at `doc/QUANT_FIT_TABLE.md` exists and is current.
