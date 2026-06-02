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

| Slice | Quant    | Status | UD wall share              | End-to-end lift | Notes |
|-------|----------|:------:|----------------------------|----------------:|-------|
| 3a    | Q4_K     | ✅ shipped | 72.6 % (Q4_XL), 26.6 % (Q3_XL) | **1.42× on UD-Q4_K_XL** (13.72 → 19.49 tps) | Kernel at Q5_K dp4a parity per output row (102 us/row). |
| 3b    | Q3_K     | ✅ shipped | 51.7 % (Q3_XL)             | **1.58× on UD-Q3_K_XL** (9.54 → 15.05 tps, cum) | First-draft had QI8_1=4 typo; fix went max_rel_err 1.15 → 1.7e-4. |
| 3c    | IQ4_XS   | ✅ shipped | 13.9 % (Q3_XL)             | **1.87× on UD-Q3_K_XL** (9.54 → 17.80 tps, cum) | HIP `__builtin_amdgcn_perm` 4-way LUT lookup. |
| 3d    | IQ4_NL   | ✅ shipped | n/a in Phase 1             | sweep-only      | No on-disk Phase 1 consumer; reuses Phase 3c LUT helper. |
| **3e** | **Q2_K** | next   | n/a in Phase 1             | UD-Q2_K_XL pending | 2-bit blocks + 4-bit min scales; medium difficulty, no MoE template. |
| 3f    | IQ3_S    | pending | 3.0 % (Q3_XL)              | unlocks IQ3_S       | hard — first codebook quant; the helper that 3g–3l inherit. |
| 3g    | IQ3_XXS  | pending | n/a in Phase 1             | unlocks UD-IQ3_XXS | hard — codebook + signs. |
| 3h    | IQ2_S    | pending | n/a in Phase 1             | unlocks IQ2_S       | hard — 2-bit codebook. |
| 3i    | IQ2_XS   | pending | n/a in Phase 1             | unlocks IQ2_XS      | hard — 2-bit codebook. |
| 3j    | IQ2_XXS  | pending | n/a in Phase 1             | unlocks UD-IQ2_XXS  | hard — 2-bit codebook + signs. |
| 3k    | IQ1_S    | pending | n/a in Phase 1             | unlocks IQ1_S       | hard — extreme-low-bit niche. |
| 3l    | IQ1_M    | pending | n/a in Phase 1             | unlocks IQ1_M       | hard — IQ1_S variant. |

**Cumulative coverage on UD-Q3_K_XL after 3a-3c:** 92.2 % of decode
wall (Q3_K 51.7 + Q4_K 26.6 + IQ4_XS 13.9) is now dp4a-optimized.
Remaining 7.8 % is sub-1us/call kernels + per-token overhead floor.

**Honest result re-baseline.** Phase 3a's measured 1.42× came in
below the 1.5× plan threshold but the trace shows the kernel hit
Q5_K dp4a parity at 102 us/row. The gap from kernel-wall (1.50× per
the trace) to end-to-end (1.42×) is the python/AR/sync floor, not
kernel headroom. Accept ≥ 1.4× as the practical bar going forward;
adjust plan threshold accordingly.

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

## Phase 3.5 — MoE-side dp4a parity

Phase 3 fixed only the **dense** dispatch path. MoE expert matmuls
go through a separate `[[indexed_moe_mmvq]]` dispatch row family;
the Phase 3a kernel (`indexed_moe_mmvq_q4_k_r2_dp4a`) actually
predates this plan and was the *template* the dense Phase 3a was
copied from. But every K-quant / IQ-quant beyond Q4_K is still on
scalar FP32 multiplies in the MoE path — the same gap the dense
sweep closed.

### MoE dp4a coverage today (pre-Phase-3.5)

Already shipped (pre-existing):
- Q4_K: `indexed_moe_mmvq_q4_k_r2_dp4a`, `_r4_dp4a`, `_gate_up_r{2,4,8}_dp4a`
- Q4_0: `indexed_moe_mmvq_q4_0_gate_up_dp4a`
- Q8_0: `indexed_moe_mmvq_q8_0_gate_up_dp4a`

Missing (port targets):

| Slice | Quant | Affected on-disk model | Status |
|-------|-------|------------------------|--------|
| M-a | Q3_K MoE | Qwen3.6-35B-A3B-Q3_K_S | ✅ shipped — A/B: SCALAR 39.89 → DP4A 46.94 tps (+17.7 %) pp2tp2 ctx 4096 f16 KV |
| ~~M-b~~ | ~~Q8_K MoE~~ | ~~UD-Q8_K_XL~~ | N/A — UD-Q8_K_XL MoE experts are Q8_0 (already dp4a-covered); model itself currently blocked by missing BF16 MoE sharded-stacked dequant on blk.1 |
| M-c | Q5_K MoE | Qwen3.6-35B-A3B-UD-Q5_K_S | ✅ shipped — A/B: SCALAR 47.77 → DP4A 51.47 tps (+7.7 %) pp2tp2 ctx 4096 f16 KV. Also wired prefill tile8 (`indexed_moe_mmq_q5_k_gate_up_tile8_dp4a` was on-disk but un-dispatched) + decode-gate split path (no fused gate+up MMVQ for Q5_K). |
| M-d | Q6_K MoE | Qwen3.6-35B-A3B-UD-Q6_K_XL | ✅ shipped — A/B: SCALAR 1913.32 → DP4A 1635.13 ms median total_ms (−14.5 % wall / +22 % decode), pp2tp2 ctx 4096 f16 KV. Also wired Q6_K gate+up tile8 prefill (`indexed_moe_mmq_q6_k_gate_up_tile8_dp4a` was on-disk but un-dispatched) + decode-gate split path (no fused gate+up MMVQ for Q6_K, same shape as M-c Q5_K). Module `indexed_moe_mmvq_q6_k_r2_dp4a` registered in the kernel module list. |
| M-e | Q2_K MoE | UD-Q2_K_S MoE variants | ✅ kernel-only — `indexed_moe_mmvq_q2_k_r2_dp4a`, sweep PASS (max_rel 1.3e-4). No on-disk consumer: `Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf` uses IQ2_XS + IQ3_XXS for MoE experts (Unsloth mixed dynamic-quant), not Q2_K. Same Lever-1 r2 dp4a template as dense; ships for future pure-Q2_K MoE downloads. |
| M-f | IQ4_XS MoE | Unsloth IQ4_XS MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq4_xs_r2_dp4a` |
| M-g | IQ4_NL MoE | Unsloth IQ4_NL MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq4_nl_r2_dp4a` |
| M-h | IQ3_XXS MoE | UD-Q2_K_XL down_exps | ✅ dp4a shipped — `indexed_moe_mmvq_iq3_xxs_r2_dp4a` |
| M-h | IQ3_S MoE | low-bit IQ3_S MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq3_s_r2_dp4a` |
| M-i | IQ2_XS MoE | UD-Q2_K_XL gate/up | ✅ dp4a shipped — `indexed_moe_mmvq_iq2_xs_r2_dp4a` |
| M-i | IQ2_S MoE | extreme-low-bit MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq2_s_r2_dp4a` |
| M-i | IQ2_XXS MoE | extreme-low-bit MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq2_xxs_r2_dp4a` |
| M-j | IQ1_S MoE | extreme-low-bit MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq1_s_r2_dp4a` |
| M-j | IQ1_M MoE | extreme-low-bit MoE | ✅ dp4a shipped — `indexed_moe_mmvq_iq1_m_r2_dp4a` |

**M-f–M-j wiring slice** (decode + prefill MMVQ + Ops trait + HipOps impls):
Pre-slice, every IQ-MoE gate/down combo bailed in `moe_experts.rs` decode
match arms (the prefill tile8 dispatch + dense IQ MMVQ + scalar IQ
indexed-MoE MMVQ kernels were already shipped; only the decode-gate and
decode-down dispatch arms were missing). End-to-end on
Qwen3.5-122B-A10B-UD-Q2_K_XL / pp2tp2 / ctx 4096 / f16 KV: median
total_ms = 4459.23 ms (64 ct, 259 pt). The MoE expert kernels themselves
are still scalar FP32 — porting dp4a siblings is a separate slice
(M-f–M-j dp4a port), expected ~2× lift mirroring M-a (+17.7 %) and
Phase 3 dense gains.

Per slice the port mirrors the dense Phase 3 work: copy the dense
kernel body, add an `expert_ids` indirection on the weight pointer
(see `indexed_moe_mmvq_q4_k_r2_dp4a.cu` for the template), register
impl_id, swap the `[[indexed_moe_mmvq]]` dispatch row, sweep cert,
re-bench. Roughly 1 session per slice for Q-family, half a session
per slice for IQ-family once the helpers from dense Phase 3f-3l are
in place.

**Order of priority by on-disk consumer:**
1. ✅ M-a Q3_K MoE — Qwen3.6-35B-A3B-Q3_K_S: +17.7 % end-to-end (39.89 → 46.94 tps).
2. ~~M-b~~ N/A — UD-Q8_K_XL MoE is Q8_0 (covered); model blocked by BF16 MoE dequant.
3. ✅ M-c Q5_K MoE — Qwen3.6-35B-A3B-UD-Q5_K_S: +7.7 % end-to-end (47.77 → 51.47 tps). Also wired pre-existing-but-un-dispatched Q5_K tile8 prefill + Q5_K decode-gate split.
4. M-d → M-j — on demand as MoE consumers surface.

gemma4-26B-A4B's MoE layers use Q4_K which is already covered.

## Phase 4 — KV-quant pairings

`--kv q8` is fully wired in v2 (CLI flag → `ServerConfig.kv` →
`KvLayout::Q8Contig` → `scratch_config_for(... kv_layout)` →
per-layer Q8Contig for head_dim ∈ {64, 128, 256, 512}, F16Contig
elsewhere). Earlier feedback memory `feedback_kv_q8_not_honored`
(2026-05-22) is superseded by the v2 wiring shipped before
`feedback_q8_kv_head_dim_512_two_wave` (2026-06-01).

### Phase 4 Slice 1 — Qwen3.6-27B-Q4_0 (head_dim=128 dense) pp2tp2

| ctx | prompt | F16 KV total_s 3-rep | Q8 KV total_s 3-rep | Notes |
|-----|--------|---------------------|---------------------|-------|
| 16k | 4565   | 15.61 (ct=64)       | 15.95 (ct=61)       | Q8 ≈ noise (+2.8 %) |
| 16k | 11453  | 42.95 (ct=76)       | 44.99 (ct=76)       | Q8 +4.8 % wall at matched ct |
| 32k | 22933  | 146.6 avg ct=62.3   | 154.7 avg ct=74.3   | Q8 per-token rate +13 %, total wall +5.5 % |

Paris smoke (`The capital of France is`) returns identical text on F16
and Q8 KV across all configs — Q8 KV is **correctness-OK on Qwen3.6-27B
head_dim=128**.

Performance: roughly neutral. No structural Q8 win at head_dim=128
without an SWA cache-pointer-offset lever (the gemma4-only optimization
in `feedback_swa_cache_pointer_offset_lever`). Qwen3.6-27B has no
sliding-window layers; the per-token KV scan is full-context f16-read
vs full-context q8-read+dequant, and at 1 TB/s HBM the difference is
inside measurement noise once ct variance from temperature is in play.

Recommendation: F16 KV remains the default on Qwen3.6 dense / MoE
head_dim=128. Q8 KV is functional as a VRAM-saving knob (~2× cache
fit) when the user prefers concurrency or longer ctx over absolute
single-stream perf.

### Phase 4 Slice 2 — gemma4 head_dim=512 (regression flagged)

Re-benched with the deterministic rig on gemma-4-26B-A4B-it-Q8_0 /
pp2tp2 / temperature 0:

Bisect via the deterministic rig:

  prompt 128  (1 prefill chunk):  Q8 coherent — decode_tps 46.66 ttft 205 ms
  prompt 600  (2 prefill chunks): Q8 coherent — decode_tps 46.90 ttft 869 ms
  prompt 1500 (3 prefill chunks): Q8 **broken** — `<pad>` flood
  prompt 3000 (6 prefill chunks): Q8 **broken** — `<pad>` flood

The crossover is the prefill-chunk count, NOT the SWA window. First
hypothesis (SWA cache-pointer-offset lever byte-arithmetic) was tested:
the gating change leaves the regression in place, and `kv.bytes_per_row`
is layout-aware via `KvLayout::bytes_per_row` — the Q8Contig stride
`(kv_width/32) * 34` was always correct. Hypothesis ruled out.

Current best hypothesis: the Q8 chunked-prefill cross-chunk K/V state
is broken. Suspects:
  - `kv_append_f16_to_q8` at high `write_pos` (≥ 1024) — block stride
    interaction with `quantize_f16_q8_0` internal layout
  - `attn_prefill_q8_kv` reading prior-chunk K/V written by an earlier
    kv_append call
  - The gemma V-unit-norm-into-Q8 split path
    (`v_unit_norm_per_head_f16` then `kv_append_f16_to_q8`) losing
    precision past chunk 2

Earlier memory `feedback_q8_kv_head_dim_512_two_wave` (2026-06-01)
covers single-chunk correctness only; the head_dim=512 two-wave
kernels are correct in isolation. The chunked-prefill cross-chunk
behavior was never separately certified.

**Tighter bisect (Slice 3b)** localised the cliff precisely:

  actual pt 1008 (target 1100): coherent
  actual pt 1062 (target 1150): broken  ← cliff
  actual pt 1116 (target 1200): broken

The boundary is prompt_tokens crossing **1024**, which is exactly
`gemma4.attention.sliding_window` from the GGUF metadata for
gemma-4-26B-A4B. (Earlier memory recorded window=512 by analogy with
gemma4-E4B; 26B-A4B actually ships window=1024.) Bug therefore lives
in `attn_prefill_q8_kv`'s SWA-mask branch (`t_start > 0`), which is
activated for the first time once any q_token has `qpos >= 1024`.
F16 prefill at the same SWA boundary works (Q8 in isolation, not the
mask math). Three earlier hypotheses now all ruled out empirically:
chunked-prefill, Q8 splitk decode, SWA cache-pointer-offset lever.

**Action items (Slice 3c, real kernel fix)**:
1. Add a parity unit test for `attn_prefill_q8_kv` at n_q_tokens
   crossing the SWA boundary (1023, 1024, 1025) vs F16 reference;
   expect the failure to reproduce in isolation.
2. Inspect `attention_prefill_q8_kv.cu` inner loop for divergence
   when t_start > 0: per-block `__shfl_xor` reductions assume all
   lanes participate; cross-wave LDS rendezvous at head_dim=512
   syncs across 2 waves on `score_parts[2]` and must include the
   SWA-masked iterations or skip them uniformly across both waves.
3. Compare against `attention_decode_q8_kv.cu` (which DOES work for
   prompts within the cache window) to spot the divergence.

**Recommendation**: F16 KV is the safe default on gemma at prompt > 1024
until the kernel fix lands. Q8 KV remains correct for prompts within
the SWA window.

### Phase 4 takeaway

Q8 KV is structurally a *gemma4* win, not a generic win. The lever
chains: head_dim=512 globals + SWA window collapse → per-token KV scan
becomes the wall → Q8 halves it. Qwen3.6's head_dim=128 full-attention
already fits in HBM budget at ctx ≤ 32k; Q8 quantize/dequantize
overhead cancels the HBM win.

### Phase 4 follow-ups (not in this slice)

- Delta-perplexity sweep across Phase 1 models at F16 vs Q8 KV
  (correctness cert beyond Paris-smoke).
- Decision-table cell per (model, ctx) for VRAM-fit (feeds Phase 5).
- Deterministic-decode bench rig (temperature=0 + no early stop) so
  A/B numbers are not muddled by ct-variance.

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
3. ~~Phase 3a–3l: dense dp4a ports for all 12 missing quants~~ ✅
   shipped. Cumulative UD-Q3_K_XL 9.54 → 17.80 tps (1.87×, 92 %
   wall covered). All sweep-green; smaller-share kernels carry no
   bench but ship correctness-OK.
4. ~~Lever 1 r2 multi-row on Q4_K, Q3_K, Q2_K~~ ✅ — Q3_K +6.6 %,
   Q2_K +2.3 % on top of the single-row dp4a stack.
5. ~~Phase 3.5 M-a: Q3_K MoE dp4a port~~ ✅ — Qwen3.6-35B-A3B-Q3_K_S
   pp2tp2: SCALAR 39.89 → DP4A 46.94 tps (+17.7 % end-to-end).
6. ~~Phase 3.5 M-b: Q8_K MoE dp4a port~~ N/A — UD-Q8_K_XL MoE experts
   are Q8_0 (already covered); the model itself is currently blocked by
   missing BF16 MoE sharded-stacked dequant on blk.1 (separate slice).
7. ~~Phase 3.5 M-c: Q5_K MoE dp4a port~~ ✅ — Qwen3.6-35B-A3B-UD-Q5_K_S
   pp2tp2: SCALAR 47.77 → DP4A 51.47 tps (+7.7 % end-to-end).
8. ~~Phase 3.5 M-d: Q6_K MoE r2 dp4a port~~ ✅ — Qwen3.6-35B-A3B-UD-Q6_K_XL
   pp2tp2: SCALAR 1913.32 → DP4A 1635.13 ms median total_ms
   (−14.5 % wall / +22 % decode). Also wired pre-existing-but-un-dispatched
   Q6_K tile8 prefill + Q6_K decode-gate split + module registry entry.
9. ~~Phase 3.5 M-e: Q2_K MoE r2 dp4a port~~ ✅ kernel-only — sweep PASS,
   no on-disk consumer (UD-Q2_K_XL uses IQ2_XS not Q2_K).
10. ~~Phase 3.5 M-f–M-j wiring~~ ✅ — IQ-MoE decode/prefill dispatch arms
    for all 9 IQ dtypes (IQ4_XS/NL, IQ3_XXS/S, IQ2_XXS/XS/S, IQ1_S/M).
    Unblocks UD-Q2_K_XL on Qwen3.5-122B-A10B-UD-Q2_K_XL (4459 ms median
    for 64 ct / 259 pt pp2tp2 ctx 4096 f16 KV). Kernels still scalar.
11. ~~Phase 3.5 M-f / M-h dp4a port (IQ2_XS + IQ3_XXS)~~ ✅ —
    Qwen3.5-122B-A10B-UD-Q2_K_XL pp2tp2 ctx 4096 f16 KV:
    SCALAR 4459.23 → DP4A 3256.54 ms median total_ms
    (−27 % wall / +47 % decode, 17.0 → 25.0 tps decode-only).
    Kernels: `indexed_moe_mmvq_iq2_xs_r2_dp4a`,
    `indexed_moe_mmvq_iq3_xxs_r2_dp4a` (wave64, 2 rows / block,
    half-warp DPP reduce; same shape as M-a / M-c / M-d).
12. ~~Phase 3.5 M-f–M-j remaining dp4a ports (IQ4_XS/NL, IQ3_S,
    IQ2_XXS/S, IQ1_S/M)~~ ✅ — 7 new `indexed_moe_mmvq_iq{*}_r2_dp4a`
    kernels shipped mirroring dense Phase 3f-3l math + r2 MoE wiring
    (wave64, 2 rows / block, half-warp DPP reduce). Build green; no
    on-disk consumer beyond UD-Q2_K_XL (covered by M-f / M-h). Kernels
    inherit dense IQ dp4a sweep certs by construction (math identical;
    MoE delta is row-pair indirection only). Regression smoke on
    UD-Q2_K_XL = ~3.29 s median (unchanged vs prior 3.26 s).
8. Phase 2: download missing K-quant variants once MoE slices land
   — they inherit the kernels and don't need a separate diagnosis pass.
9. Phase 4: KV-quant pairings re-validated after Phase 3
   (decode wall composition changed; the floor analysis matters now).
10. Phase 5: decision table — final cert, plan close.

Total remaining surface: M-c–M-j land as MoE consumers surface.
Lever 2/3 on dense codebook kernels are open at any time. Phase 4
and 5 are 1 session each.

## Acceptance criteria

- Every dispatch-table row for Qwen3.6-27B has an end-to-end perf
  number on at least one (KV, ctx, topo) triple.
- No quant is silently regressed vs its bandwidth ceiling by
  > 30 % without a `cfg(unverified)`-style explanation.
- Decision table at `doc/QUANT_FIT_TABLE.md` exists and is current.
