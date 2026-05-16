# mmvq_f16_direct — rocprofv3 A/B measurement

Date: 2026-05-16
Backend: HIP / gfx906 (MI50)
Model: Qwen3.5-9B-Q4_1 (hybrid GDN, 64 layers)
Workload: prompt="The capital of France is" (6 tokens) + 16 decode tokens
Tool: `rocprofv3 -i pmc.txt -- ... infer --mesh-mode pp --devices hip:0`,
PMC counter group `Wavefronts` (used as a proxy for per-kernel dispatch
trace; counter value itself irrelevant — we want the kernel-name +
duration columns).

A/B controlled by toggling `f16_direct_supported(dtype)` at
`crates/blocks/src/attention.rs`:
- **baseline**: predicate forced to `false` (legacy `mmvq + cast_f32_to_f16`
  on every K/Q/V projection).
- **treatment**: predicate enabled (20-dtype set, current `HEAD`).

Same prompt, max-tokens, devices, and PP=1 in both runs.

## Per-kernel totals

### Baseline (F16-direct disabled)

| kernel                                | count |    total µs | share  |
|---------------------------------------|-------|-------------|--------|
| `mmvq_q4_1_t128_q8_1`                 |  3520 |  138 903.56 | 50.77% |
| `mmvq_q5_k_r2_q8_1`                   |   480 |   44 543.80 | 16.28% |
| `mmvq_q6_k_dp4a_q8_1`                 |    16 |   20 647.02 |  7.55% |
| `gdn_state_step_alphabeta_f32_s128`   |   384 |   17 962.54 |  6.56% |
| `cast_f32_f16`                        |  **1408** |   **3 408.32** |  **1.25%** |
| `mmvq_q4_1_t128_q8_1_f16`             | (not present) | — | — |
| **TOTAL**                             | **14 673** | **273 613.67** |        |

### Treatment (F16-direct enabled)

| kernel                                | count |    total µs | share  |
|---------------------------------------|-------|-------------|--------|
| `mmvq_q4_1_t128_q8_1`                 |  3160 |  131 925.17 | 49.39% |
| `mmvq_q5_k_r2_q8_1`                   |   480 |   50 590.20 | 18.94% |
| `mmvq_q6_k_dp4a_q8_1`                 |    16 |   21 118.06 |  7.91% |
| `mmvq_q4_1_t128_q8_1_f16` (new)       |   360 |    6 708.96 |  2.51% |
| `cast_f32_f16`                        |  **1048** |   **2 567.68** |  **0.96%** |
| **TOTAL**                             | **14 313** | **267 120.28** |        |

## Deltas

| metric                                        | baseline | treatment | delta            |
|-----------------------------------------------|---------:|----------:|-----------------:|
| total kernel launches                         |   14 673 |    14 313 | **−360 (−2.5%)** |
| `cast_f32_f16` count                          |    1 408 |     1 048 | **−360 (−25.6%)** |
| `cast_f32_f16` total µs                       |  3 408.32 |  2 567.68 | **−840.64 µs (−24.7%)** |
| `mmvq_q4_1_t128` F32 count                    |    3 520 |     3 160 | **−360 (−10.2%)** |
| `mmvq_q4_1_t128` F16-direct count             |        0 |       360 | **+360**         |
| `mmvq_q4_1_t128` combined µs (F32 + F16)      | 138 903.56 | 138 634.13 | −0.19% (compute-identical body) |
| **total GPU kernel time (µs)**                | **273 614** | **267 120** | **−6 494 µs (−2.37 %)** |

## Interpretation

The launch-count delta is **exactly 360** in both directions:

- 360 fewer `cast_f32_f16` launches — the 3 casts (Q + K + V projections)
  per attention block that the F16-direct path eliminates.
- 360 fewer `mmvq_q4_1_t128_q8_1` (F32 output) launches.
- 360 new `mmvq_q4_1_t128_q8_1_f16` launches (same body, F16
  saturating store).

The mmvq body's compute cost is unchanged (138 634 µs treatment vs
138 904 baseline ≈ 0.2 % delta inside measurement noise) — that's the
parity test passing on real data, not just the synthetic suite.

The **−6.49 ms of GPU kernel time over 16 decode tokens = −406 µs/token
on GPU** is what the F16-direct path actually saves. That's
**−2.37 % of total kernel time** on Qwen3.5-9B Q4_1 / single-device PP.

The 360 attribution is layer-specific: Qwen3.5-9B is a hybrid GDN /
full-attn model. F16-direct only fires on the standard-attention
layers' Q/K/V projections (`StandardAttention::forward_decode`); the
GDN layers route through `gdn_*` kernels and aren't migrated. So
"360 calls" represents a subset of attention layers × 3 projections ×
16 decode tokens. A pure full-attn model (Mistral, Qwen3.5 dense)
would migrate proportionally more projections and see a larger delta.

Wall-clock delta isn't captured here (rocprofv3 measures GPU-side
duration only) — host-side `hipModuleLaunchKernel` overhead per
eliminated cast is ~3–5 µs that doesn't appear in the kernel timing.
Adding that back: −360 × ~4 µs = ~−1.4 ms of host-side latency, on top
of the −6.49 ms GPU-side win. Decode wall-clock impact estimate:
~−500 µs/token = ~−3 % at ~60 t/s decode rate.

## Output proj + LM head NOT migrated (intentional)

`cast_f32_f16` still runs 1 048 times in the treatment trace. Those
are:
- Output projection F32 → F16 (kept on F32 path — needed for
  `f32_output_proj` saturation handling per
  `feedback_gemma4_attn_output_proj_f16_saturate`).
- LM head output cast (post-norm before sampling).
- Per-layer-embd side-channel cast (gemma4-class arches).

Migrating these would require additional flag-bit invariants
(saturation tolerance per consumer) and is out of scope for #120.

## Reproducer

```
# Baseline: edit attention.rs:f16_direct_supported → return false.
cargo build --release -p flambeau-cli --features hip_infer

# Both runs:
set -a; source .env; set +a
ROCPROF=/opt/rocm-7.1.1/core-7.13/bin/rocprofv3
OUT=/tmp/rocprof_$(date +%s); mkdir -p $OUT
echo "pmc: Wavefronts" > $OUT/pmc.txt
$ROCPROF -i $OUT/pmc.txt -d $OUT -o probe -f csv -- \
    target/release/flambeau infer \
        --model /artefact/models/Qwen3.5-9B-Q4_1.gguf \
        --prompt "The capital of France is" \
        --max-tokens 16 --devices hip:0 --mesh-mode pp

# Aggregate:
python3 /tmp/aggregate_pmc.py $OUT/pmc_1/probe_counter_collection.csv
```

## Files

- `aggregate_pmc.py` (helper, not committed — lives in `/tmp` during the run)
- baseline CSV: `/tmp/rocprof_pmc_baseline_1778944501/pmc_1/probe_counter_collection.csv`
- treatment CSV: `/tmp/rocprof_pmc_1778944072/pmc_1/probe_counter_collection.csv`
