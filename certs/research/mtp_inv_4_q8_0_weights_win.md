# MTP-INV-4 — Q8_0 MTP linears: free 3.5 % wall-clock win, zero acceptance cost

**Status:** small clean win. Switching MTP head linear weights from
F16 → Q8_0 saves 3.5 % per-step wall and **does not reduce
acceptance** (E[accept] differs by < 0.07 % over 60 steps —
indistinguishable from rounding).

## Setup

Regenerated `Qwen3.6-27B-mtp-q8_0.gguf` via the existing converter:

```
MTP_LINEAR_DTYPE=q8_0 python3 tools/convert_qwen36_mtp.py \
    --shards-dir /artefact/flambeau/tools/mtp_cache \
    --out /artefact/models/Qwen3.6-27B-mtp-q8_0.gguf
```

Test: `mtp_acceptance_passive` with `FLAMBEAU_MTP_HEAD=...mtp-q8_0.gguf`,
prompt = default 5-token prose, 60 decode steps, Q4_0 base, pp4.

## Result

| MTP weight | E[accept] | strict greedy | wall/step |
|---|---:|---:|---:|
| F16 (current default) | **64.13 %** | 61.7 % | 60.3 ms |
| **Q8_0** | **64.11 %** | 61.7 % | **58.2 ms** |

Acceptance delta: 0.02 pp — within rounding.
Wall delta: 2.1 ms/step = **3.5 % step-time reduction**.

## Why earlier Q8_0 measurements were lower

Session 4 historical: F16 vs Q8_0 was **43.8 % vs 37.5 %** (+6.3 pp
for F16). At the time this looked like Q8_0 was meaningfully worse.

What changed since session 4:
- MTP-INV-1 fixed the parity test (Python ref had a Q/gate split
  bug — flambeau forward was always correct on this axis)
- Session 6 KV accumulation (+12.4 pp lift, applied to both Q8_0
  and F16 paths uniformly)
- Sampling-aware verify metric (MTP-INV-3) — independent of weight
  dtype but produces a more comparable number

With current full stack, the historical 6.3 pp gap collapses to
< 0.1 pp. **Q8_0 is acceptance-equivalent to F16 on this MTP head.**

## Why the speedup is "only" 3.5 %

gfx906 MMVQ benchmarks suggest Q8_0 + Q8_1 activation (DP4A int8
hardware) runs ~1.88× faster than F16 + Q8_1 act for the same
weight matrix. So MTP forward itself is ~1.88× faster.

But MTP forward is only a small fraction of step time on Mesh<4>
27B-Q4_0:
- Base decode: ~50-55 ms (4-rank pipeline)
- MTP forward: ~5-6 ms (single-rank, single MTP block)

Step-time saving = MTP_speedup × MTP_fraction ≈ (1 - 1/1.88) × ~10 % ≈ 4-5 %.
Measured 3.5 %, in the right ballpark.

The gain would be larger on:
- Smaller base models (where base decode is faster)
- Multi-step MTP=2 chained drafting (more MTP per base step)

## Recommendation

**Default `MTP_LINEAR_DTYPE=q8_0`** in the converter. The historical
default was Q8_0 until MTP-4 session 4 found F16 better; that finding
turned out to be confounded by the Python-ref bug (MTP-INV-1) plus
absent KV accumulation. Q8_0 is the correct default.

Action item: change the default in `tools/convert_qwen36_mtp.py:57`
from `f16` back to `q8_0`. Existing F16 GGUF still works (loader
handles both).

## On Qwen3.6-35B-A3B comparison

**Not possible without MTP weights.** Investigated the official
Qwen/Qwen3.6-35B-A3B HuggingFace repo (web-fetched the
`model.safetensors.index.json`): no `mtp.*` entries, no
`model_mtp.safetensors` standalone file. A blog post (stevescargall
2026-04) claimed "MTP built in" but the actual release does not
ship MTP head weights. Comparison deferred until a community
MTP-augmented 35B variant appears or we train our own head.

## Future quant levers worth measuring

| candidate | bytes/elem | expected speedup | acceptance risk |
|-----------|-----------:|-----------------:|---|
| Q8_0 (now) | 1.06 | baseline (+1.88× kernel) | none vs F16 |
| Q5_K | ~0.69 | +0.5× → ~3.2× kernel | unknown — needs converter Q5_K + measure |
| Q4_K | ~0.50 | +1.1× → ~4× kernel | likely some acceptance loss |
| Q4_0 | 0.56 | +0.9× → ~3.6× kernel | likely some acceptance loss |

Converter currently only emits F16 / Q8_0. Adding K-quant emission
needs ~30 LOC in the converter + a per-tensor selector (`fc` and
`gate_proj` are large enough to potentially benefit; `q/k/v_norm`
are tiny and stay F32).

## Code state

- `tools/convert_qwen36_mtp.py` accepts `MTP_LINEAR_DTYPE=q8_0|f16`
  (existing).
- `tests/mtp_acceptance_passive.rs` accepts `FLAMBEAU_MTP_HEAD=path`
  (added MTP-INV-4) to override the default mtp.gguf path.
- The existing F16 path remains the default until the converter's
  default is flipped.
