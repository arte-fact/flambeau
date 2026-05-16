# Correctness pass — 27B/pp2tp2 sweep configs

- **Generated:** 2026-05-07T07:42:13+00:00
- **Commit:** 27cdabc
- **Model:** qwen36-27b-q4_0 on pp2tp2 (slots=8)
- **Sampling:** greedy / seed=0 / max_tokens=64

## Per-cell sha256[:16]

| config | N | sha256[:16] | err |
|---|---:|---|---|
| `baseline` | 1 | `0d42b0c87cadaf80` | |
| `baseline` | 4 | `0d42b0c87cadaf80` | |
| `prefill_ubatch_1024` | 1 | `0d42b0c87cadaf80` | |
| `prefill_ubatch_1024` | 4 | `0d42b0c87cadaf80` | |
| `ar_fuse_q8_1` | 1 | `0d42b0c87cadaf80` | |
| `ar_fuse_q8_1` | 4 | `0d42b0c87cadaf80` | |
| `all_optins` | 1 | `0d42b0c87cadaf80` | |
| `all_optins` | 4 | `0d42b0c87cadaf80` | |

## Verdict per config

| config | N=1 vs baseline | N=4 vs baseline |
|---|---|---|
| `prefill_ubatch_1024` | **identical** | **identical** |
| `ar_fuse_q8_1` | **identical** | **identical** |
| `all_optins` | **identical** | **identical** |
