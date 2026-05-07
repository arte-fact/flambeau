# Phase 3c — `--prefix-cache` cold/warm cert

- **Generated:** 2026-05-07T17:53:43+00:00
- **Commit:** 4f29723
- **Model:** qwen36-27b-q4_0 on pp2tp2
- **Workload:** same prompt 3× sequentially, max_tokens=32

## Cells

| arm | turn | prefill ms | decode ms | err |
|---|---:|---:|---:|---|
| `cold (prefix-cache off)#0` | 0 | 11525 | 1322 | |
| `cold (prefix-cache off)#1` | 1 | 11245 | 1333 | |
| `cold (prefix-cache off)#2` | 2 | 11337 | 1333 | |
| `warm (prefix-cache on)#0` | 0 | 12882 | 1477 | |
| `warm (prefix-cache on)#1` | 1 | 112 | 1340 | |
| `warm (prefix-cache on)#2` | 2 | 113 | 1341 | |

## Verdict

- **Warm-turn speedup:** 114.7× (turn 1 = 12882 ms → turn 2 = 112 ms)
- **Turn 3:** 113 ms
- **PASS** — prefix cache is effective.
