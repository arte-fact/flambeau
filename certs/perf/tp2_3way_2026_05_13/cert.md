# Three-way bench: feature/tp2 vs main/tp2 vs llama.cpp/tp2 (and PP2 sanity)

Date: 2026-05-13
Model: Qwen3.6-35B-A3B-Q4_0
GPUs: 2× MI50 16 GB, 100 W cap
Bench: `bench_35b_concurrent.py` (one fib-gen prompt, max_tokens=96,
temperature=0, sequential vs concurrent N=2/4)

## Summary

| Stack                | Mode             | N=1 seq | N=2 conc | N=4 conc | conc/seq N=4 |
|----------------------+------------------+---------+----------+----------+--------------|
| flambeau (feature)   | tp2 (0,1)        | **64.55** | 54.85    | 54.4     | 0.85         |
| flambeau (main)      | tp2 (0,1)        | 64.5    | ~54      | ~55.4    | 0.87         |
| flambeau (feature)   | pp2 (0,2)        | 59.0    | 56.75    | 68.65    | 1.17         |
| llama.cpp            | layer (0,1) PP-ish | 55.2  | 81.8–85  | 104.1–107.6 | 1.93–1.97 |
| llama.cpp            | tensor (0,1) TP-ish | ~42  | 58–63    | 104.3–105.7 | 2.40–2.60 |

flambeau commits under test: feature = `73de6e7` (post-batching trunk),
main = `0bbaa14`.

llama.cpp commands:
```
HIP_VISIBLE_DEVICES=0,1 ... llama-server \
  --model Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  --port 8081 --n-gpu-layers 999 \
  --split-mode {layer|tensor} --tensor-split 1,1 \
  --parallel 4 --ctx-size 16384 [--cache-ram 0 for tensor mode]
```

## Notes per cell

- **flambeau tp2 vs main**: post-batching trunk and main are
  indistinguishable at tp2. This session's levers (Q4_0 row-tile fused
  gate+up, batched-slots GDN state-step, batched KV-append, batched
  GDN pass A pointwise) moved pp2tp2 +2-3% and pp2 (vs nothing
  measured) but tp2's ceiling is cross-rank AllReduce, not per-slot
  launch overhead.

- **flambeau pp2 vs llama.cpp layer (both PP-shape)**: 68.65 vs 105.85 =
  llama.cpp +54%. The forward-path kernels look fine in single-stream
  (flambeau 64.55 vs 55.2 = flambeau wins single-stream); the gap
  opens with N. Profile attribution: llama.cpp's MoE decode at N≥2
  packs N×top_k routing pairs into MoE MMQ tile launches (tile8/16);
  flambeau's MoE decode at small N goes through indexed-MMVQ (row-by-
  row per routing pair). At N=4 × top_k=4 = 16 routing pairs per
  layer per step — perfect tile8 territory.

- **flambeau tp2 vs llama.cpp tensor**: same-topology comparison.
  64.55 vs 42 = flambeau **+54%** single-stream; 54.4 vs 104.5 =
  llama.cpp **+92%** at N=4 concurrent. Crossover ≈ N=2.

- **llama.cpp tensor crashes** at session-state-save (slot management
  during concurrent slot eviction); `--cache-ram 0` works around it.
  Layer mode is stable.

## Root cause of the concurrent gap

`crates/models/qwen3-moe/src/forward/moe_tp.rs::tile8_dt_ok` gates
the MoE-MMQ tile8 path at `n_tokens >= 32` (the prefill threshold).
At batched-decode the per-call `n_tokens = N` ∈ [2, 4], but the
*total routing pair count* is `n_tokens * top_k = 8–16`, which is
exactly the tile8 input size. Flipping the gate to use
`n_tokens * top_k >= 8` (or similar) would route batched-decode
through tile8 MMQ instead of indexed-MMVQ — the same lever that
gave 6.3× prefill on the 122B in this session, now applied to the
decode batched path. Estimated: closes most of the llama.cpp gap.

## Next lever (predicted)

Wire `moe_tp.rs::tile8_dt_ok` to engage at batched-decode shapes:
- Threshold from `n_tokens >= 32` (prefill) → `n_tokens * top_k >= 8`
  (decode-aware), OR
- Add a separate `batched_decode_tile8` gate that branches on
  `flambeau_in_decode_path == true && n_tokens * top_k >= 8`.

This is a one-line dispatch change. The kernels exist (Slice C/D
landed earlier this session). Parity is already cert'd at
`certs/hip/gfx906/indexed_moe_mmq_q4_0_gate_up_tile8.json` etc.
Risk: at small (n_tokens × top_k), tile8 padding overhead can exceed
the per-slot indexed-MMVQ baseline — needs microbench to confirm
the threshold value.

## Decision

- Single-stream serving (N=1, interactive chat): **use flambeau tp2**.
  Best in class on this rig (64.55 t/s, +17% vs llama.cpp layer).
- Concurrent serving (N≥2 simultaneous chats): **use llama.cpp**
  until the MoE-tile8 decode wiring lands. Once it lands, expect
  flambeau pp2 to converge with or pass llama.cpp.
