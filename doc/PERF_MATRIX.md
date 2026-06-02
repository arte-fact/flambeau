# Model × context performance matrix

Snapshot of prefill + decode throughput across representative models on
the gfx906 (MI50 ×4, pp+tp 2×2, devices 0,2,1,3, F16 KV) reference rig.
All measurements via `scripts/bench/sweep_model_matrix.py` — temperature
0 greedy SSE streaming, prefill_tps = `prompt_tokens / ttft_seconds`,
decode_tps = `(n-1) / (last_delta_t - first_delta_t)` (interval-only,
robust to early-EOS / ct variance).

`small` ≈ 200 prompt tokens (KV-scan trivial). `large` ≈ 5000 prompt
tokens (KV-scan dominates per-token wall for full-attn layers).

## 2026-06-02 baseline

After Phase 3 dense dp4a, Phase 3.5 MoE r2 dp4a (M-a … M-j), Phase 4
gemma Q8 SWA fix, and Q5_K dense Lever A (gate+up fusion). 3-rep median
per cell.

### Decode tokens/sec

| Model | small (pt≈200) | large (pt≈5000) | Δ large vs small |
|---|---:|---:|---:|
| Qwen3.6-27B Q4_0 (dense) | 32.02 | 32.43 | +1 % |
| Qwen3.6-27B UD-Q4_K_XL (dense UD K) | 20.67 | 20.81 | +1 % |
| Qwen3.6-27B Q8_0 (dense) | 28.33 | 27.80 | −2 % |
| Qwen3.6-35B-A3B UD-Q6_K_XL (MoE) | 54.00 | 54.49 | +1 % |
| gemma-4-26B-A4B Q8_0 (MoE, SWA window=1024) | 39.07 | 50.02 | +28 % |
| Qwen3.5-122B-A10B UD-Q2_K_XL (122B IQ MoE) | 31.23 | 31.42 | +1 % |

### Prefill tokens/sec

| Model | small (pt≈200) | large (pt≈5000) |
|---|---:|---:|
| Qwen3.6-27B Q4_0 | 297 | 355 |
| Qwen3.6-27B UD-Q4_K_XL | 151 | 155 |
| Qwen3.6-27B Q8_0 | 221 | 194 |
| Qwen3.6-35B-A3B UD-Q6_K_XL | 559 | 695 |
| gemma-4-26B-A4B Q8_0 | 791 | 802 |
| Qwen3.5-122B-A10B UD-Q2_K_XL | 243 | 266 |

### TTFT and wall (3-rep median, milliseconds)

| Model | ctx | prompt_tokens | ttft_ms | wall_ms (64 decode tokens) |
|---|---|---:|---:|---:|
| Qwen3.6-27B Q4_0 | small | 200 | 673.3 | 2 514.1 |
| Qwen3.6-27B Q4_0 | large | 5000 | 14 074.5 | 16 030.7 |
| Qwen3.6-27B UD-Q4_K_XL | small | 200 | 1 326.1 | 4 190.2 |
| Qwen3.6-27B UD-Q4_K_XL | large | 5000 | 32 305.4 | 35 272.7 |
| Qwen3.6-27B Q8_0 | small | 200 | 905.7 | 3 096.4 |
| Qwen3.6-27B Q8_0 | large | 5000 | 25 759.4 | 28 024.6 |
| Qwen3.6-35B-A3B UD-Q6_K_XL | small | 200 | 357.7 | 1 524.6 |
| Qwen3.6-35B-A3B UD-Q6_K_XL | large | 5000 | 7 190.6 | 8 366.7 |
| gemma-4-26B-A4B Q8_0 | small | 200 | 252.9 | 1 703.5 |
| gemma-4-26B-A4B Q8_0 | large | 5000 | 6 232.3 | 7 415.7 |
| Qwen3.5-122B-A10B UD-Q2_K_XL | small | 200 | 824.7 | 2 838.1 |
| Qwen3.5-122B-A10B UD-Q2_K_XL | large | 5000 | 18 820.9 | 20 846.0 |

## Read

**Decode**:
- MoE wins on this rig because active params < total: A3B 54 tps (10B
  active), gemma A4B 39 → 50 tps, 122B A10B 31 tps — all faster than
  the 27B dense at 32 tps.
- Dense ladder: Q4_0 (32) > Q8_0 (28) > UD-Q4_K_XL (21). The
  UD-Q4_K_XL 35 % gap vs Q4_0 reflects the K-quant family's higher
  per-token decode cost (more complex unpack on every super-block).
- Gemma jumps **+28 %** large vs small — exactly the SWA win
  confirmed by the Phase 4 Slice 3c fix
  (`attention_prefill_flash_tile_q8_kv` NaN-init guard). Once the
  prompt exceeds the 1024 SWA window, SWA layers scan only 1024 K
  rows instead of the full prompt, so decode accelerates as ctx
  grows — distinctive among the models tested.

**Prefill**:
- Gemma A4B Q8_0 leads (790-800 tps) — head_dim=512 + tile8 + small
  active params + flash-tile prefill all stacking.
- MoE prefill scales well with prompt length (A3B 559 → 695, gemma
  791 → 802) — the chunked-prefill tile8 MMQ paths amortize launch
  overhead.
- Dense Q4_K is the slowest path (151-155 tps stable) — UD K-quant
  prefill is the consistent bottleneck.
- 122B IQ MoE prefill (243 → 266) is competitive with 27B dense at
  ~1/4 the per-token rate per active-param.

**Outliers worth a future profile**:
- Qwen3.6-27B-Q8_0 large prefill *drops* 221 → 194 tps (the only
  cell that moves the wrong direction with ctx). Suspect attention
  prefill memory pressure at ctx 5k with Q8 weights.
- UD-Q4_K_XL is the lowest cell across the board — confirms the
  Phase 3 finding that dense K-quant prefill is still the weak link.

## Provenance

- Hardware: gfx906 (MI50 ×4, threadreaper rig). pp+tp 2×2 across
  devices `0,2,1,3` (the canonical `pp2tp2` lane that avoids the
  `{2,3}` link-faulted pair).
- F16 KV throughout; Q8 KV is not yet the default on gemma (see
  `feedback_gemma_q8_kv_swa_regression`).
- Code state at `feature/quant-perf-sweep` branch tip (2026-06-02);
  `git log -1` for the exact commit.
- Raw JSON: `/tmp/sweep_matrix_results.json` (also reproducible via
  `python3 scripts/bench/sweep_model_matrix.py`).
