# GDN pass A batched (silu / split_qkv / l2_norm / scale) — cert

Date: 2026-05-13
Path: `forward_gdn_decode_batched_tp` (Stage D, "Pass A.2")

## Design

The post-conv pointwise ops in GDN decode (silu of conv_out, split into
Q/K/V, L2-norm of Q and K, scale Q by 1/sqrt(head_k_dim)) were called
inside the per-slot loop — 5 launches per slot per layer per token.

These ops are all naturally batchable: silu/scale are pointwise,
gdn_split_qkv has an `n_tokens` parameter, and l2_norm processes
`n_rows` rows independently. The per-slot scratch buffers
(`conv_out`, `silu_out`, `q_norm_f32`, `k_norm_f32`, `v_f32`) are
already slot-major contiguous in `[N, ...]` layout, so one call per
op with `n_tokens=N` / `n=N×local_*` / `n_rows=N×local_num_k_heads`
covers every slot.

Restructured the per-slot loop into three passes:
- Pass A.1: per-slot conv-trio (assemble_conv_input + causal_conv1d +
  shift_conv_history) — still per-slot because each slot has its own
  `conv_history`. Batching this requires slot-pointer-indirect
  kernels (separate lever).
- Pass A.2: 5 batched pointwise calls covering all N slots.
- Pass A.3: state-step (already batched via #70) and per-slot
  ssm_norm.

Gated by `FLAMBEAU_GDN_PASSA_BATCHED` (default ON, set to `0` to
fall back to per-slot pointwise ops).

Launch count drop per GDN layer per decode token:
- Pre: `5 × N` launches (silu, split, l2-Q, l2-K, scale-Q).
- Post: `5` launches, regardless of N.

At N=4 on Qwen3.6-35B-A3B (36 GDN layers, 25% full-attn = 12 layers
excluded), per decode token: 36 × (5×4 − 5) = 540 fewer launches.

## End-to-end (Qwen3.6-35B-A3B-Q4_0 / pp2tp2 / inflight=4 / batched-decode)

A/B with the same warm build, two measurements each:

```
| Pass A    | N=2 conc | N=4 conc | conc/seq (N=4) |
|-----------+----------+----------+----------------|
| OFF (1,2) | 53.8, 55.5 → 54.65 | 50.9, 50.3 → 50.6 | 0.89, 0.89 |
| ON  (1,2) | 55.0, 57.2 → 56.10 | 51.6, 51.5 → 51.55| 0.90, 1.01 |
| Δ         | +1.45              | +0.95             | +0.01      |
```

B.2 at N=2 hit **conc/seq = 1.01×** — concurrent decode is faster
than sequential at N=2, fully closing the overlap gap. (Other B.2
N=2 run was 0.99×.)

## Cumulative this session vs no-batching baseline

```
| Variant                                          | N=2 conc | N=4 conc |
|--------------------------------------------------+----------+----------|
| Baseline (no batching)                           | 50.7     | 50.4     |
| + Q4_0 row-tile fused gate+up                    | 52.3     | 51.0     |
| + batched-slots GDN state-step                   | 53.4     | 51.9     |
| + batched-slots KV-append                        | 54.4     | 51.8     |
| + batched GDN pass A (this cert)                 | 56.1     | 51.55    |
```

End-to-end gain: **+10.7% at N=2, +2.3% at N=4** over the
no-batching baseline. The N=2 path is approaching the
no-overhead-overlap ceiling (1.01× run observed).

## Parity / correctness

No kernel changes — only call-site argument changes. The ops are
deterministic with respect to (slot-major slot offset) ↔ (n_rows ×
single batch) equivalence, so outputs are bit-equal by construction.
Greedy + temp=0 smoke chat produced the same canonical Rust
fibonacci function as previous runs.

## Closes

The trivially-batchable subset of GDN pass A. The remaining lever in
pass A is the conv-trio (assemble + causal_conv1d + shift), which
needs slot-pointer-indirect kernels — separate session.
