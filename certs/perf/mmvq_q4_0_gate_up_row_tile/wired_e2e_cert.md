# mmvq_q4_0_gate_up_row_tile_batched — end-to-end concurrent-decode cert

Date: 2026-05-13
Model: Qwen3.6-35B-A3B-Q4_0 (qwen35moe hybrid GDN, 32 experts top-4)
Topology: pp+tp 2×2 on devices `hip:0,2,1,3`
Env: `FLAMBEAU_BATCHED_DECODE=1 --inflight-slots 4 --ctx-cap 4096`

## Setup

Wiring: `forward_gdn_decode_batched_tp` now routes Q4_0 `attn_qkv` +
`attn_gate` at `n_tokens ∈ [2, 4]` through one fused
`mmvq_q4_0_gate_up_row_tile_batched` launch instead of two separate
`run_qmatmul_from_tensor(n_tokens=N)` calls. Mirrors the m=1 fused
dispatch in `forward_gdn_decode_tp`. Gate flag
`FLAMBEAU_GDN_FUSE_Q4_0_ROWTILE` defaults ON; set to `0` to fall back
to the two-separate-launch path.

## A/B (same server warm)

```
N | path              | sequential | concurrent | conc/seq
--+-------------------+------------+------------+---------
2 | fused OFF (A)     | 54.2 t/s   | 50.7 t/s   | 0.93×
2 | fused ON  (B)     | 56.4 t/s   | 52.3 t/s   | 0.93×
4 | fused OFF (A)     | 57.1 t/s   | 50.4 t/s   | 0.88×
4 | fused ON  (B)     | 56.9 t/s   | 51.0 t/s   | 0.90×
```

Δ (B vs A): concurrent decode +1.6 t/s at N=2, +0.6 t/s at N=4 — small
positive across the board.

## Interpretation

The kernel's measured per-call gain (1.32×–1.80× vs K5 on GDN-class
shapes, see `cert.md`) only translates into a ~1–3% sustained
throughput improvement because the hybrid concurrent path is bounded
by structures outside this commit: per-slot GDN state-step + per-slot
KV-append/attention loops in `forward_decode_batched_hybrid`, exactly
as `project_p29b_i2_F_hybrid_throughput` documented. The fused
gate+up launch removes one piece of the per-slot serialisation; the
remaining bottleneck is the GDN state-step loop, which is the V2
batched-GDN lever (out of scope here).

## Latent-bug fix included

Both `mmvq_q4_0_gate_up_batched` (K5) and the row-tile sibling
dereferenced `up_w` pointers unconditionally for rows where
`row >= n_rows_up` (do_up=false). Under symmetric n_rows the test
passed; under the GDN asymmetric shape (n_rows_gate=4096,
n_rows_up=2048) those OOB reads crossed unmapped pages on real
weight buffers → `hipStreamSynchronize: illegal memory access`.
Both kernels now predicate the gate/up dereferences on do_gate/do_up.

Parity test grew an explicit asym-GDN row
(`parity_n4_asym_gdn_shape` at 4096×2048×k=2048) to lock the fix.

## Closes

End-to-end wiring side of #23. Per-slot GDN state-step batching is
the next-larger structural lever and stays open.
