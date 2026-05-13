# Qwen3.6-27B-UD-Q3_K_XL — pp2tp2 — post Phase 3a/b/c

**Date:** 2026-05-13
**Rig:** threadreaper-gfx906, 4×MI50 (16 GB), 100 W cap
**Build:** `feature/batched-mmvq-decode` @ 098d31b (Phase 3c complete) +
m_range dispatch widening (this commit)
**Topology:** `pp+tp` pp_size=2 tp_size=2 devices=0,2,1,3 (avoids the
{2,3} link fault — see [[feedback_never_tp4_use_pp2tp2]])
**Server flags:** `--ctx-cap 4096 --inflight-slots 1 --kv f16`

## Model dtype mix (`inspect-gguf`)

```
449 F32     (norms, embeddings, output_norm)
134 Q3_K    (FFN down + some attn)
119 Q4_K    (attn_q/k/v, output)
 64 IQ4_XS  (attn_gate)         ← native (Phase 3a)
 52 Q6_K    (token_embd)
 23 Q5_K    (FFN gate/up subset)
 10 IQ3_S   (FFN gate/up)       ← native (Phase 3b)
  2 IQ3_XXS (attn_q/k subset)   ← native (Phase 3b)
```

**100 % of weight tensors load via direct `mmap → hipMemcpy`.** Zero host
F32 convert. Load wall: ~8.5 s on the 4-GPU rig.

## Bench: pp ≈ 452 / tg = 64 / N = 1 / greedy / 3 runs

| Run     | wall   | pp_tokens | gen_tokens | finish |
|---------|--------|-----------|------------|--------|
| warmup  | 13.19s |       452 |          8 | length |
| run1    | 19.68s |       452 |         64 | length |
| run2    | 19.78s |       452 |         64 | length |
| run3    | 19.82s |       452 |         64 | length |

Solving the 2-point linear model `wall = prefill_s + tg × t_decode`:

- **Prefill: 36.9 tok/s** @ m=452 (MMVQ loop-per-row path)
- **Decode: 8.52 tok/s** @ m=1 (native r2 MMVQ on IQ4_XS / IQ3_S / IQ3_XXS)

Decode latency per token: **117 ms**.

## Comparison to prior cert on same rig

| Model                       | quant       | decode (pp2tp2) |
|-----------------------------|-------------|-----------------|
| Qwen3.6-27B                 | Q4_K_M      | 21 tok/s        |
| Qwen3.6-27B                 | Q4_1        | 21 tok/s        |
| Qwen3.6-27B-**UD-Q3_K_XL**  | this cert   | **8.5 tok/s**   |

The 2.5× gap vs Q4_K_M is structural — the IQ family's MMVQ ports are
scalar (no dp4a) and IQ3_S adds a 9-bit codebook lookup per element
through constant memory. Two Phase-4 levers close it:

1. **MMQ kernels for IQ family** — replace the MMVQ-loop-per-row prefill
   path. Expected: prefill 36→200+ tok/s (5-7×, K-quant MMQ envelope).
2. **dp4a-with-LUT in MMVQ** — pre-apply the sign/scale into i8 packs so
   the inner dot is `__builtin_amdgcn_sdot4`. Expected: decode 8.5→15 t/s
   (~1.8×, scalar→dp4a ratio matches Q4_K).

Both deferred to task #63 (Phase 4).

## Correctness

Output coherent across 3 runs. Sample (run1, first 80 chars):

> "amet sit Lorem amet consectetur tempor eiusmod dolor sit amet incid..."

(Continues the lorem-ipsum prompt as instructed, no glitches/repetition.)

## Notes

- Dispatch wiring needed `m_range: (1, usize::MAX)` widening for all 9
  IQ rows. Original `(1, 127)` caused prefill at m=446 to bail with
  `no QMatMul impl for dtype=IQ4_XS m=446` because no IQ MMQ kernel
  exists yet. Widening routes prefill through MMVQ loop-per-row — slow
  but correct. Phase 4 lifts this.
- HipDeviceEnablePeerAccess warnings at boot (`peer access is already
  enabled`) are pre-existing benign — the rig has system-wide peer
  access pre-enabled.
