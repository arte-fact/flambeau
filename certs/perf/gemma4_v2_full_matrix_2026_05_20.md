# gemma4-v2 — full-matrix bench vs llama.cpp (post-E4B)

Date: 2026-05-20
Branch tip: `6650808` (v2 standard_attn/dense_ffn: fuse rmsnorm_f32_to_f16 + residual add)
Prompt: 725-token technical prompt (`PROMPT_BASE`)
Decode: 64 tok (E4B, 31B) / 128 tok (26B-A4B), greedy temp=0
Backends: flambeau-v2 vs llama.cpp on the same physical 0/0,1 GPUs.

## Results

| Model              | Mesh | flambeau-v2 prefill | llama.cpp prefill | v2 prefill ratio | flambeau-v2 decode | llama.cpp decode | v2 decode ratio | output     |
|--------------------|------|--------------------:|------------------:|-----------------:|-------------------:|-----------------:|----------------:|------------|
| **E4B-Q4_0** (PLE) | SD   |               563.3 |            1053.9 |       **0.53×**  |              46.03 |            71.20 |    **0.65×**    | coherent   |
| **26B-A4B-Q8_0** (MoE) | PP2 |          464.2 |             481.0 |       **0.97×**  |              53.70 |            64.57 |    **0.83×**    | coherent   |
| **31B-Q4_0** (dense) | TP2 |             286.6 |             185.1 |       **1.55×**  |              21.49 |            21.27 |    **1.01×**    | coherent   |

(Legacy on 31B-Q4_0 ships 21.0 / 21.03 t/s — prefill is essentially a
no-op vs llama.cpp because legacy gemma4 has no prefill-batching
path. v2 leads legacy by 13.6× on prefill at this shape.)

## Reading

- **E4B is the open gap**: shipped coherent (commit `9cd22c6`) with
  the KV-share routing + per-layer-embd, then perf levers landed
  (`feecc71` GPU-side proj matmul, `0460a5c` batched build+apply +
  flash-tile SWA fix, `265d0ac` splitk tile-4, `6650808`
  fused-rmsnorm+residual). Still 0.53× / 0.65× of llama.cpp on
  prefill / decode.
- **26B-A4B is at prefill parity**, decode 0.83× — improved from
  0.78× earlier this session (commit `857e05a` gate+up fusion +
  follow-ups; the recent `3d1f693` V-rmsnorm-into-KV-append, the
  `022baff` rmsnorm+RoPE fusion, and the `265d0ac` splitk tile-4
  compound on the 26B path).
- **31B-Q4_0 leads on both axes** — prefill 1.55×, decode at parity.

## The E4B remaining gap — where to look next

E4B is the smallest model (and the only one with the per-layer side
channel), so the gap is structural to that path:

1. **Per-layer-embd cost still significant**. Even with the GPU-side
   matmul (`feecc71`) the table build runs once per token; for prefill
   that's still a serialized chain. Worth re-profiling.
2. **Shared-KV layers (24..41) still pay the Q projection cost** but
   skip K/V. Verify the kernel-time accounting matches expectation —
   should be ~Q-only cost for the tail half.
3. **head_dim=512 SWA layers** (when `key_length_swa` = 256 only on
   some layers) — verify the attention decode kernel dispatch lines
   up with the per-layer head_dim variance.
4. **Output-head Q4_K at decode** uses `mmvq_q4_k_r4` after `6ec683f`
   — confirm the route fires for E4B vocab=262144 too (large vocab
   stresses the output head differently).

The fact that prefill is 0.53× (a wider gap than decode) is the
unusual signal — prefill is normally where flambeau leads. That
points at the per-token side-channel build still being on the
critical path for prefill rather than amortising across the n-token
batch.

## Files

- `/artefact/flambeau/certs/perf/gemma4_v2_vs_legacy_vs_llamacpp_2026_05_19.json`
  (raw bench results, last run timestamp 2026-05-20T14:17Z + 14:32Z
  for E4B)
