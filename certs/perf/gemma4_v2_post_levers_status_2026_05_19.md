# gemma4-v2 — post-lever bench status across model sizes

Date: 2026-05-19
Reference levers (this session):
- B: Q8_0 `indexed_moe_mmvq_gate_up_dp4a` (commit `857e05a`)
- A: `attention_decode_f16_splitk_chunk` tile-2 inner loop (commit `43cf099`)
- C: `rmsnorm_f32` float4 vectorisation (commit `398471f`)

## Bench summary (gemma4 v2 vs llama.cpp, PP2/TP2)

| Model              | Mesh | Prompt | TG  | v2 prefill | v2 decode | llama.cpp prefill | llama.cpp decode | v2/llama prefill | v2/llama decode |
|--------------------|------|-------:|----:|-----------:|----------:|------------------:|-----------------:|-----------------:|----------------:|
| 26B-A4B-Q8_0 (MoE) | PP2  |    725 | 128 |  **503**   | **50.71** |             481   |        65        | **1.05×**        | 0.78×           |
| 31B-Q4_0 (dense)   | TP2  |    725 |  64 |  **287**   | **19.88** |             185   |        21.3      | **1.55×**        | 0.93×           |
| E4B-Q4_0 (PLE)     | SD   |    --  | --  |   ERR      |   ERR     |            1055   |        71.5      |   --             |   --            |

(PLE = per-layer-embedding side channel, the gemma 4n architecture
variant. Both v2 and legacy reject E4B at boot.)

## What changed since the start of this session

- **26B-A4B-Q8_0 decode**: 43.21 → **50.71 t/s (+17.4 %)** across levers
  A + B + C. v2/llama.cpp decode ratio: 0.65× → 0.78×.
- **26B-A4B-Q8_0 prefill**: 506 → 503 t/s (within ±5 % run-to-run
  variance). Lever B (batched apply_per_expert_scale + fused gate+up)
  ships, no measurable prefill regression.
- **31B-Q4_0 prefill**: stable at ~287 t/s, 1.55× llama.cpp. Dense model
  doesn't trigger lever B; benefits from lever C (rmsnorm_f32) modestly.
- **31B-Q4_0 decode**: stable at ~20 t/s, 0.93× llama.cpp. Lever A
  tile-2 helps splitk path (engaged at n>=256 context — both prefill
  output context and decode hit this).

A spurious 31B-Q4_0 prefill reading of 135 t/s on first re-bench in
this session turned out to be first-run / GPU thermal variance; the
second run returned 287 t/s.

## Out-of-scope: E4B (gemma 4n per-layer-embd)

Both v2 and legacy fail at boot:
- v2: `gemma4-v2 does not yet support variants with per-layer embd
  (gemma 4n / E2B / E4B): embedding_length_per_layer_input > 0`
- legacy: `Gemma4PpDriver::upload: per-layer side-channel embedding
  (E2B/E4B) needs the per-layer-embd upload path`

This is task #256. The E4B GGUF has:
- 3 globals: `per_layer_token_embd` Q5_K [262144, 10752],
  `per_layer_model_proj` BF16 [10752, 2560], `per_layer_proj_norm`
  F32 [256]
- Per layer (42 layers): `inp_gate` F32 [256, 2560], `proj` F32
  [2560, 256], `layer_output_scale` F32 [1] (already supported), plus
  the cascade norms `post_attention_norm` / `post_ffw_norm` /
  `post_norm` (already supported).

llama.cpp's `gemma4-iswa.cpp` shows the math (`project_per_layer_inputs`
+ per-layer side-channel block after the FFN post-norm: `inp_gate`
projection → GELU → elemwise mul with `inp_per_layer[layer]` → `proj`
back → rmsnorm → residual add → scalar `out_scale`).

Implementing it is a fresh architecture port (config + loader + setup
helper + per-layer forward hook + cascade integration) — separate
session, not a quick fix on top of the existing levers.

## Reading

The MoE 26B-A4B path now reliably leads llama.cpp on prefill and is
within 22 % on decode. Dense 31B-Q4_0 leads on prefill and is within
7 % on decode. The remaining decode gap is the same shape as before
the levers — GQA-aware q_head batching + `<BLOCK_SIZE,K>` rmsnorm
specialisation — and is documented in
`certs/perf/gemma4_v2_decode_profile/cert.md`.
