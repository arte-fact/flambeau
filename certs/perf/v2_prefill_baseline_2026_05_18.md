# v2 prefill baseline vs legacy (Qwen3.5-9B-Q4_1, SD, 2026-05-18)

`Session<A>::forward_prefill_logits` loops `forward_one_token` per
prompt token. That's the v1 cut. Legacy `prefill_logits` calls the
true batched-prefill kernels (`forward_prefill_pp/tp/hybrid_logits`)
which prefill in a single multi-token pass per layer, amortising
weight HBM across N tokens.

## Setup
- Model: `/artefact/models/Qwen3.5-9B-Q4_1.gguf`
- Topology: SingleDevice on hip:0 (gfx906 MI50)
- Prompt: 256× "the quick brown fox " + " Reply: ok" → 1044 prompt_tokens
- `max_tokens=1` so wall ≈ prefill (one decode step amortised)
- 3 runs after 1 warmup

## Results

| Path           | wall (s) | prefill_tok/s |
|----------------|---------:|--------------:|
| legacy (V2=0)  |    2.07  |    503        |
| v2     (V2=1)  |   25.7   |     40.5      |

**v2 is 12.4× slower than legacy on prefill.**

## Diagnosis
v2's `forward_prefill_logits` is the v1 loop-single-token cut. The
fix (P9-OUT #217) is one of:
- Add N-token methods to `ForwardCtx` + a per-arch
  `forward_prefill<C: PrefillCtx>` that runs N tokens per layer in
  one pass — full structural fix, multi-session.
- Wrap the existing `flambeau-model-ops` N-token primitives
  (`attn_prefill`, `kv_append` w/ position vector, RoPE batched)
  behind a `PrefillCtx`-shaped trait and have each arch's model.rs
  delegate to it inside `forward_prefill`. Same scope.

Note: the v2 stack is the long-term home; legacy retires in #221.
Doing #217 means re-implementing the batched-prefill kernels'
composite-level orchestration on the v2 side. No new kernels needed
(model-ops already has the N-token primitives), but every composite
needs an N-token variant + per-arch forward_prefill function +
per-topology PrefillCtx impl. Sizeable.
