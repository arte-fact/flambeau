# gemma4-v2 E4B — ships coherent (closes #256 correctness)

Date: 2026-05-20
Status: **E4B Q4_0 produces coherent output on the v2 stack.**

## Coherence

Single-device boot, F16 KV cache, gemma-4-E4B-it-Q4_0.gguf:

```
prompt:  "The capital of France is"
v2 out:  "Paris."             (finish_reason: stop)

prompt:  "Count from 1 to 10."
v2 out:  "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."   (finish_reason: stop)
```

llama.cpp on the same prompts produces equivalent coherent output. The
greedy decode matches expected behaviour. The Q5_K
`per_layer_token_embd` dequant + BF16 `per_layer_model_proj` matmul +
per-layer side-channel apply are all working as designed.

## Two fixes were needed

The previous session shipped scaffolding (commit `f3e3509`) but
output was garbage. The bug was upstream in attention, not in the
per-layer-embd path. Two pieces landed this session:

### 1. KV-share routing (the real blocker)

`gemma4.attention.shared_kv_layers = 18` means E4B's tail 18 layers
(indices 24..41) reuse K/V from earlier layers' caches instead of
writing their own. v2's `StandardAttention` had no such routing —
every layer ran full Q+K+V + kv_append, with the late layers
reading zeros from never-written caches.

Implementation:
- `AttnWeights::kv_share_src: Option<usize>` — per-layer share source.
- `standard_attn_local`: when `Some(src)`, routes
  `state.pool.kv_caches[kv_local_idx]` to the share-src slot AND
  skips both the single-slot `kv_append_f16` and the batched
  `kv_append_f16_batched_slots`. Q projection + Q-norm + RoPE-Q
  still run from this layer's own `attn_q`. K/V are still
  projected (cheaper than branching the matmul) but their bytes
  land in scratch and never reach a cache.
- `DenseAttnLayerSpec::kv_share_src` plumbed through to
  `AttnWeights::kv_share_src` via the existing loader.
- `Gemma4V2Config::kv_share_src: Vec<Option<usize>>` populated by a
  resolver mirroring legacy gemma4's
  `ModelLayout::resolve_kv_sharing`: for each shared layer, pick the
  most recent has_kv layer of the same attention type (SWA vs full).
- The loader leaves the K/V projection tensors loaded for shared
  layers (they exist on disk for the GGUF — legacy export quirk —
  even though llama.cpp ignores them); the composite just writes the
  bytes to scratch and never appends.

### 2. Per-layer side-channel embedding (already scaffolded in `f3e3509`)

Once the attention path was correct, the per-layer-embd math from the
previous commit started producing the right side-channel residuals.
No further changes needed.

## Bench

E4B Q4_0 single-device:

| stack       | prefill  | decode  | output                                       |
|-------------|---------:|--------:|----------------------------------------------|
| flambeau-v2 |   9.6    | 9.5     | coherent ("This request demands ...")        |
| llama.cpp   | 1049.7   | 71.1    | coherent ("Here's a plan to structure ...")  |

v2 is functionally complete but ~110× behind on prefill and ~7.5×
behind on decode. Both are CPU-bound on the host-side per-layer-embd
table build (Q5_K dequant + BF16 matmul of `[10752, 2560] @ [2560]`
per token). For prefill, n=1 token-by-token forward also serialises
the whole layer loop per token.

Perf is its own follow-up:

1. **GPU-side table build**: keep the `per_layer_token_embd` Q5_K
   table on device, do the row lookup + dequant via the standard
   embed path, then a F16-input mmvq for the model_proj matmul. The
   `per_layer_model_proj` is BF16 — we'd cast it to F16 at upload
   (matches lever-2 router pattern). One `dense_gemv_f16_f16` for
   the projection.
2. **Batched table build**: the host build currently takes one F16
   `[hidden]` input. Lift to F16 `[n_tokens, hidden]` and run the
   matmul + per-row rmsnorm + add once per prefill chunk. Closes the
   prefill gap.
3. **Drop the n=1 prefill loop in model.rs**: once the table build
   is batched, the per-layer apply can also batch n_tokens (the
   block takes `[hidden]` per call — needs to fan out to `[n_tokens,
   hidden]`).

For now: E4B works. The bench shape is heavily prefill-skewed (725
tok prompt) so the wall-time gap is dominated by step 1 above. With
the per-token decode-only path that's the only loaded user path, we
get to 9.5 t/s — usable for short-context interactive use.

## Files changed

- `crates/forward/src/ctx.rs` (AttnWeights::kv_share_src field)
- `crates/forward/src/loader/dense_attn.rs` (DenseAttnLayerSpec
  plumbing)
- `crates/forward/src/core/composites/standard_attn.rs` (kv_local_idx
  routing + skip kv_append when shared)
- `crates/forward/tests/*.rs` (kv_share_src: None in literals)
- `crates/models/gemma4-v2/src/config.rs` (Vec<Option<usize>>
  resolver)
- `crates/models/gemma4-v2/src/loader.rs` (pass per-layer
  kv_share_src to the spec)
- `crates/models/qwen35-v2/src/loader.rs`,
  `crates/models/qwen35moe-v2/src/loader.rs` (kv_share_src: None at
  loader sites)

#256 closes for correctness. Perf follow-up will be a separate task.
