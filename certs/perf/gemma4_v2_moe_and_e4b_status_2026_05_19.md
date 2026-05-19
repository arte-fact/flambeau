# gemma4-v2 MoE + E4B support — status

This session attempted to add MoE (26B-A4B) and per-layer-embedding
(E4B) support to gemma4-v2. Both are **not yet shipped**; this cert
documents what landed as foundation and what's still needed.

## Foundation landed (re-usable, no behavior change)

These changes apply to all v2 archs and are correctness-neutral on
existing paths (qwen3.5 / qwen3.6 / gemma4-dense).

- `MoeWeights::post_ffn_norm: Option<Tensor<F16>>` field
  (`crates/forward/src/ctx.rs`). Used by `moe_ffn` composite at the
  tail of the slow path to apply post-FFN rmsnorm before residual
  add. qwen35moe-v2 passes `None` (unchanged behavior).
- `moe_ffn` composite: AR-residual fast paths now gated off when
  `post_ffn_norm.is_some()` to give the model a chance to apply the
  post-norm before residual. Adds a per-call rmsnorm at the tail of
  the slow path when post-norm is set.
- `flambeau_forward::loader::upload_moe_experts_fused_gate_up_stacked`
  — host-side split of `[n_experts, 2*inter, hidden]` fused
  gate+up tensors (gemma4 MoE layout) into two contiguous
  `[n_experts, inter, hidden]` stacks suitable for the indexed-MoE
  kernels.
- `flambeau_blocks::SharedExpert::with_activation` — opt-in Gelu
  activation for the shared-expert / shared-MLP path. Default stays
  SwiGLU (qwen3.x pattern). Adds Gelu match arms in
  `forward_decode` + `forward_prefill`.
- `gemma4-v2` config: parses `expert_count` /
  `expert_used_count` / `expert_feed_forward_length` into
  `Gemma4V2Config::moe: Option<MoeDims>` instead of rejecting at
  load time. `embedding_length_per_layer_input` still rejected
  (E4B unsupported).
- `gemma4-v2` arch: `scratch_config` sizes the routed-expert +
  shared-MLP scratch from the MoE dims when present.
- `gemma4-v2` loader: when `config.moe.is_some()`, builds
  `MoeWeights` per layer with routed experts (fused gate+up split
  + row-sharded down) + shared MLP (separate gate/up/down with the
  dense `intermediate` width). `post_ffn_norm` set to
  `post_ffw_norm`. Activation set to `GeluTanh`.

## Blocker: gemma4 MoE forward needs a 5-norm F32 cascade

The qwen-shape `moe_ffn` composite uses a single pre-FFN rmsnorm + a
F16 partial cascade. Gemma4 MoE uses a fundamentally different shape
(`crates/models/gemma4/src/moe.rs::forward_ffn_moe`):

```text
router_input = rmsnorm_f16(attn_residual, pre_router_weight)
{ids, w} = MoE::route_decode(router_input); w *= ffn_down_exps_scale
// Shared MLP — F32 partial throughout
rmsnorm_quant_q8_1(attn_residual, ffn_norm) → gate/up/GELU/down →
  partial_shared_mlp_f32          (NO F16 cast)
cur_mlp_f32 = rmsnorm_f32(partial_shared_mlp_f32, post_ffw_norm_1_f32)
// Routed MoE — F32 partial throughout
cur_moe_input_f16 = rmsnorm_f16(attn_residual, pre_ffw_norm_2)
partial_moe_f32   = MoE::forward_decode_tp_f32(cur_moe_input_f16)
cur_moe_f32       = rmsnorm_f32(partial_moe_f32, post_ffw_norm_2_f32)
// Combine + final post-norm + residual
cur_combined_f32  = cur_mlp_f32 + cur_moe_f32
tmp_f32           = rmsnorm_f32(cur_combined_f32, post_ffw_norm_f32)
x_out = attn_residual + cast_f32_to_f16(tmp_f32)
```

Five rmsnorm tensors gemma4 MoE needs that aren't in current `MoeWeights`:

- `pre_router_weight` (F16) — separate rmsnorm just for the router input.
- `pre_ffw_norm_2` (F16) — separate rmsnorm for the MoE branch input
  (not the same as `ffn_norm`).
- `post_ffw_norm_1` (F32) — applied to the shared MLP F32 partial.
- `post_ffw_norm_2` (F32) — applied to the routed MoE F32 partial.
- `post_ffw_norm` (F16, or its F32 sibling) — applied to the
  combined F32 partial before the cast back to F16.

Plus: F32 partial cascade (no F16 cast between MoE down and the
final post-norm), which means new pool scratches for
`partial_shared_mlp_f32`, `cur_mlp_f32`, `partial_moe_f32`,
`cur_moe_f32`, `cur_combined_f32`, `tmp_f32` — 6 × hidden F32 each.

Plus: per-expert scale fold (`ffn_down_exps.scale`, F32 [n_experts])
multiplied into the router top-k weights before the indexed-MoE
forward.

## Symptom of the half-implementation

Booting `gemma-4-26B-A4B-it-Q8_0.gguf` and asking "The capital of
France is" returns:

```
로 de로get neoexcludes 싶 ARI로ക്ക് aut곤 own로로يةEbAw de-కి deceit로 much
```

The forward routes through the qwen-shape `moe_ffn` (single F16
post-norm), which gives the wrong math for gemma4 (missing 4 of 5
norms, F16 cascade overflow at hidden=2816 × 128 experts × top_k=8).

## Status of the forward path now

`gemma4-v2::model::forward` rejects MoE at the top with a clear error
quoting `crates/models/gemma4/src/moe.rs`. The dense gemma4 path is
unchanged (still produces coherent output on 31B-Q4_0 / Q8_0). The
loader still uploads MoE weights — they're just unreachable through
the forward path, ready for the composite work.

## What's needed next (separate session)

1. New composite `gemma4_moe_ffn` in `flambeau-forward` (or extend
   `moe_ffn` with a "gemma4 cascade" branch gated on extra norm
   fields). Mirrors `forward_ffn_moe` math above.
2. Extend `MoeWeights` with the 4 extra norms (or a sub-struct
   `Gemma4MoeNorms`) — only Some for gemma4.
3. Extend `ScratchPool` / `ScratchConfig` with the F32 partial
   buffers for the cascade.
4. Per-expert scale tensor on disk (`ffn_down_exps.scale`) +
   `apply_per_expert_scale_f32` op (already exists in legacy
   `flambeau-ops`).
5. Prefill path: same cascade at N tokens, batched.

## E4B / E2B (per-layer embedding)

Not attempted this session. Per-layer embd requires:
- New embedding lookup (per-token + per-layer side-channel: tensors
  `per_layer_token_embd`, `per_layer_model_proj`, `per_layer_proj_norm`).
- KV sharing across the tail `shared_kv_layers` layers (last 18 layers
  read earlier layer's KV cache).
- A new composite for the per-layer embedding fusion at each layer.

This is a separate, larger session.
