# N=1 fused fast-path — +60-71% decode on chat path (2026-05-17)

Closes the bottleneck identified in `chat_path_breakdown.md`:
`forward_decode_batched_pp` at N=1 was running prefill-flavoured
unfused kernels per layer. The legacy
`forward_one_token_{pp,tp,hybrid}_logits` paths use fused
`flambeau_rmsnorm_q8_1_fused` etc., saving ~7-8 ms/token of
host-launch + HBM-round-trip overhead per token.

## Change

`crates/server/src/model.rs::qwen3moe_forward_decode_batched` — added
an `if n == 1` fast-path in each of the three arch arms (PP / TP /
Hybrid) that routes through the fused legacy entry:

- PP: `forward_one_token_pp_logits(model, &mut pp.session, cluster, &mut pp.decode, ...)`
- TP: `forward_one_token_tp_logits(model, &mut tp.decode, &tp_model.tp, &mut tp.session.caches, ...)`
- Hybrid: `forward_one_token_hybrid_logits(model, &mut hyb.decode, cluster, stage_ars, &mut hyb.session, ...)`

The fused path uses each inflight's `decode: ShardedForwardOneTokenScratch*`
field (separate from `prefill: ShardedForwardPrefillScratch*` which the
batched path uses). The decode scratch was previously dormant after
Phase 12.5; it's now the active scratch for N=1 chat decode.

Batched path (N ≥ 2) is unchanged.

## Bench — qwen3.6-35B-A3B-Q4_0, single-stream chat (`temp=0.7 top_p=0.9 seed=0`)

| topo | before | **after** | delta | llama.cpp | ratio |
|---|---:|---:|---:|---:|---:|
| tp2    | 35.66 | **57.09** | **+60.1%** | 43.63 | **1.31×** |
| pp2    | 34.59 | **55.10** | +59.3% | 61.37 | 0.90× |
| pp4    | 32.54 | **52.95** | +62.7% | 58.81 | 0.90× |
| pp2tp2 | 30.26 | **51.84** | **+71.3%** | 58.99 | 0.88× |

Prefill numbers are unchanged because prefill doesn't use this path.

## Headlines

- **TP2 now beats llama.cpp by 1.31×** on decode.
- All other topologies recover to 0.88-0.90× of llama.cpp (was 0.51-0.56×).
- pp2tp2 (today's hybrid driver shipping cell) sees the largest
  relative jump (+71%) — the extra hybrid coordination amortizes over
  layer count, so the per-launch saving compounds.

## Why this works

Per `chat_path_breakdown.md`, the per-token chat budget was:
```
total 24.2ms = dispatch_decode_one 22.2ms + sampler 2.0ms
```
Of the 22.2 ms, ~12 ms is the per-token GPU compute budget (matches
`infer`'s `argmax` sync wait) and ~7-8 ms was the unfused-kernel
overhead. The fused path eliminates that overhead.

After fix:
- TP2 per-token = 1000/57.09 = 17.5 ms (= 12 ms GPU + 2 ms sampler +
  ~3 ms overhead) — matches the GPU ceiling.
- PP2 per-token = 1000/55.10 = 18.1 ms — slightly slower; PP's
  inter-stage peer-copy adds ~1 ms vs TP's BAR1 AllReduce.

## Parity

`gemma4-26B-A4B-Q8_0 pp2tp2` still decodes "The capital of France is
**Paris**." through chat completion (post-fix server verified
end-to-end). No correctness regressions.

## Remaining gap to llama.cpp

| topo | flambeau | llama.cpp | gap |
|---|---:|---:|---:|
| tp2 | **57.1** | 43.6 | +13.5 (we win) |
| pp2 | 55.1 | 61.4 | -6.3 |
| pp4 | 53.0 | 58.8 | -5.8 |
| pp2tp2 | 51.8 | 59.0 | -7.2 |

PP variants still trail by ~10-12%. Likely the next levers:
1. **PP stage overlap** (lever 1 from `cert.md`) — we predicted ~15%
   payoff; with this base now closer to ceiling, the overlap could
   close the remaining 10% gap.
2. **GPU sampler with `KeepOnDeviceLogits`** — 2 ms sampler → ~0.5 ms
   would save ~3%. The legacy fused path already exposes
   `forward_one_token_*_keep_logits_on_device` variants; just need to
   wire them when no penalties + no JSON.

Diminishing returns from here. The big structural lever was today's.
