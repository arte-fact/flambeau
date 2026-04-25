# V2.34 — CRITICAL: `forward_prefill_pp` at L>1 produces wrong output

## Severity: high

All prefill perf numbers recorded in V2.30.a-i3, V2.31.a, V2.32.a
measure valid kernel execution **but produce wrong token outputs**.
Never surfaced because every parity cert to date tests `L=1 prefill
+ decode loop`, not multi-token prefill.

## Repro

`crates/models/qwen3-moe/tests/prefill_L_sweep.rs` on
Qwen3.5-27B-Q4_1 (arch=qwen35 gated full-attn):

```
decode:    [9419, 11, 353, 599, 264, 3296, 883, 279, 2614, ...]
L=1:  prefill returned 11       expected 11       ✓
L=2:  prefill returned 1971     expected 353      ✗
L=3:  prefill returned 59       expected 599      ✗
L=4:  prefill returned 8270     expected 264      ✗
L=5:  prefill returned 248046   expected 3296     ✗
L=6:  prefill returned 287      expected 883      ✗
L=7:  prefill returned 12       expected 279      ✗
L=8:  prefill returned 44576    expected 2614     ✗
L=12: prefill returned 16       expected 71093    ✗
L=16: prefill returned 25       expected 281      ✗
```

Same behaviour on Qwen3-Coder-30B (arch=qwen3moe dense full-attn):
only pos 0 matches, pos 1..L-1 wild.

## What this means

**Current "parity cert" test methodology missed it**: V1.7.4.b and
V2.28.b-i3 both do `forward_prefill_pp(L=1) + forward_one_token_pp ×
N`. The L=1 case is the one L that *does* work. Once L≥2 used for
prefill, outputs are wrong.

**Shape-perf cert numbers are not invalidated** (kernels run the
advertised FLOPs in the advertised time) but the **generated token
streams are garbage** for any workload that prefilled a multi-token
prompt.

**Practical user impact**: any chat-mode user who submits a prompt of
length > 1 token gets a corrupted KV cache before decoding. Every
chat turn after the first is potentially broken. V1.8 server
(`flambeau serve`) is affected.

## Scope of affected certs

- V2.30.a bench tour — prefill numbers recorded, output streams
  invalid
- V2.31.a Coder Q5_K tile8 — "+50 % prefill" means "+50 % on garbage
  output"
- V2.31.g batched router — same
- V2.32.a Qwen3.5-27B-Q4_1 27B shipping config — decode numbers valid
  (decode path), prefill numbers produce wrong outputs
- V2.33.a-f spec decoding — blocked on this
- V2.28.b-i2 Coder forward smoke's L=2/4/16 prefill last_ids — those
  numbers are whatever garbage the buggy prefill produced; they are
  "deterministic" in the reproducibility sense but not *correct*

## Hypothesis space

Not yet diagnosed. Possible root causes (ranked):

1. **Causal mask not applied or applied wrong in attention_prefill**
   — each position attends to all L positions (including future),
   producing a non-causal output that differs radically from decode's
   strictly causal attention.
2. **KV append ordering** — if append happens AFTER attention is
   computed rather than before, attention sees nothing at all.
3. **RoPE position offset wrong at L>1** — position_0's RoPE ≠
   decode_position_0's RoPE.
4. **Hidden-state layout mismatch** — the per-layer output isn't
   laid out [L, hidden] row-major on hidden_a/hidden_b as assumed.

## Investigation path (V2.34 scope)

1. Disable flash-tile attention prefill (fall back to legacy
   attention_prefill_f16). Does L>1 work? If yes → flash-tile mask.
2. Run on L=2 with very explicit positions, print intermediate
   tensors at each layer.
3. Compare per-layer output against a decoded version layer-by-layer
   (via the `FLAMBEAU_PARITY_LAYER_DUMP` infrastructure used by
   V1.7.4.b).

## Blast radius

Blocks all speculative decoding work (V2.33.b-f), all chat-mode
prompt ingestion correctness, all "prefill improved" perf claims
from V2.30.a onward. **Must be fixed before any prefill path is
trusted.**

Filed as urgent. Existing flambeau shipment should not be used for
chat mode with prompts > 1 token until V2.34 closes.
