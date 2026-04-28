# MTP-4 session 2 — two bugs fixed via vLLM+sglang source audit; gate still failing

**Status:** Two real bugs fixed and verified against both vLLM and
sglang reference implementations. Live acceptance still 0% — there's
at least one more bug. Three unrelated unresolved items from
session 1 are now closed.

## What got resolved this session

### ✓ Concat order (the silent one)

The `pre_fc_norm_*` outputs are concatenated as `[embedding, hidden]`
per **both** reference implementations:

vLLM (`vllm/model_executor/models/qwen3_next_mtp.py:86`):
```python
hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
```

sglang (`python/sglang/srt/models/qwen3_next_mtp.py:116`):
```python
hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
```

I had been using `[hidden, embedding]` order in BOTH the Python
reference AND the Rust forward. The MTP-3.5 parity test passed
because the two wrong-orderings cancel out — they don't expose the
bug. Fix landed in `tools/mtp_reference.py` and
`crates/models/qwen3-moe/src/mtp.rs::forward_mtp_step`.

### ✓ output_norm is applied BEFORE MTP

vLLM `vllm/model_executor/models/qwen3_next.py:531` (the base
Qwen3NextModel forward):
```python
hidden_states, _ = self.norm(hidden_states, residual)
```
sglang `python/sglang/srt/models/qwen3_next.py:886` matches.

The base model applies `self.norm` (i.e. `output_norm`) BEFORE
returning the hidden state to the spec-decode caller. So when
`Qwen3NextMultiTokenPredictor.forward(hidden_states, ...)` runs, its
input is **post-`model.norm`**.

flambeau's `forward_one_token_pp` writes the **pre**-norm hidden into
`scratch.hidden_a` (output_norm is folded into the LM-head call via
`rmsnorm_quant_q8_1`), so `forward_mtp_step_with_lm_head` re-applies
`output_norm` standalone before invoking MTP. (My first MTP-4 attempt
correctly applied output_norm; my second attempt — based on a
misreading of a WebFetch summary — removed it; the third correctly
restores it after re-reading vLLM line 531.)

### ✓ Q4_0 token_embd dequant is bit-exact (ruled out)

```
[diag] embed(11751) flambeau first 8: [0.01464844, 0.024414063,
                                         0.0048828125, -0.01464844, ...]
[diag] embed(11751) python    first 8: [0.01465, 0.02441,
                                         0.00488, -0.01465, ...]
```

Match to 5 decimals — the F16 cast accounts for the trailing-digit
noise. flambeau's host-side Q4_0 dequant in
`forward_embed_decode_host` correctly produces the same row values
as gguf python's `gguf.quants.dequantize`. Embedding lookup is not
the bug.

### ✓ MROPE position-invariance for single-token attention

Position-100 parity passing was initially confusing (same h_final as
position=0 within tolerance). The reason is **mathematical**, not a
bug: in single-token self-attention with no KV history, Q and K are
both rotated by the same RoPE angle θ at the same position. The dot
product `Q · K` is rotation-invariant, so the attention output is
invariant to position. This means the single-token parity test
**cannot detect** MROPE bugs — but it also doesn't indicate any.
Multi-token MTP attention (when MTP accumulates its own KV across
spec-decode draft steps) WILL exercise MROPE; need a different test
to validate it.

## What's still unresolved — and the live acceptance is still 0%

After the concat fix + output_norm fix, the live test on a real
prompt (`[760, 6511, 314, 9338, 369]` = "The capital of France is")
shows:

```
prefill emits 11751 (ĠParis)   ← base model is sane
step 1: predicted=241617 actual=271 ✗
step 2: predicted=185675 actual=248068 ✗
...
acceptance: 0% (12/12)
```

Base produces sensible structured output (Paris + chain). MTP
predictions are still nonsense, with some attractor (`185675` repeats,
`241617` repeats). Different IDs from the pre-fix run, so the math
is different — fixes are landing — but still wrong.

**Suspects ranked by likelihood:**

1. **MTP "decoder layer" composition** — the sglang code on line 71-76
   builds the MTP block via `Qwen3NextModel(config, ..., is_nextn=True)`
   with `num_hidden_layers=1, full_attention_interval=1`. This means
   the MTP block IS literally a base-model layer (full-attention,
   gated), but WITH `is_nextn=True` flag. The flag may toggle
   architectural details we haven't accounted for (e.g. position-id
   handling, attention mask, head-dim split). Worth examining what
   `is_nextn` changes.

2. **MTP attention KV history** — vLLM/sglang likely accumulate K/V
   across spec-decode draft steps in the MTP attention. Our test runs
   MTP with a 1-token KV cache (just the current step). At the FIRST
   draft step they should match, so this isn't the root cause of step
   1's failure, but is structurally different and may compound.

3. **Hidden state normalization quality** — the base output_norm is
   F16 in flambeau (cast at load); vLLM/sglang use BF16 or F32. If
   the F16 cast adds enough noise to the post-norm hidden, the MTP
   `pre_fc_norm_hidden` (a SECOND norm) won't recover it. We have a
   parallel concern with the Q8_0 quant of MTP weights — but the
   parity test passed with these, so they should be okay.

4. **Position offset bug** — we pass `position+1` as the MTP rope
   position. For single-token attention this is invariant (see
   above), so this can't matter for our test. But it'd matter once
   we have multi-step KV.

## What to do next session (MTP-4 session 3)

1. **Compare a MIDDLE-LAYER intermediate against vLLM/sglang.** Run
   our MTP forward on a fixed (h_t, e_token) and dump h0_f32 (the
   fc output). Compare to a reference computed in pure PyTorch using
   the same dequantized Q8_0 weights. If h0 mismatches, the bug is
   in concat / fc / norms upstream. If h0 matches, the bug is in
   the transformer block downstream.

2. **Audit `is_nextn=True`** in vLLM/sglang's Qwen3NextModel — see
   what flag-gated paths it enables.

3. **Try BF16 hidden state path.** Cast h_t in F32, do output_norm
   in F32, then cast to F16 only for the MTP block. If acceptance
   jumps, the precision was the issue.

4. **Multi-step parity (the "real" gate test)**: If the single-step
   forward is correct, run on a longer prompt and bisect at which
   step things break.

## Code state

- `tools/mtp_reference.py` — concat order fixed.
- `crates/models/qwen3-moe/src/mtp.rs` — `forward_mtp_step` concat
  order fixed; `forward_mtp_step_with_lm_head` re-applies output_norm.
- `crates/models/qwen3-moe/tests/mtp_acceptance_passive.rs` — uses
  real "The capital of France is" prompt; embedding dequant
  diagnostic in tree.

## Position-invariance demonstration (helps the next debugger)

For self-attention at one token (Q = K from same position):
- RoPE rotates Q by R(θ_pos) and K by R(θ_pos), same θ.
- `Q · K = (R Q_0) · (R K_0) = Q_0 · K_0` (rotation preserves dot product).
- So the attention output is independent of position when there's no
  history.

This means our single-token parity at position=100 IS by design
identical to position=0. It is NOT a validation that MROPE is
implemented correctly — only that the math is invariant. Catch
MROPE bugs requires multi-token Q-vs-history-K attention.

## Sources

- [vLLM qwen3_next.py base forward (line 531)](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py)
- [vLLM qwen3_next_mtp.py MTP forward (lines 79-107)](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py)
- [sglang qwen3_next_mtp.py MTP forward (lines 88-130)](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next_mtp.py)
- [sglang qwen3_next.py base norm (line 886)](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next.py)
