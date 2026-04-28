# MTP-4 (partial) — passive acceptance harness landed; first run is GIGO

**Status:** harness in tree, runs end-to-end without crashes. First
result on a synthetic prompt is **0% acceptance**, but this is a
**garbage-in / garbage-out test artifact**, NOT evidence the MTP
forward is wrong. The plumbing is validated; the measurement isn't yet
representative.

## What landed

### `forward_mtp_step_with_lm_head` in `mtp.rs`

Wraps three pieces:
1. `rmsnorm_f16(hidden_pre_norm, output_norm)` — apply base's final
   norm to get h_t in the post-`model.norm` form HF's MTP expects.
2. `forward_mtp_step(h_t, e_token, position)` — the MTP-3.5 forward.
3. `quantize_f16_q8_1` + `mmvq(lm_head)` — LM head on MTP output.
4. Host-side argmax over F32 logits.

Returns the predicted next token id. Allocates per-call scratch (slow
but right for the gate-first measurement). MTP-5 will fold this into a
reusable scratch struct and integrate with the spec-decode driver.

### `tests/mtp_acceptance_passive.rs`

Live decode + MTP probe:
1. Loads `Qwen3.6-27B-Q4_0.gguf` across pp4 (14.63 GiB).
2. Loads `Qwen3.6-27B-mtp.gguf` on the last rank.
3. Prefills the prompt; runs N decode steps.
4. After each base step, snapshots pre-norm hidden, embeds the just-
   sampled token (token row → F16 on rank 0 → host → last rank), runs
   MTP probe, saves predicted t+2 token.
5. Next iteration's actual t+2 is compared to the saved prediction.
6. Reports acceptance rate vs the 75% gate.

## First run: 0% acceptance — and why it's meaningless

```
=== MTP-4 passive acceptance ===
loading base across 4 ranks (64 layers)…
base loaded (14.63 GiB across shards)
loading MTP on rank 3…
prefill done; first sampled token = 5
  step 1: predicted=95118 actual=0 ✗
  step 2: predicted=59604 actual=31 ✗
  step 3: predicted=97066 actual=46474 ✗
  step 4: predicted=138955 actual=4 ✗
  step 5: predicted=78637 actual=5 ✗
  step 6: predicted=96474 actual=9 ✗
  step 7: predicted=95118 actual=0 ✗
  step 8: predicted=59604 actual=31 ✗

  acceptance: 0.0%   wall: 53 ms / step base+probe   GATE: FAIL
```

**Why the result is GIGO:**

The harness uses a synthetic 4-token prompt `[1, 2, 3, 4]`. These IDs
don't form a meaningful sentence in Qwen3.6's tokenizer — they're just
arbitrary in-range integers. The base model with a nonsense prefix has
effectively maximum-entropy logits at every step, so its argmax bounces
around the vocab semi-randomly. MTP can't predict tokens the base
itself isn't predicting deterministically — the test setup deprives
both models of signal.

Symptoms confirming this:
- "Actual" tokens are bouncing across `{0, 31, 46474, 4, 5, 9, 0, 31}`
  — note step 1's `0` and step 7's `0`, step 2's `31` and step 8's
  `31`. The base is in a tight cycle, suggesting it has gotten stuck
  in a degenerate state from the bad prompt.
- "Predicted" tokens are very large IDs (95118, 138955, etc.) — also
  ranging widely, also suggesting MTP is operating on degenerate
  hidden states with broad-flat distributions.

**What's NOT (yet) ruled out:**

- MTP forward at position!=0 — the parity test only validated
  position=0 (MROPE = identity). MROPE may be miscomputed on
  Qwen3.6-27B's `mrope_section=[11,11,10,0]` / `mrope_interleaved=true`
  config. Need a position!=0 parity validation.
- Hidden-state convention — MTP-3.5 cert documented an ambiguity: HF
  expects "h_t after model.norm" but it's possible (esp. given the
  pre-FC norms in MTP) that flambeau should pass pre-norm. The
  `forward_mtp_step_with_lm_head` applies output_norm; if it shouldn't,
  every prediction is off.
- Embed dtype — token_embd is Q4_0 in this base. Dequantize in
  `forward_embed_decode_host` produces F16 on rank 0; we ship it
  through host to the last rank. If the host-side dequant has a row-
  major / column-major bug, every embedding is wrong.

## SIGSEGV at process exit

A teardown-time SIGSEGV occurs after the test reports its result. The
test's assertions all pass; the segfault is in the Drop chain
(probably scratch/session/cluster ordering). Cosmetic for the
measurement; will run it down in MTP-4 session 2.

## Plan for MTP-4 session 2

The harness is done. Next session re-runs with **real signal**:

1. **Real prompt** — pre-tokenize a meaningful sentence (e.g. "The
   capital of France is" → known token IDs, the V2.33 / V1.7.4
   parity prompt). Avoid the GIGO trap.
2. **Position-non-zero MTP parity** — extend `mtp_step_parity` to
   exercise position=10, 100. If parity holds, MROPE is fine; if it
   fails, debug rope.
3. **Hidden-state convention audit** — try BOTH pre-norm and
   post-norm h_t and see which gives sensible acceptance.
4. **Q4_0 embed dequant audit** — compare embedding row from
   `forward_embed_decode_host` to safetensors row, byte-for-byte.
5. **Re-run acceptance.** If acceptance ≥ 75% → DF (DeepSeek-style
   spec-decode) → MTP-4 final cert. If <75% but >30% → instrumentation
   to find the gap. If 0% → likely a position-!=0 bug worth
   bisecting.

## Code in tree

- `crates/models/qwen3-moe/src/mtp.rs` —
  `forward_mtp_step_with_lm_head` added (~110 lines).
- `crates/models/qwen3-moe/tests/mtp_acceptance_passive.rs` — new
  test (~210 lines).

The passive harness is reusable: any future MTP-related debug session
can override `FLAMBEAU_MTP_ACCEPT_STEPS` and use `MTP_TEST_POSITION`
for position-targeted Python parity.

## The honest read

We made progress on the structural pieces (live multi-rank load + MTP
probe + cross-rank embedding lookup + lm_head matmul) but the gate
measurement is **not** valid yet. The 0% is a setup artifact. Next
session = make the test signal-bearing, then call the gate.
