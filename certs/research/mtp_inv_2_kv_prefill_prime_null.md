# MTP-INV-2 — KV prefill priming: NULL on Qwen3.6-27B at this prompt

**Status:** clean null. Priming MTP attention's KV cache with the
prompt prefix neither lifts nor regresses acceptance (within 1 step
of jitter on 16-step harness). Hypothesis was wrong; the gap to
vLLM's reported >75% greedy acceptance is not from missing prefix
context.

## Setup

After MTP-INV-1 confirmed flambeau's MTP forward is structurally
correct (parity test passes vs llama.cpp-aligned Python ref), the
remaining hypothesis for the 56.2% acceptance ceiling was that
vLLM's `spec_info.hidden_states` flow primes MTP attention's KV
with the prompt prefix during prefill, while our harness only
populates KV starting at decode step 0.

`tests/mtp_acceptance_passive.rs` extended with
`FLAMBEAU_MTP_PREFILL_PRIME=1` env flag. When set, replaces
`forward_prefill_pp` with sequential `forward_one_token_pp` calls
(one per prompt token) — captures `hidden_a` after each — then runs
`forward_mtp_step` with KV append at slot `i` for each prompt
position `i`. Priming pass produces 5 KV slots (prompt_len) before
decode begins; decode loop continues from `mtp_cache_pos = 5`.

Two e-token conventions tested:

| variant | e_token at prime step `i` | acceptance (16 steps) |
|---------|---------------------------|----------------------:|
| no priming (baseline)             | n/a                                  | 9/16 = 56.2% |
| prime + lookahead                 | `embed(prompt[i+1])` (matches decode loop) | 9/16 = 56.2% |
| prime + no-lookahead (vLLM-style) | `embed(prompt[i])` (vLLM input_ids convention) | 8/16 = 50.0% |

## Diagnosis

Both priming variants change MTP's predictions at a few steps
(non-trivial KV side-effect — confirmed the priming code is
producing real values, not garbage). But the changes are
acceptance-neutral or negative:

- **Lookahead variant:** identical accept/reject pattern to baseline
  except steps 4 and 6 swap mispredicted tokens (still wrong). Net 0.
- **No-lookahead variant:** mismatched at one extra step (step 7)
  vs baseline. Net −1.

Possible explanations:
1. **MTP attention barely uses its own KV history.** The model may
   have been trained as a "single-step" MTP head; its self-attention
   over MTP-side past states adds noise rather than signal.
2. **Greedy-argmax is too brittle to benefit from the prefix.** The
   correct token is rank-1 by base; MTP either matches or doesn't.
   Adding more KV context shifts logit ranks but rarely promotes a
   different token to rank 1.
3. **Wrong KV format.** Our priming feeds (h_p, embed(token)) per
   prompt position; vLLM may use a different (h, embed) pairing or
   process the prefill in a single batched MTP call rather than
   sequentially. The two e-token variants tested cover the obvious
   candidates; deeper variants (e.g., position offsets, batched
   prefill MTP) untested.

## What's left after MTP-INV-1 + MTP-INV-2

- ~~Forward bug~~ (ruled out, parity passes)
- ~~Q8_1 quant noise~~ (ruled out by BF16 null)
- ~~F32 residual rounding~~ (ruled out by MTP-4-A)
- ~~KV prefill priming~~ (ruled out by MTP-INV-2)

Plausible remaining levers, ranked by tractability × impact:

1. **Prompt-specific ceiling.** 56.2% may genuinely be the model's
   per-token greedy acceptance on this 5-token prompt
   ("The capital of France is"). vLLM's published >75% numbers are
   typically over much longer / more diverse prompts. Cheap test:
   swap to a 50-token natural-language prompt and re-measure.
2. **Sampling-aware verify.** Production spec-decode rarely uses
   strict greedy match; vLLM's
   "verify against base distribution" rule allows MTP-drafts that
   have base-rank ≤ K to count as accepted. With K=2 or K=3,
   acceptance lifts significantly even at the same prediction
   quality. Implementation: drop `pred == next` in the loop, replace
   with "MTP draft is in base's top-K logits". Requires base logits
   download (already done in `forward_mtp_step_with_lm_head` for
   the MTP side; need to extend to the base side).
3. **Dive into vLLM source for ground truth.** Compare per-step
   MTP outputs between our forward and vLLM's at identical inputs.
   Heavy lift (separate stack), only worth it if (1) and (2) don't
   close the gap.

## Code state

- `tests/mtp_acceptance_passive.rs` — refactored to support either
  prefill path (`forward_prefill_pp` default, sequential decodes
  with MTP priming on `FLAMBEAU_MTP_PREFILL_PRIME=1`). MTP scratch
  + KV cache + embed helper hoisted before prefill block. The
  priming closure `embed_into_e_token` is reusable for the decode
  loop too (deduplicates the rank-0 → host → last-rank embed copy).
- Priming default is the lookahead convention (matches decode loop;
  matches the 56.2% baseline). Use of the env flag is optional.
- `MtpForwardScratch` fields exposed `pub` (was already done in
  MTP-INV-1) — needed by the priming code to use `h_t_post_norm`
  and `mtp_h_final` directly.

## Decision

Mark MTP-INV-2 null. Default `FLAMBEAU_MTP_PREFILL_PRIME=0` (off).
Infrastructure stays — useful for any future investigation that
wants to test a different priming strategy.

Next: try lever #1 (longer prompt) — it's the cheapest test and
would either prove a prompt-specific ceiling or establish a baseline
for further investigation.

## Sources

- [`tools/mtp_reference.py`] — Python reference (post-INV-1 fix)
- vLLM `qwen3_next.py` `Qwen3NextMTP.forward` — reference for
  spec_info.hidden_states semantics (input_ids[p] convention)
- MTP-INV-1 cert: `mtp_inv_1_python_ref_layout_bug.md`
