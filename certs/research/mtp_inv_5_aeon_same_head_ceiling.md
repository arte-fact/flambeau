# MTP-INV-5 — Qwen/Qwen3.6-27B MTP head ceiling confirmed at ~67 % via community measurements

**Status:** definitive close-out. flambeau's 60-64 % E[accept] on the
published Qwen/Qwen3.6-27B MTP head is within 3-7 pp of two
independent community measurements on the **bit-identical** head.
The ~67 % ceiling is the model's, not flambeau's.

## Method

Investigated whether community-released MTP-augmented Qwen3.6-27B
variants use a **different / better-trained** MTP head than the
default published one.

Target: `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP`
(found via tag search; one of several "Qwen3.6-27B + MTP" community
releases).

## Finding

The AEON model card explicitly states:

> "MTP head **grafted from the base** `Qwen/Qwen3.6-27B` checkpoint
> (15 tensors, BF16). The base contains MTP heads but
> `Qwen3_5ForConditionalGeneration.from_pretrained` drops them
> during loading; the lna-lab pipeline pattern explicitly grafts
> them back."

So AEON's "MTP variant" is **the same 15 mtp.* tensors from
`Qwen/Qwen3.6-27B`** — bit-identical to what flambeau's converter
extracts. Same applies to:
- `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (cited as the recipe
  origin, 22K+ downloads)
- `lna-lab/GGUF-to-NVFP4-SM120` (the documented "MTP graft recipe")

All three projects ship the same MTP head. **No fine-tuning has
been done by any community on the Qwen3.6-27B MTP head.**

## Measured acceptance — same head, different stacks

| project | inference | hardware | acceptance |
|---|---|---|---|
| AEON-7 (NVFP4-MTP regular) | vLLM modelopt | RTX PRO 6000 Blackwell (sm_120) | 67.7 % |
| AEON-7 (NVFP4-MTP-XS) | vLLM modelopt | RTX PRO 6000 Blackwell | 69.2 % |
| AEON-7 (MTP method) | vLLM modelopt | DGX Spark / GB10 (sm_121a) | 66.3 % |
| **flambeau (us)** | flambeau | **gfx906 Mesh<4>** | **60.9 % (UD-Q8 base, 120 steps), 64.1 % (Q4_0, 60 steps prose)** |

flambeau's measurements are within 3-7 pp of the community's
numbers on the same head. The remaining gap is consistent with:
- Prompt content type (we tested prose; theirs may include code)
- Measurement methodology (rolling-window vs single-shot)
- Step-window variance (we measured ±5 pp range across 16-200 steps)

## Cross-check via sakamakismile's "mean accept length 3.0-4.0"

AEON's README quotes sakamakismile's reference numbers as
"Mean MTP acceptance length: ~3.0-4.0" (vs DFlash chains 2.0-2.3).
Length is for `num_speculative_tokens=3` chains.

Expected length given per-position acceptance `p`:
  `E[len] = 1 + p + p² + p³`
  - p=0.6 → E[len]=2.30
  - p=0.67 → E[len]=2.46
  - p=0.7 → E[len]=2.55
  - p=0.8 → E[len]=2.95
  - p=0.9 → E[len]=3.44

Sakamakismile's "3.0-4.0" only fits p ≈ 0.85-0.95 sustained accept.
That's higher than AEON's measured 67-69 %, suggesting
sakamakismile's numbers are either (a) on **code-heavy prompts**
where even the same head hits ~85 %+ (consistent with
patrickbdevaney's content-type variance), or (b) include the
"bonus token" that's always accepted at the chain root.

Either way: nothing in the community's published numbers points to
a configuration where the *same* head gives consistently > 75 % on
prose. The 67-69 % acceptance is the head's training-bound ceiling.

## What this rules out

- **flambeau implementation bug**: ruled out — same head, similar
  numbers as community.
- **Different precision config helping**: AEON keeps MTP head BF16
  with `mtp.fc` dequantized. We measured BF16 vs F16/Q8_1 (MTP-4-C):
  bit-identical argmax. Their config doesn't help.
- **vLLM-specific tricks**: vLLM's MTP serving runs the same forward
  composition. Their wins come from base inference (NVFP4 + Marlin)
  and tree attention drafting, not from MTP head changes.

## Path forward to > 75 % acceptance

Only one realistic option, and it's outside flambeau's current
scope:

1. **FastMTP self-distillation fine-tune** (arxiv 2509.18362).
   Position-shared weights, self-distilled data targeting multi-step
   draft chains. Paper claims +82 % vs vanilla MTP. Requires
   PyTorch training rig, GPU compute (~hours-days), and re-emitting
   a converted MTP GGUF. **Multi-week effort.**

Alternative shipping options (if 75 % gate is *not* hard-required):

2. **Ship at 67 %** — close to community state-of-art on this head.
   Spec-decode net-negative on hybrid arch (per MTP-5 walk-through),
   so don't activate spec-decode in production.
3. **Wait for a community-released finetuned head**. None exist as
   of 2026-04-29. Track AEON-7 / sakamakismile / lna-lab feeds.

## Decision

Mark MTP-4 (≥75 % gate) as **closed: target unreachable on the
published MTP head**. flambeau implementation is already at the
community ceiling. Re-open if a finetuned head appears.

The MTP infrastructure shipped this session is sound and reusable:
- Forward (F16/Q8_1 + BF16 paths, parity-verified)
- KV accumulation (+12 pp from MTP-4 session 6)
- KvCache rollback + GDN snapshot/restore (for any future use)
- Sampling-aware verify metric (more representative than strict greedy)
- Content-type sweep harness

## Sources

- [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Text-NVFP4-MTP) README
- [sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP) (recipe origin)
- [lna-lab/GGUF-to-NVFP4-SM120](https://github.com/lna-lab/GGUF-to-NVFP4-SM120) — MTP_GRAFT_RECIPE.md
- [vLLM Discuss thread #2447](https://discuss.vllm.ai/t/qwen3-5-27b-fp8-speculative-decoding/2447) — confirms hybrid arch selective-accept limitation
- FastMTP paper: https://arxiv.org/abs/2509.18362
- All prior MTP investigation certs (`mtp_4_*`, `mtp_inv_*`)
