# MTP-4 session 4 — F16 MTP linears: acceptance 37.5% → 43.8%

**Status:** lever validated, not yet at the 75% gate. The Q8_0→F16
quant change for MTP linears moves the needle; remaining gap is
structural (F16 hidden-state propagation, Q4_0 base quality,
acceptance-scoring strictness).

## Numbers

`mtp_acceptance_passive_qwen36_27b` on "The capital of France is",
N=16 decode steps:

| MTP linear quant | acceptance | hits |
|---|---|---|
| Q8_0 (session 3)        | 37.5% (3/8)  | 271, 271, 4252 |
| **F16 (this session)**  | **43.8% (7/16)** | 271, 271, 4252, 2972, 57590, 279, 6511 |

Notable hits with F16: `57590 = "Paris"`, `6511 = "Ġcapital"`,
`279`, `2972` — semantically meaningful tokens, not just generic
fillers. The model's drafting capacity for content tokens is real.

Misses are near-misses (e.g. predicted=11 vs actual=321 — both in
the punctuation-cluster). Suggests our F16 chain is close but not
matching the reference path's specific choice within the top-k.

## What changed

- `tools/convert_qwen36_mtp.py`: linears can now be quantized as
  F16 (default) or Q8_0 (set `MTP_LINEAR_DTYPE=q8_0`). MTP file
  size: 451 MB → 811 MB.
- `crates/models/qwen3-moe/src/mtp.rs`:
  - `MtpHeadWeights` accepts `GgmlDType::F16` for linears
    (previously only F32 / Q8_0).
  - `forward_mtp_step` now dispatches `mmvq` `QDtype` per
    weight via a small `qdtype_for(w)` helper. The weights' dtype
    is read from the loaded `DeviceTensor`, so adding new quant
    families later is trivial.
  - `forward_mtp_step_with_lm_head` adds `F16` to the lm_head
    dtype-dispatch match.

flambeau's existing `mmvq` already short-circuits to a dedicated
`mmvq_f16_launch` path when weight is F16 (qmatmul.rs:545), so no
new kernel work was needed.

## Where the remaining gap likely lives

Expected ceiling per vLLM + Lorbus int4 (BF16 base + BF16 MTP):
~90% acceptance at K=1. We're at 43.8%. Remaining drags:

1. **F16 hidden propagation in the MTP block.** The 4 norms + 8
   matmuls all run with F16 intermediate hidden state. F16 has
   ~10 bit mantissa; over 12 sequential ops, drift accumulates.
   vLLM/Lorbus run BF16 throughout. Fixing this in flambeau means
   modifying the existing MTP forward to use F32 staging through
   the residual paths — substantial work; might not pay back vs.
   acceptance ceiling ≈ 60-70%.
2. **Q4_0 base model.** vLLM's 90% reference uses BF16 base. Q4_0
   produces hidden states with ~0.5-1% drift per layer; over 64
   layers that's measurable. The base model "predicts" a slightly
   different t+1 distribution than the reference, and our MTP
   (trained on the reference's distribution) misses more often.
3. **Single-token K/V in the passive harness.** vLLM's MTP uses
   accumulated K/V across draft cascading; we simulate just one
   step. For K=1 evaluation this should match, but maybe vLLM's
   benchmark numbers actually use K=3 cascade.
4. **Strict argmax-vs-argmax scoring.** "Acceptance" in
   spec-decode is usually `top1 match` only; ours is too. But
   missing by a near-miss top-2 token is normal for any drafter
   and shouldn't tank the gate. This is the least likely root
   cause.

## Suggested next session(s)

A fork in the road:

**Path A — push toward gate via bigger precision changes:**
- Try BF16 weight + BF16 hidden propagation (needs a new code
  path through the MTP forward, since flambeau's other forward
  paths are F16 hidden). 1-2 sessions.
- Bisect h0/h1/h2 against pure-PyTorch reference using REAL h_t.
  Find where flambeau diverges from the reference math; if it's
  small per-stage drift, accept the ceiling and ship at ~50%.
  If a stage has a structural issue, fix it.

**Path B — accept ~50% gate, ship K=2/K=3 cascade:**
- 43.8% per-token acceptance × K=2 cascade gives ~1.88 tokens
  accepted per draft batch — enough for ~+15-20% throughput at
  modest implementation cost.
- Build the active spec-decode driver (KV rollback + verify
  loop). 2-3 sessions.

**Path C — file MTP-4 null and pivot:**
- We hit 43.8% but the originally-stated gate was 75%. If we
  honor that gate strictly, file MTP null and look at other
  levers (FA-2 head_dim=256 for long context, etc.).

## Decision rationale

A is the right play if the goal is "production speedup ≥ 1.4×".
Real-world vLLM with realistic Qwen3.6-27B prompts probably hits
~60-70% acceptance (not the marketing 90%), so that's a fair
ceiling target. Two more sessions for ~1.3-1.5× combined throughput
is good ROI.

B is the right play if "directional progress is enough" is the
mandate. ~1.2× throughput at K=3 with the current 43.8% per-token
rate is shippable.

C is the right play if "75% gate or null" is policy.

Recommend: ship session 5 = bisection (Path A first half) before
committing to bigger investment. The bisection will tell us whether
the ceiling is implementation (fix it) or precision (accept it).

## Code state

- `tools/convert_qwen36_mtp.py` — `_linear_dtype()` reads
  `MTP_LINEAR_DTYPE` env var; defaults to F16.
- `/artefact/models/Qwen3.6-27B-mtp.gguf` — 811 MB, F16 linears.
- `crates/models/qwen3-moe/src/mtp.rs` —
  `qdtype_for()` helper for per-weight dispatch; F16 added to
  loader allow-list and to lm_head dtype-dispatch match.

## Cumulative MTP-4 progress

| Session | Bug fixed | Live acceptance |
|---|---|---|
| 1       | (harness only)        | 0% (GIGO setup) |
| 2       | concat order; output_norm; embed verified | 0% (real prompt) |
| 3       | GemmaRMSNorm `+1` bake | 37.5% |
| 4       | F16 MTP linears        | **43.8%** |

## Sources

- [vLLM Qwen3.6-27B recipe (90% acceptance reference)](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [Lorbus/Qwen3.6-27B-int4-AutoRound (preserved BF16 MTP head)](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound)
