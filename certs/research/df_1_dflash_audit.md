# DF-1 — DFlash GGUF audit (Qwen3.6-27B target + draft)

**Verdict: pivot to native MTP head.** DFlash drafter has three blockers
the original Path A scoping didn't surface:

1. F16 acceptance ceiling **43%** (per spiritbuun's README) — already
   below the DF-4 ≥50% gate, before we add Q8_0 quant + hardware drift.
2. Draft is a **block-diffusion** model that consumes **target hidden
   states from 5 specific layers** ([1, 16, 31, 46, 61]). Not a
   token-in/token-out drafter. Token-level interop requires
   instrumenting flambeau's per-layer activation pipeline to tap and
   export hidden states — invasive.
3. Draft uses **sliding-window attention** (window=2048, pattern
   `[S,S,S,S,F]`). Flambeau has no SWA today; would need new kernel.

Combined, DF-2..DF-5 is closer to 6-8 sessions than 4, and the gate is
likely to fail anyway. Native MTP from base safetensors is the
better-priced bet.

## Target — `Qwen3.6-27B-Q4_0.gguf` (15.79 GB)

- **arch:** `qwen35` (already supported by flambeau — same arch family
  as the V1-bench Qwen3.5-27B-Q4_1 entry).
- **64 blocks**, hybrid GDN + full-attn (`full_attention_interval = 4`
  → 16 full-attn + 48 GDN per the same pattern as Coder-Next).
- **Hidden 5120, ffn 17408, head_dim 256, 24 attn heads, 4 KV heads.**
- **Dense FFN** (no `ffn_*_exps`, no shared expert) — different from
  the qwen35moe / qwen3next paths.
- **MROPE** (`rope.dimension_count = 64`, `dimension_sections = [11,11,10,0]`).
- Vocab 248320, tokenizer pre `qwen35`.
- SSM (GDN): inner=6144, state=128, conv_kernel=4, group=16, time_step_rank=48.
- Tensor packing per GDN block: `attn_qkv` + `attn_gate` + `ssm_*` + dense FFN.
  Per full-attn block: `attn_q/k/v/output` + `attn_q_norm/k_norm` + dense FFN.

## Draft — `dflash-draft-3.6-q8_0.gguf` (1.85 GB)

- **arch:** `dflash-draft` (NOT in upstream llama.cpp, requires
  `spiritbuun/buun-llama-cpp` fork at commit `b9d01582b+`).
- **5 blocks** (`block_count=5`), 1.7B params, Q8_0.
- **Hidden 5120, ffn 17408** — match target.
- **Vocab 248320, tokenizer pre `qwen35`** — match target.
- **head_dim 128** (vs target's 256), 32 attn heads, 8 KV heads
  (vs target's 24/4) — DIFFERENT geometry; can't reuse target's
  attention scratch.
- **RoPE dim 128, freq 10M** (regular RoPE; NOT MROPE) — different
  from target.
- **Sliding-window attention:** window=2048,
  pattern=`[true,true,true,true,false]` — 4 SWA layers + 1 full.
  Flambeau has no SWA support today.
- **Block diffusion drafting:** `dflash.block_size = 16`,
  `dflash.mask_token_id = 248070` — generates 16 tokens at once via
  iterative masking, not greedy AR.
- **Hidden-state taps:** `dflash.target_layer_ids = [1, 16, 31, 46, 61]`,
  `dflash.n_target_features = 25600` (= 5 × 5120). Draft input is
  the **concatenation of target hidden states from those 5 layers**.
  No token embedder of its own.
- Special tensors:
  - `dflash_fc.weight [5120, 25600]` — projects concatenated
    target features down to draft hidden.
  - `dflash_hidden_norm.weight [5120]` — norm before draft layers.
  - `output_norm.weight [5120]` — norm before LM head.
  - **No `token_embd.weight`** — draft doesn't tokenize.
  - **No `output.weight`** — draft reuses the target's LM head
    (vocab matches).
- Per-block tensor families (5 of each): `attn_q/k/v/output` +
  `attn_q_norm/k_norm` + `attn_norm/post_attention_norm` +
  `ffn_gate/ffn_up/ffn_down` — standard llama-style transformer.

## Interop requirements

**What matches (good):**
- Vocab size (248320) and tokenizer (qwen35) → token-level
  pass-through OK.
- Hidden size (5120) → target hidden states feed draft directly.

**What flambeau needs to add for DF-2..DF-5:**

| Requirement | Lift | Notes |
|---|---|---|
| `arch=dflash-draft` loader path | medium | Standard llama-style 5-block transformer, but with dflash-specific tensors (`dflash_fc`, `dflash_hidden_norm`) and no token embedder. |
| Sliding-window attention kernels | **high** | New `attention_decode_f16_swa` + `attention_prefill_f16_swa`. Or extend existing kernels with mask arg + offset. SWA window=2048 needs careful KV-cache addressing. |
| Cross-rank hidden-state taps | **high** | During target forward, export hidden states from layers `[1, 16, 31, 46, 61]` to a contiguous 25 600-F16 buffer. In PP/TP topologies these layers live on different ranks → cross-rank gather (peer copy or AR). The current forward has no tap points for this. |
| Block-diffusion draft generation | **high** | Generate 16 tokens at once via mask-token replacement. Iterative refinement. Different scheduling from V2.33's greedy AR draft. |
| Target LM head reuse on draft output | low | Vocab matches; just route draft's `output_norm`-ed hidden through target's `output.weight` matmul. |
| KvCache::rollback (relands V2.33) | low | Target KV is the authoritative cache; rollback on mismatch. We had this once, it was reverted. |
| Draft KV cache | low | 5-layer KV cache for draft, separate from target's. |

## Why the F16 ceiling matters

spiritbuun's published numbers:
- Q8_0 → ~43% acceptance (matches F16 reference)
- Q4_K_M → ~28% acceptance (degraded by quant)

DF-4 gate was set at ≥50% (proxy for "≥1.3× combined throughput").
Even the F16 ceiling is below it. Real-world hardware drift + our
TP/PP topology overhead would only erode further. The acceptance
ceiling here is **architectural**, not implementation.

For comparison, vLLM with **Qwen3.6-27B native MTP** (Lorbus int4
quant that preserves MTP in BF16) reports **~90% acceptance, ~2×
throughput**. The MTP-head architecture is structurally better-suited
than block-diffusion drafters.

## Recommendation

**Close DF as filed-null.** Land this audit as the "considered &
rejected" cert. Open MTP-1..MTP-5 to pursue the native MTP head from
base Qwen3.6-27B safetensors:

- MTP-1: audit Qwen/Qwen3.6-27B safetensors — list MTP head tensor
  names, document the math, no code.
- MTP-2: custom safetensor → GGUF converter that preserves MTP
  weights under a `mtp.*` namespace flambeau recognizes.
- MTP-3: load MTP weights, smoke single MTP step (no spec decode
  yet) against a Python reference.
- MTP-4: speculative driver + acceptance-rate measurement
  (HARD GATE ≥80% — MTP's higher ceiling lets us tighten the gate).
- MTP-5: TP/PP integration on Qwen3.6-27B.

Path B has been worse-priced than originally sketched (custom
converter is an extra session) but the **acceptance ceiling alone**
makes it the dominant choice now.

## Files in tree

- `/artefact/models/Qwen3.6-27B-Q4_0.gguf` — target, already on rig.
- `/artefact/models/dflash-draft-3.6-q8_0.gguf` — draft, downloaded
  this session (1.85 GB). Keep on disk for potential future pivot if
  MTP also fails; small enough to live alongside.

## Sources

- [spiritbuun/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF) — Q8_0 ≈ F16 ≈ 43% acceptance; Q4_K_M ≈ 28%.
- [llama.cpp PR #22105 — DFlash support](https://github.com/ggml-org/llama.cpp/pull/22105) — upstream merge pending.
- [Lorbus/Qwen3.6-27B-int4-AutoRound](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — MTP in BF16, ~90% acceptance via vLLM.
