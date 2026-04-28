# MTP-2 — Integrated MTP converter landed

**Status:** shipped. `tools/convert_qwen36_mtp.py` produces a single
augmented GGUF with the Qwen3.6-27B MTP head appended. First output
verified byte-exact for base tensors.

## What landed

`tools/convert_qwen36_mtp.py` — Python script using `gguf` + `safetensors`
+ `torch` (BF16 read).

Pipeline:
1. Read MTP shards 13 + 15 from `Qwen/Qwen3.6-27B` (cache dir, ~4.2 GB).
2. Quantize MTP linears to Q8_0 via `gguf.quants.quantize`; norms stay F32.
3. Read base GGUF via `GGUFReader`; copy ALL fields and tensors verbatim
   (no dequant / requant — preserves the original Unsloth quant
   exactly).
4. Append 15 `mtp.*` tensor entries + 3 `mtp.*` metadata keys
   (`mtp.source_repo`, `mtp.num_layers`, `mtp.use_dedicated_embeddings`).
5. Write to `<basename>+mtp.gguf`.

## First conversion

| | |
|---|---|
| Base | `Qwen3.6-27B-Q4_0.gguf` (15.79 GB, 851 tensors) |
| Output | `Qwen3.6-27B-Q4_0+mtp.gguf` (16.24 GB, 866 tensors) |
| MTP overhead | +451 MB (Q8_0 linears + F32 norms) |
| Conversion time | ~32s on warm filesystem (single 16 GB stream-copy + small append) |

## Byte-exact base verification

5 representative base tensors (across dtype + size):

| Tensor | dtype | size | match |
|---|---|---|---|
| `output.weight` | Q6_K | 1042.9 MB | ✓ |
| `token_embd.weight` | Q4_0 | 715.2 MB | ✓ |
| `blk.0.attn_qkv.weight` | Q4_0 | 29.5 MB | ✓ |
| `blk.31.ffn_down.weight` | Q4_1 | 50.1 MB | ✓ |
| `blk.63.attn_norm.weight` | F32 | 0.02 MB | ✓ |

Plus structural check: 0 missing tensors, exactly 15 added (the MTP
set), all expected names, correct dtypes, expected shapes.

## MTP tensor inventory in the new file

```
mtp.fc.weight                           Q8_0   [10240, 5120]    55.71 MB
mtp.norm.weight                         F32    [5120]           0.02 MB
mtp.pre_fc_norm_embedding.weight        F32    [5120]           0.02 MB
mtp.pre_fc_norm_hidden.weight           F32    [5120]           0.02 MB
mtp.layers.0.input_layernorm.weight     F32    [5120]           0.02 MB
mtp.layers.0.post_attention_layernorm   F32    [5120]           0.02 MB
mtp.layers.0.self_attn.q_proj.weight    Q8_0   [5120, 12288]   66.85 MB  (gated: Q || gate)
mtp.layers.0.self_attn.k_proj.weight    Q8_0   [5120, 1024]     5.57 MB
mtp.layers.0.self_attn.v_proj.weight    Q8_0   [5120, 1024]     5.57 MB
mtp.layers.0.self_attn.o_proj.weight    Q8_0   [6144, 5120]    33.42 MB
mtp.layers.0.self_attn.q_norm.weight    F32    [256]            0.00 MB
mtp.layers.0.self_attn.k_norm.weight    F32    [256]            0.00 MB
mtp.layers.0.mlp.gate_proj.weight       Q8_0   [5120, 17408]   94.70 MB
mtp.layers.0.mlp.up_proj.weight         Q8_0   [5120, 17408]   94.70 MB
mtp.layers.0.mlp.down_proj.weight       Q8_0   [17408, 5120]   94.70 MB
                                                          total 451.32 MB
```

Note `q_proj` is `[5120, 12288]` (= 48 heads × 256), confirming gated
attention: half is Q, half is the gate signal (per the base model's
`attn_output_gate=true` convention). MTP-3 will need to handle the
split when wiring forward.

## Reusing the MTP blob across base variants

The expensive step is the BF16 → Q8_0 quantization (~1s for 393 MB).
For the next run on a different base variant (Q4_1, Q8_0,
UD-Q4_K_XL, etc.) we re-quantize the same MTP weights —
deterministic, identical output bytes, ~1s overhead on top of the
16-30 GB stream-copy. If we cared we could cache the quantized
blob, but the script is so fast (~30-60s per variant) that it's not
worth the complexity.

## What's gitignored

`tools/mtp_cache/` — the 4.2 GB safetensor shards. Re-downloadable
via `curl` from HuggingFace; not committed.

## Next: MTP-3

Wire the loader. flambeau needs:
- `MtpHeadWeights` struct under the existing qwen35 model.
- Optional load — if the file lacks `mtp.*` tensors, load without MTP.
- `forward_mtp_step` composed from existing kernels (rmsnorm_f32 /
  rmsnorm_f16, mmvq_q8_0, mrope, attention_decode_f16, swiglu_*,
  add_*) — no new kernels required.
- Smoke against a Python reference (HF Transformers loaded with
  `mtp_num_hidden_layers=1`).

## Sources

- [`gguf` Python package](https://pypi.org/project/gguf/) — provides `GGUFReader`, `GGUFWriter`, `GGUFValueType`, `quants.quantize`.
- [`safetensors` Python package](https://pypi.org/project/safetensors/) — read BF16 shards directly to torch tensors.
