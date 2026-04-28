# MTP-2 — MTP converter landed (thin sibling-file mode)

**Status:** shipped. `tools/convert_qwen36_mtp.py` emits a small
standalone `Qwen3.6-27B-mtp.gguf` (431 MB on disk, 451 MB tensor
content) that the flambeau loader pairs with any base Qwen3.6-27B
GGUF at runtime. Optional `--integrate-into <base>` mode produces a
single combined file for users who prefer that.

**Why thin/sibling, not integrated**: integrated mode would
write a full ~16 GB duplicate per base variant the user owns — across
7 known variants (Q4_0 / Q4_1 / Q8_0 / UD-Q3_K_XL / UD-Q4_K_XL /
UD-Q6_K_XL / UD-Q8_K_XL), that's ~3.2 GB of MTP duplication AND
several minutes of stream-copy I/O per variant. Thin mode is
**+431 MB on disk total** and **<1s to generate**, paired with all
variants for free at runtime.

## What landed

`tools/convert_qwen36_mtp.py` — Python script using `gguf` + `safetensors`
+ `torch` (BF16 read).

**Default (thin) pipeline:**
1. Read MTP shards 13 + 15 from `Qwen/Qwen3.6-27B` (cache dir, ~4.2 GB).
2. Quantize MTP linears to Q8_0; norms stay F32.
3. Write a small standalone GGUF with `general.architecture =
   "qwen35-mtp"`, the 15 mtp.* tensors, and pairing metadata
   (`mtp.target_arch`, `mtp.target_hidden_size`,
   `mtp.target_vocab_size`, `mtp.source_repo`, `mtp.num_layers`,
   `mtp.use_dedicated_embeddings`).

**Optional (`--integrate-into <base>`) pipeline:** copy a base GGUF
verbatim (no dequant of base tensors) and append the MTP entries
into a single combined file. Produces `<base>+mtp.gguf`. Earlier
session ran this once on Q4_0 and verified byte-exact base
preservation across 5 sampled tensors (Q4_0 / Q4_1 / Q6_K / F32);
deleted to save disk now that thin mode is canonical.

## Output (thin mode)

| | |
|---|---|
| File | `/artefact/models/Qwen3.6-27B-mtp.gguf` (431 MB, 15 tensors) |
| Generation time | <1 s (after the one-time ~4.2 GB shards download) |
| Pairs with | any `qwen35` base GGUF: Q4_0, Q4_1, Q8_0, UD-Q3/4/6/8_K_XL, … |
| Disk overhead across all base variants | +431 MB **once** |

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

## Pairing semantics for the loader (MTP-3)

When loading any qwen35 base GGUF, the flambeau loader looks for a
sibling MTP file via this rule:

1. If the loader sees `mtp.*` tensors in the base GGUF (integrated
   mode), use those.
2. Else, look for `<basename>-mtp.gguf` next to the base. If found,
   open as a second GGUFReader and verify pairing metadata:
   - `mtp.target_arch` matches base's `general.architecture`
   - `mtp.target_hidden_size` matches base's `qwen35.embedding_length`
   - `mtp.target_vocab_size` matches base's vocab size from
     `tokenizer.ggml.tokens.count` (or equivalent)
3. If sibling exists but mismatches, fail loud with a "MTP/base
   mismatch — re-run convert_qwen36_mtp.py" error rather than load
   silently with wrong weights.
4. If neither integrated nor sibling found: load without MTP
   (spec-decode unavailable; eager decode still works).

The user can also pass `--mtp <path>` to override the auto-pair
location.

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
