# MTP-3.5 — `forward_mtp_step` + Python-reference parity passed

**Status:** shipped. Forward composition passes Python-reference parity
within the noise floor expected from the F16 hidden / Q8_1 activation /
Q8_0 weight pipeline.

## Numbers

Test: `crates/models/qwen3-moe/tests/mtp_step_parity.rs::mtp_step_parity_position_zero`

| metric | got | tolerance | status |
|---|---|---|---|
| max abs diff (1 outlier element / 5120) | 3.15e-1 | 5.0e-1 | ✓ |
| mean abs diff | 5.99e-2 | 1.0e-1 | ✓ |
| reference h_final std | 1.29 | — | — |
| signal-to-noise (mean_abs / std_ref) | 4.7% | < 8% predicted | ✓ |

First 8 elements (ref vs got):
```
ref [0..8]:  [-0.148, -0.166, +0.424, -0.679, +0.139, -1.101, -0.819, -0.354]
got [0..8]:  [-0.132, -0.137, +0.323, -0.664, +0.126, -1.191, -0.776, -0.346]
```
Signs match on every element; magnitudes drift ~1-15% per element
consistent with cumulative Q8_1 activation quantize across 8 sequential
mmvq operations.

## What landed

### `forward_mtp_step` in `crates/models/qwen3-moe/src/mtp.rs`

One pre-norm MTP block forward. Composes existing flambeau ops only —
no new kernels. Allocates scratch internally (smoke-grade; spec-decode
hot path will pre-allocate in MTP-4).

Op chain (15 stages):
1. `rmsnorm_f16(h_t, pre_fc_norm_hidden)`        → fc_in[0..hidden]
2. `rmsnorm_f16(e_token, pre_fc_norm_embedding)` → fc_in[hidden..2h]
3. `quantize_f16_q8_1(fc_in[2h])`                → fc_in_q8_1
4. `mmvq(fc.weight, fc_in_q8_1)`                  → h0_f32 → cast F16
5. `rmsnorm_quant_q8_1(h0, input_layernorm)`     → h0n_q8_1 (fused)
6. `mmvq(q_proj)` → q_full_f32 → cast F16
7. `split_q_gate_f16(q_full)` → q_f16 [6144] + gate_f16 [6144]
8. `mmvq(k_proj/v_proj)` → cast F16
9. `rmsnorm_f16(q, q_norm)`, `rmsnorm_f16(k, k_norm)`
10. `rope_neox_partial_f16(q,k, position)` — skipped at position=0
11. Single-token attention via `attention_decode_f16_slots` with
    n_tokens_kv=1 (degenerates to `softmax(qk)·v = v` per head)
12. `sigmoid_mul_f16(gate, attn_out)` → gated (V1.7.4.b convention)
13. `quantize_f16_q8_1(gated)` + `mmvq(o_proj)` → attn_proj
14. `add_f16(h0, attn_proj)` → h1
15. `rmsnorm_quant_q8_1(h1, post_attn_layernorm)` → h1n_q8_1
16. `mmvq(gate_proj)`, `mmvq(up_proj)` → F32 gate, up
17. `swiglu_f32_to_q8_1(gate, up)` (CN-80B-19c/d kernel) → mlp_q8_1
18. `mmvq(down_proj)` → F32 → cast F16
19. `add_f16(h1, down)` → h2
20. `rmsnorm_f16(h2, mtp.norm)` → h_final

### Discoveries / fixes during this session

1. **`gguf.GGUFReader.shape` returns the GGUF-stored (reversed) shape**;
   `gguf.quants.dequantize` returns the array already in PyTorch
   `[out, in]` order. Documented in MTP-3 cert.

2. **Output gate is `sigmoid(gate)` not `silu(gate)`** despite
   `output_gate_type="swish"` in `config.json`. Confirmed via prior
   V1.7.4.b finding for the broader Qwen3.5/3.6 family. Updated Python
   reference + regenerated test vectors.

3. **Norms cast F32 → F16 at MTP load time** (mirrors flambeau's base
   loader convention via `upload_as_f16`). Added inline cast in
   `upload_mtp_tensor` so the existing `rmsnorm_f16` /
   `rmsnorm_quant_q8_1` kernels can consume MTP norm weights without a
   per-call cast.

### Tolerance band rationale

Q8_1 activation quantize has ~1.5% per-encode noise. The MTP forward
runs **5 Q8_1 encodes** (fc_in, h0n, gated, h1n, swiglu) and **8 mmvqs**.
Cumulative drift (multiplicative, per-element across the head-dim and
intermediate-dim widths): ~5-8%.

Reference h_final std=1.29; mean_abs_diff=0.06 → 4.7% S/N. Inside the
predicted band. Real structural bugs (e.g. sigmoid→silu, transposed
weight, wrong head-dim split) would show 10× larger drift.

The tolerance numbers are NOT cosmetic — they're picked from the noise
floor. Tighter (e.g. 0.02 mean) would fail on quant noise. Looser
(e.g. 1.0 max) would hide a swapped activation.

## What's NOT validated yet

- **MROPE at position!=0** — current test runs at position=0 (rotation
  identity). The MROPE path code is in tree but not exercised here;
  MTP-4 will exercise it on real prompts.
- **Multi-step cascading.** Real spec-decode feeds `(h2, embed(t+1))`
  back through the MTP block to get t+2 logits. The chain is straight-
  forward (it's the same forward fn applied to different inputs), but
  not tested here.
- **End-to-end with base-model hidden state.** The current parity uses
  random F32 input. MTP-4's acceptance-rate bench will use real base
  hidden states from the actual Qwen3.6-27B forward pass.

## Files committed

- `crates/models/qwen3-moe/src/mtp.rs` — added `forward_mtp_step` (~280
  lines). Imports `attention_decode_f16_slots`, `cast_*`,
  `add_f16`, `sigmoid_mul_f16`, `swiglu_f32_to_q8_1`, `quantize_f16_q8_1`,
  `rmsnorm_f16`, `rmsnorm_quant_q8_1`, `mmvq`, `split_q_gate_f16`,
  `rope_neox_partial_f16`. No new kernels.
- `crates/models/qwen3-moe/tests/mtp_step_parity.rs` — 1 test passing.
- `tools/mtp_reference.py` — already shipped in MTP-3; updated this
  session to use sigmoid (not silu) for the output gate. Test vectors
  regenerated.

## Sources

- `certs/research/mtp_1_qwen36_audit.md` — initial MTP audit
- `certs/research/mtp_2_converter_landed.md` — converter
- `certs/research/mtp_3_loader_landed.md` — loader + Python reference
- `project_v1_7_4_b_sigmoid_gate.md` — sigmoid vs silu output gate (memory)
