# MTP-3 — MTP loader landed (forward parity deferred to MTP-3.5)

**Status:** loader + Python reference shipped; 4 Rust integration
tests pass. Forward composition + bit-parity vs Python reference is
the next step.

## What landed this session

### Python reference

`tools/mtp_reference.py` — pure-PyTorch implementation of one MTP
step using weights loaded directly from `Qwen3.6-27B-mtp.gguf`. Runs
on a fixed-seed input and emits raw F32 binary test vectors:

```
crates/models/qwen3-moe/tests/data/mtp_ref/
  h_t.bin                (5120 × 4 = 20480 B)         — base hidden state
  e_token.bin            (5120 × 4 = 20480 B)         — next-token embedding
  expected_fc_in.bin     (10240 × 4 = 40960 B)        — pre-fc concat
  expected_h0.bin        (5120 × 4 = 20480 B)         — fc output
  expected_attn_pre_gate.bin  (6144 × 4 = 24576 B)
  expected_attn_post_gate.bin (6144 × 4 = 24576 B)
  expected_h1.bin        (5120 × 4 = 20480 B)         — post-attn residual
  expected_h2.bin        (5120 × 4 = 20480 B)         — post-MLP residual
  expected_h_final.bin   (5120 × 4 = 20480 B)         — pre-LM-head hidden
  inputs.json            — config + seed + h_final stats
```

Sanity numbers from the reference run at position=0 (MROPE = identity):
- `h_final.shape = (5120,)`, `mean = -0.018`, `std = 1.277`,
  range `[-6.3, +5.4]`. Sensible activation magnitudes.

### Discovery: GGUF shape orientation

Empirically verified that `gguf.GGUFReader.shape` returns the GGUF-stored
shape (which is REVERSED from PyTorch convention: GGUF stores `[in, out]`,
PyTorch wants `[out, in]`). However:
- `gguf.quants.dequantize(t.data, Q8_0)` returns the array **already in
  PyTorch `[out, in]` order** without any reshape.
- `np.array(t.data)` for F32 1D tensors is also in PyTorch order.

So the right pattern is: dequantize, **do not reshape** to the
GGUF-reported shape. Use the array as-is.

Confirmed by byte-by-byte compare: GGUF-loaded `mtp.fc.weight`
treated as `(5120, 10240)` matches the safetensors source `(5120,
10240)` within Q8_0 quant noise (max abs diff ~3.4e-3, mean ~5e-5 —
expected since Q8_0 is lossy vs BF16 source).

### Rust loader

`crates/models/qwen3-moe/src/mtp.rs` — new module providing:

- `MtpHeadWeights` struct with all 15 tensors as `DeviceTensor`,
  organized as `{ fc, norm, pre_fc_norm_hidden, pre_fc_norm_embedding,
  block: MtpBlockWeights }`.
- `MtpBlockWeights` for the single MTP transformer block (q/k/v/o,
  q/k norms, gate/up/down, layernorms).
- `verify_pairing(mtp_file, base_arch, base_hidden, base_vocab)` —
  checks `general.architecture == "qwen35-mtp"`, plus the
  `mtp.target_arch / target_hidden_size / target_vocab_size`
  metadata keys against the base config.
- `load_mtp_head(mtp_file, device)` — uploads all 15 tensors to a
  single device. Linears stay Q8_0; norms stay F32. ~451 MB total.
- `derive_sibling_mtp_path(base_path)` — strips trailing quant
  suffixes (Q4_0, UD-Q4_K_XL, etc.) and appends `-mtp.gguf`.
- `MTP_TENSOR_NAMES: &[&str]` — canonical list for unit tests.

### Rust integration tests (4/4 passing)

`crates/models/qwen3-moe/tests/mtp_load_smoke.rs`:

| Test | Purpose | Status |
|---|---|---|
| `mtp_load_smoke` | Load all 15 tensors to device, assert shape + dtype + total bytes (~451 MB) | ✓ |
| `mtp_pairing_metadata_present` | Verify `mtp.*` metadata keys are stamped in the GGUF | ✓ |
| `mtp_pairing_rejects_mismatch` | Wrong arch / hidden / vocab → fail loud | ✓ |
| `sibling_path_derivation` | Derives `Qwen3.6-27B-mtp.gguf` from any quant variant | ✓ |

## What's deferred to MTP-3.5 (or merged into MTP-4)

The forward composition `forward_mtp_step(target_hidden, next_token_embed, position) -> draft_logits`
plus bit-parity check against the Python reference. This is real
work because of activation-precision plumbing that the existing
flambeau ops surface for free in other forward paths but needs
custom orchestration here:

1. **Q8_1 input quantize for MMVQ.** flambeau's `mmvq_q8_0` expects
   activation in Q8_1 blocks (32 elements per block, F16 scale + sum +
   32 int8). Need `quantize_f16_q8_1` or `quantize_q8_1` (both exist)
   on `[norm_h ‖ norm_e]` and on `h0n` etc.
2. **Gated-attention split.** `q_proj`'s output is `[Q ‖ gate]` (12288
   = 6144 + 6144). The forward needs to slice the F32 output (or
   take pointer offset on the F32 buffer) before per-head reshape +
   q_norm + MROPE. The `attn_qkv` path in GDN does something similar
   for `[Q ‖ K ‖ V]`; can model after that.
3. **Single-token attention with no KV history.** The `attention_decode_f16`
   kernel expects KV cache. For MTP we either (a) feed it a
   single-step KV cache the MTP just wrote to position 0, or (b)
   inline the math (single-token attention is trivially `softmax(qk)
   · v` which collapses to just `v` for one token). For the
   isolated smoke test, (b) is simpler; for MTP-4 spec-decode we
   need (a) so the chain extends.
4. **MROPE partial @ position!=0.** The reference test uses position=0
   (rope identity). MTP-4 will exercise the full MROPE path with
   real positions; flambeau's `rope_neox_partial_f16` already
   handles this — just need to call it correctly.
5. **Bit-parity tolerance.** Q8_0 quant noise at the Python ref vs
   flambeau's F16-pipeline noise creates a baseline drift around
   1e-2 max-rel-err. Need to set the tolerance band carefully — too
   tight catches noise, too loose hides bugs.

These are 5 well-scoped items, all using existing kernels. ~3-4
hours of focused work. Reasonable to fold into MTP-4's first
session before the spec-decode driver.

## Files committed

- `tools/mtp_reference.py` — PyTorch reference + test-vector emitter.
- `crates/models/qwen3-moe/src/mtp.rs` — Rust loader (~310 lines).
- `crates/models/qwen3-moe/tests/mtp_load_smoke.rs` — 4 integration tests.
- `crates/models/qwen3-moe/src/lib.rs` — added `pub mod mtp`.
- `crates/models/qwen3-moe/tests/data/mtp_ref/*.bin` + `inputs.json` —
  reference test vectors (~210 KB total). These are regenerable from
  the Python script; checking them in lets the Rust test run without
  Python in the loop.

## Sources

- Earlier session's audit: `certs/research/mtp_1_qwen36_audit.md`
- Converter: `certs/research/mtp_2_converter_landed.md`
