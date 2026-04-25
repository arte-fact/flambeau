# V2.28.b — Qwen3-Coder-30B-A3B-Instruct loads + forwards (arch=qwen3moe)

## New architecture supported

Qwen3-Coder-30B-A3B-Instruct uses the **`qwen3moe`** arch tag (pure
transformer MoE, no GDN/SSM, no shared expert — distinct from the
`qwen35moe` hybrid we've shipped for Qwen3.6-35B).

Config (from GGUF):
- `general.architecture = qwen3moe`
- block_count=48, hidden=2048, heads=32/4 (GQA), head_dim=128
- num_experts=128, top_k=8, expert_ff_length=768
- **no `ssm.*`, no `full_attention_interval`, no shared expert**
- rope_freq_base=10M, rotated_dims=head_dim (no multi-freq sections)

Every layer is full-attention + routed-MoE. The key structural difference
from qwen35moe's "gated full-attention" layers:

|                      | qwen35moe (hybrid, full-attn layer) | qwen3moe (Qwen3-Coder) |
|----------------------|-------------------------------------|-------------------------|
| `attn_q.weight` rows | `2 * n_heads * head_dim` (Q\|gate)  | `n_heads * head_dim`    |
| post-attn gate       | `sigmoid(gate_f16) * attn_out`      | none                    |
| Q/K/V biases         | none                                | optional                |
| RoPE sections        | present (multi-freq, partial)       | absent (standard NeoX)  |

## What landed

### V2.28.b-i0 — load smoke (`certs/perf/` n/a, cert is this doc)

- `Qwen3MoEConfig::from_gguf` parses `qwen3moe` as `AttentionFamily::Dense`.
- `Qwen3MoEShardedModel::load` walks `LayerAttnBlock::Dense`, resolves
  q/k/v/output/q_norm/k_norm + optional biases, uploads across Mesh<4>.
- **Result:** `Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf` loads in
  **13.9s at 16.45 GiB total / 4.11 GiB per rank** on 4× MI50.
- Test: `crates/models/qwen3-moe/tests/load_smoke_qwen3_coder.rs`.

### V2.28.b-i1 — dense attention forward

- `forward_dense_attn_decode` + `forward_dense_attn_prefill` in
  `crates/models/qwen3-moe/src/forward/attn.rs` — trimmed copies of
  the gated full-attn pair:
  - plain Q projection (shape `[n_heads*head_dim, hidden]`, no split)
  - no `sigmoid_mul_f16` post-attn
  - Q/K/V biases assert-absent (Qwen3-Coder-30B has none; future Qwen3
    variants with biases need a bias-add step)
  - reuses existing `FullAttnScratch` / `FullAttnPrefillScratch`
    (q_fused_f16/gate_f16 unused on this path, ~128 KiB dead per layer)
- `forward_layer_decode` / `forward_layer_prefill` dispatch on
  `AttnWeights::Dense` vs `::FullAttn` variant.
- `LayerForwardScratch::new` + `LayerPrefillScratch::new` now allocate
  `GdnScratch` / `GdnPrefillScratch` only when `cfg.gdn.is_some()`
  (qwen3moe has no GDN).

### V2.28.b-i2 — Q5_K indexed-MoE MMVQ + 1-token + short-L smoke on real weights

UD-Q4_K_XL mixes Q4_K (35/48 layers) and **Q5_K (13/48 layers)** for
`ffn_down_exps`. V2.28.b adds:
- new kernel
  `crates/kernels-hip/src/kernels/indexed_moe_mmvq_q5_k.cu` —
  combines the Q4_K indexed-MoE pointer math with the Q5_K 5th-bit
  (`qh` byte, mask `(hi_half ? 2 : 1) << (2*grp)`) decoder from
  `mmvq_q5_k.cu`. 64 threads/block, 1 output row per block.
- `indexed_moe_mmvq_q5_k` registered in `crates/ops/src/hip/moe.rs`
  and `crates/backend-hip/src/impls.rs` (DirectCallKernel).
- `validate_moe_dtypes` + `run_indexed_moe_down` + MoE prefill
  dispatcher extended with Q5_K.

**Forward smoke (100 W/GPU, Mesh<4>):**

```
L=1 prefill seed 9419 → last_id = 25
greedy 4 tokens → [25, 330, 488, 9419]
L=2 prefill → last_id = 39024
L=4 prefill → last_id = 76808
L=16 prefill → last_id = 67392
```

All deterministic; test: `crates/models/qwen3-moe/tests/forward_smoke_qwen3_coder.rs`.

## Deferred

### V2.28.b-i3 — parity cert vs llama.cpp

Needs a llama-server reference run with seed 9419 on same GGUF to
compare token-by-token. **Not done this session** — thermal budget is
tight; filing as separate cert once the head-to-head is captured.

### Other follow-ups

- **Q/K/V biases in dense attention** — `forward_dense_attn_*` asserts
  `attn_q_bias.is_none()` etc. Some older Qwen3 variants have biases;
  add a `bias_add_f32` step before casting to F16 when we hit a model
  that carries them.
- **Q5_K indexed-MoE cert sweep** — the new kernel currently has no
  correctness cert under `certs/hip/gfx906/`. The `DirectCallKernel`
  entry points to a cert path that doesn't yet exist; a `bench sweep`
  run authors it. Flagged as i4.
- **Q4_1 indexed-MoE** — needed for `Qwen3-Coder-30B-A3B-Instruct-1M-Q4_0.gguf`
  whose `ffn_down_exps` promote to Q4_1 on 6/48 layers. Same pattern
  as Q5_K; punted.
- **Perf baseline** — smoke test doesn't record tok/s. A proper
  `perf_baseline_qwen3_coder.rs` is V2.28.b-i5.

## Ship status

- `Qwen3MoEConfig::from_gguf` accepts `qwen3moe`.
- `Qwen3MoEShardedModel::load` uploads the full model to Mesh<4>
  (16.45 GiB, 4.11 GiB/rank, 13.9 s).
- `forward_one_token_pp` + `forward_prefill_pp` run end-to-end on
  Qwen3-Coder-30B weights at L ∈ {1, 2, 4, 16}.
- Build green.
- Load + forward smoke tests in-tree and passing.
- Q5_K indexed-MoE MMVQ kernel added (no perf sweep / no cert file yet).
