# V2.34 — `forward_prefill_pp` at L>1 produced wrong output — **CLOSED 2026-04-26**

## Status: closed

## Root cause

`crates/ops/src/hip/qmatmul.rs:124` (pre-fix). The `qmatmul` MMVQ
row-loop computed the per-token activation stride from
`nb_per_row = k / block_elems(weight_dtype)`:

```rust
let nb_per_row = k / block_elems(dtype_weight);
let act_row_bytes = nb_per_row * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
```

`nb_per_row` is correct as the **kernel argument** (it counts WEIGHT
superblocks per row), but it is wrong for the **activation row
stride**: activations are Q8_1 with `QK8_1=32` elements per block,
not `block_elems(weight_dtype)`. The two coincide for
`Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0` (block_elems=32), but for
**K-quants** (`Q4_K / Q5_K / Q6_K`, block_elems=`QK_K=256`) the
activation stride was **8× too small**, so iteration `i ≥ 1` of the
MMVQ row-loop read partway into row 0's activation rather than from
row `i`'s activation.

## Why it stayed latent

- **Decode** (m=1) runs the loop body once with `i=0`; bug invisible.
- **MMQ paths** (m ≥ 128) take the `RecipeKind::Mmq*` branches that
  pass the row-multiplexing into the kernel via grid.y rather than
  iterating in Rust; bug invisible.
- **Prefill at small L (2..127) with K-quant weights** was the only
  path that exercised the buggy row-loop. Every prior parity cert
  ran `L=1 prefill + decode-loop`, so this path was never tested
  end-to-end until `prefill_L_sweep.rs` and
  `prefill_vs_decode_parity*` landed.

The bug **predates** the V2.30.a / V2.31.g / V2.28.b commit batch
that the user initially suspected. It has been present since the
first qmatmul implementation that bundled a Rust-side row-loop for
multi-row MMVQ. Mistakenly framed as a TP-related regression because
TP work was the first time multi-token prompt prefill was being
exercised end-to-end.

## Affected models

Any model that has a K-quant weight (`Q4_K / Q5_K / Q6_K`) reachable
through `forward_*_prefill` at L in [2, 127]:

- **Qwen3.5/3.6 hybrid** (qwen35moe arch): `ssm_out` is `Q5_K` on
  every Q4_1 / Q4_K / Q4_0 / UD-Q4_K_S GGUF — every GDN layer's last
  qmatmul. → wild outputs at every t≥1.
- **Qwen3-Coder-30B UD-Q4_K_XL** (qwen3moe dense): every attention
  projection (`attn_q / attn_k / attn_v / attn_output`) is `Q4_K`,
  every dense FFN gate/up/down is `Q4_K` or `Q5_K`. → wild outputs
  at every t≥1.

## Fix

`crates/ops/src/hip/qmatmul.rs:120-141` — separate the two strides:

```rust
let nb_per_row = k / block_elems(dtype_weight);
let act_blocks_per_row = k / 32; // QK8_1 = 32 elements per Q8_1 block.
let act_row_bytes =
    act_blocks_per_row * std::mem::size_of::<flambeau_quant::BlockQ8_1>();
```

The fix landed at the same time as this cert was closed.

## Regression cert

```
prefill_L_sweep on Qwen3.5-27B-Q4_1 (4×MI50, ROCm 7.1.1):
  L=1..16  : 10/10 prefill argmaxes match decode stream

prefill_vs_decode_parity on Qwen3-Coder-30B-UD-Q4_K_XL:
  pos 0..7 : 8/8 prefill argmaxes match decode stream

prefill_vs_decode_parity_27b on Qwen3.5-27B-Q4_1:
  pos 0..7 : 8/8 prefill argmaxes match decode stream
```

## Lessons

- "Two block-size constants that look interchangeable but are not"
  is a class of bug worth grepping for. `QK_K` (256) and `QK8_1`
  (32) differ only when the weight is a K-quant; any code that uses
  `block_elems(weight_dtype)` to compute an **activation** stride is
  suspect.
- A single-token decode parity cert is **not** a multi-token prefill
  parity cert. Every multi-row code path (MMVQ row-loop, batched
  router gemv, attention prefill, KV append) needs its own L>1
  cert — the V1.7.4.b 8-token parity that flambeau has shipped under
  is misleadingly named: it tests `L=1 prefill + 7 decode steps`,
  not `L=8 prefill`.
- Suspect-ranking failure mode: the original cert ranked
  "causal mask wrong / KV append order / RoPE position" — all
  attention-shaped hypotheses. The actual bug was in qmatmul's
  row-loop activation stride, an order of magnitude further from
  the L-axis than the attention block. The bisect that found it
  was: V/Q/K post-split match, state_step output matches,
  swiglu/out_normed/gated_f32 match, gated_q8_1 (block 0 of token 1)
  matches decode bit-equivalent, but ssm_out_f32 diverges. The
  divergence narrowed to the qmatmul invocation for the Q5_K weight,
  which let us spot the wrong stride formula.
