# gemma4-v2 E4B — lever 2 (batched build + apply): ships, fixing
# flash-tile SWA NaN-init bug

Date: 2026-05-20
Status: **shipped.** Unblocked by fixing a pre-existing NaN-init bug
in `attention_prefill_flash_tile_f16`'s online-softmax loop.

## Bench (E4B-Q4_0, SD, prompt 716 tok, decode 128 tok)

```
                    | prefill t/s | decode t/s | output
flambeau-v2 (l2)    |     558.4   |    41.1    | coherent
flambeau-v2 (l1)    |      41.0   |    41.4    | coherent
llama.cpp           |    1045.9   |    70.6    | coherent
```

**Prefill 41 → 558 t/s (13.6×).** Decode unchanged. Now 0.53× of
llama.cpp prefill (was 0.04×) and 0.58× of llama.cpp decode.

## Root cause

`attention_prefill_flash_tile_f16.cu`'s online-softmax update:

```c
if (row >= limit || row < swa_min) {
    s_j = -INFINITY;
}
const float m_new = fmaxf(m_i, s_j);
const float alpha = gfx906_fast_exp(m_i - m_new);
const float p     = gfx906_fast_exp(s_j - m_new);
```

When a warp's **first** processed K row is masked AND no prior valid
row has been seen, `m_i = -INFINITY` (initial) and `s_j = -INFINITY`
(masked). Then `m_new = -INFINITY`, and `m_i - m_new = (-inf) - (-inf)
= NaN`. `exp(NaN) = NaN`, so `alpha` and `p` are NaN, poisoning
`o_reg` for every subsequent row.

This was a known-but-undocumented hazard — the kernel's own test in
`attn_prefill.rs` carries a comment:

```rust
// n_q_tokens < 4 → oracle kernel (flash_tile SWA has a
// NaN-init bug when `block_swa_min % BC != 0`).
```

…but the test sidesteps it by forcing dispatch to the oracle kernel
(which has a separate, correct implementation).

### Why it stayed hidden until lever 2

- Lever-1 ran prefill **n_tokens = 1 per forward** (token-by-token
  loop). At n_q_tokens=1, the dispatch picked an oracle kernel
  variant that handled SWA correctly.
- Lever-2 calls the prefill kernel at n_tokens > 1 — the path that
  hits flash_tile. Once `n_q_tokens >= 4` (BR=4 SWA dispatch), the
  buggy flash_tile path is taken.
- For E4B specifically, `sliding_window = 512`, so the bug fires once
  the prompt exceeds the window. Boundary measured at ≈ 500–512
  tokens before fix. 31B uses window=1024, which is why its prefill
  worked at 716 tokens.

## Fix

`kernels-hip/src/kernels/attention_prefill_flash_tile_f16.cu`:

```c
const bool masked = (row >= limit) || (row < swa_min);
if (!masked) {
    const float m_new = fmaxf(m_i, s_j);
    const float alpha = gfx906_fast_exp(m_i - m_new);
    const float p     = gfx906_fast_exp(s_j - m_new);
    for (i = 0..D_PER_LANE) o_reg[i] = alpha * o_reg[i] + p * v_lds[…];
    l_i = alpha * l_i + p;
    m_i = m_new;
}
```

Skip the online-softmax update entirely for masked rows. This is
mathematically equivalent (masked p is 0 → no contribution) and
avoids the `(-inf) - (-inf)` arithmetic.

Regression test added: `attn_prefill_f16_flash_tile_swa_unaligned_window`
in `model-ops/src/ops/attn_prefill.rs` — runs the flash_tile path
(n_q_tokens=8) with a window choice that makes `block_swa_min %
BC != 0`. The pre-existing `attn_prefill_f16_with_swa` test's
"n_q_tokens < 4" workaround comment is dropped (the bug is fixed,
the workaround no longer needed).

## What ships in lever 2

- Batched per-layer-embd build:
  `dense_gemv_f16_f16_batched(model_proj, main_embd, …, n_tokens)`
  writes `[n_tokens, pe * n_layer]` token-major F32; host helper
  reshapes to layer-major `[n_layer, n_tokens, pe]`.
- Batched per-layer-embd apply: `PerLayerEmbedBlock::forward_n_tokens`
  runs the gate / gelu / proj / norm / add chain at any n_tokens
  using the existing `*_batched` ops.
- `PerLayerEmbedDims::pe` carried through `ScratchConfig.per_layer_embd`
  so the pool sizes `ple_*` slots to `n_tokens × {pe, hidden}`.
- `table_dev` and `proj_matmul_f32_dev` allocations bumped to
  `PLE_MAX_TOKENS = 1024` × original size in `gemma4-v2/src/loader.rs`
  (covers default prefill_ubatch=512 + headroom for overrides up to
  1024).

## Decode-side perf follow-up still standing

- E4B decode: 41 t/s vs llama.cpp 70 t/s — same ratio as before
  lever 2 (decode path is unchanged). The 1.71× decode gap is the
  next lever target (per-layer apply fusion + GPU build elimination
  remain on the roadmap).
- Prefill at 0.53× of llama.cpp leaves room: per-layer apply still
  runs 42 × 7 = 294 kernels in the layer loop; fusing the apply
  chain to one kernel per layer is the next prefill lever.
