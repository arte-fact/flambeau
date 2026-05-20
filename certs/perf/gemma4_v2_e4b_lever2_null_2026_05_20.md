# gemma4-v2 E4B — lever 2 (batched build + apply): null, reverted

Date: 2026-05-20
Status: **null result.** Batched per-token build + per-layer apply
shipped a 14× prefill speedup AND a coherence regression at
n_tokens > ~500. Reverted to lever-1 state; the stash holds the WIP
for the next session to debug.

## What was attempted

The plan (from `gemma4_v2_e4b_gpu_matmul_2026_05_20.md`):

1. Generalise the host build helper to `n_tokens > 1` — produce a
   layer-major `[n_layer, n_tokens, pe]` F32 table from
   `n_tokens` consecutive `per_layer_token_embd` rows + the batched
   GPU matmul output (`dense_gemv_f16_f16_batched`).
2. Generalise `PerLayerEmbedBlock::forward_decode` to
   `forward_n_tokens` — `dense_gemv_f32_f16_batched` for both
   projections, flat-`n = n_tokens × dim` gelu_mul / cast / add,
   `n_rows = n_tokens` for rmsnorm.
3. Drop the n=1 token-by-token prefill loop in `gemma4-v2/src/model.rs`.

All four pieces compiled and shipped. The decode path (n=1) preserved
correctness and the bench number (~41 t/s decode), matching lever-1.

## Bench at the attempted state

```
                    | prefill t/s | decode t/s | output
flambeau-v2 (l2)    |     567.9   |    42.5    | "<pad><pad>..."  ← BROKEN
flambeau-v2 (l1)    |      41.0   |    41.4    | coherent
llama.cpp           |    1050.7   |    70.9    | coherent
```

Prefill jumped from 41 → 568 t/s (1280 %, 0.54× of llama.cpp); decode
unchanged. **But output is incoherent pad-tokens.**

## Failure boundary

Coherence cuts off at exactly `n_tokens > ~500`:

| prompt tokens (n) | output                                 |
|------------------:|----------------------------------------|
|   19 ("Hi")       | `"Hello! How can I help you today?"`   |
|   29 ("…Paris")   | `"Paris."`                             |
|  319              | `"OK."`                                |
|  499              | `"OK."`                                |
|  529              | `"<pad><pad><pad><pad>…"`              |
|  559              | `"<pad><pad><pad><pad>…"`              |

The threshold sits suspiciously close to the default
`FLAMBEAU_PREFILL_UBATCH=512`, but the bug **is not chunking**: a
single-chunk run with `FLAMBEAU_PREFILL_UBATCH=4096` and a 559-token
prompt **still** produces pad output. So the bug is real-batched-path
at large n_tokens, not chunk-boundary state leakage.

## What was ruled out

- **Chunking**: single-chunk run at n=559 fails too (covered above).
- **Decode regression**: n=1 path verified working at 42 t/s.
- **KV-share interaction**: the existing kv-share fix is per-layer
  and doesn't change with n_tokens; 26B-A4B (no kv-share, no
  per-layer-embd) prefill batching is fine, so the bug is specific
  to the per-layer-embd batched path.
- **Scratch overflow**: PLE scratch is sized
  `n × max(pe, hidden) × f32/f16` from `ScratchConfig.max_prefill_tokens`
  which equals `prefill_ubatch`; for n_tokens up to that limit the
  scratch fits.

## Suspects for next session

1. **Host build layout transpose bug**: the GPU matmul writes
   `[n_tokens, pe * n_layer]` token-major; the host helper expects
   the same input but emits layer-major `[n_layer, n_tokens, pe]`.
   Re-derive the index math from the kernel docstring — there might
   be a token/layer stride swap that only surfaces when n_tokens > 1.
2. **gridDim.y at large n_tokens**: `dense_gemv_*_batched` launches
   `gridDim = (n_rows, n_tokens)`. For the build matmul, n_rows =
   `pe * n_layer = 10752`; HIP enforces gridDim.x ≤ 2³¹ but the
   per-CU work-item budget may exceed something at the high row
   count + n_tokens > 500.
3. **rmsnorm_f16 with n_rows = n_tokens**: the apply rmsnorm runs
   over `n_tokens` rows of `hidden`. The single-token decode tested
   `n_rows = 1`. For `n_rows = 559` something might overflow.

The stash is at:
`git stash show stash@{0}` →
"lever-2 batched-build-and-apply WIP (broken at n_tokens > ~500)"
covering 6 files (blocks, ctx, engine, gemma4-v2 loader+model, +
the bench JSON).

## What ships in lever-1 (commit `feecc71`)

Current state on `feature/gemma4`:

| stack       | prefill | decode | output                                |
|-------------|--------:|-------:|---------------------------------------|
| flambeau-v2 |   41.0  |  41.4  | coherent                              |
| llama.cpp   | 1050.7  |  70.9  | coherent                              |

Decode is at 0.58× llama.cpp and usable for short-context interactive
work. Prefill is 25× behind; lever 2 was the right fix in shape but
the implementation has a latent bug that needs another session.

## Reading

Lever 2 didn't ship, but the design is right: batched per-layer-embd
is the unblocker for E4B prefill perf. The decode-side change
(`PerLayerEmbedBlock::forward_n_tokens` with n=1 codepath) is
correctness-preserving and could ship independently if the apply
batching is the broken side. Recommended approach next session:

1. Isolate which of build vs apply is broken — write a CPU reference
   for the apply at n_tokens > 1 and `assert_close` it.
2. If apply is fine, run the build host helper + matmul outputs
   through a Python sanity check (FP32 reference of
   `per_layer_model_proj @ inp_batch[t]` for known t).
3. If both look right, the bug is in the rmsnorm-with-n_rows path or
   a kernel limit; instrument the kernels with `printf` at one (t, il)
   pair.
