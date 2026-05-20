# S3 piece-2 — V unit-RMSNorm + KV append fused

Date: 2026-05-20
Status: **shipped.** Coherent at short and long contexts.

## Change

New kernel `kernels-hip/src/kernels/kv_append_v_unit_norm_f16.cu`:
fused V unit-RMSNorm + K direct-copy + V normed-copy to KV-cache
slot, in one launch. Wrappers for head_dim ∈ {64, 128, 256, 512}.

For gemma4 (`weights.attn_v_unit_norm_w.is_some()`), replaces:

```
old: rmsnorm_f16(v_f16 → v_tmp)
   + DtoD memcpy(v_tmp → v_f16)
   + kv_append_f16 (2× DtoD memcpy: K + V to cache slots)
new: kv_append_v_unit_norm_f16 (1 launch, both K and V in cache)
```

Net per layer per token: 4 launches → 1 launch.

Per (token, kv_head): one block. Loads K (direct copy to cache) and
V (read for both rmsnorm reduction and the final norm-then-write).
K writes happen pre-sync; V writes happen post-reduction.

## Measurement (E4B-Q4_0 SD, prompt 291 tok, decode 128 tok, hip:3)

3-run streaming bench, same prompt:

| run | prefill t/s | decode t/s |
|---:|---:|---:|
| 1 | 584.7 | 47.3 |
| 2 | 590.5 | 47.3 |
| 3 | 591.8 | 47.3 |
| **avg** | **589** | **47.3** |

vs post-S3-1:

|  | prefill | decode |
|---|---:|---:|
| post-S3-1 | 588 | 46.8 |
| post-S3-2 | **589** | **47.3** |
| Δ | flat | **+1.1 %** |

Smaller than the 2.4 % back-of-envelope. Reason: the saved DtoD
memcpys were ~1 KB each (E4B GQA kv_width=512); driver-side launch
overhead dominates over data-bandwidth even after fusion. The
real saved cost is just the 3 launch headers (~5 µs each × 5376
layer-token = ~80 ms / 3.1 s = 2.5 % expected). The measurement
came in lower (~1.1 %) — likely because the fused kernel's per-
block work (K-copy + V-reduce + V-norm) sequentialises operations
that the GPU previously overlapped at the queue level. Still a
real win, just half the projection.

## Cumulative perf vs llama.cpp on E4B

|  | flambeau-v2 | llama.cpp | ratio |
|---|---:|---:|---:|
| prefill | 589 t/s | 1046 t/s | 0.56× |
| decode  | **47.3 t/s** | 70.6 t/s | **0.67×** |

Cumulative L1+S2+S3-1+S3-2 on E4B decode: **41.1 → 47.3 = +15.1 %**.

## Coherence

- Short: `"The capital of France is"` → `"Paris."` ✓
- Long 128-token decode on 291-token compiler prompt: coherent
  technical text matching the request.

## Files

- `crates/kernels-hip/src/kernels/kv_append_v_unit_norm_f16.cu`
- `crates/ops/src/hip/{mod,attention,ops_impl}.rs`
- `crates/ops/src/ops_trait.rs`
- `crates/forward/src/core/composites/standard_attn.rs` — branched
  switch to the fused kernel when `attn_v_unit_norm_w` is set;
  non-gemma4 archs (Qwen3, qwen3-moe) retain `kv_append_f16`.

## Remaining S3 piece-3

The attn-post-norm + residual-add fuse is the next sub-piece if S3
continues. Per-layer per-token win shape similar to S3-1
(rmsnorm of delta then add to resid, currently 2 launches + 1
intermediate buffer reuse).
