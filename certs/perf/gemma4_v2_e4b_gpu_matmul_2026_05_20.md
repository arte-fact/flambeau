# gemma4-v2 E4B — GPU-side per_layer_model_proj matmul (perf lever 1)

Date: 2026-05-20
Status: **prefill 9.6 → 41.1 t/s (+330 %); decode 9.5 → 41.4 t/s (+335 %).**

E4B's per-token side-channel build was CPU-bound on the BF16 matmul
`per_layer_model_proj [10752, 2560] @ inp_batch [2560]` (~5.5 ms /
token on scalar CPU). Moving that one step to GPU as
`dense_gemv_f16_f16` (~10 µs / token) collapses the per-token CPU
cost from ~5.6 ms to ~150 µs (Q5_K row dequant + rmsnorm + add still
on CPU).

## Bench (gemma-4-E4B-it-Q4_0, SD, 725-tok prompt, 64-tok decode)

| stack          | prefill t/s | decode t/s | v2 / llama.cpp |
|----------------|------------:|-----------:|----------------|
| flambeau-v2    |   **41.1**  |  **41.4**  | 0.039x / 0.59x |
| llama.cpp (SD) |     1047.1  |     70.5   | —              |

Decode is **0.59× of llama.cpp** — within 1.7× and usable for
short-context interactive use. Prefill is still 25× behind because
the n=1 token-by-token prefill loop is intact (the layer-by-layer
forward + per-layer-embd apply both still run once per token).

Coherence preserved: `"The capital of France is"` → `"Paris."` (greedy
temp 0, 12-tok max).

## Change

1. **Loader**: cast `per_layer_model_proj` BF16/F32 → F16 once at
   load and upload `[pe * n_layer, hidden]` F16 to device. Allocate
   `proj_matmul_f32_dev` scratch of `pe * n_layer * 4` bytes. The
   raw bytes Vec for `per_layer_model_proj` (55 MB) is dropped after
   the F16 cast so RAM stays bounded.
2. **`ForwardCtx::per_layer_embd_build_table`**: replace `model_proj_raw`
   + `model_proj_dtype` params with `model_proj_f16_dev` +
   `proj_matmul_f32_dev` device pointers. Engine impl:
   - `dense_gemv_f16_f16(model_proj_f16, main_embd.ptr,
     proj_matmul_f32, n_rows=pe*n_layer, k=hidden)`
   - DtoH the `[pe * n_layer]` F32 matmul output (~43 KB)
   - Hand off to the new
     `flambeau_blocks::per_layer_embd::build_inp_per_layer_table_with_proj`
     host helper, which finishes the build (Q5_K dequant + rmsnorm
     + add + scale) without the matmul step.
   - HtoD upload to `table_dev` (existing).
3. **New host helper**
   `build_inp_per_layer_table_with_proj(tok_embd_row_raw, …,
   proj_matmul_f32, …)` accepts the precomputed matmul output. The
   old `build_inp_per_layer_table` stays for legacy gemma4.

The new path replaces an inner CPU loop of `total × hidden` =
`10752 × 2560 ≈ 28 M` BF16→F32 multiplies (~5.5 ms scalar) with one
HIP `dense_gemv_f16_f16` launch (~10 µs at MI50's F16 throughput).

## Remaining gap

Decode is at 0.59× llama.cpp. Two further levers:

1. **Batched n_tokens table build + batched per-layer apply**: closes
   the prefill 25× gap. The model.rs prefill currently loops
   `forward(n=1)` once per prompt token because both the table build
   and the per-layer apply are decode-only (n=1). The build is
   trivially batchable (run `dense_gemv_f16_f16_batched` against
   `[n_tokens, hidden]` instead of `[hidden]`); the apply is the
   harder side, the PerLayerEmbedBlock would need an `n_tokens`-aware
   forward path.
2. **GPU-side Q5_K row lookup + rmsnorm in the build**: moves the
   remaining ~150 µs / token CPU work onto the GPU. Modest decode
   gain (~5 %) but eliminates the DtoH/HtoD pair per token.

For now: E4B is functionally complete and within 1.7× of llama.cpp
on decode. Prefill at 41 t/s is still slow on a long prompt but the
short-context decode case is usable.
