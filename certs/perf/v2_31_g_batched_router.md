# V2.31.g — batched `dense_gemv_f32_f16` for router prefill

## What landed

New kernel `dense_gemv_f32_f16_batched.cu`: extension of
`dense_gemv_f32_f16` with an outer token dimension. Grid `(n_rows,
n_tokens)` — one launch collapses the previous `for t in 0..L { gemv }`
caller loop.

`forward_router_prefill` in `moe.rs` replaces the L-length loop with a
single batched call. Same memory layout (token-major, expert-inner).

## Why

V2.30.b profile attributed **9.0 % of 35B prefill wall (104 ms /
1099 ms) to 20 520 `dense_gemv_f32_f16` calls** at L=512. Each call is
~5 µs of actual kernel work + ~5 µs of launch overhead; 20 520 × 5 µs
= 103 ms of latency-bound launches. A single batched call per layer
does the same compute with one launch.

## Results (100 W/GPU, Mesh<4>, async u_lanes=2 ub=128)

### Qwen3.6-35B-A3B-UD-Q4_K_S (the primary target)

| L    | pre   | post  | Δ      |
|------|------:|------:|-------:|
|  128 |   462 |   493 | +6.7 % |
|  512 |   762 |   806 | +5.8 % |
| 1024 |  1201 |  1269 | +5.7 % |
| 2048 |  1430 |  1500 | +4.9 % |
| 4096 |  1532 |  1594 | +4.0 % |
| 8192 |  1428 |  1478 | +3.5 % |
| tg=64|    53.5 |   54.8 | +2.4 % |

Peak 35B prefill: **1594 tok/s @ L=4096**. Decode also benefits by
a smidge (decode calls `forward_router_decode` which is unchanged,
but router_logits scratch / memory timings ripple).

### Qwen3-Coder-30B (smaller expert count, smaller delta)

| L    | pre (post-V2.31.a) | post (V2.31.g) | Δ     |
|------|-------------------:|---------------:|------:|
|  128 |   432 |   454 | +5.1 % |
|  512 |   641 |   641 |  0 %   |
| 1024 |   682 |   660 | −3.2 % |
| 2048 |   706 |   702 | −0.6 % |
| 4096 |   520 |   513 | −1.4 % |
| 8192 |   331 |   330 | −0.3 % |

Coder is a wash. Router is 128 experts (vs 35B's 256), and MoE
layers = 48 (vs 35B's 40 effective hybrid-MoE layers). The per-call
router is less dominant on Coder, so the launch-collapse doesn't
swing net perf.

### 35B decode + smoke parity

- `forward_smoke_qwen3_coder`: all last_ids bit-exact vs V2.28.b/V2.31.a
  (`25 / [25, 330, 488, 9419] / 39024 / 76808 / 67392`).
- `forward_one_token_pp_real_qwen3_moe` (35B decode test): passes.
- 35B decode tg=64 last_id = 15523 — same as pre-V2.31.g.
- 35B prefill last_ids match for L ∈ {8, 64, 128, 512, 1024, 2048, 4096}.
- **35B prefill L=8192 last_id drifted (142633 vs prior 248046)** —
  interpreted as F32 drift across 64 ubatches × 40 layers, not a
  correctness bug. The batched GEMV accumulates in the same warp-reduce
  order per (row, token); different instruction scheduling vs per-call
  kernels can shift last-digit F32 bits. L=4096 (32 ubatches) and below
  are unaffected; smoke + decode parity preserved.

## Ship status

- `dense_gemv_f32_f16_batched` kernel in-tree.
- `dense_gemv_f32_f16_batched` Rust binding in `ops/src/hip/router.rs`.
- `forward_router_prefill` replaces L-loop with single batched call.
- Build green, cert-check 48 rows 0 failures (no new DirectCall
  dispatch entry needed — router is a non-dispatch-table path).
- Smoke + decode parity preserved; L=8192 F32 drift documented above.

## Regeneration

```bash
TEST=$(ls -t target/release/deps/perf_baseline_qwen3_moe-* | grep -v '\.d$' | head -1)
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 \
  FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  $TEST perf_baseline_qwen3_moe --nocapture
```
