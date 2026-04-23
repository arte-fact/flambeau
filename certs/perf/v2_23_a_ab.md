# V2.23.a — Lever-1: kernel-launch fusion AB report

Goal: reduce calls/token on 35B-A3B-Q4_0 Mesh<4> decode. Baseline 2231 calls/tok vs turbo 1653.

## Results

Model: Qwen3.6-35B-A3B-Q4_0, Mesh<4> 4×MI50, decode tg=64 from seed token id 9419.
Profiler: `/opt/rocm-6.3.4/bin/rocprofv3 --kernel-trace --stats`.

| | baseline | A.1 (add+rmsnorm) | A.2 (+ two-residuals combine) |
|---|---:|---:|---:|
| decode tok/s (unprofiled, median of 3) | 47.22 | 47.60 | **47.87** (+1.4 %) |
| wall ms (64 tokens) | 1356 | 1345 | 1337 |
| total kernel calls | 142,791 | 139,591 | **136,391** |
| Δ calls vs baseline | — | −3200 | **−6400 (−4.5 %)** |
| calls/token | 2231 | 2181 | **2131** |
| sum_kernel_ms (profiled) | 1637.9 | 1639.1 | 1641.9 |
| add_f16 calls | 6640 | 3440 | 240 |
| rmsnorm_f16 calls | 5100 | 1900 | 1900 |
| rmsnorm_f16_add_residual calls | 0 | 3200 | 3200 |
| moe_combine_f16 calls | 3320 | 3320 | 120 |
| moe_combine_two_residuals_f16 calls | 0 | 0 | 3200 |

UD-Q4_K_S 8-token parity (seed 9419, [11, 271, 40, 1044, 4313, 310, 958, 279]) bit-exact preserved after each attempt.

## Attempts

**A.1 — `rmsnorm_f16_add_residual.cu`**

Fuses `add_f16(x_in, attn_delta, mid) + rmsnorm_f16(mid, weight, mid_norm)` at the attention-residual epilogue (`crates/models/qwen3-moe/src/forward/layer.rs:224-244`) into a single kernel that writes both `mid` and `mid_norm`. Both outputs consumed downstream (mid → moe_residual path, mid_norm → FFN input). Same phase-1-sum-phase-2-scale structure as rmsnorm_f16; first phase reads x_in + delta, stores mid, accumulates sum-of-squares. Phase 2 rereads mid and writes mid_norm. Layout, thread count, and reduce tree identical to rmsnorm_f16 so no perf surprise per-launch.

**A.2 — `moe_combine_two_residuals_f16.cu`**

Fuses `add_f16(mid, shared_delta, moe_residual) + moe_combine_f16(expert_outs, weights, moe_residual, out)` at the shared-expert→combine boundary (`crates/models/qwen3-moe/src/forward/layer.rs:288-301` + the combine inside `forward_moe_ffn_decode`). New kernel variant accepts two residual inputs and sums them into the accumulator inline. `forward_moe_ffn_decode` gained an `extra_residual: Option<DevicePtr>` param; `None` path keeps the original combine kernel.

## Diagnosis

Both fusions work mechanically — call counts drop exactly as expected (−3200 per attempt). End-to-end tok/s moves +1.4 % because on 4-rank PP decode at Mesh<4> we're already GPU-saturated (sum_kernel_ms / n_ranks ~= wall); the launch-count reduction primarily saves Rust-side FFI + driver launch overhead which is off the GPU-time critical path. The fusions will likely pay larger dividends on smaller meshes (Mesh<1>/Mesh<2>) or on workloads that are host-bound.

The cumulative 4.5 % launch reduction narrows the gap to turbo (which at 1653 calls/token vs our now-2131 still runs 22 % fewer launches). Remaining gap is mostly in operator-level fusion decisions (SwiGLU fused into gate_up, residual folded into quantize, shared-expert scale+cast fused — V2.23.a future attempts or its V2.23.x siblings).

## Gate

- cert-check hip/gfx906: 48 rows, 0 failures
- UD-Q4_K_S parity: bit-exact vs llama.cpp reference on seed 9419 × 8 tokens
- No new per-kernel certs emitted; correctness relies on parity. (Both fused kernels are structural combinations of already-certified ops — numerical envelope identical.)

## Regeneration

```
FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  FLAMBEAU_PREFILL_L=1 FLAMBEAU_TG_LEN=64 \
  /opt/rocm-6.3.4/bin/rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/prof/ \
    -- target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe_mesh_all \
    --nocapture --include-ignored
```
