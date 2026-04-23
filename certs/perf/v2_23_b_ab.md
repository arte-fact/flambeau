# V2.23.b — Lever-2: per-category drill-down AB report

Per-category delta vs turbo (flambeau-minus-turbo) from pre-V2.23 profiles on
35B-A3B-Q4_0 Mesh<4> decode tg=64:

| category        | flambeau ms | turbo ms | **Δ** |
|---|---:|---:|---:|
| MMVQ            | 815.0 | 656.5 | **+158.5** |
| GDN             | 152.8 |  47.4 | +105.4 |
| activation/gate | 100.6 |   0.9 |  +99.7 |
| norm            | 126.8 |  67.2 |  +59.6 |

Selected attempts to close the top-2 actionable deltas.

## Attempts

### B.1 — fused Q4_0 indexed-MoE gate+up MMVQ

Class: MMVQ. Target kernel: `indexed_moe_mmvq_q4_0_q8_1` (9545 calls, 317.7 ms).

`indexed_moe_mmvq_q4_0_gate_up_dp4a.cu` reads Q8_1 activation once per block and writes both gate + up outputs. Halves the per-decode MMVQ call count for Q4_0 MoE gate/up (the tile8 MMQ path doesn't fire at decode n_tokens<32; we fell through to two separate MMVQ invocations). Arithmetic byte-identical to the unfused version (same `(q-8)·y` bias-correction identity per block).

Wired via `run_indexed_moe_gate_up` in `forward/common.rs`; single-weight variant still available for down path.

### B.2 — fused `swiglu_f32_to_f16`

Class: activation/gate. Target pattern: `swiglu_f32 → cast_f32_f16 → quantize_f16_q8_1`.

`swiglu_f32_to_f16.cu` combines SwiGLU with the immediate F32→F16 cast, so the intermediate F32 activated buffer is never materialised. Swapped at two decode sites (`forward_moe_ffn_decode`, `forward_shared_expert_decode`) — prefill paths left on the unfused pair since tile8 routing + L-sized tensors have different tradeoffs.

## Results

35B-A3B-Q4_0 Mesh<4> decode tg=64, median of 3 unprofiled runs:

| | baseline | A.2 (from V2.23.a) | B.1 | **B.2** | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| decode tok/s | 47.22 | 47.87 | 49.13 | **49.30** | **+4.4 %** |
| wall ms (64 tok) | 1356 | 1337 | 1303 | 1298 | −58 ms |
| total kernel calls | 142,791 | 136,391 | 133,071 | **126,671** | **−16,120 (−11.3 %)** |
| sum_kernel_ms | 1637.9 | 1641.9 | 1587.5 | **1573.4** | **−64.5 ms (−4.0 %)** |
| calls/token | 2231 | 2131 | 2079 | **1979** | (turbo: 1653) |

Per-kernel kernel-call deltas baseline→B.2:

| kernel | baseline | B.2 | Δ |
|---|---:|---:|---:|
| indexed_moe_mmvq_q4_0_q8_1 | 9545 | 2905 | −6640 |
| indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1 | 0 | **3320** | +3320 (new) |
| swiglu_f32 | 9130 | 2730 | −6400 |
| swiglu_f32_to_f16 | 0 | **6400** | +6400 (new) |
| cast_f32_f16 | 19090 | 12690 | −6400 |
| add_f16 | 6640 | 240 | −6400 |
| moe_combine_f16 | 3320 | 120 | −3200 |

UD-Q4_K_S 8-token parity vs llama.cpp (seed 9419 → [11, 271, 40, 1044, 4313, 310, 958, 279]) bit-exact preserved after each attempt.

## Diagnosis

Cumulative V2.23.a+B wins **+4.4 % tok/s** on 35B-A3B-Q4_0 Mesh<4> decode via −11.3 % launch count and −4.0 % kernel time. We've closed roughly a third of the pre-V2.23 gap-to-turbo tok/s; remaining gaps are (i) GDN class (+105 ms, untouched here — flagged for lever-3 or a separate tuning cycle), (ii) cast_f32_f16 has 12,690 residual calls from places we haven't refactored, (iii) turbo still runs fewer MMVQ launches total (28k vs our 30.5k — mostly because their MoE is expressed in a handful of large-batch kernels rather than per-expert launches).

No new per-kernel certs — both fused kernels are structural compositions of already-certified ops (swiglu math + F32→F16 cast; same DP4A Q4_0 decode + bias correction). Correctness backstopped by end-to-end parity.

## Gate

- cert-check hip/gfx906: 48 rows, 0 failures
- UD-Q4_K_S 8-token parity bit-exact

## Regeneration

```
FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  FLAMBEAU_PREFILL_L=1 FLAMBEAU_TG_LEN=64 \
  /opt/rocm-6.3.4/bin/rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/prof/ \
    -- target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe_mesh_all \
    --nocapture --include-ignored
```
