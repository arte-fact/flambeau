# V2.23.d — Lever-4: copy/memcpy reduction AB report

Pre-V2.23 the flambeau-vs-turbo per-category breakdown on 35B-A3B-Q4_0
Mesh<4> decode showed us already 57 ms ahead on `__amd_rocclr_copyBuffer`
(51.7 vs 109.5 ms — much of that load-time weight upload). The 27B-Q8_0
decode breakdown separately showed a +20 ms gap from 3× more short
copyBuffer calls. Rather than chase the load-time discrepancy, T4 went
after residual per-token device-to-device memcpys in the decode+prefill
hot paths.

## Attempts

### D.1 — fused `gdn_assemble_conv_input_f32`

Replaces two DtoD memcpys in `forward/gdn.rs::assemble_conv_input`
(decode GDN layer) with a single elementwise kernel. Each GDN layer at
decode fires this pattern once per token — on ~30 GDN layers × 65 tokens
/ 4 ranks that's ~490 decode-time memcpy calls the driver no longer
processes (per-rank).

### D.2 — `swiglu_f32_to_f16` in prefill paths

The fused SwiGLU → F16 kernel introduced in V2.23.b.2 was applied to
the decode path only. D.2 extends it to:

- `forward_moe_ffn_prefill` (main prefill — both `tile8` and `turbo`
  branches)
- `forward_moe_ffn_prefill` Q4_0/Q8_0 fast path
- `forward_shared_expert_prefill`
- `forward_dense_ffn_decode` + `forward_dense_ffn_prefill`

Each site skips a `cast_f32_to_f16` launch between `swiglu_f32` and
`quantize_f16_q8_1`.

## Results (35B-A3B-Q4_0 Mesh<4>)

### Decode tg=64 median of 3

| | baseline | B.2 | D.1 | **D.2** | cumulative Δ |
|---|---:|---:|---:|---:|---:|
| decode tok/s | 47.22 | 49.30 | 49.85 | **49.76** | **+5.4 %** |
| wall ms (64 tok) | 1356 | 1298 | 1284 | 1286 | −70 ms |
| sum_kernel_ms | 1637.9 | 1573.4 | — | **1553.5** | **−84.4 ms (−5.2 %)** |
| total kernel calls | 142,791 | 126,671 | — | **124,031** | **−18,760 (−13.1 %)** |

(D.1 is decode-path only; per-rank profile shows ~−2500 calls vs B.2.
D.2 adds another −2640 calls on top, mostly from prefill paths not
firing at n_tokens=1 decode — the decode number doesn't move much from
D.1 to D.2.)

### Prefill grid (FLAMBEAU_PREFILL_ONLY=1)

| L | post-V2.22.b | post-V2.23.d | Δ |
|---:|---:|---:|---:|
| 8 | 99.07 | 104.80 | +5.8 % |
| 64 | 207.33 | 374.36 | +80.6 % |
| 128 | 593.92 | 594.81 | noise |
| 512 | 756.81 | 757.82 | noise |
| 1024 | 761.56 | 764.58 | noise |

L=64 jumps dramatically (+80.6 %) — the prefill path at small L fires
the swiglu + cast + quantize chain proportional to L, so fewer per-step
launches helps most there. L ≥ 128 is MMVQ/MMQ-dominated so the fusion
savings are noise-level.

UD-Q4_K_S Mesh<4> prefill: L=512 663.92, L=1024 702.84 — unchanged.
UD-Q8_K_XL Mesh<4> prefill: L=512 725.72, L=1024 768.76 — unchanged.

### Gate

- UD-Q4_K_S 8-token parity vs llama.cpp bit-exact (seed 9419) after
  both attempts
- cert-check hip/gfx906: 48 rows, 0 failures (new kernels are
  structural recompositions of certified ops — numerical envelope
  unchanged)
- 35B-A3B-Q4_0 prefill AND decode: no regression on either axis

## Diagnosis

D.1 hits a narrow per-GDN-layer win at decode. D.2 applies the B.2
swiglu fusion pattern to all remaining prefill sites — small prefill
(L ≤ 64) sees the biggest wall-time gain since launch overhead
dominates per-step cost there. At L ≥ 128 MMVQ/MMQ compute dominates
and the fusion is noise.

The original T4 premise ("flambeau makes 3× more copyBuffer calls") was
from the 27B-Q8_0 measurement and most of the observed delta is load-
time weight uploads, not decode-time hot-path copies. Decode copies
reducible without structural changes (ring buffers, aliasing
conv_history with conv_input) are limited to the `gdn_assemble`
pattern D.1 addresses.

## Attempts NOT done (out of scope this session)

- Conv1d history ring-buffer / aliasing with conv_input — requires
  reshaping `causal_conv1d_f32` to handle modular indexing or
  in-place shift with proper synchronization.
- Batch PP hand-off (one larger DtoH/HtoD carrying multiple per-layer
  needs) — would require restructuring the PP forward to collect all
  cross-rank data before handing off, a larger architectural change.

Both filed for a future V2.23.d.x cycle.

## Regeneration

```
FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  FLAMBEAU_PREFILL_L=1 FLAMBEAU_TG_LEN=64 \
  /opt/rocm-6.3.4/bin/rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/prof/ \
    -- target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe_mesh_all \
    --nocapture --include-ignored
```
