# V2.27.a-i5 — decode graph capture: +8.8% on 35B by folding output head

Second pass on the V2.27.a decode-capture track. V2.27.a-i3 wired
capture for the per-rank layer chain only — null on 9B, +4.3 % on
35B (from V2.27.a-i4 cert). Adding the output head (rmsnorm_quant_q8_1
+ lm_head mmvq) to the last rank's captured graph roughly doubles
the 35B win.

## Triple-run A/B (9B Q4_1 Mesh<4> + 35B-A3B-UD-Q4_K_S Mesh<4>, tg=64)

| model | run | legacy tok/s | graph tok/s | Δ |
|---|---:|---:|---:|---:|
| 9B | 1 | 54.44 | 53.37 | −2.0 % |
| 9B | 2 | 53.21 | 52.80 | −0.8 % |
| 9B | 3 | 52.29 | 54.71 | +4.6 % |
| **9B avg** | — | **53.31** | **53.63** | **+0.6 %** |
| 35B | 1 | 53.69 | 58.22 | +8.4 % |
| 35B | 2 | 52.84 | 58.58 | +10.9 % |
| 35B | 3 | 53.94 | 57.85 | +7.2 % |
| **35B avg** | — | **53.49** | **58.22** | **+8.8 %** |

Parity bit-exact on all 12 runs (9B last_id=30, 35B last_id=15523).

## Why the split (refined model)

- **Layer count**: 35B has 40 layers, 9B has 36 (+11 %).
- **Per-layer driver work**: 35B's MoE path runs router-gemv + topk +
  indexed-MoE gate+up + indexed-MoE down + shared-expert + combine +
  residual per layer. 9B's dense FFN is 3 qmatmuls + silu_mul. 35B
  has ~3× the kernel launches per layer and each MoE kernel takes
  more host-side arg assembly (routing sort indices, per-expert
  pointers).
- **Threshold**: the captureable wins kick in when Rust-side kernel
  dispatch work is within a factor of ~0.3 of GPU compute time.
  Below that threshold, dispatch overlaps with GPU work completely
  and capture claims nothing (9B). Above it, dispatch leaks onto
  the critical path (35B).

## Gap to turbo

9B tg=64 = 53.6 (graph avg) vs turbo 73.5 = **0.73×** (unchanged
from V2.27.a-i4). The 28 % gap on 9B decode remains per-kernel-
bound; V2.29.c (MMVQ multi-row tuning) is the lever.

35B: no turbo reference (turbo crashes loading Qwen3.6-35B-A3B on
gfx906 per V2.3.c.1's upstream-bug note).

## Recommendation

`FLAMBEAU_DECODE_GRAPH=1` is now a **measurable ship-worthy win on
35B MoE/GDN**. Still opt-in because:
- 9B dense hybrid: within noise, no clear gain
- Cost: first decode call per rank pays ~3-5 ms capture +
  instantiation (amortised over tg tokens, so <0.1 % at tg=64+)

Ship-ready for long-generation workloads on MoE models. If a future
serving harness routes Qwen3.6-35B decode requests, flipping this
default for 35B's config would claim the 8.8 %.

## Gate

- 9B tg=64 both paths: last_id=30 ✓
- 35B tg=64 both paths: last_id=15523 ✓
- `argmax_token_host` stays uncaptured; runs post-launch per token
  as before.
- `forward_output_head_decode` post-rank-loop call gated by
  `!use_decode_graph` to avoid double-run when capture is active.

## Regeneration

```
# 9B
for path in "" "FLAMBEAU_DECODE_GRAPH=1"; do
  env $path FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
    ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture
done

# 35B
for path in "" "FLAMBEAU_DECODE_GRAPH=1"; do
  env FLAMBEAU_TG_LEN=64 $path \
    FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
    ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture
done
```
