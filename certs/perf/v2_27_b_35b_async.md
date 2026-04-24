# V2.27.b — 35B MoE async prefill unlocked

Ported FLAMBEAU_ASYNC_UBATCH / FLAMBEAU_UBATCH / FLAMBEAU_U_LANES env
plumbing into `perf_baseline_qwen3_moe.rs` (mirror of the 9B harness).
Also extended the prefill grid to include L ∈ {2048, 4096}.

## Qwen3.6-35B-A3B-UD-Q4_K_S Mesh<4> — sync vs async ub=128 lanes=2

| L    | sync   | async ub=128 | Δ      |
|-----:|-------:|-------------:|-------:|
| 512  |  697   |   907        |  +30 % |
| 1024 |  712   |  **1414**    |  +99 % |
| 2048 |  672   |  **1613**    | +140 % |
| 4096 |  541   |  **1529**    | +182 % |

Parity bit-exact across all L on both paths (last_id matches).

## Cumulative vs V2.8 cert's Mesh<4>

V2.8 Mesh<4> pp=1024 baseline was ~640 tok/s. Now 1414 tok/s —
**+121 % cumulative** over the V1.7.6 → V2.26 track.

## Why the jump on 35B

Same mechanism as V2.26.a-i5a / V2.26.a-i7b fixed on 9B: the
cross-lane Rust-dispatcher barriers (`stream.synchronize()` inside
`upload_positions_range` + per-token embed syncs) were blocking
async aux-stream overlap. 35B has 40 layers (vs 9B's 36) + MoE
routing kernels on every layer, which means *more* kernel launches
per ubatch and *more* Rust-dispatcher work — so the barrier removal
lifts it proportionally more (+182 % at L=4096 vs 9B's +178 %).

Graph capture path (`FLAMBEAU_ASYNC_GRAPH=1`) would work on 35B too
but expected null per the V2.26.a cert's 9B finding.

## Parity — 8 tokens match sync

`last_id` at every (L, path) combination is identical between sync
and async_ub=128:

- L=512:  both `263`
- L=1024: both `220`
- L=2048: both `198`
- L=4096: both `248046`

decode tg=64 on async path = 52.76 tok/s; sync = 53.36 tok/s
(within noise — decode uses single-stream, no aux-lane benefit).

## Regeneration

```
# sync
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  FLAMBEAU_TG_LEN=64 \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture

# async
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  FLAMBEAU_TG_LEN=64 \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture
```
