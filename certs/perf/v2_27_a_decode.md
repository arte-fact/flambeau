# V2.27.a — decode graph capture: null on 9B, +4 % on 35B

End of the decode-capture track. Infra from V2.27.a-i1 → i3 shipped
correctly with bit-exact parity; perf result split by model.

## Head-to-head tg=64

| model | config | tok/s | Δ | last_id |
|---|---|---:|---:|---:|
| 9B Q4_1 Mesh<4> | legacy | 52.91 | — | 30 |
| 9B Q4_1 Mesh<4> | DECODE_GRAPH=1 | 52.90 | −0.02 % | 30 |
| 35B-A3B UD-Q4_K_S Mesh<4> | legacy | 53.34 | — | 15523 |
| 35B-A3B UD-Q4_K_S Mesh<4> | DECODE_GRAPH=1 | **55.64** | **+4.31 %** | 15523 |

Parity bit-exact (`last_id` matches) across all four runs.

## Why the 9B/35B split

The V2.27.a-i1 audit estimated +23 % from graph capture on 9B based
on assumptions of ~3 µs per kernel launch × ~1000 launches/token +
~50 µs per `upload_position` sync × 9 layers/rank.

Measured: 0 % on 9B, +4.3 % on 35B. Audit overcounted:
- **gfx906 kernel launches are sub-µs** for our kernel shapes, not
  3 µs. ~1000 launches × <1 µs = <1 ms/token = <5 % of the 19 ms
  token budget. Capture collapse saves a fraction of that.
- **`hipStreamSynchronize` on an idle stream is near-free** (~1 µs,
  not 50 µs) when the stream's submitted work has already completed
  by the time the host reaches the sync. Relaxed-capture sync
  swallowing recovers nothing.

35B sees +4.3 % where 9B sees 0 % because:
- 35B has 40 layers vs 9B's 36 (+11 % work).
- 35B runs MoE routing + combine kernels per layer (extra 5-8
  launches/layer over 9B's dense FFN), so per-token launch count is
  ~50 % higher.
- MoE kernel launch parameters include the expert selection
  routing; more per-kernel host-side state to assemble.

The extra Rust-dispatch time on 35B is where capture's collapse
takes hold. On 9B there isn't enough Rust work per token for
capture to claw back.

## Gap to llamacpp-turbo after i3

| model | phase | flambeau | turbo | Δ |
|---|---|---:|---:|---:|
| 9B | tg64 | 52.91 | 73.5 | −28 % |
| 35B | tg64 | 55.64 | n/a (turbo crashes) | — |

The 9B 28 % gap is at the **per-kernel compute level** (MMVQ, attention
decode, rmsnorm kernels), not the driver-overhead level. Closing it
needs V2.29.b / V2.29.c (flash-tile + MMVQ tuning) — per-kernel work
that V2.26.a's driver-side fixes couldn't reach and V2.27.a's graph
capture doesn't touch either.

## Recommendation

- **Don't flip `FLAMBEAU_DECODE_GRAPH=1` on by default.** 0-to-4 %
  upside isn't enough to justify the added complexity (graph
  instantiation on first token + split-K suppression + shadow state
  in `HipGraphExec`). Keep it opt-in — users who run heavy 35B MoE
  decode can set the flag; 9B decode gets no benefit.
- **Pivot to per-kernel tuning (V2.29.a audit → V2.29.b/c/d
  kernels)** for the real 9B decode gap.
- Graph capture infra (V2.26.a-i2 through V2.27.a-i3) stays as
  reusable primitives for the infrequent case where driver overhead
  does dominate.

## V2.26.a + V2.27.a track summary — what we actually shipped

| iteration | delivered | perf impact |
|---|---|---|
| V2.26.a-i2..i5c | graph-capture infrastructure (FFI, slot recorder, kernel + memcpy param update, shadow state) | 0% (infra) |
| **V2.26.a-i5a** | `upload_positions_range` sync removal | **2.07× on 9B prefill L=4096** |
| **V2.26.a-i7b** | batched prefill embed (2·u → 1 sync per ubatch) | small incremental |
| **V2.27.b** | 35B MoE async harness | **2.82× on 35B prefill L=4096** |
| V2.27.a-i2b | decode `positions_host` (capture prereq) | 0% (decode) |
| V2.27.a-i3 | decode graph capture wiring | 0% (9B), +4.3% (35B) |

The entire V2.26.a / V2.27.a graph-capture engineering (~2000 LOC)
netted 0 % direct perf; the incidental sync removals done while
plumbing it netted 2–3× on prefill. An accidental-rediscovery moral.

## Gate

- 9B Q4_1 Mesh<4> tg=64 bit-exact parity both paths (`last_id=30`)
- 35B-A3B UD-Q4_K_S Mesh<4> tg=64 bit-exact parity both paths
  (`last_id=15523`)
- Build clean; opt-in env flag isolated; default behaviour unchanged.

## Regeneration

```
# 9B decode
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# 9B decode + graph
FLAMBEAU_DECODE_GRAPH=1 FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=... \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# 35B decode
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  FLAMBEAU_TG_LEN=64 \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture

# 35B decode + graph
FLAMBEAU_DECODE_GRAPH=1 FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=... \
  FLAMBEAU_TG_LEN=64 \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture
```
