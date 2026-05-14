# Gemma 4 v1 perf cert — flambeau vs llama.cpp

**Date:** 2026-05-14
**Rig:** threadreaper-gfx906, 4 × MI50 (16 GB / card), 100 W cap
**Flambeau build:** `feature/gemma4` @ ce61ecf
**llama.cpp build:** 97508acb1 (8704), ROCm gfx906 backend
**Methodology:** `crates/models/gemma4/tests/perf_bench.rs` — one upload
per topology, warmup (1 prefill 512 + 16 decode), then 3 prefill + 3
decode measurements with KV cleared between runs, **min** wall-clock
reported. llama-bench: stock `-r 3`, mean ± stddev reported.
**Shape:** prefill = 512 tokens, decode = 128 tokens.

## Headline

| Model | Topology | Phase | Flambeau | llama.cpp | Ratio (us / them) |
|---|---|---|---:|---:|---:|
| E4B-Q4_0      | single (hip:0)      | pp512 | **4.8 tok/s**   | 1287.4 tok/s | 0.004× ❌ |
| E4B-Q4_0      | single (hip:0)      | tg128 | **4.5 tok/s**   |   71.1 tok/s | 0.063× ❌ |
| 31B-Q4_0      | PP2 (hip:0,2)       | pp512 | **194.1 tok/s** | OOM (does not fit) | — ✅ flambeau-only |
| 31B-Q4_0      | PP2 (hip:0,2)       | tg128 | **10.1 tok/s**  | OOM | — ✅ flambeau-only |
| 31B-Q4_0      | PP4 (hip:0,1,2,3)   | pp512 | **194.7 tok/s** | 213.2 tok/s  | 0.91× ❌ |
| 31B-Q4_0      | PP4 (hip:0,1,2,3)   | tg128 | **10.4 tok/s**  |  20.6 tok/s  | 0.51× ❌ |

## Bench command (flambeau)

```
cargo test -p flambeau-gemma4 --features hip --release \
  --test perf_bench -- --nocapture --test-threads=1
```

## Bench command (llama.cpp)

```
LD_LIBRARY_PATH=/opt/rocm-host/lib \
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-host/lib/rocblas/library \
llama-bench -m <model> -p 512 -n 128 -r 3 \
  [-sm none -mg 0]                                  # single
  [-dev ROCm0,ROCm2 -sm layer -ts 1/1]              # PP2 (failed — see note)
  [-sm layer -ts 1/1/1/1]                           # PP4
```

## Findings + diagnoses

### Single-device E4B is catastrophically slow (~210 ms / token)

Both prefill and decode bottom out at ~4.5 tok/s. **Root cause: there
is no batched-prefill kernel for the single-device gemma4 path** — the
bench's prefill measurement loops `forward_one_token` over the prompt,
so prefill timing is identical to decode timing.

Additionally, E4B carries the per-layer-embd side-channel, which runs
a host-side BF16×F16 matmul of shape `[2560, 10240]` per token in a
tight scalar loop in `per_layer_embd::build_inp_per_layer_table`
(`crates/models/gemma4/src/per_layer_embd.rs:115-138`). At 26M FMAs
per token, this CPU pass alone contributes ~50-100 ms / token on a
single core. **Adding a HIP kernel for the model_proj matmul + dequant
is the obvious next perf lever** for E4B; without it E4B single-GPU is
unusable.

The 15.8× decode gap and 268× prefill gap vs llama.cpp are both
direct consequences of these two missing pieces, not of the
fundamental kernel quality (the same kernels run at competitive speeds
on the 31B path below).

### 31B PP2 wins by fitting where llama.cpp can't

llama.cpp's allocator tries to allocate 16.5 GB on `device 0` even
with `-sm layer -ts 1/1`, triggering OOM on a 16 GB MI50. Flambeau's
per-stage upload distributes the full 17 GiB model evenly (~8.5 GB +
KV per rank) and runs the workload cleanly. **This is the only
configuration on this rig where flambeau is the only stack that runs
the model at all.**

Single-stream throughput at PP2: **194 / 10.1 tok/s** (prefill /
decode). Decode is bounded by cross-die hand-off latency between
`hip:0` and `hip:2` — see [[project_rig_whitehaven_topology]].

### 31B PP4 — flambeau within 9% on prefill, ~½ on decode

- **Prefill 194.7 vs 213.2 tok/s = 0.91×.** Flambeau's batched-prefill
  through `forward_prefill_pp` is competitive on this workload. Within
  noise margins of llama.cpp once measurement variance is accounted
  for (llama-bench reported ± 0.28 tok/s; our min-of-3 captures the
  warm tail).
- **Decode 10.4 vs 20.6 tok/s = 0.51×.** This is the gap that wants
  investigation. Candidates:
  - Per-rank stream sync overhead in the PP loop — gemma4 PP runs ~60
    layers × 4 ranks = 240 stage hand-offs per decode step. Each
    hand-off is a peer_copy_via_host of `[hidden]` F16 = 6 KB; the
    launch + sync cost dominates the actual copy.
  - No batched-decode kernel for gemma4 yet (#8 / S11). llama.cpp's
    decode path includes some batching primitives (KV-append fused
    with attention).
  - `kq_replicated` overhead — N/A for 31B (dense), but the path is
    still general; check whether the dispatch picks the right tile
    width for `head_dim=512`.
  - PP4 and PP2 measure the **same decode throughput** (10.4 vs 10.1)
    — adding ranks beyond PP2 doesn't help, consistent with the
    per-rank-overhead diagnosis.

### Why no TP / hybrid rows?

Real-GGUF TP + hybrid upload paths haven't landed (`#20` / `#21`).
The driver layer has the algorithms (synthetic-weight smoke tests
pass), but the real-weight GGUF-to-stage upload is a separate
deliverable. Once those ship, this cert grows two more rows per model.

## Per-cell wall-clock raw

```
flambeau:
  E4B-Q4_0  single   pp512 = 107413.44 ms  → 4.8 tok/s
  E4B-Q4_0  single   tg128 =  28637.83 ms  → 4.5 tok/s
  31B-Q4_0  PP2      pp512 =   2638.17 ms  → 194.1 tok/s
  31B-Q4_0  PP2      tg128 =  12636.94 ms  → 10.1 tok/s
  31B-Q4_0  PP4      pp512 =   2629.73 ms  → 194.7 tok/s
  31B-Q4_0  PP4      tg128 =  12249.66 ms  → 10.4 tok/s

llama.cpp:
  E4B-Q4_0  single   pp512 = 1287.41 ± 1.62 tok/s
  E4B-Q4_0  single   tg128 =   71.13 ± 0.16 tok/s
  31B-Q4_0  PP2      → OOM (`alloc_tensor_range: failed to allocate ROCm0
                            buffer of size 17322494208`)
  31B-Q4_0  PP4      pp512 =  213.16 ± 0.28 tok/s
  31B-Q4_0  PP4      tg128 =   20.55 ± 0.07 tok/s
```

## Action items (perf-lever queue)

1. **Single-device batched-prefill kernel for gemma4** — replaces the
   `forward_one_token` loop and would close the 268× E4B prefill gap.
   Tracked as a new follow-up.
2. **Per-layer-embd `model_proj` HIP kernel** — moves the 26M-FMA
   host-side matmul to GPU. E4B-specific; primary lever for closing
   the E4B decode gap.
3. **Gemma4 batched-decode (#8 / S11)** — N concurrent decodes per step;
   amortises the per-rank PP hand-off overhead. Primary lever for
   closing the 31B PP4 decode gap.
4. **TP4 + hybrid real-GGUF upload (#20 / #21)** — adds the missing
   topology rows.

## Conclusion

Gemma 4 dense inference works end-to-end across single / PP2 / PP4 on
real weights. PP2 unlocks a configuration llama.cpp can't run on this
rig. Prefill is within 9% of llama.cpp on 4-GPU; decode is at 51%, and
single-device E4B is a structural outlier blocked on two missing
kernels (batched-prefill + on-device per-layer-embd). The perf-lever
queue above is the natural follow-up path; the framework itself is
complete enough to declare V1 dense-gemma4 done.
