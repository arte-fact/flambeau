# D4 — batched-decode throughput on the v2 shared-Session stack (2026-05-18)

## Setup

- Stack: v2 (`FLAMBEAU_V2=1`), post-D3-A (one shared `Session<A>` with
  `max_slots = inflight_slots`, `V2Conv` ticket pool).
- Model: `Qwen3.5-9B-Q4_1.gguf` (dense + GDN hybrid).
- Prompt: identical across streams (60-noun list continuation), 24
  tokens. Forces `length`-stop at K tokens (no early EOS in any run).
- Generation: 64 tokens per stream, greedy (`temperature=0`).
- Runs per N: 3, best wall reported.
- Decode batch window: 1500 µs.
- gfx906 MI50s (per-MEMORY rig).

## Headline results

| Topology | Devices    | N=1 t/s | N=2 agg | N=4 agg | speedup N=4 |
|----------|------------|--------:|--------:|--------:|-------------|
| SD       | hip:0      |  27.58  |  26.46  |  26.27  | 0.95×       |
| PP2      | hip:0,2    |  30.97  |  29.13  |  28.55  | 0.92×       |
| TP2      | hip:0,1    |  32.08  |  30.71  |  30.54  | 0.95×       |
| PP2+TP2  | hip:0,2,1,3|  29.99  |  29.96  |  31.44  | 1.05×       |

Per-topology detail certs:
- `qwen35_9b_sd_2026_05_18.md`
- `qwen35_9b_pp2_2026_05_18.md`
- `qwen35_9b_tp2_2026_05_18.md`
- `qwen35_9b_pp2tp2_2026_05_18.md`

## Architectural verification (D3-A primitive)

`RUST_LOG=info,server.scheduler=info FLAMBEAU_TRACE_BATCH=1` confirms
the leader-drain scheduler dispatches with `pending=2` (and `pending=4`
at N=4) for every per-step batched decode after the initial prefill
phase. Both/four streams complete at the same wall time within ±5 ms,
indicating tight batch alignment after the first step.

Per-slot KV/GDN isolation works: every stream receives 64 distinct
tokens of coherent text without crosstalk, and all hit `length` stop
at exactly K=64.

## Why aggregate ≈ 1×, not N×

The per-step decode time grows **linearly with N**:

| Topology | per-step @ N=1 | per-step @ N=2 | per-step @ N=4 |
|----------|---------------:|---------------:|---------------:|
| SD       | 36 ms          | 76 ms          | 152 ms         |
| PP2      | 32 ms          | 69 ms          | 140 ms         |
| TP2      | 31 ms          | 65 ms          | 116 ms         |
| PP2+TP2  | 33 ms          | 67 ms          | 127 ms         |

`Session::forward_decode_batched(slots)` does correctly dispatch one
forward call carrying N (token, position, slot_id) tuples. Inside the
composites, however, the per-token work loops over distinct slots
serially:

- `standard_attn` decode branch: each slot has its own KV history,
  so per-slot KV-append + `attention_decode_f16(pos)` is an N-loop.
  No batched-attn kernel for the decode shape yet (the existing
  `attention_decode_f16_batched` lives on legacy qwen3-moe; v2 has
  not yet hosted a port).
- `gdn_layer` decode branch: GDN state-step + pass-A + conv update
  are per-slot loops over recurrent state slabs.

So per-step at N=K ≈ K × per-step at N=1, and aggregate stays flat.

This is the same ceiling memory-noted for the legacy qwen3-moe stack
(`project_p29b_i2_F_hybrid_throughput.md`: "3× gate structurally
blocked by per-slot GDN state-step + per-slot KV-append+attn loops").
D3-A inherits the same ceiling because the composites' inner loops
are unchanged from D1/D2.

## What D4 actually validates

This cert is **NOT** a perf win cert. It is the architectural
verification cert: the D3-A shared-session primitive routes N
concurrent decodes through one Session::forward_decode_batched, the
scheduler engages with `pending=N`, and per-slot state is correctly
isolated. Aggregate throughput equals N=1 baseline because the
underlying composite kernels still serialise per-slot.

Closing the 1× ceiling toward the architectural ~N× requires:
1. **Batched-decode attention kernel for v2 composites** — a single
   kernel that loads N slot KV slabs + N positions + N Q rows and
   emits N attention outputs. Direct port of legacy
   `attention_decode_f16_batched` into the v2 standard_attn composite.
2. **Batched-GDN companion** — GDN state-step / pass-A / conv update
   kernels that consume N slots at once. Largely a transpose of the
   existing prefill-batched GDN kernels.

Without those, raising `--inflight-slots` above 1 reduces TTFT for
queued requests (queue parallelism, which the scheduler does deliver)
but does not increase aggregate decode throughput.

## Note: pp2+tp2 anomaly

PP2+TP2 at N=4 shows a small 1.05× gain — likely the 4-GPU compute
parallelism amortising part of the per-slot serialisation cost at
N=4. With more GPUs there's more aggregate compute available per
unit of per-step wall, so the linear-N kernel scaling is softened.

## Reproduce

```
cargo build --release -p flambeau-cli --features hip_serve
python3 scripts/bench/d4_batched_decode.py \\
    --model /artefact/models/Qwen3.5-9B-Q4_1.gguf \\
    --topology <pp|tp|pp+tp> --devices <csv> \\
    [--tp-size N] [--pp-size N] \\
    --n-max 4 --n-run 1,2,4 \\
    --tokens 64 --runs 3 \\
    --out certs/perf/d4_batched_decode/<name>.md
```
