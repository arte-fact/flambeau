# V2.27.d — async PP ubatch guards for hybrid models (GDN state race)

Root-caused + guarded the async-PP parity failures V2.26.a-i7 surfaced
at `ubatch ∈ {64, 100, 192}` on 9B Q4_1 Mesh<4>.

## The race

Hybrid Qwen3.5/3.6 models have `full_attention_interval` set, so
`cfg.is_recurrent(il)` is true for 3-of-4 layers. The GDN forward
reads + writes a recurrent state tensor that lives in
`rank_session.caches[layer_idx]` — **one per rank, not one per lane**.

When `u_lanes > 1` the 1F1B dispatcher assigns alternating ubatches
to different aux streams on the same rank:

| timestep | rank 0 lane 0 | rank 0 lane 1 |
|---|---|---|
| t=0 | ubatch 0 | — |
| t=1 | — | ubatch 1 |
| t=2 | ubatch 2 | — |
| t=3 | — | ubatch 3 |

Aux streams on the same device run concurrently driver-side. Ubatch 1
on lane 1 (t=1) races with the tail of ubatch 0 on lane 0 (t=0) if
ubatch 0 hasn't finished its GDN state r/w by the time ubatch 1's
GDN kernel launches.

**Masking effect:** at `ubatch ≥ 128` each ubatch has enough per-layer
kernel work that the driver's in-flight queue naturally serialises
the races on the shared state buffer — stream scheduling happens to
order the GDN state accesses correctly. At `ubatch < 128` the
per-ubatch work is short enough that the state accesses can
genuinely overlap, producing non-deterministic wrong outputs.

## Reproduction

Before the guard:

| ubatch | lanes | L | last_id expected | last_id observed | Δ |
|---:|---:|---:|---:|---:|---:|
| 64  | 2 | 1024 | 220 | **96519** | wrong |
| 100 | 2 | 1024 | 220 | **97181** | wrong |
| 192 | 2 | 1024 | 220 | **9616** (tail=64) | wrong |
| 160 | 2 | 1024 | 220 | **9616** (tail=64) | wrong |
| 96  | 2 | 1024 | 220 | **44122** (tail=64) | wrong |
| 128 | 2 | 1024 | 220 | 220 | ✓ |
| 256 | 2 | 1024 | 220 | 220 | ✓ |
| 384 | 2 | 1024 | 220 | 220 | ✓ (tail=256 ≥ 128) |
| 512 | 2 | 1024 | 220 | 220 | ✓ |

Pattern:
- `ubatch < 128` → race present on every ubatch
- `ubatch ≥ 128` but `tail = L mod ubatch ∈ (0, 128)` → race on tail
- `ubatch ≥ 128` with tail = 0 or tail ≥ 128 → safe

## Fix

`forward_prefill_pp_async` now rejects unsafe configurations at entry
when the model has any GDN layer and `u_lanes > 1`:

1. Reject `ubatch_size < 128` outright.
2. Reject `L % ubatch_size != 0 && L % ubatch_size < 128` (tail too
   small).

Error messages name the offending config and point to this cert.
Non-hybrid models (e.g. Qwen3.5-dense or Qwen3.6-27B-dense once
#169 lands) skip the guard — no GDN, no shared state, no race.

## Real fix (deferred)

Making `u_lanes > 1` safe at small ubatches on hybrid models needs
per-lane GDN state + a deterministic merge (average? last-write-wins
via stream ordering?). Per V2.26.a's finding that `ubatch=128
u_lanes=2` is the measured optimum anyway, that refactor has no
measured perf upside. Kept the real fix unscheduled.

## Default config — unchanged

The V2.26.a cert's recommendation stood: `FLAMBEAU_UBATCH=128
FLAMBEAU_U_LANES=2` on 9B Q4_1 Mesh<4>. This sits comfortably in
the safe region; the V2.27.d guards catch only tuning-sweep
explorations that would have picked up corrupt outputs silently.

## Gate

- `ub=64 u_lanes=2` errors cleanly with the GDN race message.
- `ub=192 u_lanes=2 @ L=1024` (tail=64) errors with the tail-too-small message.
- `ub=128 u_lanes=2` (no tail) still green: L=1024 last_id=220, L=4096 last_id=62.
- Sync path + `u_lanes=1` unaffected.

## Regeneration

```
# Should error (GDN race):
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=64 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# Should error (tail race):
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=192 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=... \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# Should still work (safe regime):
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=... \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture
```
