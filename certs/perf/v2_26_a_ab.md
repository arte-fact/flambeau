# V2.26.a — graph capture: null result, but the plumbing landed a 2–3× async win

V2.26.a set out to capture the async-PP forward chain into a
`HipGraphExec` and replay with per-node param updates, targeting the
11 % infra-overhead ceiling V2.26.b projected from the `ubatch=256`
vs `ubatch=512` at `L=4096` gap.

Six iterations landed (i2 FFI → i3 recorder → i4 scalar slots → i5a
persistent positions → i5b memcpy slots → i5c full wiring). All three
paths (sync, async legacy, async + FLAMBEAU_ASYNC_GRAPH=1) are
bit-exact parity on 9B Q4_1 Mesh<4> across L ∈ {8, 64, 128, 512,
1024, 2048, 4096}.

## The actual graph-capture payoff: zero

| L | async legacy (best cfg) | async GRAPH (best cfg) | Δ |
|---:|---:|---:|---:|
| 1024 | 1816 | 1822 | +0.3 % |
| 2048 | 2059 | 2071 | +0.6 % |
| 4096 | 1988 | 1985 | −0.2 % |

(ub=128, u_lanes=2, best-tuned config per the sweep below.)

Within run-to-run noise. Graph capture neither helps nor hurts in
this configuration — the per-ubatch launch overhead V2.26.b's
projection was built on no longer exists, so there's nothing for
graph capture to collapse.

## What actually delivered — V2.26.a-i5a sync removal

V2.26.a-i5a refactored `upload_positions_range` to use a persistent
host buffer on `FullAttnPrefillScratch` so the HtoD memcpy source
would be stable across graph replays. As a by-product, the
`stream.synchronize()` inside that function was dropped — it had
been there only to keep the transient `Vec<i32>` alive across the
copy.

Under aux-stream dispatch, that sync was a cross-lane barrier: the
Rust driver thread blocked on rank X lane Y's memcpy before issuing
any further work, serialising what should have been driver-side
parallel operation across ranks.

Its removal is the real V2.26 story:

| L | Sync (baseline) | V2.26.b async (w/ barrier) | post-i5 async legacy (no barrier) | Δ vs V2.26.b async |
|---:|---:|---:|---:|---:|
| 1024 | 827 | 770 | **1816** | **+136 %** |
| 2048 | 745 | 800 | **2059** | **+157 %** |
| 4096 | 606 | 714 | **1988** | **+178 %** |

Async prefill on 9B Q4_1 Mesh<4> is now **2.20×–3.28× the sync
baseline**. The sync path itself is unchanged (826→827 at L=1024,
within noise) — the barrier only mattered when multiple lanes were
concurrently dispatching.

## (ubatch, u_lanes) sweep — new optimum

| ubatch | u_lanes | L=1024 | L=2048 | L=4096 |
|---:|---:|---:|---:|---:|
| 128 | 2 | **1816** | **2059** | **1988** |
| 256 | 2 | 1414 | 1764 | 1817 |
| 384 | 2 | 1227 | 1700 | 1673 |
| 512 | 2 | 919 | 1354 | 1468 |
| 512 | 4 | 840 | 1106 | 1070 |

With the barrier in place (V2.26.b), `ubatch=512` beat `ubatch=256`
because per-ubatch infra overhead dominated at smaller ubatches.
Barrier-free, the opposite holds: smaller ubatches fill the pipeline
sooner and amortise less fixed overhead (the per-ubatch peer-copy
is tiny relative to the layer-chain compute), so more ubatches
= more pipeline efficiency.

`u_lanes=4` regresses significantly (1070 vs 1988 at L=4096): extra
lanes add scratch memory + lock contention without adding
concurrency beyond what N=4 ranks can issue.

**New recommendation:** `FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2` on
9B Q4_1 Mesh<4>. Graph capture stays opt-in via
`FLAMBEAU_ASYNC_GRAPH=1` but isn't load-bearing.

## Known tail-ubatch gotcha (surfaced during the sweep, not blocking)

With `ubatch=192` at `L=1024` (1024/192=5.33) and `L=4096`
(4096/192=21.33), parity breaks: `last_id=9616` vs expected `220`,
`last_id=75881` vs expected `62`. The failure is reproducible on the
LEGACY async path — not a graph-capture artefact. Ubatches that
don't divide `L` cleanly produce a short tail ubatch; some path in
the async loop mishandles partial-last-ubatch boundaries. Filed as a
follow-up; unrelated to V2.26.a.

## Gate

- 8-token parity bit-exact at L=1024 async legacy + async GRAPH
  (`last_id=220`).
- `last_id` matches sync on every (ubatch, u_lanes) config where
  ubatch divides L.
- `FLAMBEAU_ASYNC_GRAPH=1` path green; produces same `last_id` as
  legacy async across the whole matrix.
- Build clean, opt-in on both env flags; default path unchanged.

## Honest close of the V2.26.a graph-capture track

Graph capture infra committed and working:
- V2.26.a-i2 (696a6ca): hipGraphGetNodes + kernel param update POC
- V2.26.a-i3 (1c09abd): thread-local launch recorder + SlotMap
- V2.26.a-i4 (0883846): scalar-slot support + node-shadow fix
- V2.26.a-i5a (fcf6429): persistent positions_host (**the actual win**)
- V2.26.a-i5b1 (43dc03a): memcpy-node FFI + MemcpySlot
- V2.26.a-i5b2a/b/c (41e848c/36373e7/11a7c4a): KvCache split API +
  kv_cache_append_hip_slot + AttnPrefillSlots plumbing
- V2.26.a-i5c (52f9d8f): forward_prefill_pp_async capture wiring

The infra is re-usable if a future bottleneck fits its shape (many
small launches on a stable DAG with pos-bearing scalars). For now it
sits idle under `FLAMBEAU_ASYNC_GRAPH=1`.

**V2.26.a tuning passes #155-157 (i8-i10) are deprioritised**:
further graph-path tuning can't beat the current 2–3× win since the
underlying bottleneck was removed, not the one graph capture was
designed to attack. If a new bottleneck surfaces (e.g. Qwen3.6-35B
MoE kernel tuning in V2.26.a-i10's #157), graph capture can be
revisited for it.

## Regeneration

```
# sync
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L="

# async (new best config, legacy path — identical to graph)
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L="

# async + graph (parity check; perf identical to legacy)
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_ASYNC_GRAPH=1 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L="
```
