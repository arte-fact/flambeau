# V2.26.b — async PP at larger L (crossover confirmed)

Extended the 9B Q4_1 Mesh<4> prefill grid to L ∈ {2048, 4096} to test the
V2.25.h projection: async-PP pays when K (ubatches per pass) is large
enough to drown out fill+drain overhead.

## Results (9B Q4_1 Mesh<4>)

Best configs after a (ubatch, u_lanes) sweep:

| L | Sync | **Async ub=512 u_lanes=2** | Δ |
|---:|---:|---:|---:|
| 1024 | **823.12** | 769.87 | −6.5 % |
| 2048 | 738.65 | **800.48** | **+8.4 %** |
| 4096 | 603.00 | **713.66** | **+18.4 %** |

Parity bit-exact across all configs (`last_id` matches sync on every L).

**Crossover at K ≥ 3 ubatches per pass** (L ≥ ~1500 at ubatch=512). At
L=1024 K=2, fill+drain dominates (only 1 steady-state timestep out of 5);
at L=4096 K=8, pipeline is 8 steady + 6 fill/drain → ~0.7× efficiency
per rank × 4 ranks = 2.8× serial-equivalent dispatch, enough to recover
the sync regression and win +18 %.

## Why sync drops past L=1024

Qwen3.5-9B is dense full-attention (no GDN). Attention is O(L²·d); at
L=4096 that's 16× the compute of L=1024 but only 4× the tokens →
throughput drops. Async hides this: while rank r works attention on
ubatch u, rank r-1 works attention on u+1 on independent layers.

## Sweep (L=4096)

| ubatch | u_lanes | tok/s | Δ vs sync |
|---:|---:|---:|---:|
| 256 | 2 | 636.22 | +5.5 % |
| 256 | 4 | 605.13 | +0.4 % |
| **512** | **2** | **713.66** | **+18.4 %** |
| 512 | 4 | 671.55 | +11.4 % |

- **ubatch=256 is worse than ubatch=512** — more K but per-ubatch FFI /
  Mutex / closure overhead (~300 µs × K × N dispatches) exceeds the
  extra pipeline depth. Compute per ubatch needs to exceed infra
  overhead per ubatch.
- **u_lanes=4 is worse than u_lanes=2** — at steady state only N=4 ranks
  can concurrently dispatch; the 3rd and 4th lanes add scratch memory
  and lock contention without extra overlap.

## Hardware P2P probe — null

Before extending L, probed `hipDeviceEnablePeerAccess` +
`hipMemcpyPeerAsync` on the 4×MI50 rig as a candidate replacement for
the pinned host-bounce. Results:

| pair | can_access | enable | bandwidth |
|---|---|---|---:|
| 0↔1 | yes | ok | 6.17 GB/s |
| 0↔2, 0↔3, 1↔2, 1↔3 | yes | ok | 4.57 GB/s |
| 2↔3, *→3, 3→* | yes | ok | **hipErrorLaunchFailure (719)** after 2–3 iters |

P2P is equal-or-slower than host-bounce (6.75 GB/s) and unstable on
GPU3 pairs on this rig. Not committed — host-bounce stays the peer
transfer path. System is P2P-configured at the OS level
(`iommu=pt`, `pcie_acs_override=downstream,multifunction`, 16 GB BAR1),
so this is a gfx906 driver/firmware limitation at sustained load, not
a capability gap.

## Why ubatch=256 loses — next lever

Infrastructure overhead per ubatch on the async path, measured from
the gap between `ubatch=256 → 636` and `ubatch=512 → 714` at L=4096:

- 16 ubatches × 4 ranks × 1 peer-copy × ~50 µs = 3.2 ms/pass peer-copy
- 16 ubatches × 4 ranks × 10 layers × ~20 µs kernel launch = 12.8 ms/pass launch
- closure + Mutex + event record/wait overhead: ~5–10 ms/pass

Total ~20 ms/pass pure infrastructure at ubatch=256. Wall clock at
ubatch=256 is 6438 ms → 0.3 % infra overhead per pass. But the net
throughput loss is ~11 % vs ubatch=512 — because the infra overhead
sits on the critical path (ubatches serialise the dispatch order
even at 1F1B; they just overlap the kernel work).

**V2.26.a** (HIP graph capture) should record the (peer-copy +
N-layer forward) subgraph per rank once at session init and replay
per-ubatch — collapses the 12.8 ms launch cost to a single replay per
ubatch (~2 µs). Expected to reopen the ubatch=256 regime and push K
higher, recovering full pipeline efficiency at L ≥ 2048.

## Gate

- 8-token parity bit-exact at L=1024 (`last_id=220`) — async path
  validated in V2.25.h, preserved here
- `last_id` matches sync on L ∈ {2048, 4096} across every (ubatch,
  u_lanes) config
- Build clean, opt-in via FLAMBEAU_ASYNC_UBATCH + FLAMBEAU_UBATCH +
  FLAMBEAU_U_LANES — default path unchanged

## Regeneration

```
# sync reference
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L="

# best async config
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=512 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L="
```
