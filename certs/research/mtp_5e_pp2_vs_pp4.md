# MTP-5e — PP2 vs PP4 topology comparison for spec-decode

Run on the same rig (4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1) with
the same model (Qwen3.6-27B-Q4_0 + Qwen3.6-27B-mtp), the same prompt,
the same prefill prime, and the same target token count (60). Greedy.

| topology | layers/rank | baseline ms/tok | spec ms/tok | spec − baseline | accept rate |
|---|---:|---:|---:|---:|---:|
| **PP4** (4 ranks, 16 layers each) | 16 | **50.22** | 56.54 | +12.6 % | 87.5 % |
| **PP2** (2 ranks, 32 layers each) | 32 | 54.35 | **63.68** | **+17.2 %** | 87.5 % |

First 16 tokens bit-identical between baseline and spec in both
topologies → correctness preserved.

## Findings

### 1. PP2 baseline is itself slower than PP4 (+8 %)

Naïve expectation was that PP2 would be faster: fewer cross-rank
PCIe hand-offs (1 hop vs 3 hops) and the same total compute. Reality:
the per-rank kernel-launch overhead amortizes better across more,
smaller PP stages. With 32 layers serial on a single rank, each kernel
launch's host-side latency stacks; with 16 layers on two parallel
ranks, the per-rank launches overlap with peer-copy across ranks.

L=1 PP forward has no token-level pipelining (it's sequential), but
fewer layers per rank means the per-rank kernel-launch chain is shorter
and less of its slack is exposed to wall.

### 2. PP2 spec ratio is *worse* than PP4 (+17.2 % vs +12.6 %)

Hypothesis was that fewer hand-offs would lower the L=2 wall
multiplier. Instead the relative cost of the L=2 verify scales worse
on PP2 — measured 1.17× spec/baseline ratio vs 1.13× on PP4. With
spec overhead (GDN snapshot, MTP draft, output head ×2) being a
constant per macro that's ~5–6 ms/macro, the relative tax depends
on how cheap baseline is. PP2's baseline is slower → constant overhead
is a larger relative fraction → worse ratio.

Plus: PP2's L=2 body has 32 layers of L=2 work serial per rank vs
PP4's 16. Doubling the per-rank serial work doubles the absolute
spec overhead per rank, and while PP2 has only 2 ranks, the scaling
isn't favourable enough to offset.

### 3. The Lever-1 reject path optimization works on both

`redo_gdn_only_pp` runs per-rank-parallel, scaling with the number
of GDN-bearing layers per rank. Same code path; same correctness on
both topologies.

## Implication for V1 deployment

For Qwen3.6-27B with spec-decode on this rig:
- **PP4 is strictly better** for both ms/tok absolute and the spec
  efficiency ratio.
- The KV cache memory pressure at PP2 (32 layers × 512 MB at full
  131072 ctx = 16 GB just for KV) means PP2 also needs aggressive
  ctx clamping (≤ 4096 for the perf A/B run); PP4 fits 32k ctx
  comfortably.
- PP2 is not a viable alternative deployment topology on this rig.

## What about TP2?

TP2 uses BAR1 P2P AllReduce (much faster than PCIe peer-copy via host)
for cross-rank communication. It's the topology most likely to flip
spec-decode net-positive — but that's MTP-5f-tp2 (#191), separate
work. The hypothesis is unchanged by this PP2 result: PP2's worse
performance is from longer per-rank serial work, not from hand-off
cost; TP2 has neither problem (full hidden replicated, AllReduce on
each layer).

## Reproducer

```
FLAMBEAU_PP_RANKS=2 FLAMBEAU_PERF_AB_TOKENS=60 FLAMBEAU_CTX_CAP=4096 \
  cargo test --release -p flambeau-qwen3-moe --features hip \
  --test mtp_spec_decode_perf_ab -- --nocapture
```

(Default rank count = all visible GPUs, default ctx cap = 4096.)
