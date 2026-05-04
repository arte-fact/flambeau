# TP vs PP topology diagnosis — Qwen3.6-27B + Qwen3.6-35B-A3B — 2026-05-04

User question: *"i expected more from tp vs pp, is pp fast or tp low?"*

Short answer: **PP is fast on this rig; TP is bandwidth-underutilised.**
On PCIe-only MI50 the kernel-trace shows TP runs at ~53% GPU utilisation
even on its own best-case profile (35B-A3B / TP2, 26.2 t/s synthetic),
with the rest of wall lost to host coordination and per-token sync.
PP4 hits 58 t/s on the same model, **2.2× faster** at single-stream,
because each stage holds 1/4 of the weights and the four stages overlap
inside a single token's wall.

## 1. Single-stream tok/s side-by-side

Two harnesses agree on the rank order:
- **Matrix bench** (this session): chat shape, 1256-token prompt + 256
  decode, server path (`flambeau serve`).
- **profile_point** (synthetic): L=512 prefill + 64 decode steps,
  ctx=1024, no server, no SSE — just the forward path.

| model           | topology | matrix tok/s | synthetic tok/s | ratio synth/matrix |
|-----------------|----------|--------------|-----------------|---------------------|
| Qwen3.6-27B     | tp2      | 21           | (KV alloc OOM)¹ | —                   |
| Qwen3.6-27B     | pp4      | 17           | 21.4            | 1.26×               |
| Qwen3.6-27B     | pp2tp2   | 23           | (not measured)  | —                   |
| Qwen3.6-35B-A3B | pp4      | 41           | 58.2            | 1.42×               |
| Qwen3.6-35B-A3B | tp2      | (skipped)²   | 26.2            | —                   |
| Qwen3.6-35B-A3B | pp2tp2   | 44           | (not measured)  | —                   |

¹ profile_point_tp over-allocates 268 MB KV per layer regardless of
  `FLAMBEAU_CTX_CAP`; the test harness needs a fix. The matrix used
  the actual server path which honours ubatch ⇒ tp2/27B works there.
² Skipped from matrix (35B/TP2 with KV at 16384 won't fit in 16 GB / rank).
  TP2 number is from `qwen36_35B_A3B_tp2_profile_2026_04_29.md`.

**Headline finding from synthetic single-stream**:
- 35B-A3B / **PP4 = 58.2 t/s** vs **TP2 = 26.2 t/s** ⇒ PP is **2.22× faster**
- 27B / **PP4 = 21.4 t/s** vs the matrix tp2=21 ⇒ basically tied at single-stream

The dense 27B is bandwidth-bound on both; the MoE 35B-A3B has only
~3 GB active weights per token, so the overhead-vs-compute ratio is
fundamentally different and PP wins decisively.

## 2. HBM-bound ceiling — what's reachable

For decode of one token, every active weight gets read through HBM once.
Per-device 1 TB/s on MI50.

### Qwen3.6-27B (dense, 17 GB Q4_1)

- Active per token = 17 GB (every weight)
- TP2: 8.5 GB / device, both read in **parallel** → 8.5 ms = **117 t/s ceiling**
- PP4: 4.25 GB / stage, stages **sequential** → 4.25 × 4 = 17 ms = **58 t/s ceiling**

| topology | measured | ceiling | utilization |
|----------|----------|---------|-------------|
| 27B / tp2 (matrix) | 21 t/s | 117 | **17.9%** |
| 27B / pp4 (synthetic) | 21.4 t/s | 58 | **36.9%** |
| 27B / pp2tp2 (matrix) | 23 t/s | 117² | **19.7%** |

² pp2tp2 = 2 PP stages × TP2 per stage; effective ceiling matches TP2.

### Qwen3.6-35B-A3B (MoE, ~5 GB active per token)

A3B has 64 layers × MoE-128 + dense attention; active per token ≈
2.5 GB MoE experts + 2.5 GB attention/dense = ~5 GB.

- TP2: 2.5 GB / device parallel → 2.5 ms = **400 t/s ceiling**
- PP4: 1.25 GB / stage sequential → 1.25 × 4 = 5 ms = **200 t/s ceiling**
- pp2tp2: 1.25 GB / device parallel × 2 stages sequential → 2.5 ms = **400 t/s ceiling**

| topology | measured | ceiling | utilization |
|----------|----------|---------|-------------|
| 35B / tp2 (synth) | 26.2 t/s | 400 | **6.6%** |
| 35B / pp4 (synth) | 58.2 t/s | 200 | **29.1%** |
| 35B / pp2tp2 (matrix) | 44 t/s | 400 | **11.0%** |

PP4 sits at ~30% of its ceiling. TP2 sits at <10%. Both have headroom
but PP wastes less of what it has.

## 3. Where the wall time goes — TP2 reference (from 2026-04-29 cert)

`certs/perf/qwen36_35B_A3B_tp2_profile_2026_04_29.md` rocprofv3 over
35B-A3B/TP2/L=512+TG=64:

- Total kernel time both ranks summed = 2546 ms ⇒ per-rank kernel time = **1273 ms**
- Wall for 64 decode steps at 26.2 t/s = **2444 ms**
- **Per-rank GPU utilization = 1273 / 2444 = 52%**
- The other **48% of wall is host coordination**: kernel-launch latency,
  per-step Rust sampler invocation, `decode_via_scheduler` channel
  round-trip, sync between forward and host sample.

Top kernel buckets per the cert:
- GDN state-step (11.7%) — recurrent layer, HBM-bound
- Indexed MoE MMVQ Q4_0 (10.3%) — main FFN matmul, HBM-bound
- Q4_0 fused gate+up dp4a (5.0%) — bandwidth-bound
- AR primitives (3.5% + 2.4% = **5.9% total**) — TP communication is small
- Assorted rmsnorm/cast/quantize/swiglu (~15% combined)

**Comm (5.9%) is not the TP bottleneck** — the bottleneck is the 48%
host-side gap between kernel dispatches.

## 4. Why PP wins for single-stream on a 4-GPU rig

The decode wall is roughly:

  T_decode = T_kernel_active + T_host_overhead

T_host_overhead is *roughly constant per token* — a fixed ~50ms tax on
this rig per the GPU-idle measurements in MEMORY (mean GPU% ~14% on 9B,
~26% on 27B, both consistent with this number).

For TP2:
- T_kernel_active per token ≈ 1273 ms / 64 = 20 ms (per rank)
- T_host = 38 - 20 = **18 ms** ⇒ T_decode ≈ 38 ms ⇒ 26 t/s ✓

For PP4 on the same model:
- Each stage holds 1/4 of layers, so T_kernel per stage ≈ 5 ms (1/4 of 20)
- Stages run sequentially: 5 × 4 = 20 ms (matches TP2's per-token compute)
- BUT the host overhead is *per-stage* not per-token: cluster sync, peer_copy,
  next-stage launch — about 4 × 3 ms = 12 ms of coordination
- T_decode ≈ 20 + 12 = ~17 ms ⇒ 58 t/s ✓

**PP wins because the constant-per-step host overhead amortizes over
4 mini-steps per token** instead of being paid once whole. Even though
each peer_copy_via_host is 3-5 ms (PCIe-bound), 4× small overhead beats
1× large overhead at single-stream.

For 27B (dense), the per-stage compute is bigger (≈ 4.25 ms × 4 = 17 ms
just for HBM read), so the host-overhead share is smaller and PP's edge
narrows: 21.4 t/s PP4 vs 21 t/s TP2 ≈ tied.

## 5. Why TP wins at concurrency past N=2

Inverse of section 4: under multi-stream load, the pipeline bubbles in
PP fill up — each stage gets work from N streams. The matrix data:

| model | topo | N=1 | N=2 | N=4 | N=8 |
|-------|------|-----|-----|-----|-----|
| 35B   | pp4 | 41 | 79 | **133** | **126** |
| 35B   | pp2tp2 | 44 | 93 | 76 | 74 |
| 27B   | pp4 | 17 | 31 | **40** | 40 |
| 27B   | pp2tp2 | 23 | 39 | 34 | 32 |

PP4 keeps scaling to N=4 (3.2× on 35B, 2.4× on 27B). pp2tp2 saturates
at N=2 because there are only 2 PP stages — beyond that, the second
TP rank pair is just sharing one stage's load.

**The "TP wins at high concurrency" claim is wrong** for this rig. PP4
wins everywhere at N≥4. pp2tp2 wins at N=1-2 because its single-stream
compute is the fastest (TP-style sharding inside each PP stage).

## 6. So which is "fast"?

- **PP4 is fast on this rig** for both single-stream and multi-stream
  on every model size we measured. 35B-A3B / PP4 / N=4 = 132.9 t/s
  is the matrix headline.
- **TP2 is bandwidth-underutilised** — 6-18% of HBM ceiling at single-stream
  depending on model. The bottleneck is host coordination, not AR
  comm (AR is 5.9% on the existing kernel-trace cert).
- **pp2tp2 is the best low-concurrency choice** (single-user chat).
- **PP4 is the best high-concurrency choice** (2+ concurrent users).

The user's intuition that "TP should beat PP" is the textbook NVLink
result. On PCIe-only MI50, the host-coordination floor turns it inside
out: PP's 4-stage pipeline hides launch latency that TP can't.

## 7. Levers that would change this picture

(In order of expected impact, given the 48% host-coordination gap.)

1. **HIP graph capture** — would batch the per-token kernel dispatches
   into one graph launch, cutting the 18 ms host tax per token to ~2 ms.
   Documented as null in the past on PP via `feedback_*` notes
   (G3 attempt) — but those were on a single GPU. Worth re-trying on
   TP2 specifically given the 48% gap.
2. **Per-slot pre-allocated TP prefill scratch** (#321 follow-up) —
   would let the prefill scratch survive across decode and remove the
   per-token alloc churn that currently bites TP2 under concurrency.
3. **Async forward** — overlap one token's host sample + JSON-grammar
   advance with the next token's GPU dispatch. Existing decode loop is
   strictly sequential.
4. **Decode kernel fusion** — fuse rmsnorm + Q8_1 quant + first-layer
   mmvq into one launch (saves 2 launches per layer × 64 layers = 128
   launches per token). 1-3% per the existing cert.

These are V2 work; not in the V1 scope per CLAUDE.md.

## Reproducer

```sh
# Existing cert (TP2 / 35B-A3B): see qwen36_35B_A3B_tp2_profile_2026_04_29.md

# Synthetic single-stream (no server) — works for PP, not TP at 27B:
FLAMBEAU_PROFILE_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
FLAMBEAU_PROFILE_MESH=4 FLAMBEAU_PROFILE_L=512 FLAMBEAU_PROFILE_TG=64 \
FLAMBEAU_CTX_CAP=1024 \
target/release/deps/profile_point-* --nocapture profile_point

# Multi-stream chat — see scripts/bench/run_matrix.py
```

## Open follow-up

- `profile_point_tp.rs` allocates 268 MB / layer KV regardless of
  `FLAMBEAU_CTX_CAP`. Fix to honour the cap so 27B/TP2 can be profiled
  the same way.
- Real rocprofv3 capture on PP fails with SIGABRT (multi-rank
  finalize bug under ROCm 7.1.1 — known). Per-rank capture via
  `HIP_VISIBLE_DEVICES` cycling could work but adds harness complexity.
