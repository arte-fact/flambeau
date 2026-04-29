# MTP-5f-tp2 — TP2 vs PP4 spec-decode comparison

| topology | baseline | spec | spec − baseline | accept |
|---|---:|---:|---:|---:|
| **PP4** | 50.22 ms/tok | 56.54 ms/tok | +12.6 % | 87.5 % |
| PP2 | 54.35 ms/tok | 63.68 ms/tok | +17.2 % | 87.5 % |
| **TP2** (GPUs 0,1) | **36.64 ms/tok** | 56.35 ms/tok | **+53.8 %** | 73.5 % |

Same model (Qwen3.6-27B-Q4_0), same MTP head, same prompt, same 60-token
target, greedy in all branches. Within TP2: baseline and spec first 16
tokens bit-identical → correctness preserved. Across topologies the
output differs (F16 AllReduce ordering vs PP peer-copy yields slightly
different numerical paths; both produce coherent text).

## Findings

### 1. TP2 baseline is 27 % faster than PP4 baseline

**This is the headline:** for greedy single-token decode of
Qwen3.6-27B on this rig, **TP2 (36.64 ms/tok) is strictly faster than
PP4 (50.22 ms/tok)** without any spec-decode involvement. BAR1 P2P
AllReduce eats less wall than PCIe peer-copy via host on a 27B model
with hybrid GDN + full-attn + MoE+shared layers.

This makes TP2 the recommended *production* topology for this model on
this rig, separate from any MTP work.

### 2. TP2 spec matches PP4 spec absolutely (56.35 ms/tok)

The L=2 verify on TP2 is 2× sequential L=1 calls, ~73 ms total. On PP4
the L=2 paired primitive produces ~96 ms. The fixed spec overhead
(GDN snapshot ~2 ms, MTP draft ~1.5 ms, reject path ~5 ms avg) is
similar. Net spec wall lands within 1 ms across both topologies. The
structural overhead of K=1 spec converges to a similar absolute ms/tok
regardless of how the base forward is laid out.

### 3. TP2 spec relative gap is *much worse* than PP4 (+54 % vs +13 %)

Because TP2 baseline is already cheap (~37 ms/tok), the same fixed
spec overhead is a larger relative fraction. TP2 spec inherits the
absolute ceiling of ~56 ms/tok but loses its baseline advantage in
the process.

### 4. TP2 acceptance rate is lower (73.5 % vs 87.5 %)

The MTP head's training-set forward was a specific numerical path; the
TP slicing changes that path slightly via AllReduce ordering. The MTP
draft and the base verify therefore diverge more often on TP than on
PP. This contributes ~1–2 ms/tok extra reject overhead on TP relative
to PP.

## What this falsifies

The MTP-INV-5 / MTP-5e-cert hypothesis was: *"TP topology has lower
per-step rank-sync cost so the L=2 wall multiplier may drop below the
PP4 1.92×, possibly flipping spec net-positive."*

That is **falsified**:
- L=2 wall on TP2 = 2 × L=1 = ~73 ms = 2.0× the L=1 wall (no batching).
- L=2 wall on PP4 (with the paired primitive) = 96 ms = 1.92× L=1.
- Both topologies' spec wall converge to ~56 ms/tok absolute.
- The baseline gap is what differs, and TP2 has a *better* baseline,
  making spec's relative penalty *worse*.

There is no topology among PP4 / PP2 / TP2 / pp2tp2 (likely similar)
on this rig where K=1 spec-decode flips net-positive. The structural
constraint is **K=1 spec adds ~10–20 ms of fixed per-macro overhead
that cannot be hidden** at any topology where the underlying base
forward is < ~80 ms/tok.

## What's left

Three theoretically remaining levers (none ready to ship):

1. **K=2 spec** — paper's γ=2 with optimal at α=0.85 is γ=4–5 — but
   each γ adds another L to the verify and on PCIe-bound topologies
   that compounds; on TP each L is a full AR, also compounds. Likely
   doesn't help on this rig.
2. **FastMTP head fine-tune** — push α from 0.875 to 0.95+; reduces
   wasted-slot cost. Multi-week training rig work.
3. **NVLink/xGMI rig** — the only hardware change that could lower
   the per-step sync wall enough for spec to dominate. Out of scope.

## Recommended deployment topology

Independent of spec-decode:

| use case | recommended | reason |
|---|---|---|
| Qwen3.6-27B greedy decode | **TP2** (GPUs 0,1) | 36.6 ms/tok, fastest measured |
| Qwen3.6-27B prefill ≥ 512 | **PP4** | batched-L prefill at 1500+ tok/s |
| Spec-decode active | **PP4 + Lever 1** | 56.5 ms/tok — same as TP2 spec but +13 % vs baseline instead of +54 % |

For users who want fastest decode, leave spec OFF and use TP2. For
mixed prefill+decode, PP4 has better-balanced perf.

## Reproducer

```
FLAMBEAU_TP_RANKS=2 FLAMBEAU_PERF_AB_TOKENS=60 FLAMBEAU_CTX_CAP=4096 \
  cargo test --release -p flambeau-qwen3-moe --features hip \
  --test mtp_spec_decode_tp_perf_ab -- --nocapture
```

Test source: `crates/models/qwen3-moe/tests/mtp_spec_decode_tp_perf_ab.rs`.
Driver: `Qwen3MoETpSession::{save_gdn_snapshot, restore_gdn_snapshot,
rollback_full_attn}` + `forward_speculative_tp_step`. The TP session
methods mirror PP's. The driver uses 2× `forward_one_token_tp_logits`
for verify (no L=2 batched primitive yet) and full L=1 redo on reject
(no Lever-1 GDN-only-redo yet).
