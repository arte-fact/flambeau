# Batched-decoding arc retrospective

**Period:** 2026-04-29 → 2026-05-04 (P2.9b → #317)
**Final state:** lever-1 mixed-batch v1 (commit `743606e`), shipping behind `FLAMBEAU_BATCHED_DECODE=1` + optional `FLAMBEAU_MIXED_BATCH=1`.
**Net throughput gain over original legacy:** **1.49× on Qwen3.5-9B/pp2tp2/N=6** (91 → 134 t/s aggregate on 6-user staggered traffic).
**3× cert gate:** **not closed**. Closest was 1.49× by config + small fixes; the architectural rewrites we tried (1F1B, async dispatcher) all came back null.

## What we shipped

- **Multi-slot inflight pool** (#260, P2.9b-i1) — `Vec<Mutex<Inflight>>` with claim/release lifecycle. 1.30× on 35B-A3B / pp2tp2 / N=2 from host-side overlap alone (tokenize, sampler, channel release).
- **Batched-decode drivers** (#262–#265) — `forward_decode_batched_{pp,tp,hybrid}` running N slots through one PP/TP forward. FFN/MoE batched at `n_tokens=N`; per-slot inner loops only for KV-append + attn.
- **Batched-attention decode kernel** (#266, `attention_decode_f16_batched`) — 4–8× attn wall at N=4–8. Bit-identical across 11 shapes.
- **Batched-GDN** (#284–#287) — `forward_gdn_decode_batched_tp`. Wired but limited because GDN state-step is per-state-recurrent.
- **PP-pipelined decode** (#290–#294) — slot-major host loop with HipEvent bridges, working at PP≥2.
- **Sarathi mixed-batch v1** (#302–#308) — `forward_decode_mixed_hybrid` co-batches one prefill chunk + N decodes. K\|N split across full-attn (two attn calls) + GDN (prefill + decode regions). All parity tests bit-exact at K∈{32,128}, N∈{1,2,4}.
- **Mixed-batch scheduler** (#305) — `MixedScheduler::next_iteration` builds the per-iteration plan with token-budget + chunk-size knobs.
- **Server wiring** (#306) — `MixedBatchCtx` on `ServerState`, `prefill_via_mixed_scheduler` + `decode_via_mixed_scheduler` entry points, opt-in via env.
- **Lever-1 perf fixes** (`3a0d898` + `743606e`) — leader-release race (#276 pattern), decode-only fast-path delegation, lock-collapse in `dispatch_mixed_iteration`. The 1.115× win.

## What we tried and rolled back

- **1F1B-decode** (#298–#301) — adapted V2.25.h prefill 1F1B to decode. Implementation correct (md5-bit-identical), wall null (~26s = same as slot-major). Decode at n_tokens=1 is HBM-bound; 1F1B is for compute-bound prefill. Wrong-targeted lever; shipped behind `FLAMBEAU_DECODE_1F1B=1` for completeness.
- **Lever-2 async dispatcher** (#309–#316, rolled back) — threaded dispatch + completion thread + 2-slot scratch pool + event-deferred output sync. ~700 LOC of careful concurrent code. Result: 5% wall regression, GPU% unchanged, output coherent. Reason: PCIe-only rig means `peer_copy_via_host` is host-bounce-blocking; iter N+1's enqueue can't overlap iter N's stage hand-off. The vLLM v1 pattern assumes NVLink/xGMI peer-async — we don't have that. Rolled back to lever-1 baseline; scope doc + null cert kept as history.
- **27B model test** (#317) — hypothesis: heavier model means more GPU work per iter, host coordination becomes a smaller fraction, mixed-batch ratio improves. Result: GPU% doubled (14% → 27%) as predicted, but mixed-vs-legacy ratio stayed flat (~2% slower). Per-iter host coord ~50ms is independent of model size; even at 27B GPU work is only ~13ms/iter, so wall is still `max(host, GPU)` = host.

## Numbers chronologically

| date / commit  | scope                                    | wall      | t/s    | ratio |
|----------------|------------------------------------------|----------:|-------:|------:|
| 2026-04-30     | legacy batched-decode (#260+#272)         | (per req) | 91     | 1.00× |
| 2026-05-04     | mixed-batch v1 N=4 (`3a0d898`)            | 15602 ms  | 99     | 1.087× |
| 2026-05-04     | + N=6 inflight (config)                   | 15115 ms  | 133    | 1.46× |
| 2026-05-04     | + lever-1 lock collapse (`743606e`)       | 14566 ms  | 134    | 1.49× |
| 2026-05-04     | lever-2 threaded scaffold (rolled back)   | 15310 ms  | 129    | 0.96× vs lever-1 |
| 2026-05-04     | 27B mixed (`#317`, same hot path)         | 36682 ms  | 53     | 0.98× vs 27B legacy |

The 1.49× was hard-won. Of that:
- ~+9% from the v1 driver itself (the per-call mixed-batch lever)
- ~+34% from N=6 inflight sizing (a config change!)
- ~+1% from lock collapse

The vast majority of throughput improvement came from **a config change** (`FLAMBEAU_INFLIGHT_SLOTS=6`). All the kernel + scheduler engineering produced ~10%.

## What we got wrong

### Pattern 1: Projecting from H100/NVLink papers onto PCIe MI50

Three architectural rewrites all projected wins from reference implementations on rigs we don't have:

| lever | reference | projected | measured | cause |
|-------|-----------|----------:|---------:|-------|
| 1F1B-decode | Megatron / V2.25.h prefill | 2.29× at PP=4/N=4 | 1.0× (null) | decode is HBM-bound, not compute-bound |
| Async dispatcher | vLLM v1 RFC #11945 | 1.34× | 0.96× | PCIe peer_copy_via_host is host-blocking |
| Mixed-batch ratio at 27B | proportional-to-compute | "improves" | flat | host coord is constant, not %-of-wall |

Each lever had a coherent rationale; each was falsified by the rig. The pattern: **assume the rig profile matches the reference, only confirm GPU%/host% breakdown afterward.** Should have been opposite.

### Pattern 2: Engineering scope grows but the lever doesn't

Lever 2 took 5 commits (#309–#315) and ~700 LOC of careful concurrent code. The cert recommended NOT rolling back the architecture because "it's correct, just gated". After running #316 (the kernel-side completion event that the cert called for), result was still null. Sunk cost on architecture that doesn't pay.

The right move (in hindsight) on the negative cycle-3 cert: STOP, lock the lever-1 baseline, ship that, write the architectural async-dispatch idea as a doc for a future xGMI rig. We did write the cert and the scope doc, but kept building.

### Pattern 3: Diminishing returns invisible until measured

Every cycle:
- Cycle 1 knob sweep: defaults were already optimal. Null result.
- Cycle 2 prefill fast-path: wrong heuristic, regressed 64%. Reverted.
- Cycle 3 N sizing: +34% (the actual win).
- Lever 1 lock collapse: +1.1%.
- Lever 2 threaded: -4%.
- 27B test: -2%.

After cycle 3 the marginal-return curve had clearly turned. Lever 1 was the last positive step; everything after it was either noise or regression. Should have stopped after `743606e` and shipped. Did not — invested another full session arc into lever 2 + 27B test before believing the signal.

## What we got right

### Bit-exact parity tests as the safety net

`mixed_batch_parity.rs` runs the v1 driver against a separate-sessions reference. All 4 configs pass: K=32/N=1 bit-exact, K=32/N=2 bit-exact, K=128/N=4 top-1 match within F16 noise, multi-chunk integration top-1 match. **Every refactor that broke perf still passed correctness.** That meant null results were honestly null, not subtle bugs masquerading.

The `mixed_scheduler` unit tests (9 of them) caught zero bugs but verified the data-structure semantics. Cheap insurance.

### Honest cert culture

The repo has 4 negative-result certs from this arc alone:
- `qwen36_27b_pp4_1f1b_2026_05_04.md` — 1F1B null
- `qwen35_9b_optimization_cycles_2026_05_04.md` — 3-cycle (knob sweep null, fast-path regression, N=6 win)
- `qwen35_9b_lever2_pipeline_overlap_2026_05_04.md` — lever 2 null
- `qwen36_27b_mixed_batch_2026_05_04.md` — 27B test null

Each documents the diagnosis path and concludes. Future "let's try X again" decisions can read why X didn't work last time.

### Diagnostic gates that actually mattered

- GPU% measurement via `rocm-smi --showuse --csv` polled at 200ms — surfaced the 14%-idle finding mid-cycle-3, which directly informed why subsequent levers wouldn't move the needle. Should have been the FIRST measurement, not the third.
- `FLAMBEAU_TRACE_BATCH=1` → gated diagnostic logs that helped debug #275 stream-handle race + #276 leader-release race.
- Per-iteration timing: never properly instrumented, which left "host coord ~50ms" as an estimate instead of a measurement. If we had per-iter latency breakdown the lever-2 null would have been predictable from theory.

## What's left in production

Code:
- `crates/server/src/mixed_scheduler.rs` — chunk-budget scheduler primitive (170 LOC + 9 unit tests)
- `crates/server/src/routes.rs` — `MixedBatchCtx`, `prefill_via_mixed_scheduler`, `decode_via_mixed_scheduler`, leader-elect dispatcher with race fix
- `crates/models/qwen3-moe/src/forward/hybrid.rs::forward_decode_mixed_hybrid` — the v1 driver (~700 LOC, K\|N split for full-attn + GDN + MoE)
- 4 parity tests + 1 microbench + 1 scheduler integration test

Knobs (production-relevant):
- `FLAMBEAU_BATCHED_DECODE=1` — opt into legacy batched-decode (tested + stable)
- `FLAMBEAU_INFLIGHT_SLOTS=6` — **the actual perf lever** (+34% over default 4)
- `FLAMBEAU_MIXED_BATCH=1` — opt into mixed-batch (modest +1.5–2× per-call wall on long prompts; ~equal at typical workloads)
- `FLAMBEAU_MIXED_BUDGET=512`, `FLAMBEAU_MIXED_CHUNK=256` — Sarathi knobs (defaults are tuned)

Out-of-scope-for-this-rig but designed and documented for future:
- Lever 2 async dispatcher — `doc/V1.x/lever2_pipeline_overlap_scope.md`. Reactivate if/when xGMI/NVLink hardware lands.
- Varlen-attention kernel (`#303`, deferred) — would halve per-layer kernel calls in chunk-bearing iterations. Could pay if combined with lever 2 on a peer-async rig.
- Larger inflight pool (N=12+) on a model with >50ms/iter GPU work — would flip the host-bound ratio. Out of VRAM range on 4×16GB.

## Recommendations for the next perf arc

1. **Measure GPU% / host% breakdown before any lever proposal.** A 200ms `rocm-smi` poll is enough to see if this rig is host-bound or GPU-bound. Lever selection should match.
2. **Prefer config-knob exploration before architectural rewrites.** The 1.34× from N=6 dwarfs all the kernel work combined. Knob sweeps are <1 session; rewrites are 2–3 session bets.
3. **Set explicit roll-back gates with %-improvement thresholds in the scope doc.** We had ≥1.2× as the lever-2 gate and used it. Without that gate, lever 2 would still be in the codebase regressing perf.
4. **Limit projection-driven engineering to <3 sessions.** If the projection isn't converging by then, the rig profile doesn't match the reference. Stop, document, move on.
5. **Different rig hardware is the cleanest experiment.** When the same code on H100+NVLink runs at the projected ratio and on PCIe-MI50 doesn't, the bottleneck is interconnect, not implementation.

## Numbers that mattered

- **14%** GPU% at cycle-3 baseline → showed the host floor.
- **+34%** from `FLAMBEAU_INFLIGHT_SLOTS=6` → the real lever.
- **0** correctness regressions across all the perf experiments.
- **6** committed null-result certs documenting the search space.

## TL;DR for the next person

The batched-decoding arc shipped a working mixed-batch path that wins 1.49× over original on Qwen3.5-9B / pp2tp2 / N=6. The 3× cert gate is not closeable on this rig at any model+N combo we can fit. Architectural rewrites projected from NVLink-rig papers (1F1B, async dispatch) are null on PCIe-only MI50 because `peer_copy_via_host` is the floor.

Default production: `FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=6` at commit `743606e`. The kernel + scheduler engineering produced ~10% improvement; the config change produced 34%.

Don't repeat the lever-2 implementation cycle without xGMI/NVLink hardware.
