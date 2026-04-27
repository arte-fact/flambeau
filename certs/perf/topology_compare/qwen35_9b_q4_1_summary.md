# Qwen3.5-9B-Q4_1 — topology comparison (2026-04-27 → AUTO-6e refresh)

Rig: 4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1.
GGUF: 5.43 GiB (Qwen3.5-9B-Q4_1).
Bench: `tests/topology_compare_qwen35_9b.rs`. tp4 omitted (rig {2,3} BAR1 fault).

## Throughput (tok/s) — fresh after AUTO-6c4 + AUTO-6e3

| topology            | pp | tp | devices       | load (s) | pp8   | pp64  | pp128 | pp512 | tg64  |
|---------------------|----|----|---------------|----------|-------|-------|-------|-------|-------|
| mesh1               | 1  | 1  | `[0]`         | 1.91     | 67.4  | 65.1  | 509.3 | 627.3 | 49.9  |
| pp2                 | 2  | 1  | `[0,1]`       | 1.82     | 73.3  | 66.3  | 515.7 | 635.3 | 59.2  |
| tp2                 | 1  | 2  | `[0,1]`       | 2.13     | 74.0  | 70.2  |  68.5 |  61.3 | 69.0  |
| **tp2 + batched**   | 1  | 2  | `[0,1]`       | —        | 123.2 | 119.7 | 733.2 | **950.7** | —     |
| pp4                 | 4  | 1  | `[0,1,2,3]`   | 1.96     | 83.6  | 68.2  | 522.6 | 629.3 | 55.8  |
| pp2tp2              | 2  | 2  | `[0,2,1,3]`   | 3.11     | 70.8  | 69.2  |  67.1 |  58.0 | 68.0  |
| **pp2tp2 + batched**| 2  | 2  | `[0,2,1,3]`   | —        | 124.7 | 122.5 | 706.2 | **920.0** | —     |

`+ batched` rows opt into the AUTO-6 L-batched TP prefill driver via
`FLAMBEAU_TP_BATCHED=1`. Pure-TP routes through
[`forward_prefill_tp_batched_logits`]; hybrid PP-of-TP routes through
[`forward_prefill_hybrid_batched_logits`] which calls the layer-range-aware
helper [`forward_prefill_tp_batched_layers`] per stage and hands off
`[L, hidden]` F16 between stages instead of one hidden vector.

## Headline vs pp4 (the prior prefill champion)

| topology              | prefill L=512   | decode tg=64    |
|-----------------------|-----------------|-----------------|
| pp4                   | 1.00× (629)     | 1.00× (56)      |
| **tp2 + batched**     | **1.51× (+51 %)** | **1.24× (+24 %)** |
| **pp2tp2 + batched**  | **1.46× (+46 %)** | **1.22× (+22 %)** |
| pp2tp2 (per-token)    | 0.09× (-91 %)   | 1.22× (+22 %)   |
| tp2 (per-token)       | 0.10× (-90 %)   | 1.24× (+24 %)   |
| pp2                   | 1.01× (+1 %)    | 1.06× (+6 %)    |
| mesh1                 | 1.00× (-0 %)    | 0.89× (-11 %)   |

## AUTO-6 gate verdicts

- **TP gate (tp2 prefill L=512 ≥ 0.7× pp4 = 440.5 tok/s):** PASS by **2.16×** (950.7).
- **Hybrid gate (pp2tp2 prefill L=512 ≥ 0.7× pp4 = 440.5 tok/s):** PASS by **2.09×** (920.0).
- Both gates close the original "TP-based topologies have no batched
  prefill" caveat from the AUTO-5 cert. Prefill choice is no longer
  about avoiding TP — it's about whether the model fits two cards.

## What changed in this round

1. **AUTO-6c4** — TP batched dispatcher generalised to all four layer
   flavors (full-attn / GDN × dense / MoE+optional-shared). Bit-exact
   vs per-token within FP32 AR-cadence reassoc tolerance.
2. **AUTO-6e1** — Layer-loop body of the TP batched driver factored
   into `forward_prefill_tp_batched_layers` taking `il_range` +
   `il_cache_offset`. The PP-of-TP hybrid driver invokes it per-stage
   with `il_cache_offset = stage.layer_range.start`.
3. **AUTO-6e3** — `forward_prefill_hybrid_batched_logits` ships:
   embed L on stage 0 → batched layers per stage → `[L, hidden]`
   inter-stage hand-off → output head on the last position. Same
   pinned-bounce DtoH/HtoD path as the per-token hybrid; the only
   thing that grows is the byte count.

## What we still give up

- **Per-request prefill scratch alloc.** Both `tp2 + batched` and
  `pp2tp2 + batched` allocate `ShardedForwardPrefillScratch{Tp,Hybrid}`
  on every prefill call and dispose on return. For high-QPS bursty
  workloads, binding the scratch on `Qwen3MoETpSession` /
  `Qwen3MoEHybridSession` would amortize this — V2.x perf lever.
- **Inter-stage hand-off scales with L.** At L=512 the hand-off is
  `tp_size_next × L × hidden × 2 = 2 × 512 × 4096 × 2 = 8 MB / hop`
  (pinned-bounce ~6.7 GB/s on PCIe 3.0). For Qwen3.6-35B at long
  contexts this becomes the dominant cost; an async overlap with the
  next stage's first kernel would close it.
- **MoE tile8 + sort-by-expert under TP** is still on the MMVQ-per-token
  fallback — a perf lever for `qwen3moe` / `qwen35moe` arches.

## When to pick which (refreshed)

- **Model fits 2 cards** → **`tp2 + batched`** is the universal best:
  best prefill (1.51× pp4) AND best decode (1.24× pp4).
- **Model needs 4 cards** (Qwen3.6-35B, Qwen3-Coder-30B at high
  context) → **`pp2tp2 + batched`**: prefill within 3 % of tp2,
  decode matches, but the model splits across 4 cards.
- **Single GPU available** → mesh1 (66 % of tp2-batched prefill, 73 %
  of tp2-batched decode; no AR / no hand-off cost).

## Next levers (V2.x)

1. **Session-bound prefill scratch.** Drop the per-request alloc by
   carrying `ShardedForwardPrefillScratch{Tp,Hybrid}` on the inflight
   session, sized to the model's `max_position_embeddings`. Saves
   ~1–3 ms per request at 4 ranks (4 × `device.alloc(L * hidden * 2 *
   4)`).
2. **Async inter-stage hand-off.** Overlap stage `s+1`'s first kernel
   with stage `s → s+1`'s pinned-bounce. Especially relevant at
   L≥1024 where hand-off bytes start dominating.
3. **MoE tile8 + sort-by-expert under TP.** For Qwen3.6-35B-A3B and
   Qwen3-Coder-30B prefill, the MMVQ-per-token fallback gives up
   ~25–30 % vs the non-TP tile8 path. Wiring tile8 into the TP MoE
   prefill needs per-rank sort scratch + a check that `expert_ids`
   replication holds under sort-by-expert ordering.
