# V2.28.a-i1 — Qwen3.6-27B loads via existing `qwen35` loader; u_lanes=2 unsafe

## Load + forward smoke (Mesh<4>)

Qwen3.6-27B-Q8_0 is `arch=qwen35` per GGUF metadata — same family as 9B.
Existing `Qwen3MoEShardedModel::load` handled it with zero code
changes. Model loads on Mesh<4> (6.75 GiB/rank weights + 2.1 GiB/rank
KV at context_length=262144 → 8.9 GiB/rank, fits 16 GiB cards).

Dims vs 9B:
- hidden=5120 (9B: 3584)
- n_heads=24, n_kv_heads=4, head_dim=256 (9B: 28/4/128)
- n_layers=64, full_attention_interval=4 → 16 full-attn + 48 GDN
  (9B: 36 = 9 full-attn + 27 GDN)
- ffn=17408, ssm_inner=6144
- weights: attn + FFN + ssm_* all Q8_0 (9B: Q4_1)

Parity smoke at L ≤ 4096 — sync + async u_lanes=1 match bit-exact:

| L    | sync  | async u_lanes=1 | last_id |
|------|------:|----------------:|--------:|
|  512 |  138  |            220  |  248046 |
| 1024 |  138  |            n/a  |  248046 |
| 2048 |  136  |            115  |  248046 |
| 4096 |  130  |            110  |      62 |
| 8192 |  OOM* |            101  |  248046 |

*Sync OOMs at L=8192 because scratch is sized for full L (570 MiB).
Async u_lanes=1 uses ubatch-sized scratch and fits.

## u_lanes=2 is UNSAFE on 27B — GDN race scales with layers/rank

Triple-run at L ∈ {4096, 8192} with `u_lanes=2, ub=128`:

| run | L=4096 last_id | L=8192 last_id |
|----:|---------------:|---------------:|
|   1 |             62 |              1 |
|   2 |         142081 |            198 |
|   3 |            516 |            198 |

**Non-deterministic** across runs. Also per-run perf varies wildly
(375 → 255 → 217 tok/s at L=4096).

Root cause: same GDN cross-lane race as V2.27.d + V2.28.c.1, but
27B has **12 GDN layers per rank** (48 GDN / 4 ranks) vs 9B/35B's
~7. More shared-state tensors per rank = more race opportunities
per ubatch. The race manifests even at K=32 on 27B (L=4096), far
below V2.28.c.1's K>64 threshold for 35B.

## Guard

Extended `forward_prefill_pp_async` entry check with a model-aware
reject for high-GDN-density models:

```rust
if gdn_per_rank > 10 {
    bail!("... gdn_per_rank={gdn_per_rank} > 10 races non-deterministically
          at all K. Use u_lanes=1.");
}
```

Thresholds:
- 9B: gdn/rank=7 → allowed (K ≤ 64 guard still active from V2.28.c.1)
- 35B: gdn/rank=7 → same
- **27B: gdn/rank=12 → u_lanes > 1 unconditionally rejected**

Post-guard:
- 27B u_lanes=1: fully deterministic across triple-run (115/110/101
  tok/s at L=2048/4096/8192, identical last_ids each run)
- 27B u_lanes=2: clean error message pointing to cert
- 9B u_lanes=2 ub=128: still works at 2352 tok/s at L=4096 (unaffected)

## Perf summary for 27B on current stack

| L    | sync | async u_lanes=1 | best valid |
|------|-----:|----------------:|-----------:|
|  128 |  123 |         123     |    **123** |
|  512 |  138 |         220     |    **220** |
| 1024 |  138 |         n/a     |        138 |
| 2048 |  136 |         115     |    **136** (sync) |
| 4096 |  130 |         110     |    **130** (sync) |
| 8192 |  OOM |         101     |    **101** |

u_lanes=1 loses to sync at mid-range L (overhead of ubatch chunking
without concurrency benefit). Users running 27B should use **sync
at L ≤ 4096, u_lanes=1 at L=8192** until per-lane GDN state lands
(V2+ work).

## Ship status

27B is **loadable + correct on Mesh<4>** post-guard. Perf is
significantly lower than 9B/35B per-token (130 tok/s at L=4096 vs
9B's 2498 tok/s + 35B's 1906 tok/s async) because:
- Q8_0 weights (larger per-layer transfer than Q4_1)
- 64 layers + GDN state (bigger forward than 9B's 36 / 35B's 40)
- Stuck on sync or u_lanes=1 (no aux-stream parallelism)

The proper perf fix is per-lane GDN state (V2+). For now 27B is
best suited to workloads that don't need >130 tok/s prefill.

## Proper fix (deferred)

Per-lane GDN state + merge logic. Would eliminate ALL three guards
(V2.27.d ubatch<128, V2.28.c.1 K>64, this V2.28.a-i1 gdn_per_rank>10)
and let 27B run u_lanes=2 for the same 2-3× prefill speedup 9B/35B
get. Multi-session kernel + forward-path rework.

## Gate

- `27B Q8_0 Mesh<4>` loads in 11 s + forwards correctly in sync +
  async u_lanes=1 paths.
- `u_lanes > 1 && gdn_per_rank > 10` → clean guard error with cert
  reference.
- 9B + 35B unaffected (existing async u_lanes=2 paths still green).

## Regeneration

```bash
# Guard fires (now errors):
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.6-27B-Q8_0.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture

# Safe (u_lanes=1 or unset):
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.6-27B-Q8_0.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b --nocapture
```
