# V2.28.c.1 — 35B async parity divergence at L=16384: long-context GDN race

Root-caused + guarded the V2.28.c-flagged divergence.

## Repro (confirmed)

9B Qwen3.5-Q4_1 Mesh<4>:
- L=16384, ub=128, u_lanes=2, sync → `last_id=198`
- L=16384, ub=128, u_lanes=2, async → `last_id=198` ✓ (V2.28.c)

35B Qwen3.6-A3B-UD-Q4_K_S Mesh<4>:
- L=16384, ub=128, u_lanes=2, sync → `last_id=220`
- L=16384, ub=128, u_lanes=2, async → `last_id=143737` ✗

## Bisection

| L     | ubatch | K = L/ub | u_lanes | last_id | verdict |
|------:|-------:|---------:|--------:|--------:|---------|
| 16384 |   128 |   **128** |       2 |  143737 | wrong |
| 16384 |   256 |       64 |       2 |     220 | ✓     |
| 16384 |   512 |       32 |       2 |     220 | ✓     |
| 16384 |   128 |      128 |   **1** |     220 | ✓     |
|  8192 |   128 |       64 |       2 |  248046 | ✓     |

Conclusive:
- **K > 64 + u_lanes > 1 → race triggered**
- K ≤ 64 at any u_lanes → safe
- u_lanes=1 at any K → safe (single-lane, no cross-lane concurrency)

## Root cause — V2.27.d race, at longer horizon

Exactly the V2.27.d GDN cross-lane race: `rank_session.caches` holds
per-layer GDN state that's NOT lane-local. Under `u_lanes > 1` aux
streams for alternate ubatches on the same rank can concurrently
read/write the same state tensor.

V2.27.d's guard said "at ubatch ≥ 128, per-ubatch kernel work
naturally serialises the race via stream scheduling". That premise
held up to K = 64. Past K = 64 the accumulated race damage per
layer per ubatch compounds enough that ONE cross-lane interleave
among the 128 ubatches is enough to push the final logits into a
different argmax.

9B didn't hit it at K=128 likely by coincidence of the argmax rank
shift — both paths diverged internally but happened to land on the
same token (198). Not a correctness guarantee; guard conservatively
rejects hybrid GDN + u_lanes > 1 + K > 64 regardless of model.

GGUF context_length on 35B is 262144 — far above our L=16384, so
this is NOT a KV-cache-capacity overflow. Confirmed via
`cargo run -p flambeau-cli -- inspect-gguf`.

## Fix

Extended V2.27.d's guard in `forward_prefill_pp_async` to also
reject `K > 64` on GDN models with `u_lanes > 1`:

```rust
let n_ubatches_val = l.div_ceil(ubatch_size);
if n_ubatches_val > 64 {
    bail!(
        "L={l} ubatch_size={ubatch_size} produces K={n_ubatches_val} \
         ubatches. On hybrid (GDN) with u_lanes={u_lanes}, K > 64 \
         accumulates enough cross-lane state-race damage that parity \
         breaks. Use u_lanes=1 or larger ubatch_size to keep K ≤ 64."
    );
}
```

Tested post-fix:
- 35B L=16384 ub=128 lanes=2 → errors (was silently wrong)
- 35B L=16384 ub=256 lanes=2 → works, last_id=220 matches sync
- 35B L=16384 ub=128 lanes=1 → works, last_id=220 (slow but correct)
- 9B long grid capped at L=8192 in the default bench; L=16384 still
  reachable via `FLAMBEAU_PREFILL_L=16384 FLAMBEAU_UBATCH=256`.

## Proper fix (deferred)

Per-lane GDN state + merge logic on lane-boundary ubatches. Same
scope note as V2.27.d: multi-session kernel-adjacent rework for
no measured perf upside (ub=256 lanes=2 at L=16384 = 1461 tok/s
vs the racy ub=128 lanes=2's 1382 tok/s — 5 % faster AND correct).

Users with long-context workloads can set `FLAMBEAU_UBATCH=256`
or higher at L ≥ 16384 and stay in the safe regime.

## Bench updates

- `perf_baseline_qwen35_9b.rs`: default L grid trimmed to ≤ 8192
  (L=16384 would violate the guard at default ub=128).
- `perf_baseline_qwen3_moe.rs`: same.
- V2.28.c's perf numbers at L=16384 documented as valid; guard
  added retroactively.

## Gate

- 35B L=16384 ub=128 lanes=2: clean error with diagnostic message.
- 35B L=16384 ub=256 lanes=2: works + parity matches sync (both
  last_id=220).
- Default bench runs (no env override) stay green up to L=8192.
- No impact on decode path, sync prefill path, or u_lanes=1 paths.

## Regeneration

```bash
# Guard fires (now errors):
FLAMBEAU_PREFILL_ONLY=1 FLAMBEAU_PREFILL_L=16384 \
  FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture

# Safe config (K=64, works + correct):
FLAMBEAU_PREFILL_ONLY=1 FLAMBEAU_PREFILL_L=16384 \
  FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=256 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  ./target/release/deps/perf_baseline_qwen3_moe-* perf_baseline_qwen3_moe --nocapture
```
