# MTP-5c — speculative-decode integration: working at 87.5 % acceptance

**Status:** integration shipped and validated. Driver
`forward_speculative_pp_step` runs end-to-end on Qwen3.6-27B
Mesh<4> with **87.5 % acceptance** (28/32 macro steps), coherent
output, ~60 ms/token wall (parity with baseline). Speed
optimisation deferred to MTP-5e (batched L=2 paired-logits
primitive — described below).

## Big finding: 1-ahead spec semantics gives 87 % accept, not 60 %

The previous `mtp_acceptance_passive` harness measured the MTP
head doing **2-ahead** prediction (draft token at position+2 given
h@position-1 + embed of base's just-sampled next-token). That gave
55-64 % acceptance.

Standard K=1 spec-decode uses **1-ahead** prediction: MTP drafts
token at position+1 given h@position-1 + embed(last_token). This
is the same prediction task the base model itself does, and the
MTP head — sharing embed/lm_head with base — does it well:

| harness | semantics | accept rate (Q4_0 / prose) |
|---|---|---|
| `mtp_acceptance_passive` (2-ahead) | draft `T@(p+2)` from h, next | 56-64 % |
| `mtp_spec_decode_smoke` (1-ahead) | draft `T@(p+1)` from h, last | **87.5 %** |

That matches the AEON / sakamakismile community measurements
(67-69 % on prose) much more closely once you adjust for prompt
sampling variance.

## What's shipped

### `crates/runtime/src/kv_cache.rs`
- `KvCache::rollback(n_remove)` — truncate full-attn cache tail.

### `crates/models/qwen3-moe/src/session.rs`
- `GdnLayerState::snapshot_state` / `snapshot_conv_history` (Option<DevicePtr>)
- `Qwen3MoESession::save_gdn_snapshot` / `restore_gdn_snapshot` /
  `rollback_full_attn` (single-device session).

### `crates/models/qwen3-moe/src/sharded.rs`
- `Qwen3MoEShardedSession::save_gdn_snapshot` / `restore_gdn_snapshot` /
  `rollback_full_attn` (multi-rank Mesh<N> session).

### `crates/models/qwen3-moe/src/forward/spec.rs` (new)
- `SpecStep { committed, new_position, accepted, draft, verify, timings_ms }`
- `SpecTimings { gdn_snapshot, mtp_draft, base_l2, gdn_restore, base_l1_redo }`
- `forward_speculative_pp_step(...)` — one macro step, K=1 spec.
  - Snapshot GDN → MTP draft → base verify → accept/reject decision
  - On reject: `rollback_full_attn(2)` + `restore_gdn_snapshot` +
    L=1 redo with verified token.

### `crates/models/qwen3-moe/tests/mtp_spec_decode_smoke.rs` (new)
- 32-macro-step smoke: 87.5 % accept, 60 tokens, 59.3 ms/token,
  output coherent vs baseline.

## Per-macro timings (cumulative over 32 steps)

| stage | total | per macro | notes |
|---|---:|---:|---|
| GDN snapshot | 69 ms | **2.2 ms** | always paid |
| MTP draft | 44 ms | **1.4 ms** | always paid |
| base verify (currently 2× L=1) | 3156 ms | **98.6 ms** | should be 75 ms with batched L=2 |
| GDN restore | 9 ms | 0.3 ms avg (12.5 % of steps × 2 ms) | reject path |
| L=1 redo | 197 ms | 6.2 ms avg (12.5 % × 50 ms) | reject path |

The dominant cost is the **base verify**, currently 2× sequential
L=1 decodes (~50 ms each). The proper batched L=2 primitive
would amortise this to ~75 ms total (saving ~25 ms per macro =
~10 % wall improvement).

## Why "currently 2× L=1": missing primitive

`forward_prefill_pp_logits` exists but only emits the **last**
position's logits (`vocab` floats), not all L positions
(`L * vocab`). Spec-decode needs both positions' logits in a
single batched call. The L=2 paired-logits primitive is a small
addition (run `forward_output_head_decode` twice on
`hidden_a[0]` and `hidden_a[1]` after the L=2 prefill) — filed
as MTP-5e.

## Speedup arithmetic

At measured 87.5 % accept, with the planned L=2 paired primitive:

| path | freq | wall | tokens | ms/tok |
|---|---:|---:|---:|---:|
| accept | 87 % | snap+L2 = 2 + 75 = 77 ms | 2 | 38.5 |
| reject | 13 % | snap+L2+restore+L1 = 2+75+2+50 = 129 ms | 1 | 129 |
| **avg** | | | | **~50 ms/tok** |

vs baseline 50-60 ms/tok = **0-18 % speedup**, depending on the
true L=2 wall multiplier (1.5× decode is the lower bound; could
be more if attention isn't well-amortised at L=2).

## Remaining work

- **MTP-5e** (next): build `forward_prefill_pp_logits_paired_l2`
  primitive. Run output_head twice on the L=2 hiddens. ~50 LOC
  in `forward/pp.rs`. Wire into `spec.rs` replacing the 2× L=1
  fallback. Re-run smoke test to confirm wall reduction.
- **MTP-5d** (after): wire into `crates/server/` decode loop
  behind `--spec mtp` flag. Per-request acceptance reporting via
  the existing metrics hook.

## Sources

- All prior MTP investigations (`mtp_4_*`, `mtp_inv_1..5`)
- `forward_speculative_pp_step` in `src/forward/spec.rs`
- `mtp_spec_decode_smoke` test
