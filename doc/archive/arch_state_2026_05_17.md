# Arch implementation state: qwen3-moe vs gemma4 (2026-05-17)

State-of-the-code audit ahead of cross-arch mutualisation. Captures
where each crate is, what the actual deltas are, and which deltas are
opportunities to push into shared code (`flambeau-blocks` /
`flambeau-server-core`) versus deltas that are legitimately
arch-specific.

## Section 1 — Structural shape

### qwen3-moe (weights ↔ session ↔ scratch are split)

```
Qwen3MoE<Sharded|Tp|Hybrid>Model         — weights only (immutable after upload)
Qwen3MoE<Sharded|Tp|Hybrid>Session       — KV cache + per-rank cache state
ShardedForwardPrefillScratch{,Tp,Hybrid} — transient prefill scratch
ShardedForwardOneTokenScratch{,Tp,Hybrid} — transient decode scratch

# Server bundles them per-request:
{Pp,Tp,Hybrid}HipSession {
    session: Qwen3MoE<*>Session,
    prefill: ShardedForwardPrefillScratch<*>,   // PP-only
    decode:  ShardedForwardOneTokenScratch<*>,
}
```

This is the shape that lets multi-slot batched decode work:
`Model` (shared weights) is shared across slots, each slot owns its
own `Session` + `decode` scratch.

### gemma4 (everything bundled in driver)

```
Gemma4<Pp|Tp|Hybrid>Driver {
    cfg, layout,
    stages: Vec<Gemma4<*>Stage {
        layer_weights: Vec<Gemma4LayerWeights>,
        kv_caches: Vec<Option<KvCache<F16Contig>>>,
        ... // all bundled, one instance = one request slot
    }>,
}
```

Hard cap: `FLAMBEAU_INFLIGHT_SLOTS = 1` enforced at boot (server/serve.rs:681).
The current `Gemma4Session::reset_for_next_request` is a **no-op with a
warning comment** — second request sees prior KV as a prefix and
produces garbage. Single-request-only by construction.

## Section 2 — Forward-path surface

| Entry | qwen3-moe | gemma4 |
|---|---|---|
| Decode-1 argmax | `forward_one_token_pp/tp/hybrid` | `driver.forward_one_token` |
| Decode-1 host logits | `forward_one_token_*_logits` | `driver.forward_one_token_logits` |
| Decode-1 **keep on device** | `forward_one_token_*_keep_logits_on_device` | **missing** |
| Batched decode (N≥1) | `forward_decode_batched_pp/tp/hybrid` | **missing** |
| Prefill (L tokens) | `forward_prefill_pp/tp/hybrid_logits` | `driver.forward_prefill_logits` |
| Prefill (async, lanes) | `forward_prefill_pp_async` | **missing** |
| Prefill (paired L=2, spec) | `forward_prefill_pp_logits_paired_l2` | **missing** (MTP dropped) |
| Prefill chunked + on_boundary | `prefill_logits` w/ boundary cb | per-call only |

## Section 3 — Server integration (`Model` + `Session` trait)

| Trait method | qwen3-moe override | gemma4 override |
|---|---|---|
| `topology()` | `"pp" / "tp" / "pp+tp"` (3 impls) | `Gemma4Model.topology` field |
| `supports_scheduler_batching()` | **true** | false (default) |
| `requires_prefill_serialiser()` | true (TP, Hybrid) | false (default) |
| `requires_tp_prefill_scratch()` | true (TP) | false (default) |
| `forward_decode_batched()` | delegates to `qwen3moe_forward_decode_batched` | uses default impl → `Session::decode_one_logits` |
| `chat_stop_markers()` | empty | `<end_of_turn>`, etc. (7 markers) |
| `Session::decode_one_logits()` | default-bails (qwen routes via batched) | real impl → `driver.forward_one_token_logits` |
| `Session::reset_for_next_request()` | clears KV+GDN | **no-op (broken)** |
| `Session::bos_id()` | default `None` | returns BOS for prepend |
| `Session::as_model_driver_mut()` | None (qwen has no `ModelDriver` impl on the inflight) | `Some(driver)` |

## Section 4 — Feature deltas

| Feature | qwen3-moe | gemma4 | Mutualise? |
|---|---|---|---|
| Multi-slot inflight | ✓ | ✗ (slots=1 hard cap) | **YES — gemma4 needs weights/session split** |
| KeepOnDeviceLogits sink | ✓ | ✗ | **YES — generic `LogitsSink` enum** |
| Async/laned prefill | ✓ (1F1B PP) | ✗ | YES (but distant — needs scratch split first) |
| Batched decode N≥2 | ✓ | ✗ | YES — once gemma4 multi-slot lands |
| Scheduler engagement | ✓ | ✗ | YES — once batched lands |
| Prefix cache | ✓ | gated off (capability bail) | YES — capability trait |
| Multiple KV layouts | ✓ (F16/Q8) | F16 only | YES — `KvCache<L: CacheLayout>` typestate already exists |
| Per-layer-embd side-channel | ✗ | ✓ (E2B/E4B) | gemma4-specific |
| iSWA + final-logit softcap | ✗ | ✓ | gemma4-specific |
| GDN hybrid (recurrent layers) | ✓ | ✗ | qwen-specific |
| MoE | ✓ | ✓ | already mutualised (`blocks::MoeExperts`) |
| Spec-decode paired-L2 | ✓ (MTP-shaped) | ✗ | NO — MTP dropped, qwen-specific |
| TP prefill serialiser | ✓ | not needed | qwen-specific |
| Real-GGUF parity tests | ✓ (`real_text_tp_correctness`) | ✓ (just shipped #94 harness) | YES — already done |

## Section 5 — Anti-pattern: arch-specific dispatch in shared files

`crates/server/src/model.rs::qwen3moe_forward_decode_batched` is a
~200-LOC dispatcher that downcasts to `as_pp / as_tp / as_hybrid` and
does arch-specific scratch lookups. It's called from each of the
three qwen3-moe `Model::forward_decode_batched` impls, all of which
just forward to it.

This **violates CLAUDE.md rule 13** ("Arch-specific glue lives in the
model-glue crate"). The right shape:

- `qwen3moe_forward_decode_batched` body moves into
  `flambeau_qwen3_moe::server_glue` (or per-topology files).
- `model_handle.rs` becomes a thin trait impl that calls into the
  model crate.
- `gemma4_handle.rs` follows the same pattern (after gemma4's
  weights/session split lands).

`qwen3moe_forward_decode_batched`'s **N=1 fused fast-path** (today's
`9e54d8b`) is also arch-specific. It lives in the wrong crate
right now.

## Section 6 — Proposed mutualisation pass

Four phases, three of them gemma4-prerequisite (gemma4 must split
state to participate in shared infrastructure):

### Phase A — `LogitsSink` enum (foundation, 1 session)

```rust
// flambeau-server-core::traits
pub enum LogitsSink<'a> {
    Host(&'a mut Vec<f32>),
    Argmax,
    KeepOnDevice,
}
```

- Both arches accept `&mut LogitsSink` in their `forward_one_token_*`
  variant (collapse qwen's three sibling fns into one).
- Server `dispatch_decode_one` and the scheduler decode loop become
  arch-agnostic for the sink choice.

### Phase B — gemma4 weights/session split (the big lift, 2-3 sessions)

Today's `Gemma4<*>Driver` becomes two pieces:

```
Gemma4<*>Model  — weights, layout, layer_weights, ...
Gemma4<*>Session — kv_caches, scratch
```

`Gemma4<*>Driver` becomes the per-request bundle (matches qwen3-moe
`{Pp,Tp,Hybrid}HipSession`).

Wins:
- `FLAMBEAU_INFLIGHT_SLOTS > 1` allowed for gemma4.
- `reset_for_next_request()` becomes real (today: no-op + warning).
- Multi-slot batched decode possible (Phase C / future).
- KeepOnDeviceLogits variant becomes trivial.

Risk: ~3 × 700 LOC refactor, touches code we just stabilised
(#102 hybrid, #106-108 fixes). Bisect carefully.

### Phase C — Move arch-specific dispatch into model crates (1 session)

- Move `qwen3moe_forward_decode_batched` body from
  `server/src/model.rs` into `flambeau-qwen3-moe::server_glue`.
- Same for gemma4 once Phase B done.
- Server's `Model::forward_decode_batched` impls become 1-line
  forwards.
- CLAUDE.md rule 13 honoured.

### Phase D — Unified scheduler engagement (1 session)

After A+B+C: `state.model.supports_scheduler_batching()` returns
true for BOTH arches. The scheduler decode loop's "PP-only fast path"
predicate (`model.topology() == "pp"` in the GPU sampler hook, etc.)
also vanishes — replaced with capability trait methods.

## Section 7 — Don't-do (anti-monkey-copy list)

- ✗ Add `forward_one_token_pp_keep_logits_on_device` to gemma4 by
  copy-paste. Do `LogitsSink` (Phase A) and have BOTH arches gain
  the variant via the enum, not a new fn per arch.
- ✗ Add `forward_decode_batched_pp` for gemma4. Without weights/session
  split (Phase B), the batched form re-uses the legacy fused kernels
  at N=1 anyway — which is what today's qwen `9e54d8b` fix routes to.
  So adding batched-decode to gemma4 SOLO buys nothing until Phase B.
- ✗ Hack arch-specific branches into `routes/decode_loop.rs` for
  gemma4 multi-slot. That's the cliff we keep walking up to (Phase
  12.5 / 12.7 / 12.10 cleanup commits in MEMORY). Drive feature
  parity through the trait surface, not branches.

## Section 8 — Recommended next session

Given the scope:

**Option A (lowest risk, sets pattern):** Phase A only —
`LogitsSink` enum. Each crate gains a single `forward_one_token_*`
that switches on sink. ~1 session. Doesn't unlock new features but
unblocks Phase C move-dispatch-into-model-crate later.

**Option B (highest near-term value):** Phase B1 — gemma4 PP-only
weights/session split. Don't do TP or Hybrid yet. PP is the simplest
of the three drivers + the test coverage is the strongest. Establishes
the pattern; TP/Hybrid follow once verified.

**Option C (smallest tactical win):** Wire qwen3-moe's existing
`forward_one_token_*_keep_logits_on_device` into the scheduler path
+ GPU sampler. ~3% chat decode improvement. No mutualisation but
real perf. Falls inside the current `9e54d8b` direction.

My recommendation: **Option A first** (sets the pattern, low risk),
then **Option B** (unlocks gemma4 multi-slot + reset fix — both are
real correctness/feature wins). C can opportunistically piggyback in
Phase A's `LogitsSink` switch.

## Files touched in this audit

- `crates/models/gemma4/src/{lib,pp,tp,hybrid,session,single_device}.rs`
- `crates/models/qwen3-moe/src/{lib,session,sharded,tp_sharded,hybrid}.rs`
- `crates/models/qwen3-moe/src/forward/{pp,tp,hybrid,batched}.rs`
- `crates/server/src/{model,model_handle,gemma4_handle,serve}.rs`
- `crates/server/src/routes/decode_loop.rs`
- `crates/server-core/src/traits.rs`

No code changes. Audit only.
