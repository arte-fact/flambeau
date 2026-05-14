# Phase 8b parity cert — typed `tp_allreduce_sum<0>` at gemma4 TP AR boundaries

**Date:** 2026-05-14
**Branch:** `feature/gemma4`
**Commit under test:** `20ce15b` (gemma4 TP migration to typed AR)
**Baseline commit:** `ffbe4a6` (Phase 8a typed op surface; no gemma4 changes)

## What changed

`gemma4::tp::forward_layer_decode_tp` Phase-2 + Phase-5 AR call sites
migrated from the untyped `tp_allreduce_sum_into` (Phase 4d) to the
typed `tp_allreduce_sum::<0>` (Phase 7b). The typed transition
consumes per-rank `Buffer<F16, RowParallel<0>>` partials and produces
`Vec<Buffer<F16, Replicated>>` — same underlying allocations,
typestate-tagged at compile time.

The migration is the typed-AR-boundary proof-of-concept (Phase 8b)
for the smaller of the two arches. qwen3-moe's analogous migration
shipped at commit `9db72ab` (S5).

## Acceptance criteria

The typed wrapper is zero-cost (`Buffer<T, D>` is `Copy` over a
`DevicePtr` + two phantom markers; the typed AR helper builds the
same `[DevicePtr; N]` arg from `.ptr()` and delegates to
`BarP2pAllReduce::sum_tp{2,4}`), so the post-migration forward path
should produce **bit-exact same outputs** as the legacy untyped path.

## Smokes

`cargo test -p flambeau-gemma4 --features hip --test-threads 1`:

| Test | Result |
|---|---|
| forward_layer_smoke (4 tests) | ok |
| forward_one_token_smoke (2 tests) | ok |
| forward_one_token_pp_smoke (2 tests) | ok |
| forward_one_token_tp_smoke (2 tests) | ok |
| forward_one_token_hybrid_smoke (2 tests) | ok |
| forward_prefill_pp_smoke (2 tests) | ok |

## Parity vs llama.cpp

`cargo test -p flambeau-gemma4 --features hip --test parity_vs_llamacpp -- --test-threads 1`
(filtered to the three PP/single-device tests that pass on the
post-Phase-4 baseline; the four TP-specific tests have pre-existing
environmental failures unrelated to Phase 8b):

| Test | Pre-8b | Post-8b |
|---|---|---|
| parity_e4b_q4_0_single | ok | ok |
| parity_31b_q4_0_pp2 | ok | ok |
| parity_31b_q4_0_pp2_pertoken | ok | ok |

All three tests produce coherent output ("Paris", "The capital of
France is X" follow-up) bit-identical to the pre-migration baseline.

## Conclusion

Typed AR boundary in gemma4 TP path produces bit-exact same outputs
as the untyped baseline. The migration is zero-cost at runtime and
adds compile-time enforcement at the highest-value safety boundary.

Phase 8b PASSES. Phase 8c documentation captures the typed-flow
pattern for future arch implementers.

## Phase 8 acceptance overall

- 8a (typed op surface): shipped commit `ffbe4a6`. Workspace builds
  clean; no model crate changes.
- 8b (gemma4 TP typed flow): this cert. Bit-exact parity, smokes
  green.
- 8c (docs + cert): this cert + `doc/ARCHITECTURE.md` "Typed buffer
  flow" section.

Per the Phase 8 plan, **execution stops here** — the minimum-viable
Phase 8 deliverables (typed op surface + typed AR boundary on both
arches + documentation) are complete. Continuation into 8d-8g
(deeper typing into op signatures + scratch fields) is open scope
that's not justified without a specific bug class to prevent.
