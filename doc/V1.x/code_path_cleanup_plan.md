# Code-Path Cleanup Plan — Optimal-Path-Driven, Not Bench-Driven

**Status:** draft. Successor to `env_purge_plan.md` (which optimised
the wrong loop: measured impact, then deleted; missed slot-count and
topology interactions).

## Goal

Shrink the project surface to **only** the code paths that the V1
target models traverse on their best-known production config.
Everything else — kernel variants nobody picks, env vars gating dead
branches, fallback paths that never fire — is candidate for deletion.

## Scope (locked by user 2026-05-07)

**Approach:** code-paths first, env vars follow. We do not start
from a bench cert. We start from the code: walk each target model's
forward path top-down on each topology, and **only** keep what those
walks visit.

**Anchor models** (the "target paths"):
- **Qwen3.5-9B-Q4_1** (dense, qwen35dense arch)
- **Qwen3.6-27B-Q4_0** + **Qwen3.6-35B-A3B-Q4_0** (qwen36moe arch:
  hybrid GDN + full-attn + MoE + shared expert)
- **Qwen3-Coder-Next-80B-Q4_0** (qwen3next arch: same hybrid forward
  path as qwen36moe — they share `crates/models/qwen3-moe/forward/`)

**Anchor topologies:** `pp4`, `tp2`, `pp2tp2`. (gfx906 / MI50 only.
gfx1031 portability canary remains a build target but is not a
deletion gate.)

**Anchor config:** `bench/profiles/optimized.toml`:
- `FLAMBEAU_BATCHED_DECODE=1`
- `FLAMBEAU_GPU_SAMPLER=1`
- `FLAMBEAU_INFLIGHT_SLOTS=4` (or 8 for max single-user perf)
- `FLAMBEAU_PREFILL_UBATCH=512`
- Race-safe fast-path lock (committed at `d7c3cc6`)

**Out of scope (unchanged from CLAUDE.md):**
- Mistral / Gemma-4 (listed but not anchor-bench targets)
- Qwen3-Coder-30B (dropped V1.x — qwen3moe arch underperforms)
- Metal, training, LoRA, vision/audio, speculative decoding, ONNX

## Why this differs from the last attempt

The env-purge plan's measure-then-delete loop produced two failure modes:

1. **Slot-count blindness.** Sweep ran each var at a single
   `INFLIGHT_SLOTS` setting (mostly 1). When we discovered that
   `INFLIGHT_SLOTS=8` was the dominant N=1 perf lever (2.58×), the
   prior null verdicts had to be re-interpreted.
2. **Path-collapse over-reach.** Reading "scheduler is correct, fast-path
   is racy" as "delete fast-path" cost 2.5–3× on N=1 across all
   topologies. The fast-path is the perf path; closing the race needs
   a 5-line lock, not a 5-PR collapse.

This plan inverts the loop: **understand the path, then delete**.
A bench is only ever the *last* check that the deletion didn't break
something we missed in code reading.

## Target forward-path map (deliverable of Phase 0)

For each (anchor model, topology) pair, we want a one-page map:

```
qwen36-35b-a3b-q4_0 / pp2tp2 / N=1 (fast-path):
  routes::run_completion_blocking_streaming
    └─ prefill_logits (TP-Hybrid, with prefill_serialiser, pooled scratch)
        ├─ models::qwen3_moe::forward::forward_prefill_hybrid_logits
        │   ├─ rmsnorm + qkv_proj (fused: FLAMBEAU_QKV_FUSED)
        │   ├─ gdn_step / full_attn (alternate per layer)
        │   └─ moe (MoE-MMVQ Q4_0 indexed)
        └─ topk_softmax_f32 (GPU sampler)
    └─ decode loop:
        for step:
          decode_logits (direct, no scheduler at N=1 streaming)
          gpu_sampler::run_gpu_topk
          push_and_emit
```

```
qwen36-35b-a3b-q4_0 / pp2tp2 / N=4 (scheduler):
  routes::decode_via_scheduler_into
    └─ batched_pending leader → forward_decode_batched_hybrid
        ├─ ... (same kernels, batched shape)
```

The map names every kernel variant chosen by dispatch + every env-var
gate that influenced the choice. Anything not on this map for any
anchor cell is a cleanup candidate.

## Phases

### Phase 0 — Inventory (read-only, no deletions)

**0a. Forward-path maps.** Walk each anchor model + topology + (N=1
fast-path / N>1 scheduler) and document the kernel-and-op call tree.
Output: `doc/V1.x/optimal_paths.md`, one section per (model, topology,
N-class).

**0b. Env-var read-site grep.** `rg 'env::var\("FLAMBEAU_'` across all
crates. For each hit, record file:line + the conditional it gates.
Output: `doc/V1.x/env_var_inventory.md` (tabular).

**0c. Kernel-impl inventory.** `rg '#\[cfg\(unverified\)\]'` plus
manual pass on each `crates/ops/*/src/hip/*.rs` to list every kernel
implementation. Annotate which dispatch row(s) reference each one.

**Deliverable:** three docs above. **No code change.** User reviews
before Phase 1.

### Phase 1 — Triage (label-only, no deletions)

For each env var (from 0b) and each kernel impl (from 0c), produce a
verdict by cross-referencing with 0a:

- **on-path** — referenced from at least one anchor's optimal path.
  Keep. (Examples expected: `BATCHED_DECODE`, `GPU_SAMPLER`,
  `INFLIGHT_SLOTS`, `PREFILL_UBATCH`, `QKV_FUSED`, `KV_F16_DST`.)
- **on-path, dispatch-table candidate** — referenced from anchor path
  but the alternate value wins on ≥1 cell. Migrate to dispatch table
  per CLAUDE.md rule #1, then delete the env read.
- **off-path** — only referenced from non-anchor paths or a branch no
  anchor selects. Delete the var **and** its dead branch.
- **dead-flag** — gates a code path that's never selected by any
  anchor and the alternate is identical / dead. Delete.
- **broken** — known to crash (e.g. `DECODE_GRAPH` on hybrid).
  Two sub-options: fix or delete the broken alternate. Default delete.

**Deliverable:** `doc/V1.x/cleanup_triage.md` with one row per env
var + per kernel impl. User reviews before any deletion.

### Phase 2 — Delete in slices with verification

Each slice = one PR-shaped commit, formatted as:
- Slice S1: delete N off-path env vars + their dead branches.
- Slice S2: delete M unused kernel impls (`#[cfg(unverified)]` first).
- Slice S3: collapse a redundant op trait or model glue layer.
- ...

**Verification gate** between every slice (mandatory):
1. `cargo build --release --features hip_serve` clean.
2. `repro_35b_divergence.py` → NO-DIVERGENCE.
3. Smoke bench: `BASELINE_27B_PP2TP2` + `BASELINE_35B_PP2TP2` at
   N=1, N=4 — must stay within ±5% of baseline (this commit `d7c3cc6`).
4. Greedy-seed=0 chat test on 9B + 35B-A3B emits coherent output.

A regression on any of these stops the slice and reverts. The lesson
from the last attempt is that *some* of these slices will surprise us.

### Phase 3 — Re-evaluate remaining env vars (post-deletion)

After Phase 2, re-run a small env_impact-style sweep on the surviving
vars only. Confirm verdicts hold under the post-deletion baseline.
Vars whose verdict flips are reverted to keep + a follow-up note.

### Phase 4 — Optimise the optimal path

Only after Phase 3. With the surface shrunk, profile the remaining
hot path. Each surviving env var either:
- Becomes a `clap` arg in `flambeau serve` (server config),
- Becomes a dispatch-table row (kernel selection),
- Or stays as a debug-only var behind a `cfg(dev_trace)` feature.

By the end, **`optimized.toml` should have ≤ 5 `[env]` keys** and the
codebase should have ≤ 10 live `FLAMBEAU_*` reads. Today there are
~50.

## Lessons re-encoded as guardrails

1. **No deletion without code-reading first.** "Bench says null"
   is not sufficient — we measured null at slots=1 and missed the 2.58×
   slots=8 lever. Read the code paths, then bench-confirm.
2. **Slot count is part of the workload, not a free variable.** Every
   bench cell records the active `INFLIGHT_SLOTS`. Triage decisions
   that don't say "at slots=N" are unsafe.
3. **Fast-path stays.** It's the perf path at N=1; the race fix is
   the lock at `routes.rs:771` (`d7c3cc6`), not its deletion.
4. **Two-stage verification.** Build clean + race-clean + perf-clean
   on the smoke bench before moving to the next slice. Any single
   miss aborts and reverts.
5. **One reversible commit per slice.** The path-collapse PRs spanned
   5 commits before any verification. That made the revert harder than
   it needed to be.

## Open question for user before Phase 0

Phase 0 inventory is the largest read-only step (a few hundred lines
of grep + reading). Two ways to run it:

- **(a) Single full pass** producing all three docs (0a/0b/0c) at
  once. ~30–60 min of agent time. Pro: one review; Con: large doc to
  digest.
- **(b) Three smaller passes**, one per doc, each reviewed before the
  next. ~15–25 min each. Pro: catch scope drift early; Con: more
  back-and-forth.

The actual deletion work in Phase 2 is incremental either way.

