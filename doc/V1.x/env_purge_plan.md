# Env-vars purge & runtime-config refactor — plan

Status: design.
Author: drafted from accumulated cert evidence (2026-05-05 audit + 2026-05-06 N-sweep + 2026-05-06 cumulative test).
Aim: shrink the ~120-var `FLAMBEAU_*` surface to a typed `ServeConfig`
+ dispatch-table layout, deleting gates that produced zero impact and
promoting the rest to a coherent Rust API.

## Why now

V1 verification gate item: "Zero env-var variant gates
(`rg 'env::var' crates/` is empty)". Currently 281 reads across crates,
with `models/qwen3-moe` carrying ~85 unique vars. CLAUDE.md rule #1
("All variant selection lives in `dispatch/<backend>/<arch>.toml`")
is violated by every kernel-variant `FLAMBEAU_*` flag.

## Evidence base

Three sweeps drive every disposition in this plan:

1. **2026-05-05 audit** (`certs/env_impact/<NAME>.md`, 21 vars triaged):
   classifies each gate as CANDIDATE-DELETE / KEEP-DEFAULT /
   CONTEXT-DEPENDENT / SHAPE-DEPENDENT / HALT-{BROKEN,DIVERGENT}.
2. **2026-05-06 N-sweep**
   (`certs/env_impact/<NAME>_N1,2,4,8.md`, 7 vars): re-tests scheduler
   /batching gates at N=[1,2,4,8] on pp2tp2/27B+35B-A3B. Surfaces
   `NO_FAST_PATH` N-flip and the 35B-A3B multi-slot nondeterminism.
3. **2026-05-06 cumulative test**
   (`certs/env_impact/CUMULATIVE_OPTINS_27B_PP2TP2_N1,2,4,8.md`):
   sets 7 small-positive-lean opt-ins simultaneously on 27B-Q4_0.
   Result CUMULATIVE-LOSS — the gates have no additive value.

Profile reference: `bench/profiles/optimized.toml` carries the
best-known production env baseline + the per-key provenance
(`[provenance]`), the `[delete_candidates]` list, and the
`[dispatch_table_candidates]` list.

## Two-phase approach

The refactor splits cleanly:

- **Phase 1 — purge.** Delete env-vars whose gates the cert evidence
  proves null, dead-path, or trivially redundant. No Rust API change;
  just code subtraction + dispatch-table updates where applicable.

- **Phase 2 — promote.** Move surviving runtime-config vars into a
  typed `ServeConfig` with clap-derive CLI flags. Move surviving
  kernel-variant vars into `dispatch/<backend>/<arch>.toml` rows.

Phase 1 ships in two PR families: 1a (unambiguous, no blockers) and
1b (HALT-DIVERGENT-tagged, gated on the 35B-A3B race fix).

## Phase 1a — unambiguous deletes (no blockers)

### 1a-i — kernel-variant fusion gates: 7 deletes

Each gate measured null in the audit and confirmed null in the
2026-05-06 cumulative test (no additive effect when all set together).
Bake the default behaviour, drop the env read.

| Var | Bake to | Cert |
|---|---|---|
| `FLAMBEAU_AR_FUSE_Q8_1` | default off (4 null cells) | `FLAMBEAU_AR_FUSE_Q8_1.md` |
| `FLAMBEAU_Q4_0_GU_T128` | default off | `FLAMBEAU_Q4_0_GU_T128.md` |
| `FLAMBEAU_Q4_0_GU_WARPCOOP` | default off | `FLAMBEAU_Q4_0_GU_WARPCOOP.md` |
| `FLAMBEAU_BATCHED_MMVQ` | default off | memory: `feedback_mmvq_batched_activation_hbm.md` |
| `FLAMBEAU_SSM_OUT_F16_DST` | default off | `FLAMBEAU_SSM_OUT_F16_DST.md` |
| `FLAMBEAU_Q8_0_MMVQ_T128_VDR2` | default off | `FLAMBEAU_Q8_0_MMVQ_T128_VDR2.md` |
| `FLAMBEAU_Q8_0_GU_T128_VDR2` | default off | `FLAMBEAU_Q8_0_GU_T128_VDR2.md` |
| `FLAMBEAU_GDN_QKV_FUSE_Q8_0` | default fused (4 null cells) | `FLAMBEAU_GDN_QKV_FUSE_Q8_0.md` |

Action per var: locate the read site (single `std::env::var` call in
each), delete the conditional, keep only the proven-best branch.

### 1a-ii — dead-path-delete: 4 deletes

Alternate value measurably worse than default; alternate code path is
dead and can be removed entirely with the gate.

| Var | Verdict | Action |
|---|---|---|
| `FLAMBEAU_TP_BATCHED` | `=0` opt-out is 13× prefill regression | Delete the env, delete the non-batched TP forward path |
| `FLAMBEAU_Q8_0_MMVQ_T128` | `=on` is -7% on 27B-Q8_0 | Delete the env, delete the T=128 dispatch fork |
| `FLAMBEAU_MOE_SORTED` | Legacy alias for `MOE_VARIANT=r4`, never wins | Delete the env, callers using `=0` should switch to `MOE_VARIANT=r4` |
| `FLAMBEAU_DECODE_GRAPH` | `=1` crashes on hybrid; null on others | Delete the env, delete the decode-graph capture path |

Note: `MOE_VARIANT` itself stays for now (Phase 2 dispatch-table
candidate); only the legacy `MOE_SORTED` alias goes here.

### 1a-iii — Class D tracing: ~25 deletes

Every `*_DUMP`, `*_PROBE`, `*_TRACE`, `DUMP_RAW_REQ`, `DUMP_PROMPT`,
`DEBUG_TOOL_RAW`, `HOST_PROFILE`, `PROFILE_DECODE`, `GRAPH_TRACE`,
`LOAD_TRACE`, `PARITY_LAYER_DUMP`, `PARITY_TOPK_LOGITS`, `KV_PROJ_DUMP`,
`KV_ROPE_DUMP`, `AR_DUMP`, `STAGE_ENTRY_DUMP`, `BATCHED_DECODE_DUMP`,
`LAYER_STATE_DUMP`, `PP_PROBE`, `TP_PROBE`, `TP_LAYER0_BISECT`,
`TP_LAYER_LIMIT`, `TRACE_BATCH`, `NO_FAST_PATH` (downgraded to debug
after Phase 2 dispatch row absorbs the perf-relevant N=8 case).

Migration: replace bespoke env reads with `tracing` spans on a
descriptive target (e.g. `flambeau::scheduler::batch=trace`). Tensor
dumps gate behind a `dev-trace` cargo feature with a single
`FLAMBEAU_DEBUG_DUMP_DIR` env (replaces ~12 per-site dump dir flags).

User-visible verbosity moves to `RUST_LOG`. The harness stays — it
just sets `RUST_LOG=info` instead of every individual flag.

### 1a-iv — test/bench/example harness inputs: ~30 deletes

`*_GGUF`, `BENCH_*`, `PROFILE_*`, `A_*`, `B_*`, `AB_*`, `RESULT_JSON`,
`MTP_TEST_*`, `LOB_*`, `FUSE_AB_ONLY`, `AB_ONLY`, `PERF_AB_TOKENS`,
`TG_LEN`, `TOPOLOGY_*`, `TEST_*`, `PREFILL_L`, `PREFILL_ONLY`,
`DECODE_ONLY`, `LONG_TEXT`, `PP_LAYERS`, `PP_RANKS`, `TP_RANKS`,
`TP_DEVICES`, `MESH_RANKS`, `U_LANES`, `HYBRID`, `HYBRID_DEVICES`,
`MTP_BASE`, `MTP_HEAD`, `MTP_PROMPT`, `MTP_ACCEPT_STEPS`,
`MTP_KV_ACCUM`, `MTP_PREFILL_PRIME`.

Migration: each lives in `tests/` or `examples/`. Move to
`cfg(test)`-gated helpers or a `BenchConfig` struct read from the
bench harness CLI. Off the runtime API surface entirely.

### Phase 1a totals

| Bucket | Count | Risk |
|---|---:|---|
| 1a-i kernel-variant fusion | 8 | Low — bake default, delete env read |
| 1a-ii dead-path | 4 | Low — alternate path is verified slower or broken |
| 1a-iii tracing | ~25 | Low — `tracing` migration is mechanical |
| 1a-iv test/bench/examples | ~30 | Low — `cfg(test)` move |
| **Phase 1a total** | **~67** | |

## Phase 1b — divergence-blocked deletes (gated on task #16)

These gates have HALT-DIVERGENT certs from the N-sweep on
35B-A3B-Q4_0/pp2tp2. The vars themselves don't *cause* divergence —
they expose a race in the multi-slot scheduler-batched-decode path.
Delete after the underlying race is fixed (task #16).

| Var | Why gated by #16 |
|---|---|
| `FLAMBEAU_GDN_NO_BATCHED` | DIVERGENT at N=2/N=8 on 35B-A3B; need to verify post-fix that the gate is null |
| `FLAMBEAU_FORCE_BATCH_WINDOW` | DIVERGENT at N=4 |
| `FLAMBEAU_BATCH_MAX` | DIVERGENT at N=8/cap=4 |
| `FLAMBEAU_ASYNC_UBATCH` | DIVERGENT at N=2/N=8 |
| `FLAMBEAU_MBATCH` | N=4 sweet-spot regression (-6.1% / -2.5%); kept until divergence is understood (it might be related) |
| `FLAMBEAU_DENSE_GATE_UP` | DIVERGENT at TP2 with `=unfused` (separate bug, possibly related) |
| `FLAMBEAU_ASYNC_GRAPH` | CONTEXT-DEPENDENT loss on 35B-A3B/pp4 |

Action: file separate tickets per gate referencing the master
divergence ticket (#16). Delete after #16 closes AND a re-run of the
N-sweep against the optimized profile shows null on the affected
cells.

## Phase 2 — typed `ServeConfig` (Class C runtime config)

These vars are runtime configuration, not kernel variants. Promote to
typed fields on `flambeau_server::ServeConfig`, surface as
clap-derived CLI flags. Optional `--config <path>.toml` overlay via
`serde + figment` for users who prefer config files.

### 2-i — server config promotion: ~14 vars

| Env var | New CLI flag | Type | Default |
|---|---|---|---|
| `FLAMBEAU_CTX_CAP` (= `FLAMBEAU_MAX_CTX`) | `--ctx-cap` | `Option<usize>` | None (use model's `context_length`) |
| `FLAMBEAU_DEFAULT_SYSTEM` | `--default-system` | `Option<String>` | None |
| `FLAMBEAU_PREFILL_UBATCH` | `--prefill-ubatch` | `usize` | 512 |
| `FLAMBEAU_INFLIGHT_SLOTS` | `--inflight-slots` | `usize` | 1 |
| `FLAMBEAU_MAX_QUEUE_DEPTH` | `--max-queue-depth` | `Option<usize>` | None |
| `FLAMBEAU_EMBEDDING_MAX_TOKENS` | `--embedding-max-tokens` | `Option<usize>` | None |
| `FLAMBEAU_BATCHED_DECODE` | `--batched-decode` | `bool` | true (was opt-in; per memory it always wins) |
| `FLAMBEAU_BATCH_WINDOW_US` | `--batch-window-us` | `Option<u64>` | None |
| `FLAMBEAU_GPU_SAMPLER` | `--gpu-sampler` | `bool` | true (per Sampler-D3/D4 cert) |
| `FLAMBEAU_KV` | `--kv-layout {f16\|q8}` | `KvLayout` enum | `F16` |
| `FLAMBEAU_PREFIX_CACHE` | `--prefix-cache` | `bool` | false |
| `FLAMBEAU_PREFIX_CACHE_MAX_GB` | `--prefix-cache-max-gb` | `f64` | 2.0 |
| `FLAMBEAU_SPEC_MTP` | `--spec-mtp <path>` | `Option<PathBuf>` | None |
| `FLAMBEAU_MTP_BF16` | `--spec-mtp-bf16` | `bool` | false |

`KvLayout` already exists in `crates/models/qwen3-moe/src/session.rs`
as `pub enum KvLayout { F16, Q8 }` with `from_env()`. Phase 2 keeps
the enum, removes the `from_env()` factory, threads a ctor parameter
from `ServeConfig` instead.

### 2-ii — env compatibility shim (one release cycle)

`ServeConfig::from_args_with_env_compat()`:

- Read each new CLI flag.
- For each `FLAMBEAU_*` var that's been migrated, read it as a fallback
  if the corresponding flag is unset, log a deprecation warning, and
  set the field.
- After one release cycle, drop the env-compat path. `rg env::var crates/`
  becomes empty (modulo `build.rs` and `cfg(test)` harnesses).

## Phase 3 — dispatch-table candidates

These gates encode kernel-variant choices that are
shape-/topology-/N-dependent. They belong in
`dispatch/<backend>/<arch>.toml` rows, not the env layer. Each row
gets a shape predicate, a chosen `impl_id`, and a cert reference.

| Var | Why dispatch row, not flag |
|---|---|
| `FLAMBEAU_QKV_FUSED` | SHAPE-DEPENDENT: 9B/pp2tp2 +2.1%, 9B/pp4 -3.6%, others null |
| `FLAMBEAU_KV_F16_DST` | CONTEXT-DEPENDENT: -2.8% on 27B/tp2 when off, null elsewhere |
| `FLAMBEAU_NO_FAST_PATH` | N-DEPENDENT: +4.3% WIN at N=8/27B, null/loss elsewhere |
| `FLAMBEAU_MOE_VARIANT` | 4-way kernel pipeline selector, none wins on tested models; collapse to `tile8`-only and delete the gate, keep alternates as `cfg(unverified)` |

Each migration: a row in `dispatch/hip/gfx906.toml` with the predicate
that captures the winning case, a cert reference under
`certs/env_impact/`, and a corresponding deletion of the env read.

## Phase ordering & PR breakdown

```
                  ┌──────────────────────────┐
                  │ Phase 1a-iii (tracing)   │ — large mechanical PR, no blockers
                  └──────────────────────────┘

                  ┌──────────────────────────┐
                  │ Phase 1a-iv (test vars)  │ — moves into cfg(test), no blockers
                  └──────────────────────────┘

                  ┌──────────────────────────┐
                  │ Phase 1a-i (8 fusion)    │ — bake default, drop env read
                  │ Phase 1a-ii (4 dead path)│   one PR per ~3 vars
                  └──────────────────────────┘

   blocks ──►     ┌──────────────────────────┐
                  │ Task #16 — fix 35B-A3B   │
                  │ multi-slot divergence    │
                  └──────────────────────────┘
                              │ unblocks
                              ▼
                  ┌──────────────────────────┐
                  │ Phase 1b (HALT-DIVERGENT │
                  │  gates, 7 vars)          │
                  └──────────────────────────┘

                  ┌──────────────────────────┐
                  │ Phase 2 — ServeConfig    │ — parallel to 1a; needs clap deps
                  │  + KvLayout enum thread  │   add ServeConfig fields, env shim
                  └──────────────────────────┘
                              │
                              ▼
                  ┌──────────────────────────┐
                  │ Phase 2-ii — drop env    │ — after one release cycle
                  │  compat shim             │
                  └──────────────────────────┘

                  ┌──────────────────────────┐
                  │ Phase 3 — dispatch rows  │ — needs DispatchPolicy plumbing
                  └──────────────────────────┘
```

Recommended PR sequence:

1. **PR-A (Phase 1a-iii):** tracing migration. Adds `tracing-subscriber`
   at server init, replaces ~25 env reads with structured spans.
2. **PR-B (Phase 1a-iv):** move test/bench env reads into harness CLI
   or `cfg(test)` helpers.
3. **PR-C (Phase 1a-i):** bake-and-delete 8 fusion gates. One commit
   per var, single PR.
4. **PR-D (Phase 1a-ii):** dead-path deletes. One PR with deletes of
   the alternate code paths plus their env reads.
5. **PR-E (Phase 2-i):** add `ServeConfig` fields + clap flags + env
   compat shim. Doesn't delete any env yet.
6. **PR-F (Phase 3):** add `DispatchPolicy` plumbing in `core` +
   `OpsRegistry::with_policy`, migrate the 4 dispatch-table candidates.
7. *(blocks on #16)* **PR-G (Phase 1b):** delete the 7 HALT-DIVERGENT
   gates after the race is fixed and re-cert is null.
8. *(after one release cycle)* **PR-H (Phase 2-ii):** remove env-compat
   shim, drop the `FLAMBEAU_*` reads from `ServeConfig`. Roadmap V1
   "Zero env-var variant gates" gate flips green.

PR-A through PR-F can land in any order (no dependencies between them).
PR-G depends on #16. PR-H depends on PR-E.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Deleting a gate that production code paths still depend on (forgotten read site) | Per-deletion: `rg FLAMBEAU_<NAME> crates/` must return empty before merging the PR. |
| Removing a gate without realising its perf claim was platform-specific (e.g. only loses on gfx906, wins on gfx908) | All certs are gfx906-only. PR description explicitly notes this. Re-test on the second arch when it lands (V2 work). |
| `ServeConfig` clap flag collision with existing flags | Audit `crates/cli/src/main.rs` flag names before naming. |
| Env-compat shim drift — users keep setting envs after migration | Log deprecation warning per env hit. Drop the shim after one release. |
| `KvLayout::from_env()` removal breaks downstream callers | Search for every `KvLayout::from_env()` call site, replace with ctor parameter pulled from `ServeConfig`. Tests stay; they'll override the ctor argument directly. |
| Phase 1b deletion before #16 closes | Tasks #16 and Phase-1b PR-G are explicitly linked; CI gate on cert-diff for the affected vars (no removal until cert is non-DIVERGENT). |
| Cumulative-effect surprise on a model/topology not tested | Cumulative test on 27B-Q4_0/pp2tp2 was null. Re-run cumulative on 35B-A3B-Q4_0/pp2tp2 after #16 closes; if a measurable cumulative win surfaces there, revisit Phase 1a-i. |

## Verification

Per V1 verification checklist (`doc/ROADMAP-V1-QWEN36-GFX906.md`):

> - [ ] Zero env-var variant gates (`rg 'env::var' crates/` is empty).

Definition of done for the env-purge:

1. `rg "std::env::var\|FLAMBEAU_" crates/*/src/` returns only:
   - `crates/cli/src/main.rs` reading clap-derived envs (e.g. `RUST_LOG`)
   - `build.rs` files (`ROCM_PATH`, `HIP_OFFLOAD_ARCH`, `HIP_SKIP_BUILD`)
   - Test fixture paths in `crates/*/tests/` and `crates/*/examples/`
   - `bench/` external-toolchain pointers (`ROCPROFV3`)
2. `bench/profiles/optimized.toml` `[env]` section contains only the
   keys in the `ServeConfig` mapping above; rest live in
   `[delete_candidates]` (now empty) and `[dispatch_table_candidates]`
   (folded into `dispatch/<backend>/<arch>.toml`).
3. Smoke matrix re-run against the post-purge build matches pre-purge
   numbers within ±2% (the cumulative-test noise floor).

## References

- `bench/env_impact.toml` — audit spec (24 var entries)
- `bench/profiles/optimized.toml` — production profile + provenance
- `scripts/bench/run_env_impact.py` — audit harness
- `scripts/bench/run_cumulative_optin_test.py` — cumulative falsifier
- `certs/env_impact/*.md` — per-var cert evidence (29 certs as of 2026-05-06)
- Memory:
  - `memory/project_env_impact_audit_2026_05_05.md`
  - `memory/project_env_impact_N_sweep_2026_05_06.md`
- CLAUDE.md rule #1 (no env-flag variant gates)
- `doc/ROADMAP-V1-QWEN36-GFX906.md` — V1 verification gate
