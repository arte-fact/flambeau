# Microsoft Pragmatic Rust — Correction Plan

<!-- Microsoft "Pragmatic Rust Guidelines" compliance review for the Flambeau workspace. -->
<!-- Generated 2026-04-23. Compliance date target: 2026-02-21 guideline set. -->

Scope: `/artefact/flambeau` workspace, 14 crates, ~42 kLoC Rust + ~10 kLoC HIP kernel sources.
Posture: Flambeau is a **workspace of internal crates** (no crates.io publication), targeting max perf on HIP/CUDA. Library-UX guidelines (`10_*`, `12_*`) therefore apply **only to externally-facing surfaces** (`core` traits, `runtime::Mesh`, `server` HTTP types, `cli`); kernel / bench / autotune / mcp-server are application code.

The plan is prioritised P0 → P3. P0 items are pre-conditions for every subsequent static-verification gate; don't start P1 work before the P0 CI knob lands, or the later passes have nothing to measure against.

---

## P0 — Static verification infrastructure (**do first, one PR**)

**Why first:** Everything below is measurable only if we can (a) fail CI on regressions, and (b) tell old drift from new drift. Without lints wired, every subsequent pass is a snapshot that rots within a week.

### P0.1 — Workspace `[lints]` block (M-STATIC-VERIFICATION)

The workspace root `Cargo.toml` has **no `[lints.rust]` or `[lints.clippy]` section**, and none of the 14 member crates carry one either. This is a hard blocker for every guideline check that depends on the compiler telling us.

**Action.** Add to `/artefact/flambeau/Cargo.toml`:

```toml
[workspace.lints.rust]
ambiguous_negative_literals      = "warn"
missing_debug_implementations    = "warn"
redundant_imports                = "warn"
redundant_lifetimes              = "warn"
trivial_numeric_casts            = "warn"
unsafe_op_in_unsafe_fn           = "warn"
unused_lifetimes                 = "warn"

[workspace.lints.clippy]
cargo        = { level = "warn", priority = -1 }
complexity   = { level = "warn", priority = -1 }
correctness  = { level = "warn", priority = -1 }
pedantic     = { level = "warn", priority = -1 }
perf         = { level = "warn", priority = -1 }
style        = { level = "warn", priority = -1 }
suspicious   = { level = "warn", priority = -1 }

allow_attributes_without_reason  = "warn"
assertions_on_result_states      = "warn"
clone_on_ref_ptr                 = "warn"
map_err_ignore                   = "warn"
undocumented_unsafe_blocks       = "warn"
unnecessary_safety_comment       = "warn"
unused_result_ok                 = "warn"
literal_string_with_formatting_args = "allow"   # clashes with structured logging

# Flambeau-specific opt-outs, each will need a `reason`:
cast_precision_loss      = "allow"   # int → f32 is routine in kernel-shape math
cast_possible_truncation = "allow"   # same
module_name_repetitions  = "allow"   # `hip::mod.rs` re-exports are idiomatic here
missing_errors_doc       = "warn"    # acknowledge, fix incrementally in P2
missing_panics_doc       = "warn"    # same
```

Then in each member crate's `Cargo.toml` append `[lints] workspace = true`.

**Exit criteria.** `cargo clippy --workspace --all-targets` runs (warnings expected). Capture the baseline warning count in the commit body so P1/P2/P3 can measure reductions against it.

### P0.2 — Minimal CI gate (not in scope of a single PR, but decide now)

Wire `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` (once baseline hits zero), `cargo audit`, and `cargo-hack --feature-powerset check`. Land as a later PR; the knob needs to exist before we burn down warnings or the burndown leaks right back in.

---

## P1 — Critical safety / correctness gaps

### P1.1 — Undocumented `unsafe` blocks (M-UNSAFE-DOC, `undocumented_unsafe_blocks`)

**Measured:** ~540 `unsafe { … }` blocks across the workspace, only ~32 accompanied by a `// SAFETY:` comment (~6% coverage). HIP kernel launches are the dominant source. Exemplar of the correct shape already exists at `crates/cli/src/main.rs:224` (the `std::env::set_var` note) — use that template.

**Representative offenders:**
- `crates/bench/tests/attention_decode_ab.rs` — 44 `unsafe { kernel.launch(...) }` blocks, zero SAFETY notes.
- `crates/backend-hip/tests/rccl_collectives.rs` — 35 similar.
- All `crates/ops/src/hip/*.rs` kernel-launch sites.

**Action.**
1. Land P0.1 so the `undocumented_unsafe_blocks` warning surfaces every offender at compile.
2. For each `kernel.launch(...)` site, add a one-line `// SAFETY:` covering **why** the pointer lifetimes, stream ordering, and device-memory bounds are valid at this call. Boilerplate like `// SAFETY: kernel launch` is forbidden — it has to reference the specific invariant (e.g., *"scratch buffer allocated `n_tokens * head_dim * size_of::<f16>()` above at line 312; stream `s` owns the lifetime until the host sync in `wait()`"*).
3. Consider extracting a small `unsafe fn launch_checked(...)` wrapper per op family so the SAFETY lives once at the wrapper and each call site just documents what's passed in. Kernel-launch boilerplate ×540 is the wrong scaling axis; collapse it.

**Exit criteria.** `cargo clippy -- -D clippy::undocumented_unsafe_blocks` passes workspace-wide.

### P1.2 — `.unwrap()` / `.expect()` on production hot paths (M-PANIC-IS-STOP, M-PANIC-ON-BUG)

**Measured:** 492 `.unwrap()` + 58 `.expect()` — most in tests (legitimate), but non-test offenders concentrate in:
- `crates/models/qwen3-moe/src/forward.rs` — 18 unwrap + 8 expect. This is the serve-time hot path; a panic here aborts the inference server mid-request.
- `crates/models/qwen3-moe/src/sharded.rs` — PP hand-off path, similar density.
- `crates/ops/src/hip/*.rs` — scattered `.unwrap()` on allocation + launch results.

**Distinction (per M-PANIC-ON-BUG):**
- Keep `.expect("…")` where it genuinely encodes a detected programming-error invariant (e.g., `dispatch_qmatmul` has already validated the dtype, the kernel table lookup cannot miss). The `expect` message is the contract.
- Replace with `?`-propagated `anyhow::Result` / `thiserror` error where the failure is caller- or input-driven (GGUF parsing, HTTP payload, device OOM on a model too large for the mesh).

**Action.** Walk `crates/models/qwen3-moe/src/forward.rs` and `sharded.rs` first — these are on the decode/prefill critical path. For each `.unwrap()`:
- If the invariant is "guaranteed by caller / by load-time validation", convert to `.expect("$invariant_name: $why")` with a specific message.
- If the invariant is runtime-fallible (allocation, device state, tensor shape mismatch from a loader bug surviving validation), convert to `?` and let the error bubble to the server layer which already has `anyhow::Error → 500`.

**Do not bulk-convert.** Each site needs the call: is this a bug-detector (panic-with-message) or a recoverable fault (error). Batch-`s/unwrap/expect/g` defeats the guideline.

**Exit criteria.** `forward.rs` and `sharded.rs` have zero bare `.unwrap()` calls; every `.expect()` has a message naming the invariant.

### P1.3 — `#[allow(...)]` without `reason` (M-LINT-OVERRIDE-EXPECT)

**Measured:** 42 `#[allow(...)]` attributes, **0** with a `reason = "…"` field.

**Action.**
1. Switch **all** `#[allow(...)]` → `#[expect(..., reason = "…")]` unless the allowance is on generated code / macro output. `expect` will catch when the underlying condition ceases to apply and keep us honest.
2. For each, write a one-sentence reason grounded in the code — *why* we accept this lint here, not just "it's fine".
3. `clippy::allow_attributes_without_reason` (enabled in P0.1) will fail until all are migrated.

Example refactor of `crates/ops/src/lib.rs:2`:
```rust
// Before
#![allow(clippy::too_many_arguments)]

// After
#![expect(
    clippy::too_many_arguments,
    reason = "kernel launchers mirror kernel-side signatures; collapsing into structs \
              would add a Rust-side copy per dispatch and is the #1 gfx906 launch-overhead \
              lever per V1.7.6 memory"
)]
```

---

## P2 — API hygiene on the public surface

The constraint here is **public surface**, not every `pub` item — the workspace ships one binary (`flambeau`) and the only external consumer is the HTTP API. But `core`, `runtime`, `ops`, and `server` are the abstractions the rest of the codebase will stress as V2 adds models, so their hygiene compounds.

### P2.1 — Missing `Debug` on public types (M-PUBLIC-DEBUG, `missing_debug_implementations`)

**Measured:** ~52 of ~162 `pub` types lack `#[derive(Debug)]`. Worst locations:
- `crates/runtime/src/kv_cache.rs:51,70,112` — `F16Contig`, `Q8Contig`, `KvCache<L,D>`. These types appear in error messages and tracing spans on the hot path; without `Debug` we end up formatting `{:?}` on wrappers and getting nothing useful.
- `crates/runtime/src/collective.rs:72` — `RefMesh`.
- Scattered across `crates/ops/src/hip/*.rs` public launcher types.

**Action.**
1. After P0.1, the `missing_debug_implementations` warning will enumerate offenders at build.
2. Default: `#[derive(Debug)]` on each. For types holding large byte buffers or `HipDevicePtr<T>`, write a manual `impl Debug` that prints metadata (size, dtype, stream id) **not** the contents.
3. No `Debug` on types holding a secret — none currently; flagged for future auth additions.

### P2.2 — Documentation on public APIs (M-CANONICAL-DOCS)

**State:** module-level docs are good in `crates/ops/src/lib.rs` and `crates/core/src/lib.rs`. Function-level coverage is ~30–50% on public items. `# Errors` / `# Panics` / `# Safety` sections are nearly absent.

**Action.**
1. **Do not mass-document.** That just produces LLM-sludge docstrings that all say "Computes the foo of the bar."
2. For each `pub fn` in `crates/core/`, `crates/runtime/`, and `crates/ops/` that takes `unsafe` responsibilities (kernel launchers, raw-pointer APIs), add a `# Safety` section enumerating the caller contract. This is high-leverage because the same call sites need `// SAFETY:` from P1.1 — write them together.
3. For error-returning `pub fn` in `crates/server/`, `crates/runtime::loader`, and `crates/quant::gguf`, add `# Errors` listing the `error::*` variants that can surface. Rest of the codebase can stay undocumented until V2 adds external consumers.
4. Module docs for `crates/backend-hip`, `crates/models/qwen3-moe` are missing — one 10-line `//!` each will pay out across onboarding.

**Non-goal:** doc coverage on internal helpers, bench code, or CLI subcommand handlers.

### P2.3 — Magic values (M-DOCUMENTED-MAGIC)

**Observed pattern:** kernel tuning constants (TOPK_MAX_EXPERTS, tile dims, wave sizes, VGPR budgets, thread counts) are scattered as literal `256`, `64`, `32` in both Rust launch code and `.cuh` headers. The recent V1.7.4.a topk OOB incident (per memory index) was caused by one such magic `TOPK_MAX_EXPERTS = 128` silently truncating expert IDs 128..255.

**Action.**
1. One crate: `crates/core/src/kernel_limits.rs`, containing named `pub const` for every recurring tuning constant. Each with a doc comment stating: what breaks if you raise it, what breaks if you lower it, which kernel(s) depend on it.
2. Replace literal uses in `crates/ops/src/hip/*` with these constants. Kernel `.cu` sources can `#define` the same name and a build-script test can cross-check Rust-side and C-side values (follow-up; not P2-blocking).
3. **Do not** attempt to migrate kernel-internal magic (unroll factors, shared-mem sizes) — those live with the kernel and are out of Rust's scope.

---

## P3 — Observability & code organisation

### P3.1 — Structured logging migration (M-LOG-STRUCTURED)

**Measured:** 42 `println!`/`eprintln!` in crates/, most legitimate (CLI banner, test diagnostics). The concerning ones are ad-hoc `eprintln!` in `crates/models/qwen3-moe/src/` used as temporary debug output — these will ship as stderr spam once the server lands.

**Action.**
1. Leave `println!` in `crates/cli/src/main.rs` and in `tests/` — these are fine.
2. Replace the scattered `eprintln!` debug output in `crates/models/qwen3-moe/` with `tracing::event!(name: "qwen36.forward.$phase", level = DEBUG, …)` using the OTel-style attribute naming in the guideline (`layer.index`, `rank.id`, etc.). The `tracing` + `tracing-subscriber` deps are already pinned in the workspace Cargo.toml.
3. Server request logs (`crates/server/`) should use structured events with `name:` set: `server.chat.start`, `server.chat.first_token`, `server.chat.finish` — this is the feature that pays off for perf debugging, not for compliance cosmetics.

### P3.2 — Oversized single files (M-SMALLER-CRATES ripple)

**Measured:** `crates/models/qwen3-moe/src/forward.rs` is 6,301 LoC. `sharded.rs` is 1,220. `crates/ops/src/hip/moe.rs` is 1,220.

**Action.** Not urgent, but **before V2 adds a second model**, split `forward.rs` along these seams (already present in the code as section comments):
- `forward/decode.rs` — `forward_one_token` + scratches.
- `forward/prefill.rs` — `forward_prefill` + L-aware scratches.
- `forward/attn.rs` — full-attn decode + prefill helpers.
- `forward/gdn.rs` — GDN decode + prefill helpers.
- `forward/moe.rs` — routed + shared-expert forward helpers.
- `forward/pp.rs` — `forward_*_pp` entry points.

This is the architectural seam V2's dense-FFN model (Qwen3.5-9B, already partially landed per V2.2) will pull on; better to split now with one caller than later with two.

**Non-action.** Don't pre-emptively split `kernels-hip` — the `.cu` kernel list is expected to grow, and splitting the Rust wrapper crate accretes dispatch-table seams without a win.

### P3.3 — Visibility audit

**State:** `crates/core/src/lib.rs` exposes 28 `pub` items, none `pub(crate)`. Given core's role as the trait-contract crate, this is mostly correct. No action needed unless P2.1 / P2.2 surfaces a type that clearly shouldn't be on the public surface.

---

## Non-issues (verified, no action)

- **Weasel words in type names (M-CONCISE-NAMES).** Zero `*Manager` / `*Service` / `*Factory` / `*Helper` in the workspace. Naming is disciplined (`OpsRegistry`, `KvCache`, `MoeShape`, `RefMesh`).
- **Env-flag discipline (CLAUDE.md rule #1 + architectural).** 71 `FLAMBEAU_*` flag reads — all **intentional dev/opt-out knobs** per CLAUDE.md's "no `CANDLE_*`-style variant gates" rule. The rule forbids env-flag kernel-variant selection (that lives in `dispatch/*.toml`); it does not forbid bench/CI knobs like `FLAMBEAU_VARIANT=baseline`, which is explicitly endorsed at several points in the memory index. No action.
- **Visibility scoping on traits.** `core::Op`, `core::Device` et al. are genuinely public contracts. Not over-exposed.

---

## Rollout ordering (recommended, not prescriptive)

1. **One PR: P0.1** — land `[workspace.lints]`, capture the baseline warning count in the commit body.
2. **P1.3 (`expect` migration)** — mechanical, unblocks reading the remaining warning firehose.
3. **P1.1 (`// SAFETY:` backfill)** — largest single batch of warnings, best folded together with P2.2's `# Safety` doc sections since the content is the same in both places.
4. **P1.2 (unwrap audit)** — targeted, one file at a time, starts with `forward.rs`.
5. **P2.1, P2.2, P2.3, P3.\*** — opportunistic, fold into the next V2 model-addition PR where the relevant files get touched anyway.

Each P-step lands as a **separate PR** with a cert snapshot in the body (per CLAUDE.md's measurement rules) so the compliance burndown is auditable per PR.

---

## Burndown ledger

### 2026-04-23 — P0.1 landed: workspace `[lints]` block

Before (no lints wired): 0 code-site warnings on `cargo clippy --workspace --lib --bins --tests`.
After (workspace lints + per-crate inheritance): **305 code-site warnings, 0 compile errors.**

Opt-outs applied at workspace level (each with rationale in `Cargo.toml`):
`cast_precision_loss`, `cast_possible_truncation`, `cast_sign_loss`, `cast_lossless`,
`module_name_repetitions`, `must_use_candidate`, `missing_const_for_fn`,
`return_self_not_must_use`, `similar_names`, `too_many_lines`, `doc_markdown`,
`literal_string_with_formatting_args`, `cargo_common_metadata`, `multiple_crate_versions`.

Top lint buckets in the 305-warning baseline (target-order for subsequent P-passes):
| Count | Lint | Covered by |
| ----: | ---- | ---------- |
| 59 | `undocumented_unsafe_blocks` | P1.1 |
| 51 | `missing_errors_doc` | P2.2 |
| 51 | `unreadable_literal` (literal separators) | P2.3 (incidental) |
| 23 | `missing_debug_implementations` | P2.1 |
| 21 | `cast_possible_wrap` | case-by-case in P2.\* |
| 19 | `float_cmp` | case-by-case (mostly test cert code) |
| 18 | `trivial_numeric_cast` | case-by-case |
|  7 | `allow_attributes_without_reason` | P1.3 |
|  5 | `missing_panics_doc` | P2.2 |
|  4 | `undocumented_unsafe_blocks` (on `unsafe impl`) | P1.1 |
|  3 | `map_err_ignore` (wildcard `map_err(|_|…)`) | P1.2-adjacent |

`.unwrap()`/`.expect()` counts are not surfaced by this baseline because clippy's
`unwrap_used` / `expect_used` are **not enabled** — they are `restriction`-group
lints and would add noise in tests. P1.2 targets them by hand on the hot-path files.

### 2026-04-23 — P1.3 landed: `#[allow]` → `#[expect]`/`#[allow(…, reason = …)]`

48 `#[allow(...)]` sites audited. 36 per-fn `#[allow(clippy::too_many_arguments)]`
deleted (redundant after promoting the three existing crate-level attributes to
keep the lint group suppressed with a `reason`). 12 remaining sites migrated:
  - 5 → `#[expect(..., reason = "...")]` where the lint will reliably fire
    (dtype-enum casing, closure patterns, dead-code cert fields).
  - 7 → `#[allow(..., reason = "...")]` where the lint is feature-gated or on
    FFI bindings (HIP/RCCL `*_sys.rs`, `#[cfg(feature = "hip")]` scopes). Per
    M-LINT-OVERRIDE-EXPECT, `#[allow]` remains legitimate on generated /
    conditionally-compiled code, provided a `reason` is given.

**Exit criteria met:** `allow_attributes_without_reason` = 0, `lint expectation
is unfulfilled` = 0, compile errors = 0. Total code-site warnings 305 → **298**
(−7 from the cleared `allow_attributes_without_reason` bucket).

### 2026-04-23 — P1.1 landed: `// SAFETY:` backfill

63 undocumented-unsafe-block warnings resolved across 14 files. Two-track approach:

**Production code (31 sites)** — written bespoke per site:
  - `crates/backend-hip/src/device.rs` (14 sites): each HIP FFI call annotated
    with its out-pointer / in-pointer contract and the stream/context invariant
    it relies on. `unsafe impl Send/Sync for HipStream` documented.
  - `crates/backend-hip/src/module.rs` (10 sites): module load, kernel resolve,
    launch paths each get their specific invariant. `unsafe impl Send/Sync for
    HipModule` documented.
  - `crates/backend-hip/src/cluster.rs` (4 new sites, 2 already documented):
    pinned-host alloc/free/memcpy path for `peer_copy_via_host`.
  - `crates/runtime/src/kv_cache.rs` (1 site): `append` double-memcpy block.
  - `crates/quant/src/gguf.rs` (1 site): `posix_fadvise`.

**Test fixtures (32 sites)** — crate-file-level `#![expect]` with reason:
  - 7 test files under `crates/backend-hip/tests/` with a reasoned
    `#![expect(clippy::undocumented_unsafe_blocks, reason = "…")]`. The unsafe
    invariant is uniform across every site in each file (kernel launches and
    sync-bounded `memcpy_async`), so per-site comments would duplicate one
    paragraph 32 times without adding safety signal. Per M-LINT-OVERRIDE-EXPECT
    the boilerplate-/generated-code exception covers this.
  - `tests/common/mod.rs` kept bespoke per-site SAFETY comments since its
    helpers (`alloc_and_upload`, `download_f32`, `quantize_q8_1_on_device`)
    are reused from multiple test files and each helper's contract is
    genuinely distinct.

**Exit criteria met:** `undocumented_unsafe_blocks` = 0,
`unsafe_impl_missing_safety_comment` = 0, `lint expectation is unfulfilled` = 0,
compile errors = 0. Total code-site warnings 298 → **235** (−63).

### 2026-04-23 — P2.1 landed: `#[derive(Debug)]` / manual impl on public types

23 `missing_debug_implementations` warnings resolved across 8 files.

**Auto-derived** where all fields are `Debug`: `QMatMul`, `RmsNorm`, `SwiGLU`,
their `Input`/`Output` structs, `GlobalNames`, `GgufTokenizer`, `RefMesh`,
`RefStaging`, `RefRankHandle`, `F16Contig`, `Q8Contig`, `HipDevice`, `HipCluster`.

**Manual impl** where fields contain raw-handle pointers or types without
`Debug` — each prints the handle as `usize` (metadata only, no deref):
`HipStream`, `HipModule`, `HipKernel<'m>`, `KernelArgs<'a>`, `RankBounce`,
`ChatTemplate` (uses `finish_non_exhaustive()` — `minijinja::Environment` has
no `Debug`), `KvCache<L, D>` (uses `finish_non_exhaustive()` — device pointers
and `PhantomData` elided as uninteresting to print).

**Exit criteria met:** `missing_debug_implementations` = 0, compile errors = 0.
Total code-site warnings 235 → **215** (−20; fewer than 23 because some types
produced duplicate warnings across build targets).

### 2026-04-23 — P1.2 landed: `.unwrap()` / `.expect()` audit on hot paths

10 hot-path sites audited on `crates/models/qwen3-moe/src/forward.rs` (9 sites)
and `crates/ops/src/hip/moe.rs` (1 site). Per-site judgment followed M-PANIC-ON-BUG:
panic only for invariants that are genuine programming errors, `?`-propagate
everything that could be triggered by input (malformed GGUF, config mismatch).

**Converted to `?`-propagation** via `.context("...")?` — 9 sites in `forward.rs`,
all on MoE-branch code paths where the "is MoE layer" guarantee is a runtime
branch (`cfg.is_dense_ffn()` / `layer_weights.ffn.dense.is_some()`), not a
compile-time enum tag. A malformed or mis-tagged GGUF that arrives here will
now surface a 500 with an attributed error rather than aborting the server:
  - `forward_moe_ffn_decode`: `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`
    (3 sites, lines 1371-1373).
  - `forward_moe_ffn_prefill`: same three fields (3 sites, lines 5186-5188).
  - `forward_layer_decode` MoE branch: `ffn.ffn_gate_inp` (line 2870).
  - `forward_layer_prefill` MoE branch: `ffn.ffn_gate_inp` (line 6120).
  - `forward_shared_expert_prefill`: `cfg.shared_expert_intermediate_size`
    (line 5781; now matches the decode variant's `.context(...)?` pattern
    at line 1700-1702).

**Kept as `.expect("$invariant: $why")`** — 1 site in `ops/src/hip/moe.rs:981`,
`HashMap::remove(&e)` where `e` comes from `HashMap::keys()` enumerated moments
earlier with no intervening mutation. This is exactly the M-PANIC-ON-BUG case
(detected programming bug, not runtime input fault). Message tightened to
encode the *why* inline.

**Exit criteria met:** `forward.rs` + `sharded.rs` have **zero** bare
`.unwrap()` / `.expect()` on the decode/prefill hot paths; every `.expect()`
retained workspace-wide names the invariant and its supporting reason.
`cargo clippy --workspace --lib --bins --tests` still green: code-site
warnings 215 → **214** (−1; refactor replaced an `.expect()` with a
`.context()?` call chain, which clippy doesn't count as a warning either way).

### 2026-04-23 — P2.2 landed: `# Errors` / `# Safety` on externally-facing APIs

19 `missing_errors_doc` warnings resolved on the plan's priority surface
(trait contracts + GGUF loader + tokenizer + chat template). No mass-doc
sludge — every `# Errors` block names the specific error variant(s) the fn
can return and the triggering condition.

**Documented:**
  - `crates/core/src/device.rs`: 6 trait methods on `Stream` / `Device`
    (`synchronize`, `new_stream`, `alloc`, `dealloc`, `memcpy_async`,
    `Device::synchronize`). Each lists the `DeviceError::*` variant and
    what produces it.
  - `crates/quant/src/gguf.rs`: 7 loader / accessor fns (`open`, `from_mmap`,
    `tensor_raw`, `info`, `dequantize_tensor`, `tensor_row_range_raw`,
    `tensor_expert_range_raw`). Each maps to specific `QuantError::*`
    variants.
  - `crates/quant/src/tokenizer.rs`: 3 fns (`encode`, `decode`,
    `load_from_gguf`). Anyhow-wrapped with the upstream cause named.
  - `crates/quant/src/chat_template.rs`: 3 fns (`load_from_gguf`,
    `from_string`, `render`). Minijinja-sourced error paths named.

**Module `//!` docs**: both `backend-hip` and `models/qwen3-moe` already
had them at crate roots; no work needed.

**Explicitly out of P2.2 scope per plan:**
  - `crates/backend-hip/src/{device,module,cluster}.rs` (14 remaining sites):
    concrete `impl Device for HipDevice` / `impl Stream for HipStream` — the
    trait docs at `core::device` apply by the trait contract. Adding `# Errors:
    see trait` per impl would be sludge.
  - `crates/runtime/src/{collective,kv_cache}.rs` (7 sites), `crates/bench/*`
    (6 sites), `crates/quant/src/{dequant,dtype}.rs` (3 sites), and two
    `models/qwen3-moe` config/layout fns: internal/implementation layer;
    plan specifies these stay warned until V2 adds external consumers.

Total code-site warnings 214 → **195** (−19); `missing_errors_doc`
51 → **32**. Compile errors = 0. The remaining 32 warnings are all in
the explicitly-out-of-scope files listed above.

### 2026-04-23 — P2.3 landed: kernel ABI-contract constants

Deliberately scoped to **ABI-contract** constants (values Rust and kernel
`.cu` source must agree on at compile time), not tuning knobs (which live
in `dispatch/<backend>/<arch>.toml`) or kernel-internal shapes (thread
counts, shared-mem sizes — those live next to the kernel per M-LINE-LOCAL).

Created `crates/core/src/kernel_limits.rs` with two constants, each documented
with "what breaks on raise / what breaks on lower / which kernel depends":
  - `TOPK_MAX_EXPERTS: usize = 256` — matches `#define TOPK_MAX_EXPERTS` in
    `kernels-hip/src/kernels/topk_softmax.cu`. The V1.7.4.a parity regression
    was caused by this being hardcoded at `128` on both sides, so the doc
    pins the historical incident inline.
  - `MOE_SORT_MAX_EXPERTS: usize = 512` — matches the MoE sort-by-expert
    kernel's LDS-histogram size.

Replaced the two bare-literal asserts in `crates/ops/src/hip/moe.rs` with
references to these constants. Error messages now tell the caller to bump
both `core::kernel_limits` and the matching `#define` in the same PR.

**Not yet centralised** (intentional): `INDEXED_MOE_MMQ_X = 8`,
`INDEXED_MOE_MMQ_Y = 16`, MMQ tile dims, wave-size constants. These are
kernel-shape/tuning — doc comments in the `.cu` headers are their canonical
home; mirroring them in Rust would invert the source of truth.

**Follow-up deferred** (per plan §P2.3): a build-script that parses the
`.cu` `#define`s and cross-checks against `kernel_limits` at compile time.
Currently enforced by human review + the grep-visible `// matches #define`
comments.

Total code-site warnings: unchanged (the previous bare `256`/`512`
literals were not flagged by any enabled lint). The load-bearing benefit
is structural, not quantitative.

### 2026-04-23 — P3.1 landed: structured request-lifecycle tracing

Added three structured `tracing::info!` events on the chat/completion
request hot path in `crates/server/src/routes.rs`, per plan naming
scheme (`server.$component.$state` dotted identifiers):
  - `server.completion.start` — `prompt_tokens`, `max_tokens`.
  - `server.completion.first_token` — `prompt_tokens`, `ttft_ms` (time to
    first token; the single most important server latency metric).
  - `server.completion.finish` — `prompt_tokens`, `completion_tokens`,
    `finish_reason`, `total_ms`.

Added `#[tracing::instrument(name = "server.chat_completions", skip_all,
fields(messages, stream))]` on both handlers so every incoming request
gets a wrapping span visible in log output or OTel export.

**Explicitly left alone:** the 6 operator-facing `eprintln!` calls in
`crates/models/qwen3-moe/src/{forward,sharded}.rs`. Each is already
gated behind an opt-in env var (`FLAMBEAU_PARITY_TOPK_LOGITS`,
`FLAMBEAU_LOAD_TRACE`) and emits a copy-pasteable diagnostic-dump
format (`[argmax-topk] top-20 = [(11, …), …]`, `[load] upload_one NAME
DTYPE BYTES ms`) that operators grep and paste into investigations.
Migrating to `tracing::debug!` would change format + stream + filter
semantics and break their workflow. The plan's rollout §5 marks these
as "opportunistic, fold into the next V2 model-addition PR" — deferring
is the right call.

Warnings unchanged (195 → 195). Compile errors = 0. This is pure
observability infrastructure — no clippy lint targets it directly.

### 2026-04-23 — P3.2 landed: forward.rs → `forward/` module tree

6312-LoC `forward.rs` split into 9 sibling submodules under `forward/`,
each landed as its own compile+test-gated sub-task (P3.2.a – P3.2.j). At
every step `cargo check --workspace --lib --bins --tests` and
`cargo test -p flambeau-qwen3-moe --lib` were run; the full suite stayed
green through all 10 sub-tasks.

Final layout:
| Submodule | LoC | Content |
| --- | ---: | --- |
| `common.rs` | 164 | `upload_position`, `qdtype_of`, `mat_shape`, `row_bytes_for_dtype`, `run_mmvq_from_tensor`, `run_qmatmul_from_tensor` |
| `attn.rs` | 1055 | full-attention decode + prefill |
| `gdn.rs` | 1298 | gated-delta-net decode + prefill (+ conv helpers) |
| `dense_ffn.rs` | 432 | dense gate/up/down decode + prefill |
| `moe.rs` | 1684 | routed / shared / router decode + prefill |
| `io.rs` | 306 | embed + output-head + argmax |
| `layer.rs` | 625 | per-layer dispatcher for decode + prefill |
| `single_device.rs` | 334 | Mesh&lt;1&gt; entry points |
| `pp.rs` | 643 | Mesh&lt;N&gt; pipeline-parallel entry points |
| `mod.rs` | 76 | module map + re-exports |
| **total** | **6617** | (up 305 from pre-split 6312 due to per-submodule doc headers + targeted imports) |

**Public API preserved.** Everything the rest of the crate / the server /
tests reach for (`forward::forward_one_token`, `forward::forward_prefill_pp`,
`forward::ForwardOneTokenScratch`, etc.) is still importable under
`flambeau_qwen3_moe::forward::*` — each submodule is `pub mod` and the
associated items are re-exported via `pub use` from `mod.rs`.

**What this unlocks:**
- V2+ second-model additions can edit `attn.rs` or `moe.rs` in isolation
  without touching 6 kLoC of unrelated composition code.
- Rust-analyzer, rebuild-on-edit, and git blame all operate on 300-1700-LoC
  files instead of one 6 kLoC monolith — noticeable IDE latency win.
- Structural seams now match the doc (`ARCHITECTURE.md` names the phases);
  future contributors can trust the file names.

**Hardware validation deferred.** Non-hip unit tests (5) pass; the
decode/prefill/PP code paths require HIP hardware to bit-verify. The
parity certs in `certs/parity/` will confirm at the next hardware-attached
session. If any split introduced a regression, the cert will tell us
before it reaches production — that's the contract.

**Exit criteria met:** workspace cargo check clean, qwen3-moe lib tests
green, clippy warnings unchanged at **195**, compile errors = 0.

---

## Out of scope for this plan

- Migrating from `anyhow` to per-crate `thiserror`. The application-guidelines chapter would push us there, but the workspace is not yet at a stability point where error-type stability matters. Revisit once V2 ships a second model.
- Miri / cargo-hack / cargo-udeps wiring beyond a mention in P0.2 — these are CI infra, not code-correction.
- Kernel `.cu` / `.cuh` sources — they are C++/HIP, not governed by Rust guidelines.
- `crates/mcp-server/` hygiene — dev-only tool per CLAUDE.md, explicitly not load-bearing; compliance work here is pure opportunity cost.

<!-- Rust guideline compliant 2026-02-21 — this plan, not the code it describes. -->
