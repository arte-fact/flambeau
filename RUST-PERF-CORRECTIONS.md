# Rust-side perf corrections — review against `rust-performance-best-practices`

Audited 2026-04-23 by Claude. Scope: CPU-side Rust on the per-token / per-layer
hot path. Out of scope: HIP kernels, build glue, bench/CLI one-shot code.

The codebase is already perf-tuned (V1.7.6.f → V2.10 cumulative is 2.92× over
the V1.7.6 scalar baseline). The findings below are the subtle remaining
Rust-side levers. Memory note V1.7.6.f hypothesised the residual 13.5 % gap
to llama.cpp on Mesh<2> decode is "Rust-FFI launch overhead" — items **C1**
and **C3** are the two most plausible contributors and the natural first
things to measure.

**No claim here is measured.** Every "impact" estimate is a hypothesis —
the rule from `CLAUDE.md` stands: profile with rocprofv3 / wall-clock before
declaring a win, and PR with cert + PMC evidence. Items are ordered by
estimated probability of being load-bearing.

---

## Tier H — plausibly ≥1 % of decode wall-time

### C1. `HipModule::kernel()` allocates a `CString` + `String` on every kernel resolution
**File:** `crates/backend-hip/src/module.rs:96-116`

Every op call site in `crates/ops/src/hip/*.rs` calls `module.kernel("flambeau_…")`
just before launch. Each call:

1. `CString::new(name)` — heap alloc + NUL-terminate (line 97).
2. `hipModuleGetFunction` — driver lookup (cached driver-side but still a
   table walk + lock).
3. `name.to_string()` — heap alloc to populate `HipKernel.name` (line 113).

`HipKernel.name` is only read by the `Debug` impl (`module.rs:139`). It is
dead weight on the hot path.

Rough scale: V1.7.6.f decode is ~17 ops/layer × 40 layers ≈ 680
resolutions/token; at 54 tok/s that's ~37 k driver lookups + ~74 k heap
allocations per second per rank.

**Fix:**
- Drop `HipKernel.name` (or make it `&'static str` carried from the call
  site — every call site already passes a string literal).
- Cache the resolved `hipFunction_t` per `(module, name)` in `HipModule`
  itself: a `OnceCell<HashMap<&'static str, hipFunction_t>>` populated lazily,
  or a small `RwLock<HashMap<&'static str, hipFunction_t>>`.
- Better: hand call sites pre-resolved `HipKernel<'m>` handles owned by the
  per-rank state struct (`Qwen3MoEModel` / `Qwen3MoEShardedModel`), resolved
  once at `load()`. The dispatch table already enumerates every entry symbol;
  resolving them all upfront is a finite cost.

Even after the lookup is cached, killing the two per-call allocations is
its own win and is trivial.

### C2. `sample_softmax_temp` and `sample_top_p` allocate a vocab-sized vec per token
**File:** `crates/runtime/src/sampling.rs:108`, `:128`

```rust
let mut probs = vec![0.0f32; logits.len()];                 // line 108
let mut probs: Vec<(u32, f32)> = logits.iter().enumerate()  // line 128
    .map(|(i, &v)| (i as u32, (v * inv_t - max_l).exp())).collect();
```

Qwen3.6 vocab = 151 936; that's a 593 KiB / 1.16 MiB allocation per sample
in temp / top-p paths respectively. Greedy (`sample_argmax`) doesn't hit this.
At 54 tok/s on default sampling settings the alloc-then-free churn is ~32 MiB/s.

**Fix:** Hold the scratch on the sampler / session struct and reuse with
`clear()` + `extend(…)`, or a single `Vec<f32>` resized once at first call.
Top-p's `(u32, f32)` variant is the bigger item — same treatment applies.
This is the canonical `alloc-reuse-buffers` pattern.

### C3. `HipCluster::ensure_bounce` takes a `Mutex` lock per peer-copy
**File:** `crates/backend-hip/src/cluster.rs:75`, `:139`

```rust
let mut slot = self.bounces[rank].lock().map_err(…)?;
if slot.bytes >= need && !slot.ptr.is_null() { return Ok(slot.ptr); }
```

Comment at `:147` already notes the "single grow to first observed size is
the common case". After warmup the lock is uncontended and degenerate, but
the acquire still costs ~150–200 ns + a memory fence per peer-copy. With
PP layer-boundary hand-offs, that's one lock per rank per token — modest
on its own, but in the same hot path as **C1** and the goal is to strip the
launch path to bone.

**Fix:** Pre-size the bounce slab at session init (max observed
hidden-state byte size is bounded by `hidden_dim * dtype_bytes`, known
at load) and replace `Mutex<RankBounce>` with a plain `RankBounce` — no
lock, no `.lock().map_err(…)?` on the hot path. If runtime growth is
genuinely needed for V2 batched flows, gate it behind an
`AtomicUsize` capacity check + a slow-path lock for grow only.

---

## Tier M — likely 0.1–1 % each, or load-bearing in V2

### C4. `forward_layer_decode` per-step `Vec`s in MoE expert routing
**File:** `crates/ops/src/hip/moe.rs:965-973` (host-side bucketing helper)

A `HashMap<i32, Vec<i32>>` is built per expert-assignment, then keys are
`copied().collect()` into a Vec, sorted, and `remove(0)`-popped. This is
prefill-only (decode uses sorted-pair indices on the GPU per V2.5.b), so
not on the per-token path — but the prefill pp=512 → 698 tok/s number is
also customer-visible.

**Fix:** Counting-sort on `expert_id ∈ [0, n_experts)` produces both the
sorted pair indices and the `offsets[n_experts+1]` array in two linear
passes, no HashMap, no key collect, no `remove(0)`. The on-GPU
`moe_sort_by_expert` infra (V2.5.a) does exactly this; the host-side
helper should mirror its shape or be deleted if it's only a fallback.

### C5. Server hot path clones every chat message before templating
**File:** `crates/server/src/routes.rs:75-82`

```rust
TmplMessage { role: m.role.clone(), content: m.content.clone() }
```

Per request, not per token, but per *every* request. Strings are typically
small (KB-range prompts) so the absolute cost is low — but the pattern is
also load-bearing once continuous batching lands.

**Fix:** Make `TmplMessage` borrow (`&'a str`) and parameterise by a
lifetime tied to the inbound request body, or use `Cow<'a, str>` if the
template engine sometimes mutates. Either way, drop the per-message alloc.

### C6. Stop-token filter re-allocates the generated-tokens vec
**File:** `crates/server/src/routes.rs:277-281`

```rust
let visible: Vec<u32> = generated.iter().copied().filter(…).collect();
```

Per response. Acceptable today; flag for V2 batched response paths where
a per-stream alloc per finish becomes more visible.

**Fix:** `generated.retain(…)` if `generated` is owned, or write the
filtered tokens into a session-owned scratch.

### C7. `iter_tensors_mut` builds a `Vec` with `Vec::new()` + push storm
**File:** `crates/models/qwen3-moe/src/weights.rs:311-356`

Many `v.push(…)` calls on a freshly-constructed `Vec::new()`. This is
load-time, not hot-path, so impact is sub-millisecond per session, but
`Vec::with_capacity(<count>)` is a one-character fix and silences the
`alloc-vec-with-capacity` rule.

---

## Tier L — style / hygiene, not load-bearing

### C8. `dims.clone()` on small `Vec<usize>` at GGUF load
**File:** `crates/models/qwen3-moe/src/sharded.rs:304, 547, 618, 700, 788`,
`weights.rs:237`

`dims` is typically ≤4 elements. Load-time only. Cosmetic — a `Cow<[usize]>`
or `[usize; 4] + len` removes the alloc, but the existing `clone()` is
honest and cheap enough at load time. Skip unless you're already in the
file.

### C9. `panic = "abort"` is not set in `[profile.release]`
**File:** `Cargo.toml:49-53`

Current profile is sound (`opt-level=3`, `lto="thin"`, `codegen-units=1`,
`debug=1` for rocprofv3 attribution). `panic = "abort"` would shave a
small amount off binary size and remove unwinding tables, at the cost of
losing the unwind path for panic diagnostics.

For a server that should treat any panic as fatal-and-restart this is the
right default. **Recommendation:** add it once V1 has shipped and there's
appetite for a +1–2 % perf-tuning pass — not before, since unwind backtraces
are useful during the V1.7.x bug-hunt phase.

LTO `"thin"` → `"fat"` is *not* recommended: link time on this
13-crate workspace would jump from ~10 s to multiple minutes, and prior
cross-crate inlining work suggests sub-1 % wall-clock impact on the
HIP-bound critical path. Don't pay it.

---

## Non-issues confirmed

- `crates/server/src/routes.rs:169` correctly wraps the sync inference
  call in `tokio::task::spawn_blocking`. ✅
- Per-layer scratch (`LayerForwardScratch`) is pre-allocated at load,
  not per-token. ✅ (`crates/models/qwen3-moe/src/forward.rs`)
- `Arc::clone(&device)` / `Arc::clone(&module)` on the launch path is just
  a refcount bump — not a data copy. ✅
- GGUF reader uses `memmap2` not buffered I/O — no syscall amortisation
  needed. ✅
- `Mutex<Option<RocBlas>>` drop-order discipline (`HipDevice`) is documented
  and correct (CLAUDE.md §"Technical lessons / Kernel / silicon"). ✅

---

## Suggested order of attack

If a V2.x.something perf cycle picks this up, do **C1 first**: kernel
resolution caching is a one-touch refactor in `module.rs` + a search-replace
across `crates/ops/src/hip/*.rs` to take a `&HipKernel<'_>` instead of
calling `module.kernel(name)`. Measure with rocprofv3 wall-clock (not
per-kernel time — that won't move; the saving is in the host gap between
launches) on a Mesh<2> tg=64 run.

**C2** + **C3** are independent and small; bundle them with C1 in the
same PR if you want one round of benchmarking instead of three. **C4–C7**
are V2 prefill / batching cleanups; not for the V1 ship.

Per CLAUDE.md measurement rules: rebuild the relevant crate, sweep the
affected impls (none of these touch kernel correctness, so cert sweep is
green by construction), then `cargo run -p bench -- matrix` at pp=512
tg=64 on Qwen3.6-35B Mesh<2> and Mesh<4>. Cert + matrix diff in the PR
body, no separate writeup.
