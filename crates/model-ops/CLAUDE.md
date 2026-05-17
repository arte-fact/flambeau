# flambeau-model-ops — Working Notes for Claude

This crate is the leaf-op vocabulary that v2 model crates will compose.
It exists because the current per-arch crates run 10–22 kLOC, ~35–40×
their llama.cpp equivalents, because each arch open-codes its forward
path three times (PP / TP / Hybrid). The fix is to make ops the leaf
primitive and have one shared executor handle topology — but the first
step is getting the leaf surface right.

Read `README.md` for the public surface and recipe. Read this file
before writing or reviewing any code in this crate.

## Rules (do not quietly break)

1. **One op per file under `src/ops/`.** Filename = op name = `pub fn`
   name. Grep-able. Discoverable.

2. **Free functions only.** No traits with one impl. No dyn dispatch.
   If two dtypes need a kernel, that's two functions
   (`rmsnorm_f16`, `rmsnorm_f32`), not a trait with two `impl`s. If
   three quant variants need a matmul, that's three functions
   (`qmatmul_q4_0`, `qmatmul_q5_0`, `qmatmul_q8_0`), not a trait.

3. **No state.** No `Stage`, no `Driver`, no `Session`, no `Cluster`,
   no `OpsRegistry` as a stored field, no `Mutex`, no `Arc`. Ops are
   pure: typed inputs in, typed outputs out, kernel launch on the
   provided context. If you need state, you're in the wrong crate
   (state lives in `crates/models/*-v2`, which calls into this).

4. **`Tensor<T>` is the only owning type.** It wraps a `DevicePtr` +
   `n_elems` + a marker for dtype. No `Layout` typestate in V1 —
   parallel ops take `&[&Tensor<T>]` slices (one per rank) and the
   per-topology executor lives in the model layer. Adding `Layout`
   typestate is a possible future move; don't pre-empt it.

5. **Every op has a co-located test on real HIP with mock data.**
   - Alloc small buffers (n_rows ≤ 8, hidden ≤ 128) on the device.
   - Upload deterministic synthetic data (linspace, eye, random with
     fixed seed).
   - Run the op.
   - Download.
   - Compare to a CPU reference function (in the same file or
     `src/testing/reference.rs`).
   - `assert_close(got, expected, tol=1e-3)` for F16, tighter for F32.
   No test = no merge. No exceptions.

6. **CPU reference functions are plain Rust over `Vec<f32>` /
   `Vec<f16>`.** They are the test-side ground truth; they don't have
   to be fast, only obviously correct. Prefer the most direct
   formulation of the math.

7. **No new struct types beyond `Tensor<T>` and dtype zsts without
   asking.** State is the bloat trap that produced the 20× LOC ratio.
   If you think you need a struct, write the op without it first, see
   whether it's actually necessary.

8. **No phase markers, no narrative comments, no commit-sha refs.**
   Same rule as the project root: code comments describe invariants,
   safety notes, or non-obvious choices. They do not narrate refactors
   or reference tasks/PRs.

9. **No backend abstraction yet.** V1 is HIP-only. The op
   signatures take HIP-specific types directly (`&HipOps`, `&Stream`).
   A CUDA-port sibling crate or a backend trait is a later phase. Do
   not pre-emptively abstract.

10. **Re-export every op from `src/lib.rs`.** Consumers `use
    flambeau_model_ops::{rmsnorm_f16, qmatmul_q4_0, ...};` — flat
    namespace, no `ops::` prefix in the public API.

## What goes in this crate (and what does NOT)

In:
- Element-wise + reduction kernels (rmsnorm, add, scale, cast, softcap).
- Matmul variants (qmatmul per quant dtype, mmvq, mmq).
- Attention primitives (rope, kv_append, attn_decode, attn_prefill).
- FFN primitives (gate/up/down/activate fused or split).
- Sampling (topk, softmax, penalty-apply).
- MoE primitives (router, topk_softmax, indexed_mmvq, combine).
- TP primitives (sum AR, residual_rmsnorm, peer_copy via host).
- Embedding lookup.

Out (lives in the calling model crate, NOT here):
- KV cache management (allocation, reset, eviction policy).
- Scratch buffer allocation, alignment, lifetime.
- Stage/Driver/Session/Cluster types.
- Per-topology forward orchestration (PP rank loop, TP AR placement,
  Hybrid stage handoff). That's the topology executor's job, sibling
  to this crate.
- Weight upload, GGUF parsing, dtype dispatch by name.
- Server-side wiring, inflight pool, scheduler.

## Adding an op — the strict recipe

1. **Decide the name** — `{op}_{dtype}` where dtype matters
   (`rmsnorm_f16`, `rmsnorm_f32`). One word ops can skip the dtype
   suffix if there's only one variant.
2. **Create `src/ops/<name>.rs`.**
3. **Write the CPU reference first** in the same file. This is your
   ground truth and forces you to think through the math before
   touching device code.
4. **Write the op signature.** Inputs first, outputs second, shape
   params third, `ops: &HipOps` last. `Result<()>` return.
5. **Write the op body** — call into `flambeau-ops::hip::HipOps` for
   the kernel launch. Do not bypass that layer to call HSACO directly;
   reuse the existing wrappers.
6. **Write the test.** Same file, `#[cfg(test)] mod tests { ... }`.
   Use the `testing` helpers (`test_device`, `upload`, `download`,
   `assert_close`).
7. **Re-export from `src/ops/mod.rs` and `src/lib.rs`.**

That's the whole loop. If you need to do anything else (allocate a
struct, register a callback, hold a lock), stop and ask.

## Don't repeat past mistakes

The repository memory and CLAUDE.md at the project root catalogue the
ways this codebase has bloated:
- Adding `Stage`/`Driver`/`Session` types per topology.
- Re-coding the forward path three times per arch.
- "Just one more arch flag / branch / trait method" that compounds.
- Narrative comments preserving session history in the source.

This crate is the place where that pattern stops. Every commit here
should make it harder, not easier, to drift back into those shapes.

If a request would require any of the "Out" items above to be added to
this crate, push back and route the work to a sibling crate. The whole
point of the layering is that this crate stays leaf-thin.
