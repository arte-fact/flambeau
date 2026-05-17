# flambeau-forward — Working Notes for Claude

This crate is the topology executor — `ForwardCtx` trait + three
impls (Pp, Tp, Hybrid) — that lets a model write ONE forward function
which runs unchanged under any topology. It sits between
`flambeau-model-ops` (leaf primitives) and `crates/models/*-v2`
(thin model crates).

Read `README.md` for the public surface and the design rationale.
Read this file before adding or modifying any code in this crate.

## Rules (do not quietly break)

1. **`ForwardCtx` has the same method set on every impl.** No method
   exists on only one topology. If a composite makes sense only on
   TP, it belongs as a free helper or as part of a different trait —
   never as a method on `ForwardCtx`. This is rule 12 of the project
   root CLAUDE.md applied locally: no closed-enum-disguised-as-a-trait.

2. **The model forward function is `<C: ForwardCtx>` generic.**
   ONE function per arch, monomorphised per topology by the compiler.
   No `dyn ForwardCtx`. No three-times-duplicated forward function per
   arch — that's the exact bloat this crate exists to eliminate.

3. **Topology-specific machinery lives inside the trait impl.**
   AR after row-parallel ops, peer-copy at PP stage boundaries,
   handoff sync between stages, intra-stage TP composition inside
   hybrid: all of it lives inside `impl ForwardCtx for *ForwardCtx`,
   not in the model. The model never knows it's running under TP.

4. **Composites call into `flambeau-model-ops` leaf primitives.**
   Do not bypass model-ops to call `flambeau-ops::HipOps` directly,
   and do not call HSACO kernels directly. Every composite eventually
   bottoms out in a model-ops free function. If a needed leaf op
   doesn't exist in model-ops, it lands there first (with its own
   parity test) before being used here.

5. **No state on the trait, request state on the impl.**
   The trait is a method contract; the per-request state (KV cache
   refs, scratch pool, layer-to-rank map, position counter) lives on
   each impl as struct fields. The state is per-request — borrowed
   from the model's KV / scratch allocators, not owned.

6. **No `Stage` / `Driver` / `Session` types in this crate.** Those
   live in `crates/models/*-v2` (the model crate's per-request
   container) and pass refs to ctx fields. This crate doesn't
   allocate weights, doesn't manage KV caches, doesn't own clusters
   — it borrows everything it needs through the ForwardCtx impl
   struct.

7. **No phase markers, narrative comments, or commit-sha refs.**
   Same rule as project root. Code comments cover invariants, safety,
   non-obvious choices. Nothing else.

8. **Every new composite gets a RecordingCtx call + a topology-parity
   test.** RecordingCtx unit-tests the model's op-sequence (no device
   needed, fast). Topology-parity test runs a tiny synthetic model
   through Pp/Tp/Hybrid ctxs on real device and confirms outputs
   match within F16 tolerance.

9. **No `Hip*` / `Cuda*` / arch-prefixed types in this crate.**
   The trait + impls are generic over the kernel backend through
   `flambeau-model-ops`. V1 only has the HIP backend; CUDA arrives
   later by adding a CUDA backend to model-ops, not by adding a
   second executor here.

10. **Open design issue — scratch lifetime — pick BEFORE the first
    composite lands.** Three options in the README. Choose explicitly,
    not by accident. Document the choice in code comments where
    relevant. Once picked, do not silently re-litigate.

## What goes in this crate (and what does NOT)

In:
- The `ForwardCtx` trait + its three impls.
- Per-impl topology orchestration: AR placement, peer-copy, handoff
  sync, layer-to-rank iteration, scratch pool management.
- Composite ops as methods on the trait (rmsnorm, standard_attn,
  dense_ffn, moe_ffn, embed, output_head, residual_add, ...).
- A `RecordingCtx` mock for unit-testing model forward functions.

Out (lives elsewhere):
- Leaf op implementations → `flambeau-model-ops`.
- Kernel launches → `flambeau-ops` (model-ops calls into it).
- Weight upload, GGUF parsing → `crates/models/*-v2`.
- Model config, layout, tokenizer plumbing → `crates/models/*-v2`.
- KV cache allocation lifecycle → `crates/models/*-v2`
  (this crate borrows KV refs from there).
- Server-side wiring, inflight pool, scheduler → `crates/server`.

## Don't repeat past mistakes

The repository memory and the project root CLAUDE.md catalogue the
bloat patterns:
- Adding `Stage`/`Driver`/`Session` per topology.
- Re-coding the forward path three times per arch.
- Trait methods with `None` defaults that only one arch overrides
  (rule 12 / parallel-op-trait variant of rule 3).
- Narrative comments preserving session history in the source.

This crate is the place where the per-topology duplication stops.
Every commit here should make it harder, not easier, to drift back
into those shapes — specifically, if you find yourself wanting to
add a method to `ForwardCtx` that only one impl can really do,
**stop**. That's the failure mode. Add a free function in the impl
module, or refactor the composite so all three impls have a sensible
shape, or push the work down to model-ops where it's topology-free.

## Adding a composite — strict recipe

1. **Define the trait method** in `src/ctx.rs`. Name it after what it
   does at the model layer (`standard_attn`, not `tp_aware_attn`).
2. **Implement on all three** topology ctxs (`pp.rs`, `tp.rs`,
   `hybrid.rs`). Each impl calls model-ops leaf primitives + inserts
   topology-specific glue (AR, peer-copy, handoff).
3. **Add a method to `RecordingCtx`** that records the call as an
   `OpCall` variant. This is mechanical and lets the model's unit
   tests assert on op sequence.
4. **Add a topology-parity test** that exercises the new composite
   under all three ctxs on a tiny synthetic model.
5. **Re-export from `src/lib.rs`** if the composite type
   (or a helper) is part of the public surface.

If you can't do step 2 (one of the three impls doesn't make sense),
the composite shape is wrong — push back instead of shipping a
half-implemented trait.
