# flambeau-model-ops

Typed, individually-testable model ops. Each op is a free function over
typed tensors. Each op has a co-located test that runs on a real HIP
device with synthetic small inputs and asserts against a CPU reference.

## Why this crate exists

The existing per-arch crates (`crates/models/qwen3-moe`,
`crates/models/gemma4`) write the model's forward path **three times**
(PP / TP / Hybrid), each open-coding `rmsnorm → attn → AR → norm → ffn
→ ...` against the kernel-launch layer. Today that costs roughly
10–22 kLOC per arch — about **35–40×** what the same architecture is
in `llama.cpp` (~300–500 LOC per arch). The duplication compounds:
each new topology multiplies it, each new feature lands three times.

The fix is to make ops the leaf primitive that everything else
composes. A model becomes a sequence of op calls. Topology is a wrapper
that runs the same op sequence under PP / TP / Hybrid, inserting
AllReduce + peer-copy + handoff as needed — without the arch code
knowing about it.

This crate is the leaf-primitive layer. Building it does NOT touch the
existing `crates/models/*` code — those stay as the working
implementation and the correctness oracle. New thin model
implementations live in a sibling crate set; they migrate over one at
a time, each gated by a parity test against the old impl. When a
model's v2 passes parity + perf, the old crate is deleted.

## Op shape

Every op is a free function with this shape:

```rust
pub fn op_name(
    // typed inputs (read-only)
    input: &Tensor<F16>,
    weight: &Tensor<F16>,
    // typed outputs (caller-owned scratch)
    output: &mut Tensor<F16>,
    // shape parameters
    n_rows: usize,
    hidden: usize,
    eps: f32,
    // launch context (HIP-specific; backend-neutral op surface is a
    // later phase — V1 ships HIP only)
    ops: &HipOps,
) -> Result<()>;
```

Properties enforced for every op in this crate:

- **No state.** No `Stage`, no `Driver`, no `Session`, no `Cluster`,
  no `Mutex`. State (KV cache, scratch buffers, per-request bookkeeping)
  lives in the model crates that *call* these ops, not here.
- **No trait surfaces with one impl.** Free functions. If a kernel
  has two implementations, that's two functions (`rmsnorm_f16`,
  `rmsnorm_f32`), not a trait with two `impl`s.
- **No `dyn`, no `Arc`.** If an op needs to take a callback, that
  becomes a generic parameter.
- **One file per op.** `src/ops/rmsnorm.rs`, `src/ops/qmatmul.rs`,
  `src/ops/attn_decode.rs`, etc. Discoverable by filename grep.
- **Co-located unit test with mock data + CPU reference.** Tests
  alloc small device buffers, run the op, download, compare against a
  CPU reference function. No GGUF / no cluster / no model fixture.

## Adding an op (recipe)

1. Pick a name following the convention `{op}_{dtype_in}_{dtype_out}`
   when ambiguous (`rmsnorm_f16`, `cast_f32_to_f16`). One file per op:
   `src/ops/<name>.rs`.
2. Write a CPU reference function in the same file (or in
   `src/testing/reference.rs` if shared across ops). The reference is
   plain Rust over `&[f32]` / `&[f16]` — no device, no kernels. It
   exists only so the test can assert closeness.
3. Write the device-side op. Call into the existing
   `flambeau-ops::hip::HipOps` for the kernel launch — that layer
   already wraps the HSACO modules.
4. Write the test:
   - alloc small inputs on the device (e.g. n_rows=4, hidden=64),
   - upload synthetic host data (deterministic, e.g. linspace),
   - run the op,
   - download the result,
   - assert `assert_close(got, reference(host_inputs), tol=1e-3)`.
5. Re-export from `src/lib.rs`.

If you find yourself wanting to add a struct that isn't `Tensor<T>` or
an enum dtype, **stop**. State is the bloat trap that produced the
~20× LOC ratio in the first place.

## Layout

```
crates/model-ops/
├── Cargo.toml
├── README.md
├── CLAUDE.md
└── src/
    ├── lib.rs           // re-exports the op vocabulary
    ├── dtype.rs         // ElemType, marker zsts: F32, F16, I32, Q8_1, Q4_0, ...
    ├── tensor.rs        // Tensor<T> — typed view over (DevicePtr, n_elems)
    ├── error.rs         // crate-local Result / Error aliases
    ├── testing.rs       // alloc/upload/download/assert_close (test-only)
    └── ops/
        ├── mod.rs       // one `pub mod <name>;` line per op file
        ├── rmsnorm.rs
        ├── qmatmul.rs
        ├── rope.rs
        ├── attn_decode.rs
        ├── attn_prefill.rs
        ├── kv_append.rs
        ├── ffn.rs
        ├── embed_lookup.rs
        ├── cast.rs
        ├── sample.rs
        ├── tp_sum.rs
        ├── tp_residual_rmsnorm.rs
        ├── peer_copy.rs
        ├── moe_router.rs
        ├── moe_topk.rs
        └── moe_experts.rs
```

Flat under `src/ops/` until there's clear navigation pressure — easier
to grep, easier to discover. ~25–30 op files is the expected steady
state.

## Running tests

```
cargo test -p flambeau-model-ops --features hip
```

Tests require a HIP device. CI configuration follows the same pattern
as the other HIP crates (`bench`, `kernels-hip`).

## Relationship to existing crates

| crate | role | status |
|---|---|---|
| `flambeau-kernels-hip` | HSACO kernel sources + module loader | reused |
| `flambeau-ops` | per-kernel launch wrappers (`HipOps::rmsnorm_f16`, etc.) | reused as the kernel-call layer |
| `flambeau-model-ops` | composite blocks + topology orchestrators (the bloated layer this work replaces) | NOT a dependency of model-ops; will be retired once v2 models migrate |
| `crates/models/qwen3-moe`, `crates/models/gemma4` | current per-arch impls | untouched; correctness oracle; deleted when v2 lands |
| `crates/model-ops` (this crate) | leaf op vocabulary | new |
| `crates/models/*-v2` (future) | thin model crates composing these ops | new |
