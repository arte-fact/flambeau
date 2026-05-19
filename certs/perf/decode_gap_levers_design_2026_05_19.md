# Decode-gap closing levers — implementation design

Companion to `decode_gap_v2_vs_legacy_vs_llamacpp_2026_05_19.md`. Three
independent levers, each contained to one layer of the v2 stack.

## Lever 1 — F16-output mmvq variants at output_proj / down_proj

### What's there

`flambeau-ops` already exposes the legacy-fast variants on `HipOps` /
`Ops` trait (`crates/ops/src/hip/qmatmul.rs`):

| op                          | kernel entry                                  | use                                  |
|-----------------------------|-----------------------------------------------|--------------------------------------|
| `mmvq_q4_0_kv_f16dst`       | `flambeau_mmvq_q4_0_kv_f16dst_dp4a_q8_1`      | Q4_0 → F16 output (fused cast)       |
| `mmvq_q4_0_t128`            | `flambeau_mmvq_q4_0_t128_dp4a_q8_1`           | Q4_0 → F32, `_t128_dp4a` fast path   |
| `mmvq_q4_0_gate_up_t128`    | `flambeau_mmvq_q4_0_gate_up_t128_dp4a_q8_1`   | fused gate+up Q4_0, `_t128_dp4a`     |

Plus the F16-direct kernels exist in `kernels-hip/src/kernels/`:
`mmvq_q4_0.cu` ships both `flambeau_mmvq_q4_0_q8_1` (F32 out) and
`flambeau_mmvq_q4_0_q8_1_f16` (F16 out, "saturating, consumer-direct,
no cast launch needed").

V2's `QuantWeight::qmatmul()` always goes through dispatch which
selects the F32-output variant.

### Design

**Add typed-op wrappers in `flambeau-model-ops`** (one op per file, no
state, mirrors existing `qmatmul_q4_0`):

```rust
// crates/model-ops/src/ops/qmatmul_to_f16.rs
pub fn qmatmul_q4_0_to_f16(
    weight: &QuantWeight,
    act_q8_1: &Tensor<Q8_1>,
    out: &mut Tensor<F16>,
    m: usize, k: usize, n_rows: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    // Validates m=1 (mmvq path), dtype=Q4_0; dispatches to
    // ops.mmvq_q4_0_kv_f16dst for n_rows==kv_width (the gemma4 KV
    // projection case) OR — when we add it — a fused-cast Q4_0 path
    // for general F16 output. For now: bail when m>1 (caller stays
    // on the F32+cast path).
}
```

**Composite call-site changes** (no logic duplication — composites
just pick the typed-op variant when output target is F16 and m=1):

- `standard_attn_local`: when the AR fast path is active and the
  output is going to `pool.delta` F16, use `qmatmul_q4_0_to_f16`
  (output writes directly to delta, drop the `cast_f32_f16` call).
- `dense_ffn_local` / `moe_ffn_local`: same shape for the `ffn_down`
  / MoE `down` partial when targeting F16.

**Fused gate+up `_t128_dp4a` variant**:

- `flambeau-blocks::DenseFfn` and `flambeau-blocks::SharedExpert`
  internally call `ops.mmvq_q4_0_gate_up()`. The block already
  picks fused-gate-up via dtype match (line 510 of delta_net.rs;
  same pattern for ffn). The block doesn't know about the
  `_t128_dp4a` variant — it always picks `mmvq_q4_0_gate_up`
  (entry: `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1`).
- Add `mmvq_q4_0_gate_up_t128` dispatch inside the block:
  shape-based pick at gate+up dispatch (same shape as legacy uses).
  This is a `flambeau-blocks` change, NOT a composite change.

### What this gains

- ~12 288 cast_f32_f16 launches eliminated on 27B / 128 decode tokens.
- ~5 ms/tok GPU kernel time recovered.
- No composite-shape changes; just substituting which leaf op
  the composite (or the block) calls.

### Architecture rule compliance

- Rule 1 (no env flags): variant selection in composites and the
  block based on dtype + shape, not a runtime flag.
- Rule 2 (cert per kernel): the kernels are existing, cert'd already.
- Rule 3 (one contract, many impls): new typed ops are sibling free
  functions in model-ops — no parallel `Op` trait.
- Rule 4 (models are glue): all changes in `flambeau-model-ops`,
  `flambeau-blocks`, and the v2 composites. No new kernel under
  `crates/models/`.

---

## Lever 2 — F16 router weight at load time

### What's there

Legacy `tp_target_dtype` in `crates/models/qwen3-moe/src/tp_sharded.rs:518`:

```rust
if name.ends_with("ffn_gate_inp.weight") {
    return Some(GgmlDType::F16);
}
```

Converts F32 → F16 at upload. The composite's router fires
`dense_gemv_f16_f16` (no input Q8_1 quantize step needed) instead of
`mmvq_q8_0` (which requires F16 → Q8_1 quantize per call).

V2's `qwen35moe-v2/src/loader.rs` calls `upload_quant_weight` for the
router, which host-quantizes F32 → Q8_0. The composite then fires
`mmvq_q8_0` + extra `quantize_row_f16_q8_1` of x_norm per layer.

### Design

**Loader-side change in `flambeau-forward/src/loader/shard.rs`** (already
imports `upload_dequant_to_f16`):

Add a sibling helper `upload_dequant_to_f16_quant_weight` (or extend
the existing one to return a `QuantWeight` with `dtype = F16`).
Or even simpler: add a top-level `upload_router_f16` helper —
explicitly named for the use case so the model loader is readable.

```rust
// crates/forward/src/loader/shard.rs
pub fn upload_router_f16(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let ptr = upload_dequant_to_f16(file, device, name, n_elems, allocs)?;
    Ok(QuantWeight { ptr, dtype: QDtype::F16, n_elems })
}
```

**Model loader change** (1-line swap):

In `qwen35moe-v2/src/loader.rs` line ~165 (the `ffn_gate_inp` upload
inside `MoeWeights`), replace `upload_quant_weight` with
`upload_router_f16`. Apply the same to other MoE archs that gain it.

**Composite compatibility**:

The router call is `weights.router.qmatmul(&x_q8_1, ..., &mut router_logits, 1, hidden, n_experts, &ops)`.
The `QuantWeight::qmatmul()` dispatcher in `flambeau-quant` already
maps F16 dtype to `ops.dense_gemv_f16_f16` (this is what legacy uses).

**Caveat — the input quantize**: v2's moe_ffn currently fires
`quantize_f16_to_q8_1(x_norm_f16, &mut norm_q8_1)` before the router
call. When the router moves to F16, that quantize step is still needed
for the MoE indexed expert matmuls (Q8_1 activations). So the quantize
launch stays; we save the router's own mmvq → dense_gemv swap.

### What this gains

- Router becomes `dense_gemv_f16_f16` (faster) instead of `mmvq_q8_0`.
- Eliminates extra `quantize_row_f16_q8_1` launches that were only
  for the router (the post-attn-norm quantize stays for experts).
- 35B-class win: ~1 ms/tok GPU.

### Architecture rule compliance

- Rule 1: dtype choice is a loader decision (per-arch, in the arch's
  loader.rs), not a runtime flag.
- Rule 4: no kernel files under `crates/models/`.
- Reuses the existing `upload_dequant_to_f16` plumbing — no new
  upload primitive needed for the F16 path.

---

## Lever 3 — Event-based AR ordering (no host blocking)

### What's there

v2's `bar_ar_residual_f16` and `bar_ar_residual_rmsnorm_f16` in
`crates/forward/src/runtime/ar.rs`:

```rust
flambeau_core::Stream::synchronize(stream)?;   // ← blocks host
{ p.lock(); p[rank] = Some(partial_f16); }
coord.barrier.wait();
// ... snap peers, launch BAR1
```

Legacy uses `flambeau_blocks::cross_rank_event_barrier(cluster,
&cores)` which records a `producer_done_event` per rank, then on each
peer's stream issues `stream_wait_event(peer_event)`. GPU-side
ordering; host never blocks.

HIP events exist in `flambeau-core` / `flambeau-backend-hip`:
- `flambeau_backend_hip::HipEvent` (re-exported from `device.rs`).
- `Stream::wait_event(&event)` API — to confirm in `flambeau-core`.

### Design

**Extend `BarArCoordinator` with per-rank events** (the shared state
already includes the per-rank partial slab + a barrier; events join
that slab):

```rust
pub struct BarArCoordinator {
    pub bar: Arc<BarP2pAllReduce>,
    partials: Mutex<Vec<Option<DevicePtr>>>,
    events: Vec<Mutex<HipEvent>>,     // NEW: one event per rank,
                                       // pre-allocated on the rank's device.
    barrier: Barrier,
}
```

The events are constructed once per AR coordinator (alongside the
per-rank cluster device), reused across many AR calls.

**Replace `Stream::synchronize` with event record/wait**:

```rust
pub fn bar_ar_residual_f16(coord, rank, residual_inout, partial_f16, n_elems, dev, stream) -> Result<()> {
    // 1. Record producer-done event on this rank's stream — NON-BLOCKING.
    coord.events[rank].lock().record(stream)?;

    // 2. Publish own partial pointer.
    { let mut p = coord.partials.lock().unwrap(); p[rank] = Some(partial_f16); }
    coord.barrier.wait();

    // 3. Snap peer pointers AND have this rank's stream wait on every
    //    peer's event before launching BAR1.
    let peers_snapshot: Vec<DevicePtr> = { /* … */ };
    for r in 0..coord.ranks() {
        if r == rank { continue; }
        let peer_evt = coord.events[r].lock();
        stream.wait_event(&peer_evt)?;     // GPU-side ordering.
    }

    // 4. Launch BAR1.
    unsafe { coord.bar.residual_tp2_rank(...)?; }

    // 5. Barrier so the partials slab is safe to reset for next call.
    coord.barrier.wait();
    if rank == 0 { /* clear slab */ }
    coord.barrier.wait();
    Ok(())
}
```

Same shape for `bar_ar_residual_rmsnorm_f16`.

**Extract a shared helper** to avoid duplication between the two AR
functions (both have the same publish-events / wait-peers /
barrier prologue and the same slab-reset epilogue):

```rust
// Private helper in runtime/ar.rs
fn ar_bar1_prelude(coord, rank, partial_f16, stream) -> Result<Vec<DevicePtr>> {
    coord.events[rank].lock().record(stream)?;
    { coord.partials.lock().unwrap()[rank] = Some(partial_f16); }
    coord.barrier.wait();
    let peers = { coord.partials.lock().unwrap().iter().map(|o| o.unwrap()).collect() };
    for r in 0..coord.ranks() {
        if r != rank { stream.wait_event(&coord.events[r].lock())?; }
    }
    Ok(peers)
}

fn ar_bar1_epilogue(coord, rank) {
    coord.barrier.wait();
    if rank == 0 { /* clear slab */ }
    coord.barrier.wait();
}
```

Both `bar_ar_residual_f16` and `bar_ar_residual_rmsnorm_f16` become:
prelude → launch BAR1 with appropriate args → epilogue. No
copy-pasted sync/publish logic.

### What this gains

- 33 155 `hipStreamSynchronize` calls / 128 decode tokens removed
  (27B trace; similar magnitude on 35B).
- Host time spent in `hipStreamSynchronize`: 8.4 s → ~0 s across the
  bench.
- More importantly: host can queue subsequent kernels while GPU runs
  AR. CPU/GPU overlap recovered, matching legacy's behavior.

### Architecture rule compliance

- Rule 3 (one contract, many impls): `bar_ar_residual_*` keep the
  same public signature; the impl swaps host-sync for events.
- Rule 7 (explicit async): event record/wait is the canonical
  explicit-async pattern. Replacing host sync with event-stream-wait
  RESTORES the rule's intent (the host sync was a violation in
  spirit).
- No new abstraction: per-rank events are a private detail of
  `BarArCoordinator`; the outer trait surface (`TopologyHooks::
  ar_residual_*`) is unchanged.

### Risk / fallback

- If `HipEvent` ↔ `HipStream::wait_event` semantics on gfx906 don't
  give us the ordering we need, fall back to the current sync path
  (gate inside `bar_ar_residual_*` with a runtime feature probe at
  coordinator construction; same shape as `try_build_bar_ar` already
  does for BAR1 itself).
- Single-rank case (n_ranks == 1) stays a no-op as today.

---

## Cross-lever notes

- Levers are independent and can ship in any order. Each closes a
  distinct chunk of the gap; combined effect on 27B / 35B decode
  expected to bring v2 within ~2-3% of legacy.
- No changes to composite trait surfaces (`TopologyHooks`,
  `ForwardCtx`). All changes are inside existing impls + leaf ops.
- No changes to dispatch tables (`dispatch/hip/gfx906.toml`):
  Lever 1 picks the variants directly in composites/blocks based on
  shape + dtype, not through the runtime dispatch row. (If we want to
  add a dispatch row for the `_t128_dp4a` variant later, do it as
  a separate refactor.)
- No new public types beyond what each lever needs internally.
