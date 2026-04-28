# CN-80B-20 — Hybrid graph capture on Coder-Next pp2tp2

**Status:** lever blocked at ROCm 7.1.1 runtime, not at framework level.

## TL;DR

The right HIP primitive for multi-stream capture with cross-stream events
(`hipStreamBeginCaptureToGraph`, ROCm 7.1.1 beta API) **exists** at
`/opt/rocm/include/hip/hip_runtime_api.h:7946`. But its multi-device
implementation in ROCm 7.1.1 cannot return a usable graph from the
end-capture call, and capture attempts permanently corrupt the involved
streams for the rest of the process. Verified on 4× MI50 PCIe rig
(devices `[0, 2, 1, 3]`, pp2tp2 hybrid).

A re-attempt is unblocked the moment a ROCm release fixes the beta API;
the framework wiring (`HipGraphExec::capture_into_shared_graph`,
`ShardedForwardOneTokenScratchHybrid::decode_graphs`,
`forward_one_token_hybrid_inner` capture/replay branch) is in tree.

## Workload

- Model: `Qwen3-Coder-Next-Q4_0.gguf` (45 GiB, 80 B params,
  `arch=qwen3next`, hybrid GDN + full-attn + MoE-with-shared).
- Topology: pp2tp2 (`pp_size=2`, `tp_size=2`, devices `[0,2,1,3]` per
  rig topology — `{2,3}` BAR1 fault avoided by stage-major rank
  pairing).
- Test: `crates/models/qwen3-moe/tests/coder_next_pp2tp2_decode_graph_ab.rs`
- Bench: prefill 32-token synthetic prompt, then 2 warmup decode steps,
  then 32 timed decode steps. Per-run wall time, take min over 2 runs.

## Eager baseline (control)

Run twice in a fresh process (`FLAMBEAU_AB_ONLY=eager`):

| run | wall (32 tok) | tok/s |
|-----|---------------|-------|
| 0   | 708.5 ms      | 45.16 |
| 1   | 706.4 ms      | 45.30 |

Min: **45.30 tok/s**. Steady-state matches the CN-80B-12 topology
summary (Coder-Next pp2tp2 decode 45 tok/s).

## Graph capture attempt (ROCm 7.1.1 multi-stream behavior)

`FLAMBEAU_DECODE_GRAPH=1` triggers `capture_into_shared_graph` on each
stage's TP-rank streams, expecting cross-stream events from
`BarP2pAllReduce::ar_residual` to capture as internal graph edges.

Empirical findings on this rig (verified across multiple iterations of
the implementation):

1. **`hipGraphCreate` + `hipStreamBeginCaptureToGraph` per stream**
   captures all kernels in the closure (24 layers traced). Closure
   runs to completion, both AR record/wait events captured. ✓

2. **`hipStreamEndCapture` on each captured stream returns 904**
   (`hipErrorStreamCaptureUnmatched`) — there is no "primary"
   stream returning success. Each call writes a non-null pointer
   into `out_graph`, but those handles are NOT usable:
   `hipGraphGetNodes` and `hipGraphInstantiate` SIGSEGV on them.
   Our pre-allocated `shared_graph` is also empty/orphaned.

3. **Calling `hipStreamEndCapture` on a second stream** after the
   first returned 904 SIGSEGVs the runtime when the closure
   succeeded (clean session). When the closure aborted early,
   end-capture on each stream returns Invalid (1) and does NOT
   SIGSEGV — but neither path produces a usable graph.

4. **Streams are unrecoverable post-attempt.** Even after
   end-capture on every stream and `hipGraphDestroy` of every
   handle, subsequent kernel launches on those streams fail with
   `hipModuleLaunchKernel: invalid argument`. Eager fallback
   inside the same process cannot continue. The model is
   effectively single-shot under capture mode.

5. **Plain `hipStreamBeginCapture`** (separate per-stream capture
   sessions, no shared graph) FAILS on the first kernel launch in
   multi-device mode with the same "invalid argument" error.
   Single-stream PP-only capture works fine (already shipped in
   `forward/pp.rs`).

The relevant ROCm doxygen warning at
`/opt/rocm/include/hip/hip_runtime_api.h:7941`:

> "This API is marked as beta, meaning, while this is feature
>  complete, it is still open to changes and may have outstanding
>  issues."

This is one of those outstanding issues.

## Lever sizing — what we'd recover if it worked

Per CN-80B-23, the unaccounted ~7 ms/token host overhead on Coder-Next
pp2tp2 decode is roughly `1500 launches × 5 µs driver dispatch`. At
~45 tok/s baseline (~22.2 ms/tok), 7 ms is **~31% of the per-token
budget**. A working multi-stream graph capture would therefore deliver
at most a ~1.45× speedup on this topology — a meaningful win, but
gated entirely on ROCm fixing the beta primitive.

Per-rank PP-only graph capture was already measured null on Coder-Next
in the prior CN-80B-20 attempt (see
`memory/feedback_decode_graph_null_coder_next.md`): the per-rank kernel
chain dispatch overhead is comparable to what eager dispatch costs
already, so per-stream-isolated capture eliminates the wrong overhead.
The cross-rank glue is where the host time lives.

## Code in tree (forward-compatible)

- `crates/backend-hip/src/sys.rs` — FFI bindings for
  `hipStreamBeginCaptureToGraph` and `hipGraphCreate`.
- `crates/backend-hip/src/device.rs` — `HipGraphExec::capture_into_shared_graph`
  implements the multi-stream capture path. Currently surfaces a
  documented error every time on ROCm 7.1.1; works as-is once the
  runtime fixes the beta API.
- `crates/models/qwen3-moe/src/hybrid.rs` —
  `ShardedForwardOneTokenScratchHybrid::decode_graphs: Vec<Option<HipGraphExec>>`
  one shared graph per stage.
- `crates/models/qwen3-moe/src/forward/hybrid.rs` —
  `forward_one_token_hybrid_inner` capture/replay branch gated on
  `FLAMBEAU_DECODE_GRAPH=1`. Falls through to eager on capture failure.
- `crates/models/qwen3-moe/tests/coder_next_pp2tp2_decode_graph_ab.rs`
  — A/B harness, `FLAMBEAU_AB_ONLY` env var to isolate eager from graph.

## Decision

- **Do not enable `FLAMBEAU_DECODE_GRAPH` on TP/hybrid topologies under
  ROCm 7.1.1.** It will (a) error out and (b) corrupt the streams so
  no further decode can run.
- Re-test the moment a newer ROCm rev lands. The framework wiring is
  ready.
- Move CN-80B-20 to the V2 backlog as **"revisit on ROCm > 7.1.1"**.

## Reference: end-capture trace

```
[capture_into_shared_graph] end capture stream 0
[capture_into_shared_graph] end stream 0 -> code 904 out_graph 0x7bbcb8ff5650
```
(out_graph non-null + return 904 + handle unusable for hipGraphGetNodes —
that's the bug.)
