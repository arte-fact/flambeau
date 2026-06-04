# TP AllReduce Non-Determinism Investigation

**Date:** 2026-06-04
**Branch:** feature/tool-calling-fixes
**Trigger:** gemma-4-26B-A4B-Q8_0 on pp2tp2 intermittently degraded
into scaffolding/garbage spray ("ok the weights are shit?"). Same
prompt, same `temperature=0`, different output run-to-run.

## TL;DR

The forward pass is **not bit-deterministic under TP**. The
non-determinism originates in the **BAR1 P2P TP AllReduce**
(`crates/forward/src/runtime/ar.rs`), not the weights, the sampler, or
the tokenizer. At `temperature=0` the sampler is pure argmax with no
RNG, so the only way two greedy runs diverge is the logits themselves
differing — and they do, because the AllReduce sums partials in a
non-deterministic way and/or reads peer partials before they are
visible across the BAR1 aperture.

On **confident** tokens the flip is invisible (argmax margin >> AR
noise). On **near-tied** tokens — exactly the regime gemma-4-26B-A4B
spends time in once it starts a list/structured reply — the noise
flips the argmax, and one wrong token cascades into full scaffolding
spray.

Two distinct bugs were isolated:

- **Bug A — event-path BAR1 visibility race.** The small-`n` event
  ordering path (`ar_publish_with_events`, used when
  `n_elems ≤ EVENT_PATH_MAX_ELEMS = 65_536`, i.e. *every decode step*)
  lets a rank read a peer's `partial` via BAR1 before that peer's
  write is guaranteed visible across the aperture. `record`/
  `stream_wait` orders the *kernel completion*, not the *cross-device
  memory write becoming visible to the reader*. Forcing the host-sync
  path (`ar_publish_with_host_sync`, full `Stream::synchronize`)
  removed the fully-random behaviour.

- **Bug B — period-2 parity alternation.** With the host-sync path
  forced, output stopped being fully random but became **period-2
  alternating**: odd runs clean, even runs sprayed (two stable
  hashes). A second, lower-frequency non-determinism remains —
  request-parity-dependent, consistent with inflight-slot round-robin
  or mixed-batch state leaking between back-to-back requests. Not
  isolated this session (the `--inflight-slots 1` boot kept failing on
  sandbox GPU contention).

## Experiment table

| Exp | Path | Config | Result |
|-----|------|--------|--------|
| EXP1/2 | `/v1/completions` | pp2tp2, confident prompt | deterministic ×N |
| EXP5/6 | `/v1/chat` temp=0 | pp2tp2, structured reply | **non-deterministic** (fully random) |
| EXP7 | `/v1/chat` temp=0 | pp2tp2, confident prompt | deterministic (margin hides it) |
| EXP8b | `/v1/chat` temp=0 | **pp-only (no TP AR)** | **deterministic + clean ×6** |
| EXP9 | `/v1/chat` temp=0 | pp2tp2 + **host-sync AR** (`EVENT_PATH_MAX_ELEMS=0`) | random → **period-2 alternating** (clean `a4fe4a4b` vs sprayed `a9045126`) |
| EXP10 | `/v1/chat` temp=0 | pp2tp2 host-sync + `--inflight-slots 1` | **inconclusive** — server boot failed (sandbox GPU contention) |

The decisive contrast is **EXP8b vs EXP5/6**: identical model, prompt,
sampler, KV layout; the *only* difference is whether the TP AllReduce
runs. pp-only is clean and reproducible; pp2tp2 is neither.

EXP9 is the decomposition: switching the AR producer-ordering from the
event path to the host-sync path collapsed the entropy from "fully
random" to "two states" — that delta is Bug A; the residual two states
are Bug B.

## Why this manifests as garbage, not just non-reproducibility

`temperature=0` → greedy argmax → no RNG. Float-sum non-associativity
in the AllReduce (`sum_tp2_f32_rank` accumulation order across BAR1
reads is not pinned) produces logit deltas at the ULP scale. That is
harmless until two top candidates are within that delta. gemma-4-26B-
A4B (small active params, MoE routing) produces flatter logit
distributions on structured/list tokens than the dense 31B, so it hits
the near-tie regime constantly. Q4_0 31B "works" partly because it
rounds out the spikes and runs hotter margins; Q8_0 26B-A4B preserves
them. This matches the existing memory note pattern
(`feedback_gemma4_attn_output_proj_f16_saturate`,
`feedback_gemma4_moe_f16_overflow`): gemma4-MoE is the canary for
numeric-margin bugs that the dense path survives.

## Root cause (Bug A, well-supported)

`ar_publish_with_events` (ar.rs:173):

```
record event on own stream
publish partial ptr to shared slab
host barrier
stream_wait on each peer event
launch sum_tp{2,4}_f32_rank (reads peer partials via BAR1)
```

`HipEvent::record` + `stream_wait` enforce that the *producer kernel
has completed* before the consumer kernel launches. They do **not**
enforce that the producer's writes to its `partial` buffer are
*visible to a peer reading them across the BAR1 PCIe aperture*. On
gfx906 BAR1 P2P, kernel-complete and cross-device-write-visible are
not the same barrier — there is no system-scope release fence between
them on this path. The host-sync path happens to work because
`Stream::synchronize` + the subsequent host barrier inserts enough
ordering (and a full device drain) that the writes have landed.

This is consistent with the existing AR memory notes:
`feedback_bar_p2p_sum_write_target` (callers must synchronize every
rank's stream before AR because "peer BAR1 reads aren't ordered
against the peer's Phase-1 writes otherwise") documents the **same
class of bug** on the write-target side — qwen3-moe handled it with a
`producer_done_event`, the gemma4 TP path did not. The event here is
present but insufficient: it orders execution, not memory visibility.

## Fix directions (NOT implemented — needs sweep cert + the user's call)

1. **System-scope release fence in the producer / acquire in the
   consumer** (correct fix). The `sum_tp{2,4}_f32_rank` kernel needs a
   `__threadfence_system()`-equivalent acquire on the peer-pointer
   reads, and the producer needs a system-scope release after writing
   its partial. This is the real fix — it keeps the event fast path
   (no host `Stream::synchronize`, no decode perf regression) while
   making the BAR1 read see committed peer writes. Kernel change +
   correctness sweep + a determinism regression test (md5 of greedy
   decode ×N must be identical). Multi-session, cert-gated.

2. **Pin the accumulation order** in `sum_tp{2,4}_f32_rank` so the
   float sum is associative-stable run-to-run (rank 0 + rank 1 +
   ... in fixed order, no atomics). Removes the ULP-jitter even if
   visibility were perfect. Cheap, complementary to (1).

3. **Ship the host-sync path for decode** (stopgap, NOT recommended as
   the final answer). Setting `EVENT_PATH_MAX_ELEMS = 0` made EXP9
   period-2 instead of random — i.e. it fixes Bug A but costs the
   decode-decoupling win the event path exists for (see the ar.rs
   comment: event path is the gfx906 decode lever). It also does NOT
   fix Bug B. Use only as a temporary correctness-over-perf toggle if
   a release is blocked on this.

Bug B must be isolated first (re-run EXP10 with `--inflight-slots 1`
on a quiet rig; if deterministic → per-slot state carries across
requests; if still period-2 → mixed-batch
(`supports_mixed_batch()` is true for gemma4) leaks state between the
prefill-leader and decode followers).

## What was reverted

The diagnostic `EVENT_PATH_MAX_ELEMS = 0` edit in `ar.rs` was reverted
to `65_536`. It was a probe (fix-direction 3), not a ship — it carries
a decode perf cost and only addresses Bug A.

## Reproduce

```
cargo build --release -p flambeau-cli --features flambeau-cli/hip_serve
flambeau serve --model <gemma-4-26B-A4B-Q8_0.gguf> \
  --mesh-mode pp+tp --pp-size 2 --tp-size 2 --devices hip:0,2,1,3 \
  --kv q8 --ctx-cap 2048 --port 8080
# fire the same temp=0 chat completion ~6× under a structured prompt,
# md5 the response bodies — they will not all match.
# Then re-run with --mesh-mode pp --devices hip:0,2 (pp-only):
# all 6 match and stay coherent.
```
