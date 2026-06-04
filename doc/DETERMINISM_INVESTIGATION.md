# TP AllReduce Non-Determinism Investigation

**Date:** 2026-06-04
**Branch:** feature/tool-calling-fixes
**Trigger:** gemma-4-26B-A4B-Q8_0 on pp2tp2 intermittently degraded
into scaffolding/garbage spray ("ok the weights are shit?"). Same
prompt, same `temperature=0`, different output run-to-run.

## TL;DR

The forward pass is **not bit-deterministic under TP**. The
non-determinism originates in the **BAR1 P2P TP AllReduce read path**
(`crates/forward/src/runtime/ar.rs` → `sum_tp{2,4}_f32_rank`), not the
weights, the sampler, the tokenizer, the PP stage boundary, or any
per-request slot state. At `temperature=0` the sampler is pure argmax
with no RNG, so the only way two greedy runs diverge is the logits
themselves differing — and they do, because the AllReduce produces
ULP-jittered sums run-to-run.

On **confident** tokens the flip is invisible (argmax margin >> AR
noise). On **near-tied** tokens — exactly the regime gemma-4-26B-A4B
spends time in once it starts a terse/open-ended reply — the noise
flips the argmax, and one wrong token cascades into full scaffolding
spray (e.g. `c-a-t (cat) 运动控制系统…`).

**There is one bug, not two.** An earlier 6-sample run made it look
like forcing the host-sync AR path produced a clean period-2
alternation (suggesting a second, request-parity bug). Re-running at 20
samples showed that "period-2" was a small-sample illusion: the
host-sync path is still **multi-modal random** (6–10 distinct outputs
per 20 runs). Host-sync fixes the *producer* half of the ordering but
the non-determinism survives, which localizes the bug to the
**consumer / reader side** of the BAR1 transfer.

## Experiment matrix (all gemma-4-26B-A4B-Q8_0, temp=0, seed=0)

| Topology | AR path | Prompt | Result |
|----------|---------|--------|--------|
| pp-only (no TP AR) | n/a | terse | deterministic + clean (EXP8b, ×6) |
| tp2 (hip:0,2) | host-sync | confident¹ | **20/20 identical, clean** (`9cc96483`) |
| tp2 (hip:0,2) | host-sync | terse² (near-tie) | **6 distinct / 20 — random** |
| pp2tp2 (hip:0,2,1,3) | host-sync | confident¹ | **20/20 identical, clean** (`9cc96483`) |
| pp2tp2 (hip:0,2,1,3) | host-sync | terse² (near-tie) | **10 distinct / 20 — random** |
| pp2tp2 | event-path (default) | terse | fully random (original symptom) |

¹ "confident" = `"Spell the word cat letter by letter, then count to
five."` → `C-A-T. 1, 2, 3, 4, 5.`
² "terse" = `"Spell cat"` → near-tied continuation after `c-a-t (cat)`.

"host-sync" = the diagnostic binary with `EVENT_PATH_MAX_ELEMS = 0`,
which forces every AR (including decode-shape) through
`ar_publish_with_host_sync` (`Stream::synchronize` before publish).

### What the matrix proves

- **Source = TP AllReduce.** pp-only (no TP AR) is deterministic and
  clean. Any TP topology (tp2 *or* pp2tp2) is non-deterministic on
  near-tie prompts.
- **Not PP-boundary, not slot state, not mixed-batch.** The bug
  reproduces on a single pure-TP2 cluster with one stage and
  `--inflight-slots 2`. `claim_slot_blocking` hands sequential requests
  slot 0 every time, and `reset_for_next_request` only resets GDN state
  (a no-op for gemma4, which has no recurrent layers) — yet identical
  back-to-back prompts still diverge. The state that differs is on the
  device, inside the AR, not in any Rust-side per-request structure.
- **Producer ordering is insufficient.** Host-sync drains each rank's
  producer stream before the barrier, so all partials are committed to
  HBM before any peer reads them. The output is still random → the race
  is on the **reader** side: the consumer kernel reads peer partials
  across the BAR1 aperture without an acquire fence and can observe
  stale / not-yet-coherent peer data even after the producer drained.
- **Invisible on confident tokens.** Both topologies are 20/20
  identical and coherent on the confident prompt — the ULP jitter never
  crosses an argmax margin there. The bug only ever surfaces as a token
  flip on near-ties; gemma4-MoE hits near-ties constantly on
  open-ended/list replies (flatter logits than dense 31B-Q4_0, which
  rounds out spikes and runs hotter margins — same canary pattern as
  `feedback_gemma4_attn_output_proj_f16_saturate` /
  `feedback_gemma4_moe_f16_overflow`).

## Root cause

`bar_ar_sum_f32` → `sum_tp{2,4}_f32_rank` reads peer `partial` buffers
through the BAR1 PCIe aperture. Neither the producer nor the consumer
issues a system-scope (`__threadfence_system()`) fence around those
cross-device reads. On gfx906 BAR1 P2P, "the producer kernel
completed" / "the producer stream drained" does not imply "the
consumer's view of peer HBM is coherent" — the reader can pull stale or
partially-updated bytes from its own L2 / the aperture. The result is
ULP-scale jitter in the summed partials, run-to-run.

Note: 2-operand float addition is *commutative* (`p0+p1 == p1+p0`), so
for the tp2 path the per-rank accumulation order is **not** the source
— ruling out "unpinned reduce order" for tp2. The jitter is in the
*operands themselves* being read non-coherently, not the order they are
summed.

This is the same bug class as `feedback_bar_p2p_sum_write_target`
("peer BAR1 reads aren't ordered against the peer's writes"), but one
layer deeper: there the fix was a producer-side `producer_done_event`;
here even full producer drain is not enough, because the missing
ordering is a consumer-side *acquire*, not a producer-side *release*.

## Attempted fix #1 — reader-side coherent load (NULL, 2026-06-04)

Hypothesis: the reader's L2 holds a stale line for the BAR1-mapped peer
address (the partial buffers are reused every layer), so a coherent
(glc) load on the consumer would fix it. Implemented by marking every
peer pointer in the AR kernels (`p2p_allreduce_residual*.cu`,
`p2p_allreduce_residual_rmsnorm*.cu`) `volatile` — staging F16/__half2
reads through a width-matched integer load + bit-cast helper
(`p2p_coherent_load.cuh`) since HIP's `__half2` can't copy-construct
from a volatile lvalue. `volatile` global loads emit the `glc` bit on
gfx906.

Result — **NULL**. No measurable change:

| Config | Without fix | With reader-glc |
|--------|-------------|-----------------|
| pp2tp2, event path | (random) | **8 distinct / 20** |
| pp2tp2, host-sync | 10 distinct / 20 | **6 distinct / 20** |

Reader-glc combined with host-sync (full producer drain to HBM) is
still 6 distinct / 20 — indistinguishable from host-sync alone. The
change was reverted (it adds a glc/L2-bypass cost for zero benefit).

**What the null tells us:** the bug is *not* reader-side L2 staleness,
and *not* producer-kernel-completion ordering — it survives both a
reader acquire *and* a full producer `Stream::synchronize`. The only
remaining gap is a **producer-side L2→HBM writeback**: `synchronize`
guarantees the producer *kernel finished*, but on gfx906 the partial
can still sit in the *producer's* L2 unwritten-back, while a peer reads
the producer's *HBM* through the BAR1 aperture and gets stale bytes. No
reader-side fence can close that.

## Attempted fix #2 — producer-side L2 writeback (NULL, 2026-06-04)

Hypothesis from the #1 null: the partial sits in the *producer's* L2,
unwritten-back, so a peer's BAR1 read of the producer's HBM gets stale
bytes. Implemented a coordinator-level writeback: a dtype-agnostic
kernel (`flambeau_p2p_l2_writeback`) that re-stores every 32-bit word of
the partial through a `volatile` store and then issues
`__threadfence_system()` (system-scope release), launched on each rank's
own stream after the producer `synchronize` and before peers read
(threaded a byte-count through `ar_publish_with_host_sync` + all 5 AR
entry points; forced via `EVENT_PATH_MAX_ELEMS = 0`).

Result — **NULL**. pp2tp2 host-sync + L2-writeback: **10 distinct / 20**
— no improvement over host-sync alone. Reverted.

## Re-opened diagnosis (both AR fixes are null)

Two AR-targeted fixes — reader-side acquire (#1) and producer-side
release/L2-writeback (#2) — are *both* null, on top of host-sync (full
producer drain) also being null. A reader acquire, a producer drain,
*and* an explicit producer L2-writeback-plus-system-fence all fail. That
is strong evidence the bug is **not the BAR1 read/visibility of the
partial at all.**

The pp-only-vs-pp2tp2 contrast localizes the bug to the **TP path
broadly**, which is *both* the AR *and* the sharded row-parallel matmuls
that **produce** the partials. The AR has now been heavily ruled out, so
the prime remaining suspect is **non-deterministic partial production
under TP** — e.g. an `atomicAdd`-based or otherwise order-unstable
reduction in the TP-sharded attention-output / FFN-down / MoE-down
projection kernels that does not run (or runs deterministically) in the
single-device pp-only path.

**The diagnostic that actually localizes it (next step, not yet run):**
checksum each rank's `partial` buffer *before* the AR sum, across N
identical runs. If the per-rank partials differ run-to-run → the
producer (sharded matmul) is non-deterministic and the AR is exonerated.
If the partials are bit-identical but the post-AR result differs → it is
genuinely the AR. This converts blind kernel-patching into a localized
fix and should precede any further AR or producer kernel change.

## Partial-checksum diagnostic — DECISIVE (2026-06-04)

Instrumented the AR entry points to DtoH-copy and FNV-hash the local
partial *before* the AR sum (`AR-CHK`) and the result *after* (`AR-OUT`),
per-rank-sequenced. The DtoH is stream-ordered after the producer, so it
reads the true produced value. Ran the terse prompt twice on pure tp2
(`hip:0,2`, unambiguous 2-thread rank labels), `max_tokens=1`.

gemma4-26B-A4B's decode AR runs entirely through `bar_ar_sum_f32`
(`sum_tp2_f32_rank`), 292864-byte (73216-F32) payload — the host-sync
path (`n > EVENT_PATH_MAX_ELEMS`).

**Result at the very first AR call (#0), both ranks:**

| | rank0 | rank1 |
|---|---|---|
| INPUT partial (DtoH) | `b27c232b…` **run1 == run2** | `7d3db574…` **run1 == run2** |
| OUTPUT (DtoH, post-sum) | `208cc4e8` (run1) vs `90620e0d` (run2) — **DIFF** | `90620e0d` (run1) vs `3daeb3de` (run2) — **DIFF** |

Two airtight conclusions:

1. **The producer is exonerated.** Both input partials are *bit-
   identical* across runs at the first AR — partial production
   (the sharded matmuls) is deterministic. This kills the "non-
   deterministic partial production" hypothesis from the re-opened
   diagnosis above.

2. **The AR is confirmed as the source.** Identical inputs → different
   outputs run-to-run. And within a *single* run the two ranks
   **disagree on the sum**: rank0=`208cc4e8`, rank1=`90620e0d`.
   `sum_tp2_f32_rank` computes `p0+p1` on rank0 and `p1+p0` on rank1 —
   commutative, so they *must* be bit-identical. Two ranks getting
   different results from identical operands is **direct proof the BAR1
   cross-device peer read is incoherent**: at least one rank's BAR1 view
   of the peer's partial ≠ the peer's true partial (which DtoH reads
   correctly).

This **reinstates the AR** as the culprit and explains why attempts #1
and #2 were null — they were measured by output-text determinism but
targeted the wrong sub-mechanism. The mechanism is now precise: the
producer's partial is correct in its own L2/DRAM (its own DtoH reads it
fine, deterministically), but a **peer reading that partial through the
BAR1 PCIe aperture observes a stale/incoherent value** — and neither a
reader-side `glc` load (#1) nor a producer re-store + `__threadfence_
system()` (#2) made that aperture read coherent on gfx906. The likely
physical cause: the producer's write is resident in its L2 (which its
own DtoH and same-device reads see) but the BAR1 aperture maps DRAM, and
the L2→DRAM writeback that a peer needs is not produced by
`threadfence_system` on this silicon.

## Fix SHIPPED — `--deterministic` host-bounce AR (2026-06-04, gate PASSED)

Implemented fix direction #1 below. New server flag `--deterministic`
(env `FLAMBEAU_DETERMINISTIC`, off by default) sets
`LaunchParams.deterministic_ar`; the TP/Hybrid launchers then build
`bar = None`, so the F32 AR callback falls back to the host-bounce
`ArCoordinator` (DtoH → CPU sum in fixed rank order → HtoD) and every
BAR-only fast path (`ar_sum_f16`, `ar_residual_f16`, postattn fused)
reports unsupported and uses the host-bounce split. No BAR1 aperture
read on any AR path.

**GATE PASSED:** gemma-4-26B-A4B-Q8_0, pp2tp2 (`hip:0,2,1,3`), `--kv q8`,
`"Spell cat"` temp=0 ×20 → **20/20 identical md5** (`e6b35b31`), output
coherent (`c-a-t (cat)…`). Without the flag the same prompt gave 6–10
distinct hashes. Commit `c06b189`. Off by default — it trades the
DtoH/HtoD decode-throughput cost for determinism; enable per deployment.

**Throughput cost (measured 2026-06-04).** Same GGUF / GPUs / driver /
prompt (~3020-char, pp ≥ 512), tg = 128, streaming decode-rate isolation
(inter-token interval, excludes TTFT), 5-run median:

| AR path | decode tps | TTFT |
|---------|-----------|------|
| default (BAR1 P2P) | **40.72** (40.10–41.17) | ~1.19 s |
| `--deterministic` (host-bounce) | **30.57** (28.66–31.25) | ~1.84 s |

≈ **−25 % decode throughput** (BAR1 is 1.33×) and ~**+0.65 s TTFT**
(prefill ARs also route host-bounce). The cost is the DtoH→CPU-sum→HtoD
bytes per AR call that BAR1 P2P exists to avoid — which is exactly why
fix direction #2 (a real producer L2→DRAM writeback keeping the on-device
BAR1 path) is the long-term win if determinism is needed without the hit.

## Fix directions (post-diagnostic)

1. **Host-bounce AR for the affected path (guaranteed-correct fallback —
   SHIPPED above as `--deterministic`).** The diagnostic *proves* DtoH reads the
   true partial deterministically. The existing host-bounce coordinator
   `ArCoordinator::ar_sum_f32` (DtoH → CPU sum in fixed rank order →
   HtoD) sidesteps the BAR1 aperture entirely and is therefore
   deterministic by construction. Route gemma4's TP AR (or all TP AR
   when a determinism mode is requested) through it instead of
   `BarArCoordinator`. Cost: the DtoH/HtoD bytes BAR1 was introduced to
   avoid — a real decode-throughput hit, so gate it (per-arch or a
   `--deterministic` flag), don't make it unconditional.

2. **Make the BAR1 aperture read coherent (proper fix, research-grade).**
   Force the producer's partial out of L2 to DRAM with a primitive that
   actually does an L2→DRAM writeback on gfx906 (candidates: a
   write-through / streaming store on the *producer* so the partial
   never caches in L2; an explicit `buffer_wbinvl2` via inline asm;
   uncacheable mapping of the partial pool). `__threadfence_system()`
   was insufficient (#2). Needs hardware-doc spelunking + the sweep
   harness; multi-session.

The acceptance gate is unchanged: near-tie prompt (`"Spell cat"`) decoded
×20 at temp=0 collapsing to a single md5.

## Earlier fix directions (superseded)

1. **Producer-side system-scope release** (the real fix, per the null
   above). Each kernel that produces an AR partial (attention output
   proj, MoE/FFN down proj, …) must `__threadfence_system()` after its
   final store *or* the AR path must insert an explicit L2 writeback
   (`__builtin_amdgcn_s_dcache_wb` / `buffer_wbinvl2`-equivalent, a
   tiny per-rank launch) between producer and the peer reads. This is
   the invasive part — it touches every producer feeding an AR, or adds
   a flush op to the AR coordinator — which is why it is multi-session,
   cert-gated kernel work (CLAUDE.md rule 2 + "ask before destructive
   actions"). A reader-side acquire alone is proven insufficient.

2. **Pin accumulation order** in `sum_tp4_f32_rank` (tp4 only — tp2 is
   commutative). Complementary; not sufficient alone for the cases here.

3. **Host-sync everywhere** (`EVENT_PATH_MAX_ELEMS = 0`) is **NOT a
   fix** — it leaves the output multi-modal random and costs the
   decode-decoupling lever. Do not ship it as a correctness toggle.

The acceptance gate for any fix: the near-tie prompt (`"Spell cat"` on
gemma-4-26B-A4B-Q8_0) decoded ×20 at temp=0 must collapse to a single
md5. Confident prompts already pass with the bug present.

## A note on the determinism regression test

Whatever the fix, the acceptance gate is a near-tie prompt
(`"Spell cat"` on gemma-4-26B-A4B-Q8_0 reproduces reliably) decoded ×20
at temp=0 collapsing to a single md5. Confident prompts pass even with
the bug present, so they are useless as a gate — the cert harness must
use a prompt that sits on an argmax knife-edge.

## What was reverted

The diagnostic `EVENT_PATH_MAX_ELEMS = 0` edit in `ar.rs` was reverted
to `65_536`. It was a probe to mask the producer-ordering variable, not
a ship.

## Reproduce

```
# diagnostic binary (forces host-sync so the result isn't confounded
# by the event-path producer race — though both paths are non-det):
#   set EVENT_PATH_MAX_ELEMS = 0 in crates/forward/src/runtime/ar.rs
cargo build --release -p flambeau-cli --features flambeau-cli/hip_serve

# minimal repro is pure TP2 (no PP needed):
flambeau serve --model <gemma-4-26B-A4B-it-Q8_0.gguf> \
  --mesh-mode tp --tp-size 2 --devices hip:0,2 \
  --kv q8 --ctx-cap 2048 --port 8093

# fire the near-tie prompt ×20, md5 each response body:
#   {"messages":[{"role":"user","content":"Spell cat"}],
#    "temperature":0,"max_tokens":64,"seed":0}
# -> 6+ distinct hashes / 20. Swap to --mesh-mode pp --devices hip:0,2
#    (pp-only, no TP AR) -> single hash, coherent.
```
