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

## Determinism made unconditional — flag removed (2026-06-05)

Chased a zero-cost always-on path (so the `--deterministic` flag could be
dropped without the DtoD overhead). Two decisive nulls:

- **Cheap in-kernel fix (flush + `glc` reader load) is insufficient for the
  in-place F32 sum.** gemma4-26B-A4B with the F32 sum kernel's peer read
  marked `volatile` (glc) *and* the partial flushed L2→DRAM before the
  in-kernel read: **5 distinct / 20** (16/20 land on the correct value).
  The gfx906 PCIe BAR1 *aperture read itself* isn't reliably coherent even
  from fresh DRAM, and the in-place `partial_local += peer` has a
  cross-rank read-vs-write race that — unlike the DtoD copy — can't be
  fenced read-all-then-write-all (read and write are fused in one kernel).
  Only the copy engine is fully coherent.
- **The F16 path was never genuinely coherent — false-green by margin.**
  Qwen3.5-27B-Q4_0 (flat quant → flatter logits → more near-ties) on the
  **plain BAR1** F16 path, open-ended prompt: **9 distinct / 16**. The
  earlier "Qwen F16 is already deterministic" was Q8 confident-prompt
  margin luck; the F16 DtoD extension was necessary, not defensive.

**Decision (user):** since there's no zero-cost path, make the coherent
DtoD the **unconditional default** and **remove the flag** (no fence
trim). `deterministic_ar` is hardcoded true; the `--deterministic` CLI
flag + `FLAMBEAU_DETERMINISTIC` env + `ServeConfig.deterministic` are
deleted. The DtoD cost (gemma4 −5.5%, Qwen −7.4%) is now always paid — the
price of coherent cross-device AR on this silicon. Commit `ab1fc3a`.

GATES (no flag, default): gemma4-26B-A4B-Q8_0 "Spell cat" ×6 → 1 md5;
Qwen3.5-27B-Q4_0 (the flat-quant false-green) open-ended ×16 → 1 md5
(was 9/16 on plain BAR1). Vestigial follow-up: `LaunchParams.deterministic_ar`
+ the non-dtod BAR1 branches + `coord.dtod` are now reachable only from
the 9 model-crate test constructions (still `false`); a cleanup can drop
the bool + dead branches and align tests to the coherent path.

## Squeeze + scope refinement (2026-06-05)

Tried to close the residual −5.5 % gemma4 gap and characterized the
determinism scope precisely.

**Barrier-reduction squeeze — NULL (reverted).** The DtoD path runs 4
host-barrier rendezvous/AR (publish, pull-done, 2×epilogue). Dropping the
redundant epilogue slab-clear + barrier (the pull-done barrier is already
a valid post-snapshot rendezvous) *looked* safe and stayed within-session
deterministic (20/20) — but the result md5 **changed** (`e6b35b31` →
`1da90277`), i.e. it computed a different AR. The epilogue barriers are
load-bearing for cross-call ordering in a way the static analysis missed
(host-thread interleaving / memory ordering around the `DevicePtr` slab on
this rig). Reverted — correctness over ~5 %.

**Determinism scope is WITHIN-SESSION, not cross-process.** The shipped
event-path dtod is deterministic within a running server (10/10, 20/20,
12/12) — the actual requirement (the original bug was non-reproducibility
*within* a session). But the near-tie token can differ *across server
restarts* (`e6b35b31` vs `1da90277` = `_ ` vs `_\n` after `c-a-t (cat)`).
Localized decisively:
- pp-only (no AR): cross-boot **stable** (`9fbf4085` ×2 boots).
- prefill AR (host-sync path): cross-boot **bit-identical** (input
  partials + outputs, via DtoH checksum ×2 boots).
- decode AR (event path): cross-boot **variable** — the event ordering
  (`stream_wait` on the producer event) does **not** guarantee the
  partial is flushed to DRAM before the copy engine reads it, so the
  copied value is timing-(boot-)dependent.
- forcing host-sync for *all* dtod (`EVENT_PATH_MAX_ELEMS = 0`):
  cross-boot **stable** (`1da90277` ×2 boots) and coherent — confirming
  the event path is the source. The host-sync result is the *correct*
  one; the event path occasionally freezes on a slightly-stale near-tie.

**Speed/coherence tradeoff (gemma4-26B-A4B-Q8_0 pp2tp2, tg128 median):**

| AR path | decode tps | determinism | coherence |
|---------|-----------|-------------|-----------|
| BAR1 (default, non-det) | 40.72 | none | broken |
| **event-path dtod (shipped `--deterministic`)** | **38.50** (−5.5 %) | within-session | mild decode residual |
| host-sync dtod (`EVENT_PATH_MAX_ELEMS=0`) | 32.06 (−21 %) | within-session **+ cross-boot** | full |
| host-bounce | 30.57 (−25 %) | full | full |

The shipped event-path dtod's speed comes precisely from skipping the
per-AR `Stream::synchronize`; full cross-boot coherence via host-sync
costs 4× the penalty (−21 % vs −5.5 %). The cheaper proper fix landed —
see below.

### Targeted L2→DRAM flush — cross-boot determinism at event-path speed (2026-06-05, SHIPPED)

A tiny per-rank flush kernel (`flambeau_p2p_l2_flush`) re-stores every
word of the partial — `volatile`, so the value becomes the flush
thread's own write, since a separate kernel's `__threadfence_system()`
cannot flush the *producer* kernel's writes — then issues
`__threadfence_system()` to release past L2 to DRAM. It is launched on
the producer stream before the producer event is recorded, **only on the
event (decode) path** (the host-sync prefill path already drains). The
peer's copy-engine pull then sources the producer's value coherently
from DRAM. Confirms `__threadfence_system()` *does* do an L2→DRAM
writeback on gfx906 when the data is the fencing thread's own write
(fix #2's null was the wrong context).

**GATES (pp2tp2, `--kv q8`, `"Spell cat"` temp=0):**

| Model | cross-boot determinism | decode tps | vs BAR1 |
|-------|------------------------|-----------|---------|
| gemma4-26B-A4B-Q8_0 | **6/6 boots identical** (event path was 2/3) | 38.16 | −6.3 % |
| Qwen3.6-27B-Q8_0 | **2/2 boots identical** | 26.43 | −2.7 % |

So `--deterministic` is now **within-session AND cross-process
deterministic + fully coherent** (matches the host-sync ground-truth
result) at essentially event-path speed — the −21 % host-sync penalty is
gone. `flambeau_p2p_l2_flush` (`bar_p2p.rs::l2_flush`, called from
`dtod_ar_sum_f32`); no dispatch row / cert (a fixed backend utility
kernel like the AR sum kernels). The earlier within-session-only
characterization above is superseded.

### Extended to the F16/fused AR paths (2026-06-05, SHIPPED)

The F32 work covered `ar_sum_f32` only. **Correction:** the earlier note
"gemma4/qwen3.6 are F32-only AR" is right for gemma4 (its `post_*_norm`
are set, so decode routes `ar_sum_f32` → dtod) but **wrong for Qwen3.x**:
Qwen (qwen35-v2 GDN hybrid, `post_*_norm` = None) routes decode AR through
the **F16 fused** paths — `ar_residual_rmsnorm_f16` on every GDN+FullAttn
layer, `ar_residual_f16` on every dense-FFN layer — which were wired
directly via the hooks and never consulted `coord.dtod`, so under
`--deterministic` they still did the in-kernel BAR1 read.

The F32 dtod core is now factored into a shared `dtod_publish_pull`
helper (flush + publish + copy-engine pull + pull-done fence;
`peer_bytes_per_elem` sizes the F16 vs F32 flush words / pull bytes), and
all four F16/fused paths (`bar_ar_sum_f16`, `bar_ar_residual_f16`,
`bar_ar_residual_rmsnorm_f16`, `bar_ar_postattn_residual_rmsnorm_f32_to_f16`)
route through it under `coord.dtod`. Zero kernel changes. The residual
paths reassemble the canonical (rank0, rank1) partial order with the
staged peer copy in the peer's slot; the F16 flush word count is
`n_elems.div_ceil(2)`.

So `--deterministic` now routes **every** TP AR (F32 + F16 + fused)
through the coherent DtoD+flush mechanism — a complete coherence
guarantee, not just F32. GATES (pp2tp2, `--kv q8`): Qwen3.6-27B-Q8_0
open-ended prompt ×10 → 1 md5, cross-boot 2/2, output **unchanged** vs the
F16-BAR1 path (verifies the canonical reassembly + F16 dtype sizing are
correct and non-regressive — the two traps the design review flagged),
decode 25.15 vs BAR1 27.17 (−7.4 %, the per-layer residual AR
copy+flush+fence). gemma4 F32 control unchanged (6/6). The conservative
pull-done fence is kept on the out-of-place residual/rmsnorm/postattn
paths (the partial is read-only there, so it's optional — eliding it is
the open perf lever for Qwen's −7.4 %). `ar_sum_f16` + `postattn` have no
current consumer → shipped template-symmetric, not behaviorally gated.

## Fix SHIPPED v2 — `--deterministic` DtoD copy-engine AR (2026-06-04, near-BAR1 speed)

The host-bounce fix below works but costs −25 %/−38 % decode. A research
workflow + GPU validation produced a far better fix that keeps the
throughput: **route the F32 AR through the DMA copy engine instead of the
in-kernel BAR1 shader load.** `--deterministic` now selects this (commit
`685506c`); host-bounce remains only as the no-BAR fallback.

Mechanism: each rank pulls every peer's partial into rank-local scratch
via `hipMemcpyPeerAsync` (`HipDevice::memcpy_peer_in_async`, a consumer-
stream peer read) — the *same copy engine* the diagnostic proved reads
peer memory coherently — then runs the **existing** `sum_tp{2,4}_f32_rank`
kernel with its `peer` arg pointed at that local scratch, so the kernel
reads only coherent rank-local memory. **Zero kernel changes.** Wiring:
`BarArCoordinator { dtod, recv_staging, pull_events }`,
`dtod_ar_sum_f32`, `make_bar_ar_callback` dispatch.

Critical subtlety found on the rig: the AR sum is *in place* (`buf +=
peer`), so a peer can clobber its `buf` with its own sum while another
rank is mid-pull — **~30 % of AR calls raced** (7 distinct/20 on the
first cut). Fixed with a **read-all-then-write-all GPU fence**: each rank
records a pulls-done event after copying, and waits on every peer's
pulls-done event before its in-place sum. The `ar_epilogue` host barriers
cannot do this (they order host threads, not async GPU copies/sums).

**GATES (pp2tp2 `hip:0,2,1,3`, `--kv q8`):**

| Model | determinism | DtoD decode | vs BAR1 | vs host-bounce |
|-------|-------------|-------------|---------|----------------|
| gemma-4-26B-A4B-Q8_0 | **20/20** (matches host-bounce result) | 38.50 tps | −5.5 % (40.72) | +26 % (30.57) |
| Qwen3.6-27B-Q8_0 | **12/12** | 27.12 tps | −0.2 % (27.17) | +61 % (16.82) |

DtoD recovers ~78 %/~100 % of the host-bounce loss while staying
deterministic and coherent (`"Paris."` on a confident prompt). pp≥512
tg128, 5–7-run median. The F16/fused AR paths (`ar_sum_f16`,
`ar_residual_f16`, postattn fused) still use BAR1 — gemma4/qwen3.6 are
F32-only AR (gates confirm), so this is full coverage for them; extending
DtoD to the F16/fused paths is the follow-up for models that use them.

## Fix SHIPPED v1 — `--deterministic` host-bounce AR (2026-06-04, superseded by v2)

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

**gemma-4-26B-A4B-Q8_0** (pp2tp2 `hip:0,2,1,3`, `--kv q8`):

| AR path | decode tps | TTFT |
|---------|-----------|------|
| default (BAR1 P2P) | **40.72** (40.10–41.17) | ~1.19 s |
| `--deterministic` (host-bounce) | **30.57** (28.66–31.25) | ~1.84 s |

≈ **−25 % decode** (BAR1 1.33×), ~+0.65 s TTFT.

**Qwen3.6-27B-Q8_0** (same pp2tp2 / `--kv q8` / prompt / tg):

| AR path | decode tps | TTFT |
|---------|-----------|------|
| default (BAR1 P2P) | **27.17** (25.99–27.60) | ~6.7 s |
| `--deterministic` (host-bounce) | **16.82** (14.38–18.00) | ~7.9 s |

≈ **−38 % decode** (BAR1 1.62×), ~+1.2 s TTFT. The cost is steeper than
gemma4-26B-A4B: the dense 27B issues more full-attention AR calls per
token, so the fixed per-AR DtoH→CPU-sum→HtoD overhead eats a larger
fraction of its (lower) baseline.

The cost is the host-bounce bytes per AR call that BAR1 P2P exists to
avoid — which is exactly why fix direction #2 (a real producer L2→DRAM
writeback keeping the on-device BAR1 path) is the long-term win if
determinism is needed without the hit. The penalty scales with AR-calls-
per-token × hidden, so it is model/topology-dependent (−25 % to −38 %
measured here).

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
