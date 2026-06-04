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

## Fix directions (refined; NOT implemented — cert-gated, user's call)

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
