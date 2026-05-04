# Handoff — 2026-05-04 (part 5): #294 PP ≥ 2 refactor + new PP=4 blocker

Continuation of `handoff_2026_05_04_part4.md`. Part 4 shipped the
shape-aware `mmq_q4_1_wave64` routing for batched-MMVQ at GDN-out
shapes. The 3× cert gate combined ceiling stood at ~1.93×; pure-PP=4
pipelining (2.3-2.9× ceiling at N≥4) was queued as the path to clear
3× cleanly.

## What this session shipped

| commit  | task | result |
|---------|------|--------|
| 5b5afe8 | **#294** | extend `forward_decode_pipelined_hybrid` to PP ≥ 2 (was PP=2 hardcoded). Regression-tested at PP=2/N=2 — md5 59f23a62 holds. PP=4 e2e validation BLOCKED. |

The function refactor:
- Replaced hardcoded stage-0/stage-1 logic with a `for src_stage in 0..n_stages` loop, peer_copy fan-out between each adjacent pair.
- Lazy bridge-event init now runs on EVERY non-final stage's rank 0 (was only stage_0). Each non-final stage holds N events.
- `routes.rs` gate: `n_stages == 2 && n >= 4` → `n_stages >= 2 && n >= 4`.

PP=2 regression check (live, Qwen3.6-27B / pp2tp2 / SLOTS=2 / N=2):
- N=1: md5 59f23a62 ("thunderous sound")
- N=2 both slots: md5 59f23a62 (matches N=1)

No regression. The refactor preserves PP=2 correctness.

## New blocker found: PP=4/TP=1 hybrid path hangs

Tried PP=4/TP=1 with `FLAMBEAU_CTX_CAP=8192` to fit KV under the
16GB-per-MI50 budget at SLOTS=4. Server boots successfully (model
loaded, /health responds OK). But the FIRST `/v1/chat/completions`
request to a tiny prompt hangs indefinitely (>7 minutes observed
before kill).

Critical observation: **the hung request was the cert harness's
warmup call at N=1**, which falls through to
`forward_decode_batched_hybrid` (or its prefill) — NOT the pipelined
function. So the hang is in the **existing PP=4/TP=1 hybrid path**,
not in this session's #294 refactor.

PP=4 has previously only been live-validated in pure-PP mode
(`--mesh-mode pp`), which loads `LoadedModel::Pp`. The
`LoadedModel::Hybrid` arm at PP=4/TP=1 (i.e., `--mesh-mode pp+tp
--pp-size 4 --tp-size 1`) appears to be untested — and broken in some
way that surfaces as a hang during prefill or decode.

To validate pipelining at PP=4, this hang needs to be debugged first.
Likely candidates (none verified):
1. Prefill at PP=4/TP=1 hybrid: maybe the per-stage layer iteration
   or the stage-boundary peer_copy is broken at TP=1 (sub_cluster has
   1 rank, so AR-residual / collective-sync paths take the world=1
   degenerate arm — possibly a missing init).
2. Decode at PP=4/TP=1 hybrid in `forward_one_token_hybrid_inner`:
   similar path through the per-stage loop.
3. A scheduler / mutex deadlock specific to N=1 + PP=4 + Hybrid +
   `FLAMBEAU_DECODE_PIPELINE=1` interaction (less likely; pipelining
   doesn't engage at N=1 due to the n>=4 gate).

## Where the 3× cert gate stands

Unchanged from part 4:

| lever                       | win                  | status                                |
|-----------------------------|---------------------:|---------------------------------------|
| #266c batched-attn          | 1.05×                | landed                                |
| #287 batched-GDN wired      | 1.03×                | landed                                |
| #290+#294 pipelining        | ≤2.3× at PP=4/N=4    | code shipped, PP=2 holds, PP=4 blocked |
| #288-v2 wave64 routing      | ~1.15× combined      | opt-in; breaks bit-id within batch    |

Combined ceiling with pure-PP=4 + #266c + #288-v2: ~3.5×. Clears 3×
**if PP=4/TP=1 hybrid hang is resolved**.

## Open path forward (priority order)

### NEW pending: debug PP=4/TP=1 hybrid hang

This is the gate to validating the entire pipelining lever. Reproduces
deterministically with:

```bash
FLAMBEAU_BATCHED_DECODE=1 FLAMBEAU_INFLIGHT_SLOTS=4 \
  FLAMBEAU_CTX_CAP=8192 \
  ./target/release/flambeau serve \
  --model /artefact/models/Qwen3.6-27B-Q4_1.gguf \
  --devices hip:0,2,1,3 --mesh-mode pp+tp --pp-size 4 --tp-size 1 \
  --port 8080
# Send any /v1/chat/completions request — hangs > 7 minutes.
```

Bisect plan:
1. Try `--mesh-mode pp` (pure PP=4) to confirm the existing path
   works on this rig.
2. Try `--mesh-mode pp+tp --pp-size 2 --tp-size 2` (pp2tp2 — works in
   prior sessions). Confirms pp+tp path itself is fine.
3. Diff: `pp+tp / pp_size=4 / tp_size=1` is the unique combination.
   Check whether ar_residual / sub_cluster init / stage-boundary
   peer_copy has a TP=1-specific bug.

Likely files: `crates/server/src/serve.rs` (cluster setup), `crates/models/qwen3-moe/src/forward/hybrid.rs` (per-stage forward at TP=1), `crates/models/qwen3-moe/src/hybrid.rs` (model load).

### After PP=4 unblocks: re-cert

Run the 4-conc-vs-4-seq cert recipe at PP=4/N=4 with
`FLAMBEAU_DECODE_PIPELINE=1`. Target ≥ 1.6× speedup (pipelining
ceiling at PP=4/N=4 is 2.3×; per-slot host overhead and other
inefficiencies expected to bring it down to 1.5-1.8× actual).

### v3 wave64-decode kernel (independent track)

Still open from part 4. Forces FMA contraction off OR symmetrically
unrolls the per-col loop — preserves 1.27× wave64 win AND maintains
bit-identical-within-batch.

## Diagnostic toggles updated

- `FLAMBEAU_CTX_CAP=<N>` — already existed; clamps the model's KV-cache
  ceiling (handy at PP=4/TP=1 where default 128k ctx → 8GB/slot KV
  quickly exceeds 16GB-per-MI50).

(Plus all toggles inherited from prior handoffs.)
