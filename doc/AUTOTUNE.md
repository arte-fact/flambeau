# Autotune — measurement-driven dispatch, with optional crowdsourced tuning DB

**Status:** Draft v0. Plan-only. The third sub-plan of `doc/PORTABILITY-ROADMAP.md`
(with `CUDA-PORT-PLAN.md` and `KERNEL-TEMPLATING.md`). Builds the harness that
turns a variant candidate pool into a per-arch dispatch table **by measurement
instead of hand-tuning**. Aligns with architecture rule 1 (dispatch is a
reviewed, checked-in artifact — not runtime/env selection), rule 2 (cert-gated),
rule 4 (no perf claim before a green cert), and the anti-overfit "coarse table"
principle.

## What it does

Closes a loop that is currently manual: today `bench sweep` produces correctness
certs + PMC → a human reads the numbers → hand-edits `dispatch/<arch>.toml`.
Autotune automates the middle:

```
sweep (correctness)  →  rank (measured perf)  →  emit (dispatch .toml)
   eligibility filter        tiebreak only          reviewed artifact
```

It is **build-time codegen**, not runtime selection: the emitted table is
committed and reviewed (rule 1). The cert is the *eligibility filter* — a
fast-but-wrong kernel is excluded automatically; perf only ranks among already-
certified candidates (rule 4).

## What already exists (~70%)

- **Correctness gate** — `bench sweep` certs each impl vs the CPU dequant + F32
  reference (`max_rel_err` / tolerance); the cert JSON carries `pass` + `pmc`.
- **The candidate filter is in the type system** — `core::op::KernelImpl` has
  `applies(input, cfg) -> bool`, `cert_path()`, `ID`. The autotune inner loop is
  literally: *for a probe shape, collect every impl where `applies()==true`, keep
  those with a green cert, time each, rank.*
- **PMC + measurement discipline** — `bench` captures VGPR/occupancy/MemBusy; the
  CLAUDE.md measurement rules (warmup, ≥512-token prefill, locked clocks, median,
  wall-clock cross-check) are the protocol.
- **Dispatch + cert gate** — `bench/src/dispatch.rs::cert_check` already asserts
  every live dispatch row has a green cert (build-time).

## What's missing (the build)

1. **Candidate registry.** Today only the *active* variants have `KernelDescriptor`
   rows; dominated variants live as launch arms + bench code. Autotune needs every
   variant registered as a `KernelImpl` candidate. `KERNEL-TEMPLATING.md` delivers
   this for free — **each instantiation in the manifest is a candidate** — so the
   manifest *is* the candidate list. Until templating lands, register the existing
   variants explicitly.
2. **Backend-generic `bench`.** `bench` still hard-names `HipDevice` (the decouple
   genericized `forward`/`server` but deferred `bench`). Genericize the harness
   over the `core::Device` seam so one tool drives gfx906, sm_86, gfx1031, … .
3. **The rank-and-emit loop** — `bench autotune`.

## The harness: `bench autotune`

```
bench autotune --arch <arch> [--op <op>] [--dtype <dt>] [--model <id>]
               [--out dispatch/<backend>/<arch>.toml]
```

Per `(op, dtype, shape-bucket)`:
1. **Enumerate** — all registered impls where `applies(probe_shape, cfg)`.
2. **Gate on cert** — keep only impls with a green cert for this arch (sweep to
   produce any missing cert first; a candidate that can't certify is dropped, not
   ranked).
3. **Measure** — time each surviving candidate on the target GPU across the
   bucket's shapes (warmup + median-of-N, locked clocks).
4. **Rank + pick** — fastest certified candidate; record the *full leaderboard*,
   not just the winner.
5. **Emit** — write/update the dispatch row (`impl`, shape predicate, `cert`) and
   the result record.

- **Shape buckets**: start with the existing coarse buckets (`m=1..16` / `1..127`
  / `>=128`). Auto-discovering crossover boundaries (finer `m` grid, detect the
  rank-flip point) is an optional later refinement — the table **stays coarse by
  design** (dispatch-overfit is a named risk).
- **Sweep targets**: a synthetic `(m,k,n)` grid for breadth **plus the real
  per-model layer shapes** — gemma4 and qwen3 prefer opposite rows at the same
  dtype (CLAUDE.md), so `--model` feeds the actual shapes a model uses.

## Measurement protocol (the part that makes or breaks it)

The dispatch is only as trustworthy as the timings. Required:

- **Locked clocks** (`nvidia-smi -lgc` / `rocm-smi --setperf`) — otherwise boost
  and thermals dominate the signal.
- **Warmup + median-of-N** (not mean; drop outliers).
- **Hysteresis** — switch the incumbent only if a challenger beats it by **> a
  noise margin** (default 3%). Prevents the table thrashing between near-ties.
- **Determinism re-run** — re-measure any bucket whose ranking isn't stable across
  runs; flag it rather than commit a coin-flip.
- **PMC sanity** (rule 6) — attribute the win to the expected bottleneck
  (bandwidth / compute / latency); a "win" with an implausible PMC profile is
  suspect.
- **Wall-clock cross-check** — the kernel-level winner must also help end-to-end;
  if kernel-time and wall-clock disagree, trust wall-clock.

## The result schema (also the upload format)

Per measured cell — this record *is* what a contributor would upload:

```json
{
  "backend": "cuda", "arch": "sm_86",
  "op": "qmatmul_mmvq", "dtype_weight": "Q4_K", "dtype_activation": "Q8_1",
  "bucket": { "m": "1..127", "k": "any", "n": "any" },
  "leaderboard": [
    { "impl_id": "qmatmul_q4_K_mmvq_r2_dp4a_sm86", "rank": 1, "rel_time": 1.00, "cert_hash": "sha256:…" },
    { "impl_id": "qmatmul_q4_K_mmvq_dp4a_sm86",    "rank": 2, "rel_time": 1.14, "cert_hash": "sha256:…" }
  ],
  "pmc": { "vgpr": 28, "occupancy": 0.5, "mem_busy_pct": 64 },
  "rig": "RTX3090-driver580", "harness_version": "1", "n": 5
}
```

`rel_time` is **normalized to the bucket winner (1.00)** — relative *within one
rig*, never an absolute millisecond, so it is comparable across uploads from
different machines. The rank-1 `impl_id` becomes the dispatch row.

## Integration

- Emits/updates `dispatch/<backend>/<arch>.toml`; the existing `cert_check` +
  `dispatch_toml_roundtrip` tests gate it (every emitted row must carry a green
  cert).
- **CI ratchet** — a `bench autotune --check` mode regenerates the table and diffs
  vs. committed. Drift (a new variant would now win, or a row lost its cert) is a
  CI signal, not silent staleness.

## Crowdsourced tuning database (the upload design)

The network-effect layer: contributors run the autotuner on their hardware and
upload results so others get a tuned table for their exact arch without running
the sweep. Proven model — CLBlast ships a community-contributed per-device tuning
DB. Built on the cert grid, it can be made *safe* in a way a raw param DB cannot.

### Shape: a contribution pipeline, **not** a runtime dependency

The trap to avoid is the engine pulling-and-applying dispatch from a live server —
that breaks rule 1 (variant selection must live in a reviewed `.toml`) and is a
supply-chain + reproducibility hole. The shape that keeps the principles:

```
contributor `bench autotune` ──upload──► aggregation server
                                              │ consensus per (arch, op, dtype, bucket)
                                              ▼
                                  curated dispatch PR ──review──► repo (source of truth)
                                              │
  user `flambeau tune-pull --arch sm_86` ◄────┘  (optional, BUILD-time, pinned, cert-verified)
```

- The **repo stays the source of truth**; the server is a staging/aggregation
  layer that *feeds PRs* (a bot opens "update `cuda/sm_89.toml` from N reports").
  Rule 1 holds — still reviewed.
- A user may optionally **pull a tuned table at build/install time**, pinned to a
  version and cert-verified — "download a tuned config," not "trust a live oracle."

### Aggregate rankings, not absolute times

Crowdsourced timings come from incomparable rigs (clocks, driver, thermals, load),
so the server stores, per `(backend, arch, op, dtype, bucket)`, a **leaderboard of
`(impl_id, relative_rank, n_reports, cert_hash)`** and takes the **consensus
winner** across reports. Relative ranking within a rig is robust; cross-rig
absolute time is discarded. Aggregation is **per-arch** (sm_86), not per-SKU (3090
vs 3080 vs A6000) — per-SKU variation is within-bucket noise, and per-arch keeps
the DB small and respects the coarse-table principle.

### Trust model (why the cert grid matters here)

- **Correctness never leaves the local cert gate.** The server moves dispatch
  *rows* (`impl_id` + shape predicate) + **cert hashes** — never kernel *code*.
  Source always comes from the reviewed repo. Worst case, a compromised server
  routes you to a *different in-repo, still-cert-passing* kernel — a **perf
  regression, not an RCE**. Bounded blast radius.
- A pulled row is accepted only if its `cert_hash` matches a committed cert for
  that impl on your build — an unverifiable "winner" is ignored.
- Uploads are **opt-in, attributed, signed**; the fingerprint (arch + driver +
  PMC) is benign telemetry but still opt-in.

### Sequencing

Build the **local autotuner first** — early on, running it on your own rig
(~minutes) beats waiting on a sparse DB. The **result schema second** (so local
result files are forward-compatible). The **server/aggregation last**, once
there's a community to populate it. Do not front-load the server before the thing
it aggregates exists.

## Risks & mitigations

- **Measurement noise → table thrash.** Hysteresis margin + locked clocks +
  determinism re-run.
- **Sparse candidate pool (pre-templating).** Autotune can't pick where a cell has
  one candidate; `KERNEL-TEMPLATING.md` is the prerequisite that makes the pool
  dense and uniform per arch.
- **Overfit to a SKU/driver.** Per-arch coarse buckets + consensus across reports.
- **Crowdsourced trust / supply-chain.** Cert-hash gate; server moves rows not
  code; opt-in + signed.
- **Server before community = dead weight.** Sequence it last.

## Non-goals

- **Runtime auto-pull-and-apply dispatch** — breaks rule 1; supply-chain + repro
  hole.
- **Per-SKU dispatch tables** — overfit; the table stays per-arch coarse.
- **Autotuning correctness tolerances** — the cert tolerance is fixed; autotune
  only ranks perf among already-correct kernels.
- **A live perf oracle the engine depends on at inference time.**

## Phasing (onto PORTABILITY-ROADMAP)

- **P1 (spine)** — backend-generic `bench` + the rank-and-emit, proven on the pilot
  (`mmvq` Q8_0): reproduce gfx906's hand-tuned rows *and* emit the first
  `cuda/sm_86.toml` rows.
- **P2** — used per family as templating lands the dense candidate pool.
- **P6** — adding an arch (sm_80/sm_89, gfx1031/gfx908+) becomes "run autotune,"
  not hand-tuning.
- **post-P6** — the crowdsourced DB: result schema as upload format → aggregation
  server → `tune-pull`.

## Decisions to lock

1. **Candidate registry** — recommend every `KERNEL-TEMPLATING` instantiation
   auto-registers as a `KernelImpl` candidate; the manifest is the candidate list.
2. **Hysteresis margin + N** — recommend 3% / N=5, both tunable.
3. **Bucket strategy** — recommend fixed coarse buckets first; crossover-discovery
   deferred.
4. **Schema versioning from day one** — version the result record even before the
   server exists, so local result files (and later uploads) stay forward-compatible.
