# Flambeau — Roadmap

**Status:** Live tracker. Single source of truth for open work. Historical
plans live in [`archive/`](archive/README.md); the per-iteration optimization
log lives in project memory (`opt-loop-gemma4-12b.md`).

**North star:** beat llama.cpp on gemma-4-12b-q4_0, on both target rigs
(gfx906 MI50, sm_86 RTX 3090), across prefill / decode / **concurrent
throughput**. llama.cpp is the correctness oracle and the perf bar — not a
ceiling (see CLAUDE.md "Target silicon, not llama.cpp").

Every perf claim in this file must satisfy the CLAUDE.md pre-claim checklist
(rebuild → sweep-green → PMC bottleneck match → wall-clock at pp≥512/tg≥64 →
cert diff). Null results are first-class.

---

## Current state (measured, gemma-4-12b-qat-q4_0, RTX 3090 sm_86)

| Axis | flambeau | llama.cpp | ratio | verdict |
| --- | --- | --- | --- | --- |
| Prefill single-stream (931-tok) | **1674 tok/s** | 1701 | 0.98× | **parity** |
| Decode single-stream | ~65 tok/s | 82 | 0.79× | near silicon wall (ceiling ~130) |
| Throughput N=1 / N=2 / N=4 | 53 / 70.5 / **103.0** | 77.5 / 146.3 / **228.8** | **0.45×** @ N=4 | **the open frontier** |
| Batch scaling (N=1→4) | 1.94× | 2.95× | — | flambeau under-scales |

Prefill is solved; decode is at the wall; **concurrent throughput is where a
server competes and where flambeau loses 2.2×.** That gap is Track A.

Baseline commit: `d8703a89` (lever-3 dp4a batched Q6_K lm_head, +7.5% @ N=4).

**Legend:** `[ ]` open · `[~]` in progress · `[x]` done · `[-]` blocked/deferred.
Each item lists **Done when** (checkable) and **Target** (measurable).

---

## Track A — Concurrent throughput (the frontier)

The N=4 gap decomposes (iter14 nsys) into three fixable levers plus CUDA graphs.
Ranked by ROI. **Track-level target: flambeau N=4 throughput > llama 228.8 tok/s**
(or, interim, batch-scaling ≥ 2.5×).

- [ ] **A1 — Batched split-K decode attention** *(kernel proven; wiring bug)*
  Batched decode attention is single-block-per-(head,slot), serial-KV (20% of
  N=4 decode). Split-K kernel is verified **2.8–5.7× and bit-exact** in isolation
  (iter16); the e2e wiring produces wrong output on the longer slot in concurrent
  variable-length decode (iter17), matching a recurring pre-existing bug.
  - Artifacts: `memory/artifacts/lever2-batched-splitk.patch` + `.cu` (verified kernel).
  - **Refined diagnosis (2026-07-07, static):** the split-K chunk kernel is proven
    **line-by-line identical** to the verified serial batched oracle
    (`attention_f16.cuh:82`) — same window mask (`swa_min = n_tokens - window_size`),
    same ring read (`t_phys = t % ring_depth`); chunks tile exactly `[swa_min,
    n_tokens)`; combine zero-weights empty chunks. Buffer sizing is adequate
    (single-stream splitk d512@2448 works ⇒ `q_width` uses max head_dim 512).
    `n_chunks = ceil(max(host_n_kv)/chunk_size)` is consistent across all layers.
    **All three original suspects (mixed-head_dim overflow, SWA slid-pointer,
    per-call buffer variance) are REFUTED.** gemma SWA layers are *ring* layers
    (`is_ring_layer`, no slide), so `any_offset` is never set; the untested regime
    is nt > window (ring wrap + masking), which the iter16 microbench never hit
    (all its lengths < the 1024 window).
  - **Empirical result (2026-07-07, built + served + needle-swept):** the bug is
    **deterministic in `n_chunks`**, not multi-slot and not the slide path. With SWA
    masking active (n_tokens > 1024 window), `n_chunks ∈ {5,6}` corrupts the
    **max-length** slot (progressively worse with length); `n_chunks ≥ 7` is correct.
    `chunk_size=512` (n>2048) always yields n_chunks=5 → BAD. **Reproduces solo
    (n_slots=1)** — the earlier "unequal-length" framing was coincidental. Full table
    + fast repro (`/tmp/solo.py`, words=900 BAD / 1200 OK) in
    `memory/artifacts/lever2-diagnosis-2026-07-07.md`.
  - **Definitive next step:** per-layer e2e oracle-diff — inside
    `attention_decode_f16_batched_splitk`, also run the serial batched kernel into a
    scratch out-buffer and device-compare max-abs per layer on the `words=900`
    (n_chunks=5) forward; the first diverging layer pins it. Suspects: (1) combine
    recombination when the leading chunk is window-clamped at small n_chunks; (2)
    off-by-one in chunk coverage at small n_chunks. Kernel math is proven equivalent
    to the working serial/single-stream paths — it's a narrow combine/boundary bug.
  - **⚠ port note:** 8080 is held by the `claw-rwire` app (a different project) — do
    NOT kill it; serve flambeau on a free port (e.g. 8091).
  - **Done when:** concurrent 2-slot needle at unequal lengths (1898 vs 1548) is
    correct on both slots; kernel re-certed on sm_86 (serial fallback on gfx906);
    e2e throughput measured vs the 103 tok/s @ N=4 baseline.
  - **Target:** N=4 ≥ **112 tok/s** (attn 20% × ~3× → ~+10%), no correctness regression.

- [ ] **A2 — Continuous batching / coalescence** *(biggest structural lever)*
  ~30% of GEMV work falls to batch=1/2/3 because the 4 requests drift out of
  lockstep (different prefill-finish, different lengths). llama `-cb` admits/evicts
  slots mid-flight and always runs max batch; flambeau's K4c scheduler +
  decode-batch-window leaves gaps. This is the main cause of 1.94× vs 2.95× scaling.
  - Scope: scheduler change in the server decode loop; keep the batched kernels.
    No new env flags (rule 1) — behavior lives in the scheduler, not a `CANDLE_*`.
  - **Done when:** nsys N=4 shows the batch=1/2/3 fallback GEMV share drop from
    ~30% to <10%; correctness (concurrent coherence + needle) intact.
  - **Target:** batch scaling ≥ **2.5×** (N=4 ≥ **132 tok/s**), stretch → beat llama.

- [ ] **A3 — CUDA graphs for the batched decode step**
  llama boots `USE_GRAPHS=1`; flambeau has none on CUDA → per-step launch overhead
  on ~480 tiny kernels/token. Capture/replay the batched decode step.
  - Note: `CudaGraphExec` capture/replay already exists (P4 core). Prior finding:
    hip-graph on gfx906 was null-to-negative — **CUDA/sm_86 is a different regime**,
    measure fresh; do not assume the gfx906 result.
  - **Done when:** graph-captured decode step is correctness-clean; wall-clock A/B
    vs non-graph at N=1 and N=4.
  - **Target:** ≥ +5% at N=1 and N=4; keep if >parity, else `cfg(unverified)` + diagnosis.

- [ ] **A4 — Generalize dp4a-batched lm_head to K-quant models**
  Lever-3 fixed Q6_K; `mmvq_q4_k_batched` / `mmvq_q5_k` still use scalar FP32
  superblock decode (compute-bound, ~linear scaling) — same flaw, same dp4a fix.
  Hits any Q4_K/Q5_K-headed model (e.g. gemma Q4_K).
  - **Done when:** dp4a batched Q4_K/Q5_K kernels bit-match the certified single-row
    path; e2e throughput on a Q4_K-head model shows the batched lm_head no longer
    scaling linearly.
  - **Target:** +5–8% N=4 throughput on a K-quant-head model (mirror the Q6_K result).

---

## Track B — Single-stream (pull ahead of llama)

Prefill is at parity, decode near the wall. These push prefill **ahead** and
recover the last decode kernel. Lower priority than Track A.

- [ ] **B1 — MMQ 69 → ~100 TOPS** *(the last ~8% of prefill; multi-session)*
  MMQ is 66% of prefill. My 256×128 tile = 1 block/SM (33% occ) stalls at every
  `__syncthreads`; llama runs 2 blocks/SM with register-resident A swept across the
  token dim + stride-76 interleaved layout. Every cheap variant (warp-config sweep,
  scale-hoist) already ruled out null.
  - Plan/harness: `memory/rtx-prefill-opt-plan.md` (Lever A), `/tmp/pf3.cuh` sketch,
    `/tmp/sweepA.cu` (dp4a-vs-mma real-shape sweep, tol 0). Block MUST stay ≤512 threads.
  - **Done when:** microbench beats 69 TOPS on real gemma shapes (M=512, tol 0 vs
    dp4a), cert green on sm_86, e2e prefill A/B measured.
  - **Target:** prefill > **1701 tok/s** (beat llama); stretch ≥ 1850.

- [ ] **B2 — Decode `attn_decode_splitk` vectorization**
  The one off-peak decode kernel (~20% of decode): scalar f16 K/V loads + per-token
  `__syncthreads` rendezvous → 250 GB/s (d512) / 127 (d256 SWA), 3–6× off peak.
  Fix: half2-vectorized loads + warp-local accumulation (kill the block sync),
  llama fattn-vec style.
  - **Done when:** kernel bit-exact vs current splitk on identical memory; e2e decode A/B.
  - **Target:** single-stream decode toward 72–75 tok/s (bounded — decode is near
    the ~130 silicon wall, so realistic gain is small; file honestly if null).

- [ ] **B3 — HIP Q8_0 prefill: port `mmq_q8_0_4warp_lds`**
  gfx906 prefill lever: the 4-warp LDS-tiled gold-standard MMQ exists for Q4_0/Q4_1
  but not Q8_0 (~15–20× gap; Q8_0 wave64 MMQ runs at 4–6% of int8 peak). Decode side
  is closed (bandwidth-bound; dp4a microbench win doesn't translate → kept `t128_vdr2`).
  - Detail: memory `q8_0-decode-dispatch-finding.md`.
  - **Done when:** kernel certed on gfx906; verify `tile8 > tile16` end-to-end before
    any dispatch change; re-rank via the MMQ autotune path.
  - **Target:** Q8_0 gfx906 prefill within striking distance of the Q4_0 4warp_lds rate.

---

## Track C — Breadth & hardware-gated

Not bugs — waiting on silicon or lower priority. Portability program P0–P5 is
complete (see `archive/PORTABILITY-ROADMAP.md`); this is the remaining P6 tail.

- [-] **C1 — gfx1031 (RDNA2 wave32) runtime** — compile-portability done (207 kernels,
  wave32-clean). Blocked: runtime certs + `dispatch/hip/gfx1031.toml` + LDS/perf tuning.
  **Card not in rig** (`rig-hardware-2026-06`). Done when the card is present and the
  cert grid + dispatch table are green.
- [-] **C2 — sm_80 / sm_89 cert grids** — forward-compatible today; needs the arch cert
  grids + the ~30 `sm_86`/`gfx906` cli/bench arch-string hardcodes generalized (only
  bite when running sweeps on a new arch).
- [-] **C3 — CUDA multi-GPU (P4)** — NCCL `FusedAllReduce` (needs CUDA `Ops` layer),
  graph slot-map, 2-GPU P2P cert + multi-GPU forward. **Needs a 2nd sm_75+ NVIDIA card.**
  Single-GPU parts (`CudaCluster`, `DeviceAllReduce`, `CudaGraphExec`) done.
- [-] **C4 — CUDA embeddings + decode profiler** — `EmbeddingHandle` is HIP-only
  (CUDA path returns 503); CUDA decode profiler absent. Low priority.

---

## Housekeeping

- [ ] **H1 — Stash triage.** `stash@{0}` (feature/gemma4) "TP-shard fused MoE gate+up
  loader" — evaluate: land or drop. `stash@{1}` lever-2 — superseded by
  `memory/artifacts/lever2-batched-splitk.patch`, drop after A1 lands.
  `stash@{2,3,4}` old env-purge WIP on main — drop.
- [ ] **H2 — Branch audit.** Reconcile unmerged branches: `feature/gemma4`,
  `feature/iq-quants-and-batched-decode`, `feature/quant-perf-sweep`,
  `feature/tool-calling-fixes`, `refactor/multi-backend-blocks`, `path-collapse-pr2-pr5`.
  Current work is on `feature/cuda-decouple`.
- [x] **H3 — Q8_0 decode dispatch** — investigated; dp4a microbench win does not
  translate e2e (bandwidth-bound) → kept `t128_vdr2`. Closed.

---

## Sequencing (recommended)

1. **A1** — debug lever-2 (proven kernel, oracle-diff method ready; highest value/effort).
2. **A2** — continuous batching (the big structural closer of the scaling gap).
3. **A3 + A4** — CUDA graphs + K-quant lm_head generalization (self-contained).
4. **B1** — MMQ 69→100 (pull prefill ahead; multi-session).
5. **B2 / B3 / H1 / H2** — as time allows.
6. **Track C** — as hardware arrives.

## Pointers

- `ARCHITECTURE.md` — framework design (crates, traits, dispatch, KV families).
- `CUDA-PORT-PLAN.md`, `KERNEL-TEMPLATING.md`, `AUTOTUNE.md` — live mechanics references.
- `archive/` — completed plans + the prior portability roadmap (phase history).
- memory: `opt-loop-gemma4-12b.md` (per-iteration log), `rtx-prefill-opt-plan.md`
  (MMQ/attention detail), `q8_0-decode-dispatch-finding.md`, `rig-hardware-2026-06.md`.
