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

- [x] **A1 — Batched split-K decode attention** *(CLOSED: kernel certified correct; e2e throughput = measured NULL; wiring reverted)*
  Batched decode attention is single-block-per-(head,slot), serial-KV (20% of
  N=4 decode). Split-K kernel is verified **2.8–5.7× and bit-exact** in isolation.
  Artifacts: `memory/artifacts/lever2-batched-splitk.patch` + `.cu` (verified kernel),
  `oracle_diff_batched_splitk.cu` (kernel-vs-oracle proof).
  - **⚠ The "wiring bug" was a MISDIAGNOSIS (2026-07-07).** The needle-recall test
    used to "find" it is invalid: forcing `use_splitk=false` (split-K never launched)
    AND testing the **fully clean baseline** (no patch) both reproduce the *identical*
    failure pattern (1242 tok empty, 1642 OK, 2127/2463 garbage). The split-K kernel is
    not involved. The earlier "deterministic in n_chunks" table and "serial-overwrite
    bisection" were confounded by the invalid test. Full retraction +
    the 5 decisive experiments in `memory/artifacts/lever2-diagnosis-2026-07-07.md`.
  - **No valid local oracle:** the only CUDA llama.cpp build (`build-cuda`) misreads the
    gemma-4 GGUF (emits `<|channel>` harmony tokens / `011111` garbage for "2+2"); HIP
    builds are gfx906 + need libomp. Needle results are prompt-formulation noise (mayor
    recalled at 962 tok; number-distractor code fails at 667 tok) — borderline QAT-model
    behavior, not a kernel signal.
  - **Correct path (per CLAUDE.md "Run sweep first"):**
    1. ✅ **DONE — sweep cert green.** `sweep --arch sm_86 --op attention_decode_batched_splitk`
       → `certs/cuda/sm_86/attention_decode_f16_batched_splitk_sm_86.json` (pass=true, 8
       shapes, worst max_rel 6.7e-5 vs 2e-2 tol, VGPR 40). Covers exactly the configs the
       needle test flagged: hd512/max_n2127 (cs512 nc5), hd256/1242 (nc5) + 1382 (nc6) w/
       SWA window=ring=1024 wrap, and multi-slot unequal `[2127,1242]` / `[2043,545]`.
       Kernel + sweep: `crates/kernels-cuda/src/kernels/attention_decode_batched_splitk.cu`,
       `crates/bench/src/sweep_attention_batched_splitk.rs`.
    2. ✅ **DONE — re-enabled + benched → measured NULL (throughput-neutral).** Applied the
       wiring, verified e2e-correct (self-consistency smoke: mayor@962tok = split-K active →
       reproduces the serial baseline answer), then a **same-session on-vs-off A/B** on the
       1024-tok and 3000-tok concurrent harness (`scripts/bench/bench_pp1024_tg128_concurrent.py`,
       and a 3× extension):
       | ctx | N | off | on |
       |---|---|---|---|
       | 1024 | 1/2/4 | 65.3 / 63.9 / 62.2 | 65.9 / 64.1 / 62.8 |
       | 3000 | 1/2/4 | 62.0 / 60.2 / 57.0 | 62.2 / 60.4 / 58.3 (3-run mean) |
       On ≈ off at every N/context — but this A/B is **INVALID, not a null (retracted 2026-07-07).**
       Two independent reasons the "on" config never actually ran split-K on the batched path:
       (1) the harness (`bench_pp1024_tg128_concurrent.py`) **streams**, and pre-slice-1 the
       streaming path bypassed the coalescing scheduler → each request ran N=1 single-slot decode,
       so the batched decode attention kernel was never invoked (note the "off" numbers *decrease*
       with N: 65→62 — the streaming-bypass signature, now fixed by A2 slice 1). (2) the wiring
       patch (`lever2-batched-splitk.patch`) wires split-K **only into `standard_attn_mixed.rs`**
       (the Sarathi mixed path, env-gated OFF by default) — it never touches the pure batched
       decode path (`standard_attn.rs:1227`). So both on/off ran the identical serial single-slot
       kernel. on≡off by construction.
    3. **Disposition (REOPENED 2026-07-07 by nsys profile):** the kernel is CERTIFIED CORRECT
       (sweep green) + bit-exact; the throughput conclusion was wrong. Post-slice-1, nsys of steady
       batch=4 decode at ~1000-tok ctx shows **`attention_decode_f16_batched` = 60.4% of GPU time**
       (699µs/call; grid = n_heads_q·n_slots = 64 blocks each serially scanning ~1000 KV →
       under-occupied, latency-bound). Split-K (chunk KV → ~384 blocks) targets exactly this.
       **Real fix = wire `attn_decode_f16_batched_splitk` into the PURE batched decode path**
       (`standard_attn.rs:1227`, mirroring the mixed-path wiring already in the patch), size the
       partials scratch for `n_slots × n_chunks`, then A/B at LONG ctx through the now-coalescing
       streaming/non-streaming path. This is the A2 slice-2 lever (see below); A1 and A2-slice-2
       are the same work now.
  - **⚠ port note:** 8080 is held by the `claw-rwire` app (a different project) — do
    NOT kill it; serve flambeau on a free port (e.g. 8091).
  - **Done when:** sweep cert green ✅ + split-K wired into the **pure** batched decode path +
    long-ctx N=4 A/B shows on > off (attention share drops, aggregate tok/s rises). REOPENED.
  - **Note (separate track, not A1):** flambeau's long-context recall is mediocre on
    these tests, but with no valid oracle it cannot be attributed to a bug vs the QAT
    model. If pursued, get a WORKING oracle first (fresh llama.cpp CUDA w/ gemma-4, or a
    non-QAT reference), then A/B identical prompts on the clean baseline.

- [~] **A2 — Continuous batching / coalescence** *(biggest structural lever)*
  ~30% of GEMV work falls to batch=1/2/3 because requests drift out of lockstep.
  llama `-cb` always runs max batch; flambeau leaves gaps. **Slice 1 (streaming→scheduler)
  landed +92% N=4 streaming; slice 2 (staggered-prefill coalescing) open — see below.**
  - **⚡ DIAGNOSIS SHARPENED (2026-07-07) — the dominant gap is the STREAMING PATH.**
    Measured (short prompt, agg = total_tok/wall, gemma-4-12b sm_86, inflight-slots 4):
    | N=4 mode | agg tok/s | scaling |
    |---|---|---|
    | streaming (`stream:true`) | 49.0 | **1.0×** (serialized) |
    | non-streaming | 95.3 | **1.95×** (coalesces) |
    The coalescing scheduler *works* (non-stream 1.95×, matches the recorded baseline), but
    `run_completion_blocking_streaming` (`decode_loop.rs:1174`) **bypasses it entirely**: it
    `blocking_lock`s the slot for the whole loop (`:1234`) and calls N=1 `dispatch_decode_one`
    (`:1413`) — no `scheduler_can_engage` check. **Real chat traffic streams → gets half the
    throughput.** (My A1 A/B used `stream:true`, which is why it showed flat ~0.9× — that was
    the streaming bypass, not attention.)
  - **✅ Slice 1 LANDED + MEASURED (2026-07-07).** Added `run_completion_streaming_scheduler_pp_blocking`
    (`decode_loop.rs`): mirrors `run_completion_scheduler_pp_blocking` (coalescing via
    `decode_via_scheduler_into`, no slot mutex held) with the streaming path's incremental-
    detokenise `push_and_emit` SSE. `run_completion_blocking_streaming` gates on
    `scheduler_can_engage` and routes here; JSON/logprobs still fall through to the legacy N=1 loop.
    Measured (gemma-4-12b sm_86, short prompt, agg=total_tok/wall, slots 4):
    | N | stream before | stream after | non-stream ref |
    |---|---|---|---|
    | 1 | 48.5 | 49.3 | — |
    | 2 | — | 66.6 (1.35×) | — |
    | 4 | **49.0 (1.0×)** | **94.1 (1.91×)** | 95.4 |
    | 8 (slots 4) | — | 93.1 (saturated) | — |
    **Streaming N=4 +92% (1.0×→1.91×), now matches non-streaming.** Correctness: 4 concurrent
    distinct streaming prompts each answered correctly (no per-slot cross-contamination).
    - **rule-14 debt accepted (user call):** this is a 4th near-duplicate decode loop. Follow-up
      to unify all four onto one parameterised step fn = **task #67**.
  - **Slice 2 — split-K batched decode attention (= reopened A1). Profiled 2026-07-07.**
    Three hypotheses were **measured false** before the profile settled it:
    - *Not scheduler drift.* N=4 stream vs request-start stagger: 0ms→95.4, 250ms→95.1
      (fully recovers), 1000ms→78.4 (tail only). Coalescing is robust to realistic stagger.
      Findings (b)/(c)/(d) + the `FLAMBEAU_MIXED_BATCH` rule-1 env-gate are second-order — defer.
    - *Not weight re-streaming.* Q4_0 batched decode uses `mmvq_q4_0_row_tile_batched` (R=4 row-tile,
      one weight read → N cols), compiled for CUDA too (`instantiations.toml` → `kernels-cuda/build.rs:40`).
    - *Not fusion.* nsys of steady batch=4 decode at ~1000-tok ctx: the unfused-path candidates
      (rmsnorm + 2 quants + cast, incl. the "wasted" `q8_1_mmq`) are **~3% of GPU time TOTAL** — noise.
    - **✅ Real cause — batched decode ATTENTION at long ctx.** nsys `cuda_gpu_kern_sum`
      (gemma-4-12b-qat, sm_86, N=4, ~1000-tok ctx): `attention_decode_f16_batched` = **60.4%**
      (699µs/call, 6198 calls), `mmvq_q4_0_row_tile_batched_n4` = 31.8%, `mmvq_q6_k_batched_n4`
      (lm_head) = 3.5%, everything else <1%. The attention kernel's grid = n_heads_q·n_slots = 64
      blocks, each serially scanning the whole KV → under-occupied on 82 SMs, latency-bound; cost
      ∝ KV length. (Kernsum: `/tmp/a2_batch4_kernsum.txt`.)
  - **DONE — split-K wired into the pure batched decode path; e2e = VALID NULL (iter28, 2026-07-07).**
    Wired `attn_decode_f16_batched_splitk` at `standard_attn.rs:1227` (gate `!is_ring_layer &&
    max_dec_kv>256 && n_chunks∈(1,MAX] && partials!=null`), + `!is_ring_layer` guard added to the
    mixed path (patch lacked it → latent ring bug). Sweep green (8 shapes).
    - **Thermally-controlled interleaved A/B (N=4, TG=192, ~1.2k prompt):** serial mean **47.64**,
      split-K mean **47.65** tok/s — identical within noise. (A naive first A/B showed "+12%" that
      was **100% thermal artifact**: same split-K config = 55.0 cold vs 47.6 hot; the 90s rebuilds
      between configs were cooldowns. No clock-lock perm → fix was two md5-distinct binaries
      alternated with order flipped, temp/clk logged.) Deep 2.4k prompt also null (43.8 vs 44.5).
    - **Engagement proven (not a silent fallback).** nsys `cuda_gpu_kern_sum` of the split-K binary:
      `attention_decode_f16_batched` (SERIAL) = **59.3%** (11052 inst, 979µs), `..._splitk_chunk` =
      7.1% (2211 inst, 587µs), `..._splitk_combine` = 0.0%. Ratio **11052:2211 = 5.0:1** =
      gemma-4 `sliding_window_pattern` (5 local : 1 global; window 1024, 48 layers = 40 SWA + 8 global).
    - **Root cause of the null:** split-K fires on **exactly the 8 global layers** (1/6 of attention
      calls), 1.66× faster each — but the **40 SWA/local layers dominate (59.3% GPU) and can't be
      split-K'd** (ring/modular KV addressing). iter27's "attention=60%" was right; ~all of it is the
      untouchable SWA kernel, not the global one split-K addresses.
  - **REAL LEVER (new) = SWA/local-layer batched decode attention.** Window 1024, 40 layers, the same
    64-block under-occupied serial scan, 59% of N=4 GPU time. Needs a **ring-aware** occupancy fix
    (split-K inside the window's modular address space, or a restructured windowed-attention kernel) —
    bigger than global-only split-K. This is the frontier for gemma-4 N=4 decode.
  - **Split-K kernel is correct + certified** and would fire on 100% of layers for a **dense-attention**
    model (Mistral/Qwen dense, all-global) → potential long-ctx win there, **UNVALIDATED** (no dense A/B).
  - **Disposition pending user:** (a) revert hot-path wiring to clean serial; (b) `cfg(unverified)` the
    kernel + wiring with the null diagnosis (rule 10); or (c) keep the wiring (correct, low-overhead,
    may help dense models) + validate on a dense model later. Binaries/harness in `/tmp`.
    Secondary (A4): lm_head `mmvq_q6_k_batched_n4` = 1.93ms/call (3.5%), heavy per-call.

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

- [ ] **B4 — gfx906 generic LDS-tiled MMQ: close the remaining ~1.5× to llama**
  `mmq_lds_tiled.cuh` (one template + per-dtype traits) now owns m>=128 for Q4_K,
  Q5_K, Q6_K, Q3_K, IQ4_XS, IQ4_NL, Q8_0 at a 64×64 tile (2 blocks/CU). Qwen3.8-27B
  UD-Q4_K_M, 2×MI50 TP2, q8 KV: prefill 37 → 229 t/s (pp512), 33 → 208 (pp4096);
  llama.cpp `-sm layer` same GGUF: 206 / 313. MMQ is still ~90% of prefill GPU time.
  - Open: IQ3_S / IQ3_XXS / IQ2_* / Q2_K traits (still wave64; they dominate
    UD-Q3_K_XL); fold Q4_0/Q4_1 4warp_lds and the dormant 128×16 Q4_K into the
    template; NWARPS=8 (128×64, 512 threads — llama's GCN pick for IQ4_XS/Q5_K);
    PP2 does not pipeline chunks across stages (131 t/s vs TP2 229).
  - Ruled out (measured): 128×16 and 64×16 tiles (LDS/amortization), VGPR cap via
    `__launch_bounds__` alone (spill-free but e2e-null). Unrolling the k01 loop
    spills past 128 VGPR and regresses — keep `#pragma nounroll`.
  - Harness: `scripts/bench/qwen38_27b_vs_llamacpp.py`.

- [ ] **B5 — Speculative decode (MTP, then DFlash): close the gap to llama.cpp**
  `--mtp N` loads the GGUF's embedded nextn block and runs draft → verify →
  accept cycles (SD/TP, non-streaming legacy handler; default 0). It now beats
  plain decode on gfx906 TP2; the remaining gap to llama.cpp is acceptance rate
  and the plain decode step.
  - Cycle (llama.cpp `draft-mtp` alignment): the draft head pairs the newest
    sampled token with the target hidden of the position before it, one target
    forward over `[last, d1..dk]` verifies every draft, heads are evaluated
    lazily up to the first rejection, and a rejected suffix is rolled back by
    replaying the GDN state step from saved inputs (no second forward). Output
    is not bit-identical to plain decode: the multi-row path quantizes
    activations through F16 and differs from the single-row path by ~1e-3, so
    greedy text can fork at near-ties.
  - Small-batch forward (Qwen3.8-27B UD-Q4_K_M, 2×MI50 TP2, ms for an N-row
    forward incl. ~20 ms request overhead; decode step = 41 ms):
    N=2: 93 → 72 · N=3: 118 → 80 · N=4: 118 → 85 · N=5: 120 → 90 · N=8: 123 → 114
    (llama.cpp `-sm layer`: 50 / 63 / 75 / 210 / 259). Row-tiled batched dp4a
    MMVQ for Q4_K / Q5_K / IQ4_XS at m=2..8 (`mmvq_dp4a_batched_k.cuh`): the
    8-column LDS tile costs a fixed ~1.6 decode passes to stage weights, the
    row tile keeps weights in registers and stages only the activation strip.
    Null results: a 16-row x8 tile (lane = weight row, so 3/4 of the lanes
    idle: 31.3 → 25.1 t/s) and a plain batched dp4a MMVQ reading activations
    from global memory (~15 ms per extra column vs ~3 for the row tile).
  - MTP e2e, same config, greedy, mean of 3 chat prompts (code / math / prose),
    t/s: plain 22.2 → N=2 31.4 · N=3 32.7 · N=4 35.1 (42.0 / 37.7 / 25.6) ·
    N=6 32.2. llama.cpp on the same GGUF: plain 25.9 → MTP-4 41.2 (50.1 / 42.0
    / 31.6). Same ×1.6 gain; llama accepts 92 / 76 / 53 % of its drafts, we
    accept 76 / 70 / 40 %.
  - UD-Q3_K_XL (IQ3_S / IQ3_XXS / IQ2_S row-tile traits added), same rig and
    prompts: plain 19.7 → MTP-4 28.9 (32.3 / 33.1 / 21.3). llama.cpp on that
    file: 20.6 → 39.6.
  - Sampling: accept / residual draws and the draft sampler are seeded from the
    request seed, so TP ranks reach identical verdicts and a seed reproduces the
    text; `Session::mtp_step` fails the request if rank outputs ever differ.
    Temp 1.0 / top-p 0.95 / top-k 20, Q4_K_M: 21–32 t/s (plain 22).
  - Not covered: Q6_K / Q3_K / IQ4_NL / IQ2_XS / IQ2_XXS / Q2_K at m=2..8 (x8
    tile or per-row MMVQ), scheduler + SSE paths, stop-mask rules inside the
    cycle (MTP stays off for JSON mode, logprobs and reasoning budgets).
  - To do: (1) acceptance — fill the draft layer's KV with `(token, target
    hidden)` pairs for the prompt and for accepted tokens (llama's `process()`
    catch-up; today it only holds its own draft steps), and stop drafting below
    a confidence floor (`p_min`); (2) row-tile traits for the remaining
    dtypes; (3) scheduler + SSE paths; (4) DFlash2 drafter (`dflash` GGUF arch:
    5 layers, block 8, target taps at layers 6/20/34/48/62, selector rank 256 /
    top-k 16) — reuses this verify cycle; (5) sm_86 e2e.
  - Open stability item: GPU page faults on host↔device copies of transient
    host buffers (three on 2026-11-18, fault addresses inside the mimalloc
    arena). The GDN reset no longer uses a host buffer; per-step logits copies
    still target a per-command Vec (~1 MB). Not reproduced in isolation.
  - **Done when:** speculative decode beats plain decode e2e on at least one backend
    (met for MTP on gfx906 TP2; open for DFlash and for parity with llama.cpp).

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

1. ~~**A1**~~ — CLOSED. Kernel certified correct (sweep green); e2e throughput a measured
   null (attention not the N=4 bottleneck); wiring reverted. Re-openable post-A2 if the
   bottleneck moves. The A/B *confirmed* A2 is the real lever (flat ~0.9× scaling).
2. **A2** — continuous batching / coalescence — **now the active frontier.** The A1 A/B
   directly implicates it: N=4 decode scaling is ~0.9× regardless of attention kernel.
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
