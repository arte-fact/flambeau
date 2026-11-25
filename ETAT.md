# flambeau — status

Max-perf inference server for modern LLMs on AMD HIP + NVIDIA CUDA, exposed over an
OpenAI-compatible HTTP API. Scope, build and first run: [`README.md`](README.md); architectural
rules and measurement discipline: [`CLAUDE.md`](CLAUDE.md); framework design:
`doc/ARCHITECTURE.md`. This file is the project state; the detailed journal of a workstream
lives next to the workstream.

## Where the code stands

- `main` is at `fccd031` (2026-06-12), the merge of `perf/long-context-speed`: long-context
  split-K, prefix cache (host-RAM clamped, gemma4 SWA slot snapshots, zero-copy restore) and
  tool-calling (`tool_choice` forcing by prefix injection, lenient qwen-coder and gemma4
  parsers).
- **Open pull request: #3, `feature/cuda-decouple`** (draft, HEAD `cd3ab132`, updated
  2026-11-22). CUDA backend decoupled from the HIP one (sm_86), MTP in streaming, batched MMVQ
  for m ≥ 2 on sm_86, `argmax_f32` on the GPU, the CUDA decode graph for qwen35, MoE
  certificates regenerated, orphan dispatch rows and MoE kernels without an execution path
  sorted out. Merging is Benoît's call.
- Branches still open: `chore/comment-debt`, `coord/chantier-ecarts-handoff` (handoff of the
  workstream below), `feature/cuda-decouple-v4` (`9c6621b7`), `feature/cuda-decouple-chantier-ecarts`
  (`838f3db3`), `feature/gemma4`, `feature/iq-quants-and-batched-decode`.

## Workstream: bringing the gap against llama.cpp to zero

Model under test: Qwen3.8-27B-UD-Q4_K_M, KV q8, greedy, on 2× MI50 (gfx906, TP-2) and one
RTX 3090 (sm_86). **Suspended on 2026-11-22 in the evening, at Benoît's request (token
budget): everything is stopped and the GPUs are free.** State of the workstream, measures and
diagnostics: `~/Workspace/flambeau-exp/ETAT.md` on threadripper, with the results in `v1/` to
`v4/` and the analysis documents in `coord/`; a copy is pushed on
`coord/chantier-ecarts-handoff` (folder `chantier/`).

| t/s, greedy | RTX 3090, no MTP | RTX 3090, MTP-4 | MI50 TP-2, no MTP | MI50 TP-2, MTP-4 |
|---|---|---|---|---|
| llama.cpp, 2026-11-19 | 43.5 | 78.0 | 25.7 | 37.3 |
| llama.cpp HEAD `ffc87c3a7` | 43.5 | **89.2** | 25.1 | 35.6 |
| flambeau V2 `cd3ab132` | 44.70 | 82.49 | 28.00 | 41.78 |
| flambeau V4 `b0a8d697` (local) | 44.69 | **86.57** | — | — |

- MI50 wins both ways; RTX wins without speculatation and stays 3 % behind on MTP-4 against
  llama.cpp HEAD.
- V3 `50864a2f` regresses MI50 by 3.4 % (≈ 90 % from the Q4_K row-tile x8 at m = 5, ≈ 8 %
  from double quantification). It is in `feature/cuda-decouple-v4`, so **that branch must not
  be merged as it stands**: it carries the gfx906 regression from `6b69f618`.
- The remaining MTP-4 gap on the RTX is in the n = 5 verification: attention (prefill tile),
  Q8_0 alpha/beta and launch gaps; ours reclaims 3.6 ms on the drafts.
- Known bugs, older than the workstream: `gemma-4-12b Q4_0` answers HTTP 500 on sm_86
  (`mmvq_q4_k_r4` hard-called, no CUDA port, `output_head.rs:82-99`), and
  `real_gguf_cuda_parity` (12B Q4_K) diverges from llama.cpp at the fifth token — the suspect
  is the MMA prefill attention on sm_86.

## Decisions waiting for Benoît

- Merge of PR #3 (reserved to him) and whether `feature/cuda-decouple-chantier-ecarts` is
  deleted.
- Whether the workstream resumes, and in which order: V5 measurement first, or the gemma bugs.

Provisional decisions taken by the team, which he may reverse: push V2 on
`feature/cuda-decouple`; MI50 thermal guard relaxed to ≤ 56 °C; sm_86 few-MMA bands Q4_K 3..24,
Q5_K 4..24, Q3_K 3..24 with a batched fallback on mixed sites; wiring the few path before
F3/F4; DFlash2 continues on the RTX only, gfx906 only if the m = 8 draft microbench justifies
it.

## Next steps, in order

1. On `9c6621b7`: build, all-dispatch on sm_86, CUDA forward and parity tests for the Q5_K/Q3_K
   few bands (nothing of that was run: the last all-dispatch run was invalidated by a rebuild).
2. D3: put the gfx906 Q4_K row-tile back to m = 2..8, with `qmatmul-bench` at the TP-2 per-GPU
   shapes, quantification counted; any line of `6b69f618` that loses goes back to V2; then
   `cert-check` 90/0 on gfx906 and a full all-dispatch.
3. V5 measured by a Vibe session following `coord/v5-consigne.md` (MI50 MTP-4 V2 against V5,
   RTX V4 against V5, ornith). If MI50 ≥ V2 and RTX ≥ V4: fast-forward push on
   `feature/cuda-decouple`, update PR #3 and the `mr-flambeau-3.md` article.
4. Then the rest of the RTX MTP-4 gap, the 32-row tile for m = 9..16, the gemma parity
   experiment, the Q3_K batched anomaly (n4 > n5), then step 6 DFlash.

## Measurement traps worth keeping

- A gfx906 microbench at the full shapes is misleading: always the TP-2 shapes per GPU, with
  the quantification counted.
- One GPU process at a time; no build and no clippy while a measurement runs (CPU load skews
  it); never touch `sd-server` on the RTX.
- Never `checkout`, `commit`, `stash` or `reset` inside `~/Workspace/flambeau` or
  `~/Workspace/flambeau-impl`: they are shared worktrees.
- The 79–80 test failures without a GPU are GPU tests run with the cards masked
  ("no ROCm-capable device"), not a regression.
- Compare only alternating runs (hot room, MI50 drifts by ±2 %); on MI50 the `mtp.cycle` lines
  are doubled in the logs (take `[::2]`).

## Process

Never push to `main`; one branch per subject, one pull request, merge reserved to Benoît.
Commits, code and documentation in English, no agent signature. Measurements carry the binary
they were taken on.
