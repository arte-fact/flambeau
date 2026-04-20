# Roadmap V1 — Qwen3.6 on gfx906, Max-Perf Vertical Slice

**Status:** Active. First deliverable of the project.
**Arch:** AMD gfx906 (Radeon VII / MI50). Rig: 1× / 2× / 4× MI50 validated in-phase.
**Model:** Qwen3.6 MoE (`Qwen3-30B-A3B` class — 32 layers, 128 experts, top-8, GQA 32/4, head_dim 128), GGUF `Q4_K_M` mix.
**Server:** OpenAI-compatible `/v1/chat/completions` with SSE streaming.
**Posture:** Max perf from the first commit. Multi-device (TP + EP) is in V1, not deferred. No "ship single-warp, optimise later" — we port the first-class kernels directly.

## Why max-perf-first + multi-device-first

Candle shipped in two passes: correct-but-slow, then tune. Result: ~150 kernel variants, ~19 env flags, and a year of "close the gap". Flambeau inherits the destination, not the journey. Every V1 kernel is ported from the known-best variant (candle's latest + llama.cpp turbo-port + llamacpp-turbo), measured against gfx906 theoretical ceilings, and certed before it ships.

Multi-device is in V1 because the abstractions (mesh, collectives, sharded KV cache, distributed MoE router) are architectural, not perf-optimisation. Candle learned this the hard way at X8 (Gemma-4 TP+EP) — bolting it on after single-GPU-shaped traits cost an entire ladder of refactors. We pay the cost once, up front.

## Success criteria

V1 is done when **all** of the following hold:

1. `flambeau serve --model qwen3.6-30b-a3b-q4_k_m --devices hip:0,1,2,3 --port 8080` runs on a 4× MI50 rig. Same binary runs on 2× and 1× with `--devices` change.
2. An OpenAI-compat client (`curl` + Aider or Continue) can chat-complete, non-streaming and SSE.
3. Logit cosine similarity vs llama.cpp on the same GGUF ≥ 0.9999 over a 128-token canonical prompt.
4. **Perf targets (gfx906 ceilings, not llama.cpp-relative):**
   - Decode tg64 on 1× MI50: ≥ 60 tok/s (candle P29 baseline is 52.7; first-class port should exceed).
   - Decode tg64 on 4× MI50 with EP: ≥ 3.0× the 1× number (experts shard linearly on a 128-expert model).
   - Prefill pp512 on 1× MI50: ≥ 600 tok/s (candle B2+C1 got 467; 4-warp LDS-tiled + stream-K + L2-prefetch port should exceed).
   - HBM utilisation during decode ≥ 70% (rocprofv3 `MemBusy` — the real ceiling on MoE decode is HBM; if we're below 70%, a kernel is not first-class).
5. Every kernel in `dispatch/hip/gfx906.toml` has a passing correctness cert **and** a PMC snapshot (VGPR, occupancy, MemBusy, VALUBusy) under `certs/hip/gfx906/`.
6. No `CANDLE_*`-style env flags anywhere in the runtime.
7. Failover: killing one of N GPUs mid-session fails cleanly (no hang, error returned). Graceful multi-device is an architectural property, not a ticket.

Non-goal: "beating llama.cpp". We target the **silicon**. llama.cpp is a reference oracle for correctness and a lower-bound for perf. If we match the HBM ceiling and llama.cpp is faster, they're the ones with room; we file a diagnosis, we don't ship a regression to "catch up".

## Kernel inventory — Qwen3.6 needs (first-class variants only)

Every kernel below is ported from the best-known variant we've identified. Single-warp fallbacks are **not** shipped in V1.

| Family | Dtype(s) | Port target | First-class features |
|---|---|---|---|
| `QMatMul` MMVQ | Q4_K, Q5_K, Q6_K, Q8_0 | candle P29 `*_nw1_r{2,4}` | DPP multi-row reduce (half/quarter-warp), 128-bit Q8_1 loads, dp4a inner, merged-accum |
| `QMatMul` MMQ prefill | Q4_K, Q6_K, Q8_0 | llamacpp-turbo 4-warp LDS-tiled | 4-warp (256-thread) cooperative, stream-K fixup, L2 prefetch, vectorised X/Y tile load |
| `IndexedMoE` MMVQ | Q4_K, Q5_K, Q6_K | candle P29 multi-row + P30 gate+up fused | Multi-row DPP reduce, gate+up single-launch (halves launch count) |
| `IndexedMoE` MMQ | Q4_K, Q6_K | candle P37/P38 turbo_dense pattern applied to MoE | 4-warp LDS-tiled, stream-K, mmq_x tuned per dtype (8 for K-quants on gfx906) |
| `Quantize<Q8_1>` + RMSNorm | F16 → Q8_1 | candle D1 `rmsnorm_q8_fused` | One-launch norm + quant, saves 2× dispatch per layer |
| `SwiGLU` fused gate+up | F16 | candle F-series fused FFN | `silu(gate) * up` single kernel, F16 intermediate |
| `RoPE` | F16 | candle gemma4 interleaved-pair | Q + K in one kernel, rope_theta from GGUF |
| `Softmax` masked | F16/F32 | candle F5 `masked_softmax_scale_fused` | Causal mask + scale fused; `n_rows < 24` fallback to decomposed path |
| `Attention` prefill | F16 / Q4_K weights | candle `flash_attn_v2_kt` | K-transposed layout pre-materialised, V1-correctness then perf |
| `Attention` decode (F16 KV) | F16 KV | candle `gqa_decode_mv_fast_d{128,256}` | Single fused kernel, F32 accumulator |
| `Attention` decode (Q8 KV) | Q8 KV | candle `gqa_attention_decode_q8` + F3 fused-softmax | 9-kernel decomposition where it wins, fused softmax inside |
| `TopK` router | F32 | novel (no candle parity) | Bitonic top-k on-chip, 128 experts → top-8 per token |
| `MoE combine` | F16 | candle C1 `moe_combine` pattern | Weighted sum fused with residual add |

KV cache: **F16 and Q8 both ship in V1.** Qwen3.6 context grows fast; Q8 KV is ~2× HBM saving during decode and decode is HBM-bound. Quality cert (delta-perplexity on `wikitext-2` + multi-turn chat smoke) gates the Q8 layout per model. Turbo-quant (Q4/Q5 KV) is V2 — V1 does not need the extra compression and the quality-cert harness for Q4 KV needs more surface than a vertical slice affords.

## Multi-device plan (single design, from V1.0)

- **Topology.** `runtime::Mesh<N>` is the single abstraction. `Mesh<1>` is a degenerate single-GPU; `Mesh<2>` and `Mesh<4>` are the V1 physical targets.
- **Parallelism modes.**
  - **TP** (tensor parallel) across attention projections, dense MLPs if any, and the MoE router's linear. Each rank holds `1/N` of the output dim; `all_reduce` after the down-projection.
  - **EP** (expert parallel) across the 128 experts — each rank holds `128/N` experts. `all_to_all` sends tokens to the rank that owns their selected expert, compute happens locally, `all_to_all` gathers results.
  - Qwen3.6 ships with TP on attention + EP on experts as the default composition. Dense MLPs (router gate + output head) stay TP.
- **Collectives.** `ops::collective::{AllReduce, AllGather, AllToAll, Broadcast}` are ops like any other — they register backends (RCCL first-class, host-bounce fallback for correctness in the cert harness).
- **KV cache shard.** `KvCache<L>` over a `Mesh<N>` is head-sharded: each rank holds `num_kv_heads / N` heads. No cross-rank KV traffic during decode.
- **Scheduler v1.** One driver thread per rank, `crossbeam` channels for barriers. Launch-and-record on every rank, sync only at collective boundaries. (Candle's X8 host-bounce all-reduce gap closes naturally here because we never bounce.)

The mesh + collectives exist as trait surface from V1.0 so every downstream component is mesh-generic. Running on `Mesh<1>` never branches on `N == 1`; it just takes the trivial path through the same traits.

## V1 breakdown (step order is execution order)

### V1.0 — Workspace + mesh skeleton
- Cargo workspace, crates per `doc/ARCHITECTURE.md`.
- `runtime::Mesh<N>` + `collective::{AllReduce, AllGather, AllToAll, Broadcast}` trait surface with a host-bounce CPU-reference impl. Real RCCL impl lands in V1.2.
- `dispatch/hip/gfx906.toml` exists, empty `[[qmatmul]]` / `[[attention]]` / `[[collective]]` sections.
- `cargo build` clean.
**End:** `flambeau --help` prints subcommands. `cargo test -p runtime` exercises `Mesh<1>` / `Mesh<2>` / `Mesh<4>` against the reference collectives.

### V1.1 — GGUF loader + CPU dequant reference
- `crates/quant` parses GGUF v3, CPU dequant for every dtype Qwen3.6 ships.
- `flambeau inspect-gguf` round-trips a K-quant block bit-for-bit vs llama.cpp.
- Weight loader supports **sharded-on-load** (tensor-range reader per rank — candle X5 pattern) so 30B weights never peak at 2× on multi-rank init.
**End:** Qwen3.6 GGUF loads on 1, 2, 4 ranks with steady-state VRAM = load-time VRAM.

### V1.2 — HIP device + RCCL collectives
- `crates/backend-hip`: `HipDevice`, `HipStream`, allocation, copy H↔D.
- RCCL bindings for `all_reduce_f32`, `all_gather_f16`, `all_to_all_f16`, `broadcast_f16`.
- Collective correctness certs: reference-vs-RCCL match across `Mesh<2>` and `Mesh<4>` on a 2×MI50 and 4×MI50 rig.
**End:** All four collectives green on physical multi-GPU. Subsequent kernel work targets either single-rank or mesh-generic from here.

### V1.3 — First-class MMVQ (Q4_K / Q5_K / Q6_K / Q8_0)
- Port `dequantize_mul_mat_vec_*_q8_1_*_nw1_r{2,4}` variants + `quantize_row_q8_1` from candle (P29-level).
- DPP half/quarter-warp reduce via `arch_primitives/gfx906.cuh`.
- `KernelImpl<QMatMul, HipDevice>` registrations for all four dtypes.
- `bench sweep --op qmatmul --arch gfx906`: correctness (tol 5e-3 × max(|ref|,1.0)) **and** PMC (VGPR, occupancy, MemBusy) logged per shape.
- Shape grid is Qwen3.6-specific: `M ∈ {1,8,16,128,512}`, `K, N ∈ {2048, 5120, 15360, 128256}`.
**End:** Four MMVQ certs green with PMC snapshots. Multi-row default confirmed r2 for Q4_K/Q5_K, r4 for Q6_K (per candle P29).

### V1.4 — First-class MMQ prefill (Q4_K / Q6_K / Q8_0, 4-warp LDS-tiled)
- Port llamacpp-turbo's 4-warp LDS-tiled MMQ, stream-K fixup, L2 prefetch loop — straight to `kernels-hip/mmq.cu`. Include `mmq_tile.cuh` in `kernels-shared/`.
- `mmq_x` tuned per dtype at load time against PMC snapshot (candle P32/P37 showed K-quants want `mmq_x=8` at VGPR=99).
- Cert grid: `M ∈ {128, 512, 2048}`, `K, N` spanning Qwen3.6 attention and MoE projection widths.
- PMC target: ≥ 2 waves/SIMD at `mmq_x=8`, MemBusy ≥ 60% at pp512.
**End:** Three MMQ certs green. Prefill pp512 benchmark on 1× MI50 hits the ≥ 600 tok/s success gate.

### V1.5 — MoE kernels: TopK router, indexed MMVQ/MMQ, fused gate+up, combine
- `TopK` bitonic top-8 over 128 F32 logits — single kernel.
- `IndexedMoE` Q4_K MMVQ r2 (gate, up, down) + fused gate+up nw1_r2 (candle P30 pattern).
- `IndexedMoE` Q4_K MMQ — port of the 4-warp turbo-dense pattern adapted for ids+bounds indirection (candle P37 adapted to MoE).
- `MoE combine` — weighted sum fused with residual add (candle C1).
- Cert: an isolated MoE-layer forward matches CPU reference (dequant + F32 topk + F32 experts + F32 combine) bit-level stable-sort tie-break.
**End:** Every MoE-path kernel is first-class and certed. No single-warp MoE kernel lives in V1.

### V1.6 — Fused decode path + attention + KV cache families
- `RMSNorm + Quantize<Q8_1>` fused (candle D1).
- `SwiGLU` fused gate+up-post (if separate from V1.5's pre-matmul fusion).
- `RoPE` Q+K one-kernel.
- `Softmax masked+scale` fused + `n_rows < 24` decomposed fallback (candle F5).
- Prefill attention: flash-attn-v2 with K-transposed pre-materialised KV.
- Decode attention: `gqa_decode_mv_fast_d{128,256}` for F16 KV; `gqa_attention_decode_q8` + fused softmax for Q8 KV.
- `KvCache<F16Contig>` and `KvCache<Q8Contig>` both instantiable. Quality cert for Q8 on Qwen3.6: delta-perplexity on wikitext-2 ≤ 0.5% and chat smoke passes on 3 multi-turn conversations.
**End:** Every non-collective op Qwen3.6 touches has a certed first-class impl.

### V1.7 — Qwen3.6 model + TP/EP composition + token parity
- `crates/ops/{attention, mlp, moe, norm, pe}` expose mesh-generic building blocks; TP shard dims come from the `Mesh` type.
- `crates/models/qwen3_moe`: weight-name map + block list + `forward_one_token` + `forward_prefill`. Single source, `Mesh<N>`-generic.
- Per-layer sharded loader (V1.1's infrastructure).
- Token-parity cert: first-token logits cosine ≥ 0.9999 vs llama.cpp on `Mesh<1>`, `Mesh<2>`, `Mesh<4>`. Any cross-mesh divergence > cosine 1e-5 fails the gate.
- `bench matrix --devices 1,2,4`: perf numbers snapshotted. Scaling: tg on `Mesh<4>` / tg on `Mesh<1>` ≥ 3.0 is the quantitative gate.
**End:** `flambeau infer --devices hip:0[,1[,2,3]] --prompt <p>` prints a correct next-token with measurable EP scaling.

### V1.8 — Tokenizer, sampler, chat template, OpenAI server
- Tokenizer from GGUF-embedded blob via `tokenizers` crate.
- Chat template via `minijinja` from the GGUF metadata.
- Sampler: CPU-side temperature + top-p on last logit row.
- `crates/server` (`axum`): `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` with SSE.
- `flambeau serve --devices hip:0,1,2,3 --port 8080`.
- `bench/server_smoke.sh` asserts `curl` non-streaming, `curl` SSE, and at least one real OpenAI-compat client (Aider or Continue).
- No continuous batching yet (V2) — single-session server with a queue is enough to validate the server layer over a multi-device mesh.
**End:** All V1 success criteria green. Tag `v0.1.0-qwen36-gfx906`.

## Parallelisation hint (when two can work at once)

V1.0 → V1.1 → V1.2 is strictly sequential. After V1.2:
- V1.3 (MMVQ) and V1.4 (MMQ) can proceed in parallel.
- V1.5 needs V1.3's Q4_K kernels.
- V1.6 can start once V1.5 is half-done (norms/RoPE/softmax don't depend on MoE).
- V1.7 needs everything above; V1.8 is sequential with V1.7 but small.
- **T-track (warmup-tuner) and M-track (MCP server) run alongside from V1.2.** See below.

## Side-tracks — develop alongside main V1 flow

These are independent tracks that share nothing with the main-line kernel work except the cert / bench artefacts produced by V1.2–V1.6. They should be staffed concurrently, not after V1.8.

### T-track — Warmup-tuner (runtime shape-tuned dispatch)

**Premise.** Static `dispatch/hip/gfx906.toml` is authored from the cert grid's median performance; it cannot be optimal for every shape a real Qwen3.6 session hits. A runtime warmup pass that micro-benchmarks the certed variants for the exact shapes this session uses, then writes a machine-local override, captures the last ~10–20% of kernel perf without touching the architecture. Runtime **tuning** (selection between pre-certed variants) is compatible with zero-cost Rust; runtime **codegen** (JIT) is not, and is not on this roadmap.

**Non-negotiable guardrails.**
- Tuner only selects among variants that carry a committed correctness cert. No un-certed kernel ever runs, including during tuning.
- Source of truth is always the committed `dispatch/hip/gfx906.toml`. The tuner writes to a **separate** file (`dispatch/hip/gfx906.local.toml` — gitignored) that layers on top at session init.
- Tuner state is never load-bearing for correctness. If the local file is missing/corrupt, the session falls back to the committed table and logs.
- Tuning is bounded: ≤ 30 s at session start for a target model's shape set; a cache key (model id + mesh shape + driver version) lets subsequent starts skip the pass.

**Steps.**

- **T1 — Shape recorder.** `runtime` hooks into the dispatcher to record every `(op, dtype_tuple, shape)` seen during a first inference. Dump to `shapes/<model_id>.json`. Depends on V1.2.
- **T2 — Variant enumerator.** For each recorded shape, enumerate the certed impls whose predicate matches. Wire into `bench sweep` as `--shapes-from <file>`. Depends on V1.3.
- **T3 — Micro-bench runner.** Per `(op, shape, impl)` triple, run N warmups + M timed iterations, record median wall-clock + PMC snapshot, pick the winner. Output: `dispatch/hip/gfx906.local.toml` rows that override the committed table for this rig's measured-best impl.
- **T4 — Session integration.** `flambeau serve` / `flambeau infer` reads the local override after the committed table; logs any row that differs. `--no-autotune` skips the warmup.
- **T5 — Auto-promote.** If the same local override lands across ≥ 3 independent rigs (CI cluster + dev boxes), promote the row into the committed table via a PR that carries the cert + PMC diff. Tuner data never silently modifies the committed table.

**End:** Qwen3.6 session on any 1/2/4× MI50 rig warms up in ≤ 30 s and runs at ≥ 3% faster tg than the committed-table baseline on that rig. Tuner state is reproducible (same rig, same model → same local file).

### M-track — MCP server for bench/profile/cert loop

**Premise.** Candle's kernel cycles were ~30 min each: edit → rebuild → run → rocprof → grep → hypothesise. An MCP server wrapping the same tooling lets Claude drive the loop in seconds, not minutes, while keeping every finding round-trippable into a committed artefact. This is a dev-velocity tool, not a prod component; prod `flambeau serve` never talks to the MCP server.

**Non-negotiable guardrails.**
- Every MCP-driven finding lands as a committable artefact (cert, matrix snapshot, dispatch row, PMC JSON). No ephemeral live-tune state a prod server reads from.
- No destructive repo actions (no `git push`, no cert rewrites) without explicit user approval per call.
- Server runs locally on the dev rig; never exposed to the internet, never part of deployment.

**Tool surface (MCP tools exposed to Claude).**

- `flambeau_sweep` — `{arch, op?, dtype?, shapes?}` → runs `bench sweep`, returns pass/fail table + path to new/updated certs.
- `flambeau_matrix` — `{models, prompt_lens, tg_lens, devices}` → runs `bench matrix`, returns tok/s table + path to committed snapshot.
- `flambeau_profile` — `{binary_args, counters}` → runs `rocprofv3` with the requested PMC counters, returns structured JSON (per-kernel VGPR, VALUBusy, MemBusy, MemStall, calls, total time).
- `flambeau_dispatch_ab` — `{arch, op, impl_a, impl_b, shapes}` → runs A/B on the two certed impls over the shape list, returns perf table + PMC deltas.
- `flambeau_inspect` — `{artifact: "gguf"|"hsaco"|"ptx", path}` → structured output of the CLI `inspect-*` subcommands.
- `flambeau_cert_diff` — `{impl_id, baseline, current}` → structured diff between two cert JSONs (correctness tolerances, PMC deltas).
- `flambeau_tune_dry` — same as T-track warmup but reports proposed overrides without writing; lets Claude see what the tuner would pick.

**Steps.**

- **M1 — Crate skeleton + protocol.** `crates/mcp-server` with MCP protocol handshake; no tools yet. Depends on V1.0.
- **M2 — Sweep + matrix + inspect tools.** Wrap `bench sweep`, `bench matrix`, `cli inspect-*` directly — no new logic, just structured I/O. Depends on V1.3 (first real certs to sweep).
- **M3 — Profile + dispatch_ab tools.** `rocprofv3` structured PMC output (JSON, not the default CSV); A/B harness that reuses T-track's micro-bench runner. Depends on V1.4 and T3.
- **M4 — Cert diff + tune_dry.** Structured diffs; tuner dry-run that doesn't write `.local.toml`. Depends on T4.
- **M5 — Claude onboarding.** Short `doc/mcp-usage.md` with tool descriptions, example workflows ("find the slowest kernel in decode", "propose a dispatch row change with PMC evidence"). Skill/subagent wiring for Claude Code is out of this repo, but the server exposes the tools cleanly.

**End:** A Claude session on this project can: rebuild an impl, run sweep + profile, diff PMC vs baseline, propose a dispatch change, all without leaving the chat and all producing committable artefacts.

**Owner hint.** T-track and M-track can be the same person or different people. They're independent; neither blocks the other. T1 can start as early as V1.2; M1 can start at V1.0. Both should land before V1.8 so the end-to-end measurement story is coherent when V1 ships.

## Non-goals for V1 (file V2 tickets, do not slip)

- Turbo-quant KV (Q4/Q5 KV cache) — V2.
- Continuous batching — V2.
- Pipeline parallelism (PP) — V2+, if ever.
- `/metrics`, function calling, tool use, logprobs, `/v1/embeddings` — V2.
- Any model other than Qwen3.6 — V2.
- Any HIP arch other than gfx906 — V2.
- CUDA backend — V2.
- gfx1031 portability canary — V2 (it is the portability proof, which lands with the second arch, not the second roadmap).
- Speculative decoding — deferred.

## Risks specific to V1

- **4-warp LDS-tiled MMQ port correctness.** llamacpp-turbo's version is complex (stream-K, L2 prefetch, vectorised X/Y tile loads). Risk: port correctness holes visible only on rare shapes. Mitigation: cert grid includes K-not-multiple-of-QK and N-odd edge cases; compare PMC snapshot against llamacpp-turbo's kernel on the same shapes — divergence > 2× = port bug.
- **EP all-to-all throughput on PCIe Gen3.** Candle's X8 landing showed host-bounced all-reduce was the bottleneck on 4× MI50. Mitigation: RCCL direct GPU↔GPU over PCIe P2P; fall back to host-bounce only if P2P is disabled, and log loudly. Target: all-to-all overhead ≤ 10% of a MoE-layer wall-clock.
- **Q8 KV quality gate.** delta-perplexity ≤ 0.5% is a tight threshold. Mitigation: quality cert is part of V1.6 gate; if Q8 KV fails quality on Qwen3.6 specifically, ship F16 KV only and open a V2 ticket — **do not** relax the gate.
- **Chat template correctness on Qwen3.** Qwen3 has thinking-mode variants (`<think>…</think>`). Mitigation: byte-for-byte match against llama.cpp `common_chat_apply_template` on at least 5 multi-turn conversations including thinking-mode prompts.
- **Multi-device load-time VRAM spike.** 30B weights at 2× peak on an 8×2GB rank init would OOM. Mitigation: per-rank tensor-range reader (V1.1) reads only its shard from GGUF mmap.
- **RCCL version drift.** ROCm 7.1.1 RCCL is the pinned version per candle memory (7.2.1 is slower on gfx906). Document and CI-check.
- **T-track overrides drifting from committed table.** If the local autotuned overrides silently become the de-facto source of truth, the committed table rots. Mitigation: T5 auto-promote flow is the only path from `.local.toml` to committed, gated on ≥ 3 independent rigs agreeing; `bench cert-check` warns if committed rows have been overridden on this rig for > N sessions.
- **M-track tooling becoming load-bearing.** Risk: a "quick MCP tune" lands in a cert without human review. Mitigation: MCP tools that write to the repo require explicit user approval per call; `flambeau_sweep` and `flambeau_matrix` write to their normal repo paths, not MCP-specific ones, so they show up in `git status` the same way manual runs do.

## Verification checklist (V1 done gate)

- [ ] `cargo build --release` clean, no warnings gate.
- [ ] `bench sweep --arch gfx906` all green, PMC snapshots recorded, certs committed.
- [ ] Quality cert for `KvCache<Q8Contig>` on Qwen3.6 green (delta-ppl ≤ 0.5%, chat smoke passes).
- [ ] `flambeau infer` token-parity cosine ≥ 0.9999 vs llama.cpp on `Mesh<1>` and `Mesh<4>`.
- [ ] `bench matrix` recorded: Qwen3.6 pp512 / tg64 on 1×, 2×, 4× MI50. Scaling tg `Mesh<4>` / `Mesh<1>` ≥ 3.0.
- [ ] Decode HBM utilisation ≥ 70% via rocprofv3.
- [ ] Single-GPU decode ≥ 60 tok/s on Qwen3-30B-A3B Q4_K_M, pp512 ≥ 600 tok/s.
- [ ] `flambeau serve --devices hip:0,1,2,3` + `bench/server_smoke.sh` green.
- [ ] At least one real OpenAI-compat client round-trips successfully.
- [ ] Zero env-var variant gates (`rg 'env::var' crates/` is empty).
- [ ] `dispatch/hip/gfx906.toml` rows each match a cert (`bench cert-check` passes).
- [ ] `kill -9` on one of 4 rank drivers returns a clean 500 to the client; no runtime hang.
- [ ] **T-track:** warmup-tuner produces a `dispatch/hip/gfx906.local.toml` on a reference rig in ≤ 30 s; repeated runs on the same rig cache and skip. Session with `--no-autotune` runs unaffected.
- [ ] **T-track perf gate:** tuner-on run ≥ 3% faster tg than committed-table-only run on the reference rig for Qwen3.6.
- [ ] **M-track:** MCP server exposes at least `flambeau_sweep`, `flambeau_matrix`, `flambeau_profile`, `flambeau_dispatch_ab`, `flambeau_inspect`, `flambeau_cert_diff`, `flambeau_tune_dry`. A scripted Claude-driven session successfully rebuilds one kernel variant, runs sweep + profile, commits a cert update.

## What V2 looks like (not in scope, for context)

Gemma-4 and Mistral dense land; turbo-quant KV arrives with quality-cert expansion; CUDA sm_86 on RTX 3090 ports the V1 kernel set; gfx1031 (RX 6750 XT) arrives as the wave32 portability canary; continuous batching unlocks the multi-slot server; function calling and tool use join the server surface as they're needed by concrete target clients.
