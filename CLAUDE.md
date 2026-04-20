# Flambeau — Working Notes for Claude

Inference-only framework for modern LLMs on HIP + CUDA, served over an OpenAI-compatible HTTP API. Greenfield sibling of `/artefact/candle` — we keep candle's kernel learnings, drop its architectural accretion. **Metal is out of v1 scope** — do not add Metal-shaped hedges to trait design.

**Current roadmap: `doc/ROADMAP-V1-QWEN36-GFX906.md`** — a vertical slice shipping end-to-end Qwen3.6 inference on a single MI50 with an OpenAI-compat server. Read it before starting any work. The slice is deliberately narrow (one model, one arch, F16 KV, single-warp MMQ) so the *architecture* gets exercised top-to-bottom before we spread horizontally. Horizontal expansion (more models, more archs, more KV layouts, perf closure) is V2+.

Read `doc/ARCHITECTURE.md` for the architectural rules and `doc/ROADMAP-V1-QWEN36-GFX906.md` for the execution plan before touching anything substantive. The rules below encode several sessions of candle-fork lessons; violate one only with a measured reason.

## Vertical-slice posture (V1): max perf + multi-device from day one

V1 is a vertical slice — one model, one arch — but **not** a "parity-first, tune-later" slice. The first commit targets max perf; multi-device (TP + EP) is in, not deferred.

- **One model in V1: Qwen3.6 MoE (30B-A3B class), Q4_K_M GGUF.** The hardest target — MoE + GQA + RoPE + mixed K-quants. Other models are V2+.
- **One arch in V1: HIP gfx906 (MI50).** CUDA (RTX 3090) + gfx1031 (RX 6750 XT) are V2+; do not write CUDA or RDNA code in V1.
- **Two KV layouts in V1: F16 and Q8.** Q8 KV is first-class because decode is HBM-bound on MoE and Q8 is ~2× bandwidth saving. Quality cert (delta-ppl ≤ 0.5% on wikitext-2 + chat smoke) gates Q8 per model. Turbo-quant (Q4/Q5 KV) is V2.
- **First-class kernels only.** No single-warp MMQ placeholder. V1 ports the best known variants directly:
  - MMVQ: candle P29 multi-row DPP-reduce (`*_nw1_r{2,4}`) — r2 for Q4_K/Q5_K, r4 for Q6_K.
  - MMQ prefill: llamacpp-turbo 4-warp LDS-tiled + stream-K + L2 prefetch.
  - MoE: P29 multi-row MMVQ + P30 fused gate+up + P37-pattern 4-warp MMQ.
  - Fused decode path: D1 rmsnorm+Q8_1, F5 masked-softmax, F3 fused-Q8-attention where it wins.
- **Multi-device is architectural, not V3.** `runtime::Mesh<N>` + collectives are V1.0 trait surface; RCCL collectives are V1.2. Every downstream component is mesh-generic from inception; `Mesh<1>` is a degenerate instance through the same trait. V1 ships on 1×, 2×, and 4× MI50 with a `≥ 3.0×` EP scaling gate at `Mesh<4>`.
- **Perf bar is silicon, not llama.cpp.** Target ≥ 70% HBM utilisation on decode, ≥ 60 tok/s tg64 on 1× MI50, ≥ 600 tok/s pp512. llama.cpp is a correctness oracle and a lower-bound reference; if we match the silicon ceiling and they're higher, we file a diagnosis, we don't ship a regression to "catch up".
- **Minimal server surface in V1.** `/v1/chat/completions` + SSE, `/v1/completions`, `/v1/models`, `/health`. `/metrics`, function calling, tool use, logprobs, `/v1/embeddings` are V2+.

V1 touches every layer (loader → kernels → ops → mesh → model → runtime → server → client), at the first-class perf point, so the abstractions get real stress from real load. If a layer feels bent out of shape to accommodate Qwen3.6 at max perf on 4× MI50, fix the abstraction in V1 — not later under a second model's pressure.

**Side-tracks (dev-velocity, not V1 blockers, land before V1.8):**
- **T-track — warmup-tuner** (`crates/autotune`, `cli tune`): session-init micro-bench of certed variants, writes a gitignored `dispatch/<backend>/<arch>.local.toml`. Runtime selection only, never runtime codegen. Source of truth stays the committed table; promotion to committed requires ≥ 3 independent rigs agreeing + a PR with cert + PMC evidence.
- **M-track — MCP server** (`crates/mcp-server`, `cli mcp`): dev-only, wraps `bench sweep`, `bench matrix`, `rocprofv3`, A/B dispatch, `inspect-*`, cert diff, tune dry-run. Every finding round-trips into a committable artefact (cert, matrix snapshot, dispatch row). Never load-bearing for production — `flambeau serve` in prod does not talk to the MCP server.

See `doc/ROADMAP-V1-QWEN36-GFX906.md` §Side-tracks for step-by-step T1–T5 and M1–M5.

## Scope reminder

- **Models:** Mistral/Devstral dense, Gemma-4 dense+gated, Qwen3.5 dense, Qwen3.6 MoE, Qwen3-Coder MoE, Qwen3-Next (GDN hybrid). Nothing else.
- **Backends:** HIP + CUDA only. **Metal is out of v1 scope.**
  - HIP test arch: **gfx906 (MI50)** — daily target, wave64, GCN cross-lane.
  - HIP portability canary: **gfx1031 (RX 6750 XT, RDNA2, Navi 22)** — wave32, different DPP, no matrix cores, smaller LDS. Physically available and exercised every Phase 2+ gate. If the abstraction is wave-leaky, gfx1031 is where it surfaces first.
  - Other HIP targets: gfx908, gfx90a, gfx942, gfx1100 — dispatch-matrix inheritance + `arch_primitives/*.cuh`, not kernel rewrites.
  - CUDA dev arch: **sm_86 (RTX 3090)**. Baseline sm_80 (A100); sm_89/sm_90 forward-compatible.
- **Quant:** GGUF only. **Full coverage from Phase 2** — Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, BF16, F32. No "only-if-needed" escape.
- **KV cache:** three layout families, all from v0: **F16** (baseline), **Q8** (quantX, candle shape), **turbo-quant** (Q4/Q5 llamacpp-turbo scheme). Static typing; turbo-quant requires a quality cert.
- **Server:** OpenAI-compatible HTTP API from Phase 4 (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE, `/health`, `/metrics`). Function calling / tool use / structured outputs are v2 features.
- **Out:** Metal, training, LoRA, vision/audio, speculative decoding, ONNX, non-target model families.

If a request points outside scope, surface the mismatch before implementing.

## Architectural rules (do not quietly break)

1. **No `CANDLE_*`-style env flags.** All variant selection lives in `dispatch/<backend>/<arch>.toml`, reviewed in PRs. If you feel the need to gate behavior on an env var, the dispatch table is the right place. The warmup-tuner may write a machine-local override at `dispatch/<backend>/<arch>.local.toml` (gitignored), but **only** for rows whose impls carry a committed correctness cert — never a new impl, never a new shape-predicate.
2. **No kernel ships without a cert.** Every `impl_id` referenced from `dispatch/*.toml` must have a matching `certs/<backend>/<impl_id>.json` produced by the sweep harness. Uncertified kernels stay behind `#[cfg(unverified)]` and are unreachable from the runtime.
3. **One contract per op, many impls.** `core::op::Op` is a trait; kernel impls register against it. Never add a parallel op trait for "the fast path" — write another impl and let dispatch pick.
4. **Models are glue, not kernels.** `crates/models/*` compose blocks from `crates/ops/*`. A PR that adds a kernel file under `crates/models/` is wrong — push the kernel into `crates/ops` (if genuinely new) or reuse.
5. **Shared `.cuh` stays backend-neutral.** `kernels-shared/include/*.cuh` has **zero** `#ifdef __HIP_PLATFORM_AMD__` / CUDA intrinsic calls. Intrinsic use goes in `kernels-hip/` or `kernels-cuda/` respectively. Algorithmic core only in shared. HIP arch divergence (DPP lane configs, gfx906-only intrinsics, wave64↔wave32 on RDNA) lives in `kernels-hip/arch_primitives/{gfx906,gfx908,gfx90a,gfx942,gfx1031,gfx1100}.cuh` — same signatures, arch-specific bodies. **Never hard-code `WAVE_SIZE = 64`** anywhere; use the per-arch constant from `arch_primitives`. gfx1031 will catch you if you do.
6. **KV-cache layout is a type, not a flag.** `KvCache<F16Contig>`, `KvCache<Q8Transposed>`, `KvCache<TurboQ4Contig>` are distinct types. Don't collapse them into a runtime enum — the compiler catches invalid dispatch paths for free. Turbo-quant layouts additionally require a quality cert (delta-perplexity + chat smoke) on top of the correctness cert before they appear in any dispatch row.
7. **Explicit async.** Kernel launches take `&Stream`. No hidden `hipDeviceSynchronize` except session boundaries and in `sweep`.
8. **No `alloc_zeros` in hot paths.** Use `alloc` and let the kernel write every output lane. This was a P24/P29-era candle cleanup; start right.
9. **Naming convention for kernel variants:** `{op}_{dtype}_{backend}_{shape_tag}_{variant}`. Grep-able. No `_v2f_tile32_repacked` suffix drift.
10. **`#[cfg(unverified)]` is the only place a broken-but-kept kernel lives.** If a kernel is null on all target models (dp4a-fattn, Q5_K v3, fused Q8 attention null), either it gets deleted or it lives behind `cfg(unverified)` with a one-line `why:` comment. No env-flag resurrection.

## Measurement rules (inherited from candle `feedback_*` memories)

### Before claiming any perf result

1. **Rebuild before measuring.** Stale binaries cost a session in candle ("template == B2"). `cargo clean -p kernels-hip && cargo build --release` when touching kernels, then `cargo run -p bench -- matrix`.
2. **Use ≥512-token prompts for prefill.** Short prompts invert real ratios. Bench at pp=512 minimum; pp=128 only for decode-focused reports, noted as such.
3. **Compare like-for-like.** Same GGUF, same `--tg-len`, same GPU, same driver. Log all four in the result line.
4. **Run `sweep` first.** No perf claim before the correctness cert is green on the impl under test. A "3× faster" kernel that silently wrong is not faster.
5. **Profile with rocprofv3 (HIP) or Nsight (CUDA).** Attribute wall-clock gains to specific kernels. If the gain is "everywhere 2%", it's noise or scheduler variance.
6. **PMC check before proposing a perf lever.** `waves_per_eu`, VGPR tuning, LDS double-buffer — none of these are automatic wins. Check:
   - `VGPR headroom to next wave threshold` (at-floor = no-op)
   - `VALUBusy` (compute utilization)
   - `MemBusy` + `MemStall` (bandwidth vs latency)
   Propose the lever that matches the bottleneck, not the one that sounds good.
7. **No projection without a measured basis on this hardware.** "This should give 2×" is not acceptable without a measured microbench citing the same silicon.

### When reporting a result

- **Null results are first-class.** Write them up with the diagnosis (why it didn't work) and land the kernel behind `cfg(unverified)` if it might still pay off on different silicon or in a fused form.
- **Don't stop at parity.** The user's bar is beating llama.cpp, not tying. After any 0.95×–1.05× result, immediately profile for the next lever.
- **The cert + PMC snapshot IS the report.** Paste the diff in the PR body; no separate markdown writeup unless the user asks.

## Technical lessons from candle (don't relearn these)

### Kernel / silicon

- **VGPR count is first-order on gfx906.** Crossing 2 → 10 waves/SIMD beats ILP gains from unrolling. Q4_K turbo sits at 2 waves/SIMD; the next structural win is 4-warp LDS-tiled, not more unrolling.
- **MMQ turbo 4-warp LDS-tiled + stream-K + L2 prefetch is the gold-standard prefill kernel.** llama.cpp has this for every K-quant; candle shipped single-warp for most. Port the 4-warp structure directly, don't hand-roll.
- **Multi-row DPP reduce** (`gfx906_half_warp_reduce_sum_dpp`, `gfx906_quarter_warp_reduce_sum_dpp`) is the MMVQ decode pattern. Per-kernel 1.8× on Q4_K/Q5_K/Q6_K MoE (P29). Default r2/r4, not r1.
- **Q8 KV cache + decomposed GEMV** (`gqa_decode_gemv_{v,qk}_q8`) is a win on models where the F32 fused kernel *isn't* already compute-light. Gemma-4's `gqa_decode_mv_fast_d{256,512}` beats Q8 direct-GEMV by 8.8% — same wire-up, opposite sign. **Baseline shape matters more than the new kernel's shape.**
- **dp4a wins when Q-quant amortizes over many rows (MMVQ).** dp4a loses on decode attention (single Q row) because per-token int→float + Q-quant overhead swamps FMA savings. P10 was null for this reason.
- **Fusion is not automatic.** Fusing rmsnorm+Q8-quant was a win (D1) because the unfused path was bandwidth-bound. Fusing Q8 attention into one kernel was null (F4) because the baseline F32-fused kernel was already compute-light. **Profile the baseline before proposing fusion.**
- **Zero-fill elimination** (converting `alloc_zeros` → `alloc` in MMQ) saved 320ms GPU on gemma4-26B. Start with `alloc`.
- **rocBLAS handle drop-order.** Declare the `blas` field *before* `stream/modules` in `HipDevice` — Rust drops in declaration order. `Mutex<Option<RocBlas>>` for explicit take.
- **Small-M → cuda1 template.** Q4_K/Q5_K MMVQ `wc` (64-thread) loses to cuda1 (256-thread) at M ≤ 16. Dispatch table row, not a flag.

### Measurement / profiling

- **Stale-default env flag audits pay repeatedly.** Candle had 5 stale-default wins in one session (P24/P31/P32/P33/P35). In flambeau the equivalent is: **dispatch table rows that disagree with the latest cert**. Before adding a kernel, sweep existing rows.
- **CANDLE_LAYER_PROFILE style sync-based profilers inflate phase times.** Real attribution goes through rocprofv3/Nsight kernel-level counters, then wall-clock end-to-end. If the two disagree, trust wall-clock.
- **MemBusy ≈ 65% + MemStall ≈ 1% = bandwidth-bound.** Extra waves don't help. VGPR-reduction doesn't help. The fix is structural (LDS tile, prefetch, or lower-bandwidth kernel).
- **Progressive dispatch already overlaps Rust+GPU.** Hip-graph capture ("G3") on gfx906 is null to slightly negative. Don't propose it as a perf win on MI50.

### Architectural

- **Model-specific wins don't generalize.** Gemma-4 and Qwen3.x often prefer opposite dispatch rows at the same dtype. That's what the dispatch table is for — don't force a global "best" kernel.
- **Structural perf work takes multiple sessions.** 4-warp LDS-tiled MMQ is not a one-session port. Resist the end-of-session urge to simplify scope; multi-session work stays multi-session.
- **Correctness-sweep harness FIRST.** Candle's Q5_K v3 + C2/C3 template extension kept destabilizing because there was no per-variant correctness harness. In flambeau, the `sweep` CLI is a Phase-0 deliverable, not an afterthought.

## Collaboration norms

- **Be honest about null results.** Don't reshape a null into a partial win. Null kernel gets `cfg(unverified)` + a one-line diagnosis, not a flag-gated ship.
- **Never rush near end of session.** If context is filling and the work isn't done, stop and write a handoff, don't cut scope to produce a "ship" illusion.
- **No summary paragraph after a diff.** The diff is the work. One-sentence end-of-turn if anything.
- **Ask before destructive actions.** `cargo clean`, deleting kernels from `cfg(unverified)`, force-pushing, rebuilding certs — all require confirmation. Auto mode doesn't override this.
- **Respect user's gfx906 focus.** Perf claims are calibrated against gfx906 theoretical ceilings (1 TB/s HBM, 13.4 TFLOPS F32, 26.5 TOPS int8), not against llama.cpp on A100.
- **Target silicon, not llama.cpp.** The bar is the HBM / compute ceiling on gfx906. llama.cpp is a correctness oracle and a lower bound. If we hit the silicon ceiling and llama.cpp is faster, they have room; we file the diagnosis, we do not ship a regression to "catch up".

## Build / run conventions

- Workspace root: `/artefact/flambeau`. Sibling of `/artefact/candle`.
- `cargo build --release` for perf work; debug builds are misleading for anything kernel-adjacent.
- Kernel source changes: rebuild `kernels-hip` or `kernels-cuda` explicitly (`cargo clean -p kernels-hip` then build). `.hsaco` caching has bitten us.
- `cargo run -p cli -- infer …` — inference.
- `cargo run -p bench -- sweep --arch <gfx906|sm_86|…> [--op qmatmul|attention|…]` — correctness + PMC sweep, updates certs.
- `cargo run -p bench -- matrix [--models …] [--prompt-len 512] [--tg-len 64]` — perf regression matrix.
- `cargo run -p bench -- quality --model <id> --kv-layout <turbo_q4|turbo_q5>` — delta-perplexity + chat smoke for turbo-quant layouts; updates `certs/quality/`.
- `cargo run -p cli -- inspect-gguf <path>` — tensor listing, dtype audit.
- `cargo run -p cli -- inspect-hsaco <path>` — kernel symbols + VGPR budgets (HIP).
- `cargo run -p cli -- inspect-ptx <path>` — kernel symbols + register usage (CUDA).
- `cargo run -p cli -- serve --model <id> --port 8080` — start OpenAI-compatible HTTP server.
- `cargo run -p cli -- tune --model <id> --devices hip:0,1,2,3` — T-track warmup-tuner: records session shapes, micro-benches certed variants, writes `dispatch/<backend>/<arch>.local.toml`. `--dry-run` reports proposed overrides without writing.
- `cargo run -p cli -- mcp --port 9090` — M-track MCP server. Dev-only; never expose to the network. Tools: `flambeau_sweep`, `flambeau_matrix`, `flambeau_profile`, `flambeau_dispatch_ab`, `flambeau_inspect`, `flambeau_cert_diff`, `flambeau_tune_dry`.

## Quick-reference: pre-claim checklist

Before any "this is faster" / "this is correct" claim:

- [ ] Rebuilt the crate that owns the changed code (`cargo clean -p <crate>` if kernels touched).
- [ ] Sweep green on the impl (`cargo run -p bench -- sweep --impl <id>`).
- [ ] PMC snapshot matches the expected bottleneck (bandwidth-bound vs compute-bound vs latency-bound).
- [ ] Wall-clock end-to-end bench at pp≥512 tg≥64 on at least one target model per backend.
- [ ] Diff vs previous cert snapshot documented in the PR body.
- [ ] Null results filed honestly (no "slight regression" rewording).

## Pointers

- `doc/ARCHITECTURE.md` — framework design (crates, traits, dispatch, KV cache families, phasing).
- `doc/ROADMAP-V1-QWEN36-GFX906.md` — current execution plan; step-by-step V1.0 → V1.8.
- `/artefact/candle/` — source framework; read for kernel prior art, *not* for architectural patterns.
- `/artefact/candle/candle-hip-kernels/src/` — the kernels we port (gfx906_primitives.cuh, quantized.cu, mmq_turbo.cu, attn_q8_kv.cu, gated_delta_net.cu).
- `/artefact/llama.cpp/` — correctness oracle + perf benchmark. `ggml-cuda/mmq*.cu` is the 4-warp LDS-tiled reference for V2.
- `/artefact/llamacpp-turbo/` — additional reference for turbo kernel shape and turbo-quant KV-cache layout.
- `/home/sandbox/.claude/projects/-artefact-candle/memory/` — accumulated candle-side learnings. Relevant files are linked in candle's `MEMORY.md`; flambeau's own memory lives under `/home/sandbox/.claude/projects/-artefact-flambeau/memory/` once this project accumulates history.
