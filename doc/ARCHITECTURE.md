# Clean-Slate Inference Framework — Architecture Draft

**Status:** Draft v0. Supersedes the D0-consolidation plan (kept below as Appendix A).

## Context

Candle's HIP backend has become a fast but entangled codebase: ~150 quant-kernel variants, ~19 `CANDLE_*` env flags, parallel dense/MoE drivers, model-by-model accretion, no per-kernel correctness contract, no data-driven dispatch. The wins we've shipped (P24/P29/P31/P32/P34/P35/P37/P38) prove the silicon has headroom; the drag is architectural, not algorithmic.

A greenfield framework inherits candle's *learning* — kernel layouts, DPP/SFU primitives, Q8 KV cache, MMQ template shape, rocBLAS integration, GGUF layout — but drops:

- env-flag variant selection (replaced by a data-driven dispatch matrix),
- parallel implementations without correctness certificates,
- every model family we no longer care about (Llama1/2, T5, Falcon, RWKV, Phi, Stable-LM, BERT, Whisper, Stable-Diffusion, Segment-Anything…),
- train-only ops and dynamic-shape autograd infrastructure,
- the Python-parity tensor-algebra surface (we only need what inference asks for).

Goal: a lean, trait-based, inference-only framework that runs **modern LLMs** (Mistral dense, Gemma4 dense+gated, Qwen3.x dense + MoE + Qwen3-Next GDN) on **HIP + CUDA** with a single composable dispatch story, exposed over an **OpenAI-compatible HTTP server**.

## Scope (what ships)

- **Backends.** HIP + CUDA only. Metal is **permanently out of v1 scope** (not deferred, not a later phase — do not add Metal-shaped hedges to trait design).
  - **HIP test arch:** gfx906 (Radeon VII / MI50). All PMC budgets, cert baselines, and default dispatch tables are tuned here first.
  - **HIP portability:** gfx908 (MI100), gfx90a (MI210/250), gfx942 (MI300) on the CDNA line, and **gfx1031 (RX 6750 XT, RDNA2, Navi 22)** + gfx1100+ (RDNA3) on the consumer line are second-class but first-principled. Kernel code uses `#if __gfx906__` guards only around true ISA divergence (DPP lane configs, matrix-core intrinsics, **wave size**). Adding an arch = adding dispatch rows + certs, not rewriting kernels.
  - **RDNA2 portability is a first-class test.** gfx1031 (RX 6750 XT) runs **wave32**, has different DPP semantics, no matrix cores, and a smaller/differently-structured LDS vs gfx906's wave64 + GCN-style cross-lane ops. If our wave-width and cross-lane abstractions are leaky, gfx1031 is where it surfaces. Treat gfx1031 as the "does the abstraction actually work across arches" canary, second only to gfx906 in priority.
  - **CUDA dev arch:** RTX 3090 (sm_86, Ampere). sm_80 (A100) is the stated baseline; sm_89 (Ada) and sm_90 (Hopper) are forward-compatible via `__CUDA_ARCH__` guards only for primitives (TMA, wgmma) that genuinely need them.
- **Models.** Mistral-7B / Devstral-24B (dense), Gemma-4 (dense + gated attention + sliding window), Qwen3.5 dense, Qwen3.6 MoE, Qwen3-Coder-MoE, Qwen3-Next (GDN + MoE hybrid).
- **Formats.** GGUF only. Users with safetensors convert upstream (e.g. llama.cpp `convert.py`). One loader, one quant story, no runtime-quantize path.
- **Quant — full coverage from Phase 2.** Every GGUF block dtype in production use on target models ships at v1: **Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K** (K-quants), **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0** (legacy), plus **F16, BF16, F32** passthrough. No "only if a model needs it" escape hatch — coverage is a deliverable, not a follow-up.
- **KV-cache layouts — three families, all from v0:**
  - **F16** — `F16Contig`, `F16Transposed` (llama.cpp-parity baseline; correctness reference).
  - **Q8 (`QuantX`)** — `Q8Contig`, `Q8Transposed` (candle's Q8 KV cache shape; ~2× memory saving, near-F16 quality).
  - **Turbo-quant** — packed Q4/Q5 KV cache layouts ported from llamacpp-turbo's KV-quantization scheme. Higher compression, accepted quality cost, gated per-model via the dispatch table.
  - Selection is per-model in the runtime config (not per-call, not env-flag).
- **Parallelism.** Single GPU, TP (attention + dense MLP), EP (MoE experts), no PP (Phase-8 optional).
- **Runtime.** Synchronous decoder for a single session; batch-of-one first, continuous-batch second.
- **Serving.** OpenAI-compatible HTTP server as a first-class deliverable (Phase 4+): `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming. Compatible with anything that speaks the OpenAI REST dialect (llama.cpp server, vLLM clients, LM Studio, Continue/Cline, Aider, LangChain).

Explicitly **out**: training, LoRA, quantization-aware training, vision/audio, speculative decoding (deferred), ONNX, TensorRT, **Metal/MPS** (no Apple-silicon path in v1).

## Architectural principles

1. **Contracts over implementations.** Each op (MatMul, QMatMul, Attention, RMSNorm, RoPE, Softmax, Silu, QuantizeQ8, Dequantize) is a `trait` with typed inputs/outputs. Concrete kernels `impl` the contract. Many impls per contract; one contract per op.
2. **Data-driven dispatch.** A `DispatchTable` loaded at init — TOML/RON — maps `(backend, dtype_tuple, shape_predicate) → impl_id`. No `if env::var(...)`, no `match dtype` branches buried in Rust. Overridable via config file, never env var.
3. **Correctness-cert gating.** Every `impl_id` in the dispatch table must carry a `CorrectnessCert` artifact (generated by the variant-sweep harness vs a GPU-dequant+F32 reference). An uncertified kernel compiles behind `#[cfg(unverified)]` and is unreachable from the dispatcher.
4. **Zero-cost typing.** Const generics for head-dim (`D: usize`), block-quant size (`QK: usize`), rows-per-block. Type-state for KV-cache shape (F16 vs Q8, transposed vs not). The compiler — not the dispatcher — rejects invalid combinations when the combination is statically known.
5. **Explicit async.** Every kernel call takes `&Stream`; sync is the caller's choice. No hidden `hipDeviceSynchronize` except at session boundaries.
6. **Single source of truth per kernel family.** One `.cu` per kernel family per backend, parameterized by traits/templates — *not* hand-duplicated dense↔MoE pairs (current candle carries two `mul_mat_q4_K_*` drivers that differ only in id/bounds indirection).
7. **Measurement-first.** `bench` is a first-class crate, not an example. Correctness + PMC (VGPR, occupancy, MemBusy) counters captured per kernel, stored alongside the cert, shown in `DISPATCH.md`.

## Crate layout

```
workspace/
├── crates/
│   ├── core/            # Device-independent: Tensor, DType, Shape, StorageView, Op traits, DispatchTable, Registry
│   ├── quant/           # GGUF block layouts, CPU dequant reference, QTensor metadata
│   ├── kernels-hip/     # .cu sources targeting HIP (gfx906+). Includes gfx906_primitives.cuh (DPP, SFU).
│   ├── kernels-cuda/    # .cu sources targeting CUDA (sm_80+). Includes ampere_primitives.cuh (cp.async, WMMA).
│   ├── kernels-shared/  # Algorithmic-core .cuh only: MMQ tile structure, block-quant unpack, softmax math.
│   │                    #   Zero backend-specific intrinsics; both kernels-hip and kernels-cuda #include from here.
│   ├── backend-hip/     # HIP Device/Stream + kernel module + impl registrations
│   ├── backend-cuda/    # same, sm_80+
│   ├── ops/             # High-level fused ops (Attention, MoE, FFN, GatedDeltaNet) composed from core traits
│   ├── models/          # Model-family implementations — block compositions only, zero new kernels here
│   │   ├── mistral/
│   │   ├── gemma4/
│   │   ├── qwen3/
│   │   ├── qwen3_moe/
│   │   └── qwen3_next/
│   ├── runtime/         # KV cache (typed F16/Q8/turbo-quant), session, sampler, tokenizer glue, TP/EP primitives
│   ├── bench/           # First-class bench + correctness-sweep harness; emits certs
│   ├── server/          # OpenAI-compatible HTTP server (/v1/chat/completions, /v1/completions, /v1/models, SSE)
│   └── cli/             # `infer`, `bench`, `sweep`, `inspect-gguf`, `inspect-hsaco`, `serve`
├── dispatch/
│   ├── hip/
│   │   ├── gfx906.toml  # Primary tuned matrix (test target, wave64, GCN cross-lane)
│   │   ├── gfx908.toml  # MI100 (inherits + overrides gfx906)
│   │   ├── gfx90a.toml  # MI210/MI250
│   │   ├── gfx942.toml  # MI300
│   │   ├── gfx1031.toml # RX 6750 XT (RDNA2, Navi 22, wave32 — portability canary)
│   │   └── gfx1100.toml # RDNA3 (wave32 + WMMA)
│   └── cuda/
│       ├── sm_86.toml   # RTX 3090 (primary dev target)
│       ├── sm_80.toml   # A100 baseline
│       ├── sm_89.toml   # Ada
│       └── sm_90.toml   # Hopper (TMA/wgmma opt-ins)
└── certs/               # One JSON per (arch, impl_id): correctness + PMC snapshot, reviewed in PRs
```

## Core traits (sketch)

```rust
// crates/core/src/device.rs
pub trait Device: Send + Sync + 'static {
    type Stream: Stream;
    type Allocator: Allocator;
    fn default_stream(&self) -> &Self::Stream;
    fn alloc<T: Pod>(&self, n_elems: usize) -> Result<Storage<T, Self>>;
    fn synchronize(&self) -> Result<()>;
}

// crates/core/src/tensor.rs
pub struct Tensor<'d, D: Device> {
    storage: StorageView<'d, D>,
    shape: Shape,
    dtype: DType,
    strides: Strides,
}

// crates/core/src/qtensor.rs
pub struct QTensor<'d, D: Device> {
    storage: StorageView<'d, D>,   // raw packed quant blocks
    layout: QLayout,               // dtype + block_size + superblock metadata offsets
    shape: Shape,
}

// crates/core/src/op.rs
pub trait Op {
    type Input<'a, D: Device>;
    type Output<'a, D: Device>;
    type Cfg;
    fn contract(cfg: &Self::Cfg, input: &Self::Input<'_, impl Device>) -> OpContract;
}

// The contract is the typed promise the dispatcher queries.
pub struct OpContract {
    pub output_shape: Shape,
    pub output_dtype: DType,
    pub min_tolerance: Tolerance,   // max_abs / max_rel vs reference
}

// A concrete kernel implements the op for a specific backend.
pub trait KernelImpl<O: Op, D: Device>: Send + Sync {
    const ID: &'static str;
    fn applies(input: &O::Input<'_, D>, cfg: &O::Cfg) -> bool;
    fn launch<'s>(
        &self, stream: &'s D::Stream,
        input: O::Input<'_, D>, cfg: &O::Cfg,
    ) -> Result<O::Output<'s, D>>;
    fn cert() -> &'static CorrectnessCert;
}
```

Concrete ops to define early: `MatMul`, `QMatMul`, `Quantize<Q8_1>`, `Dequantize`, `RMSNorm`, `RoPE`, `Softmax`, `Silu`, `Gelu`, `Attention` (one contract, multiple impls — rocBLAS-decompose, fused-flash, fused-Q8-decode, tile-Q8), `MoE` (router + indexed-expert-matmul + combine).

## Dispatch table (example)

```toml
# dispatch/hip.toml
[[qmatmul]]
dtype    = "Q4_K"
shape    = { m = ">=128", k = "any", n = "any" }
impl     = "mul_mat_q4_K_turbo_dense_4warp_x8"
cert     = "certs/hip/mul_mat_q4_K_turbo_dense_4warp_x8.json"

[[qmatmul]]
dtype    = "Q4_K"
shape    = { m = "1..16", k = "any", n = "any" }
impl     = "dequantize_mul_mat_vec_q4_K_q8_1_cuda1"
cert     = "certs/hip/dequantize_mul_mat_vec_q4_K_q8_1_cuda1.json"
```

The dispatcher selects the most-specific-matching row; two rows matching the same shape = build-time error. `sweep` generates/refreshes certs; CI fails if a live table row has no cert.

## Model-layer split

Models import **only** from `ops`, never from `backend-*` or `kernels-src`. A model file is ~300–800 lines: block composition, weight-name mapping, forward step. Shared building blocks live in `ops`:

- `ops::attention::StandardAttention` (GQA, sliding-window, attn-sinks).
- `ops::attention::GatedAttention` (Gemma-4 style).
- `ops::mlp::DenseMlp` (gate/up/down, fused-decode path selected by dispatcher).
- `ops::moe::MoE<const NUM_EXPERTS: usize>` (topk router + `indexed_expert_matmul` + combine).
- `ops::gdn::GatedDeltaNet` (Qwen3-Next).
- `ops::norm::RmsNorm`, `ops::pe::RoPE`.

A new model family = glue code + a block-list; it ships no new kernels unless the block-list needs one genuinely missing from `ops`.

## Runtime

- **KV cache.** `KvCache<L: CacheLayout>` where `CacheLayout` is a type.
  - `F16Contig`, `F16Transposed` — baseline, matches llama.cpp bit layout. Used for correctness reference and any model where the quality delta of Q8 KV has not yet been certed.
  - `Q8Contig`, `Q8Transposed` — candle's Q8 KV cache scheme (per-group scale + zero, `QK_K=32`). ~2× memory saving; Phase-3 cert grid confirms per-model quality.
  - `TurboQ4Contig`, `TurboQ5Contig` — packed lower-bit KV layouts ported from llamacpp-turbo (block-quant K with shared scale per group, V interleaved for GEMV friendliness). Higher memory saving at measurable quality cost; shipped only for models whose cert clears the quality bar.
  - Selection happens once per session from config; the model type picks an `AllowedLayouts` set, the runtime narrows to a concrete layout, the dispatcher resolves attention impls against that concrete type. No runtime polymorphism, no downcasts.
- **Parallelism.** `tp::Mesh` + `ep::Mesh` implement collective primitives as ops (`AllReduce`, `AllGather`, `AllToAll`). Models parameterize attention/MLP/MoE over a mesh. Single-GPU is `Mesh<1>`.
- **Scheduler.** v0: one session, single-threaded driver, progressive dispatch. v1: multi-thread TP rank drivers (candle's X8 lesson — host-bounce all-reduce is the remaining gap). v2: continuous-batch scheduler driving the server crate.

## Serving (OpenAI-compatible HTTP)

`crates/server` exposes an `axum`-based HTTP API that speaks the OpenAI REST dialect. Minimum v1 surface:

- `GET  /v1/models` — lists loaded models (GGUF path, arch, quant, KV layout, max context).
- `POST /v1/chat/completions` — chat-template-aware completions; supports streaming via SSE (`stream: true`).
- `POST /v1/completions` — raw-prompt completions (legacy clients).
- `GET  /health` / `GET  /metrics` — liveness + Prometheus counters (tok/s, queue depth, KV utilization).

Chat-template application happens in a single place (`server/src/chat_template.rs`) reading the `tokenizer.chat_template` field from the GGUF metadata — no per-model hardcoded prompts. The server holds a single `Session` per request slot and routes through the v1 sampler + v2 continuous-batch scheduler once landed.

Compatibility bar: any client that works against llama.cpp's `server` binary works against `flambeau serve` with a host/port change, tested against at minimum: `curl`, Aider, Continue/Cline, LM Studio, LangChain's OpenAI provider.

## What we port from candle learnings

Keep (algorithmic value):
- MMQ turbo 4-warp LDS-tiled template (in progress in D1 there).
- MMVQ multi-row DPP-reduce pattern (P29).
- Q8 KV cache + `gqa_decode_gemv_{v,qk}_q8` decomposition.
- Fused FFN decode for Q4_0.
- `fused_ffn_decode`, `rmsnorm_q8_fused`, `masked_softmax_scale_fused`.
- Gemma-4 fused `gqa_decode_mv_fast_d{256,512}` (the reason dp4a-fattn lost to F32).
- rocBLAS drop-order fix (`Mutex<Option<RocBlas>>` idiom).
- Small-M → cuda1 template dispatch (P34).

Formalize (was implicit in candle):
- Variant naming convention: `{op}_{dtype}_{backend}_{shape_tag}_{variant}` — grep-able, no ad-hoc `_v2f_tile32_repacked` suffix drift.
- VGPR + occupancy budget per kernel, annotated in source, checked in cert.
- Dispatch table lives in repo as a reviewed artifact.

Drop (pure cost):
- Every non-target model family.
- `CANDLE_*` env flags. Config file instead.
- Parallel dense↔MoE MMQ drivers. One templated driver.
- `alloc_zeros` in hot paths (candle converted; framework starts that way).

## Phasing

- **Phase 0 — skeleton.** Workspace, `core` traits, `quant` layouts for **every** shipping dtype (F16/BF16/F32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K), GGUF reader, CPU dequantize reference, `inspect-gguf` CLI. No GPU kernels yet. End: load any target GGUF, enumerate tensors, dequant every supported dtype to F32 on CPU and match llama.cpp bit-for-bit.
- **Phase 1 — one kernel, two backends.** `QMatMul<Q4_0>` impl on HIP (gfx906) and CUDA (sm_86, RTX 3090). Dispatch matrix (`dispatch/hip/gfx906.toml`, `dispatch/cuda/sm_86.toml`) + cert pipeline (`bench sweep` CLI). End: `infer --op=qmatmul --dtype=q4_0` works on both, logit cosine ≥ 0.9999 vs CPU reference; `sweep` produces a green cert per `(arch, impl_id)`.
- **Phase 2 — full quant coverage.** Every shipping dtype × {MMVQ, MMQ, MoE} on HIP-gfx906 first, then CUDA-sm_86, then **HIP-gfx1031 (RX 6750 XT)** as the wave32/RDNA2 portability canary. The gfx1031 pass is mandatory in Phase 2 — if dispatch-matrix inheritance or `arch_primitives/*.cuh` can't carry a kernel family across wave64↔wave32, the abstraction is wrong and fixing it here is cheaper than in Phase 4+. End: 100% of target-model weight dtypes have a certed kernel on gfx906, sm_86, and gfx1031; dispatch inheritance works for gfx1031 without touching any kernel-algorithmic code.
- **Phase 3 — attention + norms + KV cache families.** `RMSNorm`, `RoPE`, `Softmax`, `Attention` (decode + prefill) across all three KV-cache families: `F16*`, `Q8*`, `TurboQ{4,5}*`. Per-model quality cert (perplexity delta vs F16 KV) required before the dispatcher will hand out a turbo-quant layout. Flash-attention-v2 default prefill on CUDA; tile-Q8 on HIP gfx906 for decode.
- **Phase 4 — Mistral / Devstral dense + server v0.** First end-to-end model inference. Token-for-token match vs llama.cpp on canonical prompts. Perf target: parity with llama.cpp on MI50 and on RTX 3090. `crates/server` ships with `/v1/chat/completions` (non-streaming + SSE), `/v1/completions`, `/v1/models`, `/health`. Validated against `curl` + at least one OpenAI-compat client (Aider or Continue).
- **Phase 5 — Gemma-4.** Adds gated attention, sliding window, attn-sinks if applicable. Server picks up the new model from the same config.
- **Phase 6 — MoE.** Qwen3.5 dense, then Qwen3.6 MoE + Qwen3-Coder MoE. `MoE` op + `indexed_expert_matmul` impls. MoE routing metrics surfaced in `/metrics`.
- **Phase 7 — TP + EP + continuous batching.** `tp::Mesh`, `ep::Mesh`, multi-thread scheduler. Continuous-batch scheduler in `runtime` driving the server (v2), so multiple HTTP requests share one model instance without serializing. Ship on 4×MI50.
- **Phase 8 — Qwen3-Next.** `GatedDeltaNet` op, hybrid GDN+FullAttn blocks.

Each phase ends with: cert-backed dispatch table (per arch), `bench` regression baseline snapshotted, end-to-end inference demo on all applicable target models, correctness vs llama.cpp documented, and — from Phase 4 onward — the server smoke-tested against one or more OpenAI-compat clients.

## Risks & unknowns

- **Kernel source duplication HIP↔CUDA.** Decision locked: separate `kernels-hip/` and `kernels-cuda/` trees, with `kernels-shared/` for algorithmic-core `.cuh` (MMQ tile loop, block-quant unpack, softmax math) — zero backend intrinsics in shared headers. Candle's `gfx906_primitives.cuh` becomes `kernels-hip/gfx906_primitives.cuh`; CUDA gets its own `ampere_primitives.cuh` (cp.async, WMMA, `__shfl_sync` with masks). Accepted cost: each new algorithm authored twice. Mitigation: the cert harness is the contract; two divergent impls is fine as long as both certify.
- **HIP cross-arch portability.** gfx906 is the daily test target; gfx908/gfx90a/gfx942/gfx1031/gfx1100 must stay runnable. Risk: gfx906-only intrinsics (`__builtin_amdgcn_mov_dpp`, `__builtin_amdgcn_ds_bpermute`) and lane-width assumptions silently break on RDNA (wave32) or newer CDNA (matrix cores). Mitigation: `kernels-hip/arch_primitives/{gfx906,gfx908,gfx90a,gfx942,gfx1031,gfx1100}.cuh` provide same-signature intrinsics — wave64 ones hand-mapped to wave32 equivalents on RDNA, matrix-core primitives present as opt-ins on gfx908+. The dispatch matrix inherits rows from `gfx906.toml` and overrides only where certs force it. CI runs a compile-only check for every HIP target even without hardware; gfx1031 is exercised on the physical RX 6750 XT every Phase 2+ gate.
- **Turbo-quant KV cache quality.** Q4/Q5 KV cache is lossy in a way Q8 KV is not. Risk: dispatcher hands out turbo-quant to a model where it degrades perplexity or chat quality. Mitigation: turbo-quant KV layouts require a **quality cert** (delta-perplexity on a fixed eval set + a short chat smoke test) in addition to the correctness cert. No quality cert, no dispatch row.
- **Dispatch-table overfit.** If shape predicates get finer than `(m, k, n) buckets`, the table explodes. Mitigation: cert harness records per-shape perf, the table stays coarse, rare shapes use a fallback impl.
- **Token-parity with llama.cpp.** Floating-point order of ops in attention varies per kernel; bitwise parity is unrealistic. Bar: logit cosine ≥ 0.9999 on a full sequence, verified per model per phase.
- **OpenAI compat surface drift.** The OpenAI REST dialect is large and evolving (function calling, tool use, structured outputs, vision). Risk: scope creep into implementing every optional field. Mitigation: the v1 server covers chat/completions/models/health/metrics + SSE streaming only. Function calling, tool use, structured outputs, logprobs are explicit v2 features, filed when a concrete target client needs them.

## Critical files (greenfield)

New repo. Representative, not exhaustive:
- `crates/core/src/{device,tensor,qtensor,op,dispatch,registry}.rs`
- `crates/quant/src/{gguf,layouts,reference}.rs`
- `crates/kernels-hip/src/{mmq,mmvq,attention,norm,rope,softmax,moe}.cu`, `gfx906_primitives.cuh`
- `crates/kernels-cuda/src/{mmq,mmvq,attention,norm,rope,softmax,moe}.cu`, `ampere_primitives.cuh`
- `crates/kernels-shared/include/{mmq_tile,block_quant,softmax_math,reduce_portable}.cuh`
- `crates/backend-hip/src/{device,stream,impls/*}.rs`
- `crates/backend-cuda/src/{device,stream,impls/*}.rs`
- `crates/ops/src/{attention,mlp,moe,gdn,norm,pe}.rs`
- `crates/models/{mistral,gemma4,qwen3,qwen3_moe,qwen3_next}/src/lib.rs`
- `crates/runtime/src/{kv_cache,session,tp,ep,scheduler}.rs`
- `crates/runtime/src/kv_cache/{f16,q8,turbo_q4,turbo_q5}.rs`
- `crates/bench/src/{harness,sweep,cert,pmc,quality_cert}.rs`
- `crates/server/src/{router,chat,completions,models,stream,chat_template,metrics}.rs`
- `dispatch/hip/{gfx906,gfx908,gfx90a,gfx942,gfx1031,gfx1100}.toml`
- `crates/kernels-hip/arch_primitives/{gfx906,gfx908,gfx90a,gfx942,gfx1031,gfx1100}.cuh`
- `dispatch/cuda/{sm_80,sm_86,sm_89,sm_90}.toml`
- `certs/hip/<arch>/**/*.json`, `certs/cuda/<arch>/**/*.json`, `certs/quality/<model>/<layout>.json`

## Verification (framework-level)

- Every `impl_id` in any `dispatch/<backend>/<arch>.toml` has a matching `certs/<backend>/<arch>/<impl_id>.json` — build-time check.
- Every turbo-quant KV-cache dispatch row additionally has a `certs/quality/<model>/<layout>.json` — build-time check.
- `cargo run -p bench -- sweep --arch gfx906` (or `--arch sm_86`) runs the full correctness grid on that device, updates certs.
- `cargo run -p cli -- infer --model=mistral-7b-q4_k --prompt=<canonical>` produces logits whose cosine similarity to llama.cpp on the same GGUF ≥ 0.9999 over the full sequence.
- `cargo run -p bench -- matrix` reproduces the candle-bench `matrix` subcommand semantics: one `(model, dtype, kv_layout, prompt_len, tg_len)` row per target → pp / tg numbers, regressions flagged against the last snapshot.
- `cargo run -p cli -- serve --model=mistral-7b-q4_k --port=8080` boots the server; a `curl` chat-completions round-trip and an SSE streaming round-trip are asserted in CI (`bench/server_smoke.sh`).

## Decisions locked

1. **HIP+CUDA source sharing** → separate `kernels-hip/` and `kernels-cuda/` trees with `kernels-shared/` for algorithmic-core `.cuh`. Each kernel family authored twice; cert harness is the equivalence contract.
2. **Metal** → **out of v1 scope.** No deferred Phase 9, no Metal-shaped trait hedges. If Metal ever returns, it's a v2 project with its own architecture doc.
3. **HIP archs** → gfx906 is the daily test target; **gfx1031 (RX 6750 XT, RDNA2, wave32) is the portability canary** exercised in every Phase 2+ gate; gfx908/gfx90a/gfx942/gfx1100 are additional portability targets via dispatch-matrix inheritance + `arch_primitives/*.cuh`. Adding an arch = new `.toml` rows + certs, not a new kernel family.
4. **CUDA archs** → sm_86 (RTX 3090) is the daily dev target; sm_80/sm_89/sm_90 are forward-compatible via `__CUDA_ARCH__` guards for genuinely new primitives only.
5. **Quant coverage** → full from Phase 2 (Q2_K through Q8_K + legacy Q*_0/Q*_1 + F16/BF16/F32). No "only-if-needed" escape.
6. **KV cache layouts** → F16 + Q8 + turbo-quant (Q4/Q5) all ship in v0. Selection is per-model config, statically typed, cert-gated; turbo-quant additionally requires a quality cert.
7. **Model input** → GGUF only. Users convert safetensors upstream.
8. **Tokenizer** → `tokenizers` crate. Not a hot path, not a reinvention target.
9. **Serving** → OpenAI-compatible HTTP server is a first-class deliverable from Phase 4 (v1: chat/completions/models/health/metrics + SSE). Function calling / tool use / structured outputs are v2, filed per target client.

Next step: write Phase-0 skeleton plan (workspace `Cargo.toml`, `core` trait surface, `quant` full-dtype loader with CPU dequant reference, `inspect-gguf` CLI) as a second plan file.

---

## Appendix A — Prior D0-consolidation plan (superseded)

Kept as a record of the candle-side cleanup that the rewrite would make unnecessary; ship it anyway if the rewrite is >1 quarter away, since the same certs/harness feed into Phase 1 of the new framework.

(Previous content: unified correctness harness, `DISPATCH.md`, delete orphans `mul_mat_q4_1_gfx906_v2c_tile32`, `mul_mat_q4_0_gfx906_v2f_tile32_repacked`, `mul_mat_q4_0_turbo_x*`, retire dominated opt-in flags. See earlier version of this file in git if needed.)
