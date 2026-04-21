# V2 backlog

V1 shipped (tag candidate: `v0.1.0-qwen36-gfx906`). Everything below is
scoped for V2 or later — each item has a honest multi-session estimate.
Listed in rough priority order for a perf-closure-focused V2.

See git log for V1 ship details:
  * `6bceeb2 feat(v1): Qwen3.6-35B decode on HIP/gfx906 at 86.5% of llama.cpp (Mesh<2>)`
  * `6dbda94 feat(v1.8): OpenAI-compat HTTP server + tokenizer + chat template + sampler`

## V2 perf work

### V2.1 — Batched kernel launch queue (FFI overhead attack)

Working hypothesis for the residual 13.6% Mesh<2> gap to llama.cpp:
Rust-FFI per-launch cost (~1-2 µs × 747 launches × 2 ranks ≈ 1.5-3
ms/token). llama.cpp's cert showed our Q8_0 + Q4_K MoE MMVQs are at least
as fast as theirs structurally, so the gap must be in small-kernel
aggregates + orchestration.

**Estimate**: 2-3 sessions.
1. Instrument FFI layer with per-call timing. Compare to rocprof's
   `hipModuleLaunchKernel` API time (current: 3.67 µs/call aggregate).
2. Prototype batched dispatch wrapper that queues N kernel launches
   and fires them in one FFI call.
3. Benchmark vs per-call launch. Decide on API surface.

### V2.2 — Mesh<1> via Qwen3.5 dense-hybrid loader

Unlocks the original 1×-MI50 perf gates from the V1 roadmap (≥60 tok/s
tg64, ≥600 tok/s pp512). Qwen3.6-35B is 19.45 GiB and doesn't fit 16 GiB;
Qwen3.5-9B-Q4_1 is 5.5 GiB and fits.

**Estimate**: 2 sessions.
1. New `crates/models/qwen35-dense` (or dense-hybrid variant of
   qwen3-moe) for arch=`qwen35`: GDN + full-attn + **dense** FFN
   (no MoE experts, no router).
2. Wire loader. Parity test vs llama.cpp on Qwen3.5-9B.
3. `bench matrix --devices 1,2,4` snapshot for scaling-ratio gate.

### V2.3 — Q8 KV quality cert (V1.7.8 per original roadmap)

Halves decode HBM traffic. Quality gate: wikitext-2 delta-ppl ≤ 0.5% +
chat smoke.

**Estimate**: 2-3 sessions.
1. Flesh out `KvCache<Q8Contig>` in runtime (stub exists).
2. Per-head Q8 quantize kernel at KV-write time.
3. `attention_decode_q8_kv` path is already certed (V1.5) — verify it
   composes with the new write path.
4. `bench quality --model X --kv-layout q8` harness.

### V2.4 — Speculative decoding (draft + verify)

Industry-standard lever. Draft Qwen3.5-9B (depends on V2.2 loader) on
one rank, verify large Qwen3.6-35B in parallel. Literature suggests
1.5-2.5× effective throughput.

**Estimate**: 3-4 sessions (after V2.2).

### V2.5 — SSE streaming + full-logit sampler

V1.8 ships non-streaming greedy only. For real OpenAI compatibility:
1. `forward_one_token_pp` variant returning the full logit row.
2. Server's `stream: true` path emits SSE chunks.
3. Server-side temperature/top-p uses the existing
   `flambeau_runtime::sample` on those logits.

**Estimate**: 2 sessions.

## V2 dev-loop tooling

### V2.6 — T-track warmup-tuner

Per ROADMAP-V1 §T1-T5: session-init micro-bench of certed variants
writes a gitignored `dispatch/<backend>/<arch>.local.toml`. Runtime
selection only. Committed table stays source of truth; promotion
requires ≥ 3 independent rigs + a PR.

**Estimate**: 3 sessions. Crate shell at `crates/autotune`.

### V2.7 — M-track MCP server

Per ROADMAP-V1 §M1-M5: dev-only MCP server wrapping `bench sweep`,
`bench matrix`, `rocprofv3`, A/B dispatch, `inspect-*`, cert diff,
tune dry-run. Every finding round-trips into a committable artefact.

**Estimate**: 3 sessions. Crate shell at `crates/mcp-server`.

## V2 model families

(From V1 scope reminder — Mistral/Devstral dense, Gemma-4 dense+gated,
Qwen3.5 dense, Qwen3-Coder MoE, Qwen3-Next.) Each is its own
model-crate port once V2.2 establishes the dense-hybrid pattern.
