# flambeau

Max-perf inference server for modern LLMs on AMD HIP + NVIDIA CUDA, exposed over an OpenAI-compatible HTTP API.

Sibling of [`/artefact/candle`](../candle). We inherit candle's kernel learnings (DPP reductions, multi-row MMVQ, 4-warp LDS-tiled MMQ, Q8 KV cache, fused decode path) and drop its architectural accretion (env-flag variant gates, parallel dense/MoE drivers, one-off model families, autograd plumbing).

---

## Why this exists

Candle shipped correct-but-slow and tuned incrementally. The result is ~150 kernel variants, ~19 `CANDLE_*` env flags, model-by-model accretion, and a per-kernel correctness story that lives mostly in memory. Flambeau inherits the destination, not the journey: data-driven dispatch, correctness-cert-gated kernels, first-class kernels from the first commit, multi-device from day one.

See `doc/ARCHITECTURE.md` for the full rationale.

---

## Scope

- **Backends:** HIP + CUDA. Metal is **out**.
- **Models:** Mistral/Devstral dense, Gemma-4 dense + gated, Qwen3.5 dense, Qwen3.6 MoE, Qwen3-Coder MoE, Qwen3-Next (GDN hybrid).
- **Input:** GGUF only.
- **Quant:** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K + legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + F16/BF16/F32.
- **KV cache:** F16, Q8, turbo-quant (Q4/Q5) — all statically typed, cert-gated.
- **Parallelism:** single GPU + TP + EP on the same binary.
- **Serving:** OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming, `/health`.

Out: training, LoRA, Metal, vision/audio, ONNX.

---

## Build

```sh
cd /artefact/flambeau
cargo build --release -p flambeau-cli --features hip_serve
```

---

## First run

```sh
# Inspect a GGUF file
flambeau inspect-gguf ~/models/qwen3.6-30b-a3b-q4_k_m.gguf

# Correctness sweep for a backend/arch
flambeau sweep --arch gfx906 --op qmatmul

# Single-prompt inference
flambeau infer --model qwen3.6-30b-a3b-q4_k_m --prompt "Hello" --devices hip:0,1,2,3

# OpenAI-compatible server
flambeau serve --model qwen3.6-30b-a3b-q4_k_m --devices hip:0,1,2,3 --port 8080

# Then, from any OpenAI client:
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-30b-a3b-q4_k_m","messages":[{"role":"user","content":"hi"}]}'
```

---

## Where to start

1. **New here?** Read this README, then `CLAUDE.md`, then `doc/ARCHITECTURE.md`.
2. **Porting a kernel?** `doc/candle-prior-art.md` maps each kernel to its source file in `/artefact/candle/`, `/artefact/llamacpp-turbo/`, or `/artefact/llama.cpp/`. Open that file; don't grep the world.
3. **Writing a cert or dispatch row?** See `dispatch/hip/gfx906.toml` and the JSON files in `certs/hip/gfx906/` — schemas are concrete, not prose.

---

## Repo layout

```
flambeau/
├── CLAUDE.md                    # Working notes / architectural rules / measurement rules
├── README.md                    # this file
├── Cargo.toml                   # Workspace manifest
├── rust-toolchain.toml          # Pinned stable
├── crates/
│   ├── core/                    # Device-independent traits
│   ├── quant/                   # GGUF + block-quant layouts + CPU dequant reference
│   ├── kernels-shared/          # Backend-neutral algorithmic-core .cuh
│   ├── kernels-hip/             # HIP .cu sources + build.rs (hipcc)
│   ├── backend-hip/             # HIP device + kernel-impl registrations
│   ├── ops/                     # Attention, MoE, MLP, RMSNorm, RoPE (Mesh-generic)
│   ├── models/qwen3-moe/        # Qwen3.x MoE composition
│   ├── runtime/                 # Session, KV cache (typed), Mesh<N>, scheduler
│   ├── bench/                   # Sweep + matrix + cert-diff harness
│   ├── server/                  # OpenAI-compatible HTTP
│   └── cli/                     # `flambeau` binary
├── doc/
│   ├── ARCHITECTURE.md          # Framework design
│   ├── GLOSSARY.md              # Plain-English glossary
│   ├── Q8_KV_DEQUANT_ANALYSIS.md  # Q8 KV cache analysis (research)
│   └── candle-prior-art.md      # Port-source map
├── dispatch/
│   └── hip/gfx906.toml          # Dispatch matrix
└── certs/
    └── hip/gfx906/              # Correctness + PMC JSON per impl
```

---

## Pointers

- `CLAUDE.md` — architectural rules, measurement discipline, technical lessons from the candle sessions. Read before touching anything substantive.
- `doc/ARCHITECTURE.md` — framework design (crates, traits, dispatch, KV layouts).
- `doc/GLOSSARY.md` — plain-English glossary of every technical term used across the project docs, aimed at beginners. Start here if the jargon is unfamiliar.
- `doc/candle-prior-art.md` — where in `/artefact/candle/` + `/artefact/llamacpp-turbo/` to look for each port target.
- `/artefact/candle/` — source framework; kernel prior art, not architectural patterns.
- `/artefact/llama.cpp/` + `/artefact/llamacpp-turbo/llama-cpp-gfx906-turbo/` — correctness oracle + port targets for 4-warp LDS-tiled MMQ.

## License

Apache-2.0 (matches candle's licensing).
