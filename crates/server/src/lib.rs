//! flambeau-server — V1.8 OpenAI-compatible HTTP API.
//!
//! v1 surface: `GET /health`, `GET /v1/models`, `POST /v1/completions`,
//! `POST /v1/chat/completions` (+ SSE when `stream: true`).
//! Chat template from GGUF `tokenizer.chat_template` via `minijinja`.
//!
//! Function calling, tool use, logprobs, `/v1/embeddings`, `/metrics` are V2.
