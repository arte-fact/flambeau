//! flambeau-server — V1.8 OpenAI-compatible HTTP API.
//!
//! v1 surface: `GET /health`, `GET /v1/models`, `POST /v1/completions`,
//! `POST /v1/chat/completions` (non-streaming only in V1.8.B; SSE is V1.8.C).
//! Chat template from GGUF `tokenizer.chat_template` via `minijinja`.
//!
//! Single-session model: one shared `ModelState` behind a Tokio mutex.
//! Concurrent HTTP requests serialise through the mutex — continuous
//! batching is V2. The V1 target is "works end-to-end with real OpenAI
//! clients", not throughput at load.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod agent_stats;
pub mod api;
pub mod mcp_client;
pub mod state;
pub mod tool_call_parser;

#[cfg(feature = "hip")]
pub mod model;
#[cfg(feature = "hip")]
pub mod routes;
#[cfg(feature = "hip")]
pub mod serve;

pub use api::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionRequest, CompletionResponse, FunctionCall, FunctionDef, ModelObject,
    ModelsListResponse, ToolCall, ToolChoice, ToolChoiceFunction, ToolChoiceNamed, ToolDef, Usage,
};
pub use state::{ModelDefaults, SamplingParams};

#[cfg(feature = "hip")]
pub use model::{decode_logits, prefill_logits, Inflight, LoadedModel};
#[cfg(feature = "hip")]
pub use serve::{serve, MeshMode, ServeConfig};
