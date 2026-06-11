//! OpenAI-compatible request/response types.
//! shipped chat completions, text completions, models list, health.
//! V2 tool-calling track (T1.1 — ROADMAP-V2-TOOL-CALLING-AND-MCP.md) adds
//! the wire surface for `tools[]`, `tool_choice`, `tool_calls[]`,
//! `role="tool"`, and `finish_reason="tool_calls"`. The types are wired
//! here but not acted upon yet — T1.2/T2.x fill in the behaviour.
//! Logprobs, embeddings, structured outputs (response_format=json_schema),
//! vision/audio tool results stay V2+.

use serde::{Deserialize, Serialize};

// ---- /v1/models ------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ModelsListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// **#235 P3.15** — operational context window in tokens (already
    /// reflects the `FLAMBEAU_CTX_CAP` shrink, not the architectural
    /// max from the GGUF). Surfaced so clients can size prompts
    /// without a separate `/v1/models/<id>/details` round-trip.
    pub context_length: u32,
    /// **#235 P3.15** — server-enforced ceiling on `max_tokens` per
    /// request (`SamplingParams::from_parts` clamps at this value).
    pub max_output_tokens: u32,
    /// **#235 P3.15** — model architecture tag from the GGUF
    /// `general.architecture` key (`qwen35moe`, `qwen36moe`,
    /// `qwen3next`, etc). flambeau-only field; OpenAI clients ignore.
    pub architecture: String,
    /// **#235 P3.15** — quantization label derived from the GGUF
    /// `general.file_type` integer. `None` when the GGUF doesn't
    /// carry the field (rare; older converters).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// **#235 P3.15** — feature surface this model + server combo
    /// supports. Each entry is a stable token (`"chat"`,
    /// `"completion"`, `"infill"`, `"embeddings"`, `"tools"`,
    /// `"thinking"`). Clients can use `.includes("thinking")` to
    /// know whether to expose the `enable_thinking` request flag in
    /// their UI.
    pub capabilities: Vec<&'static str>,
    /// **#235 P3.15** — wire shape the parser expects for tool-call
    /// model output (`"hermes"`, `"qwen_coder"`). Set even when the
    /// caller doesn't pass `tool_call_format`; matches the boot-time
    /// detection from the chat template.
    pub tool_call_format: &'static str,
}

// ---- /v1/chat/completions --------------------------------------------------

/// OpenAI chat-completions request. Fields the OpenAI spec defines but a
/// single-model self-hosted server cannot meaningfully honor are deliberately
/// not modeled (and ignored if sent): `n` (>1 choices), `service_tier`,
/// `store`, request `metadata`, and per-model routing on `model`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,

    // ---- Sampler fields (T4.b — ships with T1). Behaviour wired in T4.b.2. ----
    /// Per-token repetition penalty applied over the accumulated history.
    /// llama.cpp convention: `logit /= penalty` when `logit > 0`. Off at
    /// 1.0. Qwen3-Coder ships `1.05` in its `generation_config.json`;
    /// other Qwen3 instruct variants leave it at 1.0.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Additive presence penalty: subtracts `value` from the logit of any
    /// token already emitted this turn. Range -2..2 (OpenAI). Qwen's own
    /// Best Practices warn that values >1.0 cause "language mixing and a
    /// slight decrease in model performance" — a previous default of 1.5
    /// reproduced exactly that failure on long generations. Default off.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (OpenAI-compat). Distinct from `presence_penalty`:
    /// scales with occurrence count rather than presence.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Top-k logit filter. `None` = no filter. Qwen default ~20.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Min-p logit filter. `None` = no filter. Qwen default 0.0 (off).
    #[serde(default)]
    pub min_p: Option<f32>,

    // ---- Tool-calling fields (T1.1). Behaviour wired in T1.2 / T2.5. ----
    /// Tool definitions to surface to the model. Rendered into the prompt
    /// via the GGUF-embedded Jinja `tokenizer.chat_template`.
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    /// OpenAI `tool_choice`: `"auto" | "none" | "required"` or a named
    /// `{"type":"function","function":{"name":"..."}}` selector.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may emit multiple tool calls in one assistant
    /// turn. Default true per OpenAI. When false, decode truncates at the
    /// first `</tool_call>`.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Override the parser family. `None` / `"auto"` inspects the loaded
    /// model architecture (Qwen3.6 `qwen35moe` → `hermes`). Future-proofs
    /// the wire for the Qwen3-Coder XML parser (V2.x, see ROADMAP-V2 §T2).
    #[serde(default)]
    pub tool_call_format: Option<String>,

    // ---- Structured-output fields (P0.1) -----------------------------
    /// OpenAI `response_format`. When set to `{"type":"json_object"}`,
    /// the sampler is constrained at every step to keep the running
    /// output structurally-valid JSON (no token can break a brace
    /// balance, escape a string mid-codepoint, etc.). Used heavily by
    /// OpenWebUI's auto-prompts (search-query-gen, follow-ups, title,
    /// tags) and by Aider/Continue/LangChain JSON modes.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,

    // ---- Logprobs (P1.7) ---------------------------------------------
    /// OpenAI `logprobs`: when `true`, the response carries the
    /// per-token log-probability of every generated token. Default
    /// `false`. V1 implementation is non-streaming, host-path-only:
    /// requests on the GPU sampler path or spec-decode path silently
    /// return `null` with a warn-once log.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// OpenAI `top_logprobs`: number of top alternative tokens to
    /// return per generated step. 0..20. Capped at 20 by OpenAI; we
    /// cap at the same value. Implies `logprobs=true`.
    #[serde(default)]
    pub top_logprobs: Option<u32>,

    // ---- Streaming options (P0.3) ------------------------------------
    /// OpenAI `stream_options`. When `include_usage=true`, the server
    /// emits a final `choices=[]` SSE chunk carrying the `usage` block
    /// after the finish-reason chunk. This is the canonical OpenAI
    /// shape that the openai-python SDK and LangChain expect; without
    /// it, some clients silently drop the `usage` field embedded in
    /// the finish chunk. Default `None` (no extra chunk).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    // ---- Reasoning mode (P3.13) --------------------------------------
    /// **#233 P3.13** — opt into Qwen3.6 reasoning / extended-thinking
    /// mode. When `Some(true)`, the chat template renders without the
    /// suppression block, the model emits
    /// `<think>{cot}</think>{answer}`, and the server splits the two:
    /// `{cot}` returns in `message.reasoning_content`, `{answer}` in
    /// `message.content`. Default `None` ⇒ thinking suppressed
    /// (legacy behaviour: server passes `enable_thinking=false` to
    /// the chat template). Mirrors HF Transformers `chat_template`
    /// kwarg; OpenAI's o-series + Anthropic extended-thinking surface
    /// the same toggle via different field names — this is the
    /// OpenAI-compat shape.
    #[serde(default)]
    pub enable_thinking: Option<bool>,

    // ---- Newer OpenAI fields ----------------------------------------
    /// OpenAI `max_completion_tokens` — the current name for `max_tokens`
    /// on the chat endpoint. When both are sent, `max_tokens` wins (it is
    /// the legacy explicit field); otherwise this is used.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// OpenAI `logit_bias`: map of token-id (as a string key) to an
    /// additive bias in roughly [-100, 100]; -100 effectively bans a
    /// token, +100 forces it. Applied to the logits before temperature.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    /// OpenAI `reasoning_effort` (`"low"|"medium"|"high"`, o-series).
    /// Any non-`"none"` value enables the thinking channel; the per-level
    /// thinking-token budget is applied in a later slice.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// OpenAI streaming-options envelope. Today only `include_usage` is
/// honoured; future fields (e.g., `include_logprobs`) plug in here
/// without a wire break.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// OpenAI-spec response format selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Default — no structural constraint.
    Text,
    /// Constrain output to be parseable as a single top-level JSON value.
    JsonObject,
    /// Constrain to a specific JSON schema (more strict than JsonObject).
    /// V1 implementation: treat same as `JsonObject` (schema-aware
    /// constraint is a follow-up; the structural constraint already
    /// catches 95 % of malformations).
    JsonSchema {
        #[serde(default)]
        json_schema: serde_json::Value,
    },
}

/// One chat message on the wire.
/// T1.1 extends this to carry tool-call payloads: a prior `role="assistant"`
/// turn may carry `tool_calls` with no `content`; a `role="tool"` turn
/// carries `tool_call_id` + `content` (the tool's reply). OpenAI sends
/// `content: null` in those cases, hence `Option<String>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Serialises as `null` when the message is a tool-call-only assistant
    /// turn with no text. Defaults to `None` on the wire for backward
    /// compatibility with clients that omit the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Set on `role="tool"` messages: the `id` of the tool call this
    /// message replies to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Set on `role="assistant"` messages that invoked one or more tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// **#233 P3.13** — chain-of-thought returned separately when the
    /// request set `enable_thinking=true` and the model emitted a
    /// `<think>{cot}</think>` block. Mirrors the de-facto convention
    /// shared by DeepSeek-R1 / vLLM / sglang / OpenWebUI for
    /// reasoning models: same shape as `content` but in a sibling
    /// field so clients can render it collapsibly. Always `None`
    /// when thinking was suppressed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// Convenience for call sites that want the text body as a `&str`
    /// and treat a missing `content` as the empty string. Keeps the
    /// `Option<String>` surface honest without forcing every caller to
    /// `as_deref().unwrap_or("")`.
    pub fn content_str(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

// ---- Tool-calling types (T1.1) --------------------------------------------

/// A tool the client exposes to the model. OpenAI shape: currently only
/// `type="function"` exists; we mirror that rather than over-generalising.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Always `"function"` today. Kept as a tagged field to match OpenAI
    /// and leave room for future tool kinds without breaking clients.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

/// One function tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the function's parameters. We keep it as opaque
    /// `serde_json::Value` — T1.2's Jinja template render + T5's optional
    /// llguidance mask generator are the only readers.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// `tool_choice` request field. Either a string mode or a named selector.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto" | "none" | "required"`. Free-form string keeps the door
    /// open for vendor extensions without a struct change.
    Mode(String),
    /// `{"type":"function","function":{"name":"..."}}` — force a specific
    /// function call.
    Named(ToolChoiceNamed),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceNamed {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolChoiceFunction,
}

impl ToolChoice {
    /// True for OpenAI `tool_choice == "none"` — the model MUST NOT
    /// emit a tool call. Server enforces this by dropping the tools
    /// section from the rendered prompt and refusing to lift any
    /// `tool_calls[]` from the model's output.
    pub fn forbids_tools(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s.eq_ignore_ascii_case("none"))
    }

    /// Whether the model must be forced to emit a tool call, and which.
    /// `None` = don't force (`auto`/`none`). `Some(None)` = force *some*
    /// call (`required`). `Some(Some(name))` = force the named function.
    pub fn force_target(&self) -> Option<Option<&str>> {
        match self {
            ToolChoice::Mode(s) if s.eq_ignore_ascii_case("required") => Some(None),
            ToolChoice::Named(n) => Some(Some(n.function.name.as_str())),
            _ => None,
        }
    }
}

impl AnthropicToolChoice {
    /// Anthropic counterpart of [`ToolChoice::forbids_tools`].
    pub fn forbids_tools(&self) -> bool {
        matches!(self, AnthropicToolChoice::None)
    }

    /// Anthropic counterpart of [`ToolChoice::force_target`]. `any` forces
    /// some call; `tool` forces the named one.
    pub fn force_target(&self) -> Option<Option<&str>> {
        match self {
            AnthropicToolChoice::Any => Some(None),
            AnthropicToolChoice::Tool { name } => Some(Some(name.as_str())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// One tool call emitted by the model and echoed back in prior-turn
/// assistant messages.
/// **`arguments` is a JSON-encoded string, never an object.** This guards
/// llama.cpp #20198, where returning `arguments` as an object broke the
/// openai-python SDK. The wire contract stays string; any structural
/// constraint lives in the parser / grammar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded parameter object. See the note on `ToolCall`.
    pub arguments: String,
}

// ---- Chat-completion response ---------------------------------------------

/// Build identifier surfaced as OpenAI `system_fingerprint` so clients can
/// detect a backend change across otherwise-identical requests.
pub const SYSTEM_FINGERPRINT: &str = concat!("fp_flambeau_", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    pub system_fingerprint: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    /// OpenAI `finish_reason`. Valid values today: `"stop"`, `"length"`,
    /// and — with T2.5 wired — `"tool_calls"`.
    pub finish_reason: String,
    /// **P1.7** — per-token log-probabilities. `None` when the request
    /// did not enable logprobs, or when the path didn't support them
    /// (GPU sampler, spec-decode). Skipped from JSON when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

/// OpenAI `choices[*].logprobs` envelope.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    pub content: Vec<ChatLogProbContent>,
}

/// One entry per generated token. `bytes` carries the UTF-8 byte
/// sequence of the token (or the byte-fallback bytes for unrenderable
/// tokens — same shape OpenAI emits for tiktoken's byte-level BPE).
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogProb>,
}

/// One alternative token at a generation step.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

// ---- /v1/completions -------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    /// **P1.6c** — OpenAI suffix-style Fill-in-the-Middle. When set
    /// (and the model carries FIM tokens), `/v1/completions` switches
    /// to the same PSM-token assembly used by `/infill`: `prompt` is
    /// the prefix and `suffix` is the suffix, the model fills the gap.
    /// Without `suffix` the endpoint behaves exactly as before.
    #[serde(default)]
    pub suffix: Option<String>,
}

// ---- /infill (P1.6b — llama.cpp-compat fill-in-the-middle) ----------------

/// llama.cpp-compatible `/infill` request body. The endpoint composes a
/// FIM prompt of the form `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`
/// (PSM = prefix-suffix-middle, the canonical Qwen-Coder / StarCoder
/// layout) and decodes a completion. Optional `input_extra` carries
/// repo-context files; when present they are inlined ahead of the FIM
/// block via `<|repo_name|>…<|file_sep|>…`, the Qwen-Coder PSM extension.
#[derive(Debug, Clone, Deserialize)]
pub struct InfillRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// Code BEFORE the cursor.
    #[serde(default)]
    pub input_prefix: String,
    /// Code AFTER the cursor.
    #[serde(default)]
    pub input_suffix: String,
    /// Optional middle prefix — text that the model should treat as
    /// already-emitted at the cursor. llama.cpp accepts this as
    /// `prompt`; we honour both names so OpenWebUI / Continue / Cursor
    /// configurations can target the same endpoint.
    #[serde(default, alias = "middle")]
    pub prompt: Option<String>,
    /// Repo-context files (Qwen-Coder PSM). Each carries a `filename`
    /// and `text`; emitted ahead of the FIM block in the order given.
    /// Ignored when the model's vocab lacks `<|repo_name|>` or
    /// `<|file_sep|>`.
    #[serde(default)]
    pub input_extra: Vec<InfillExtra>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// llama.cpp uses `n_predict`; we also accept the OpenAI-style
    /// `max_tokens` for client convenience.
    #[serde(default, alias = "max_tokens")]
    pub n_predict: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InfillExtra {
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Breakdown of the completion tokens. Emitted only when the turn
    /// produced reasoning (so the legacy `/v1/completions` path stays
    /// byte-identical).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// OpenAI `completion_tokens_details`. Only `reasoning_tokens` is tracked.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
}

// ---- Anthropic /v1/messages (P1.8a) ---------------------------------------
// Mirrors the request envelope at https://docs.anthropic.com/en/api/messages
// closely enough for Claude Code, Cursor, and the anthropic-sdk-python /
// anthropic-sdk-typescript clients to talk to flambeau without a shim.
// V1 scope (P1.8a — this commit): text-only content blocks, non-streaming
// path, no tools mapping. Tools (`tool_use` / `tool_result` content blocks)
// are P1.8c; SSE event stream is P1.8b.

/// Anthropic /v1/messages request body.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    /// **Required** under the Anthropic spec — clients always send it.
    pub max_tokens: u32,
    /// Top-level system prompt; Anthropic carries it outside `messages`.
    /// Accepts either a string or an array of `{type:"text",text:"…"}`
    /// content blocks (the latter is the canonical shape used by Claude
    /// Code's prompt-caching path).
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    /// **P1.8c** — tool definitions exposed to the model. Each carries
    /// `name`, `description`, and an `input_schema` (JSON Schema). The
    /// shape mirrors Anthropic's spec; the handler translates them
    /// into the OpenAI `ToolDef` shape that the chat-template Jinja
    /// renderer already consumes.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    /// **P1.8c** — Anthropic `tool_choice`. Forms:
    /// - `{"type":"auto"}` (default) — model decides
    /// - `{"type":"any"}` — model must use a tool
    /// - `{"type":"tool","name":"..."}` — force a specific tool
    /// - `{"type":"none"}` — model must not use tools
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    /// Extended-thinking control. `{"type":"enabled","budget_tokens":N}`
    /// turns on the reasoning channel; `{"type":"disabled"}` or omitted
    /// leaves it off. `budget_tokens` is accepted; the thinking-length cap
    /// is applied alongside `reasoning_effort` in a later slice.
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
}

/// Anthropic extended-thinking control block.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub budget_tokens: Option<u32>,
}

impl AnthropicThinking {
    /// True when the client asked for the reasoning channel.
    pub fn is_enabled(&self) -> bool {
        self.kind == "enabled"
    }
}

/// One Anthropic tool definition. `input_schema` is opaque JSON Schema.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Anthropic `tool_choice`. Tagged on `type`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
    None,
}

/// Anthropic system block — string or content-array.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Plain(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicSystem {
    /// Flatten any text content into one plain string. Non-text blocks
    /// are dropped (P1.8a is text-only; vision arrives later).
    pub fn to_plain(&self) -> String {
        match self {
            Self::Plain(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// One Anthropic message. `content` may be a plain string OR a list of
/// content blocks. Both forms appear in the wild.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Anthropic message-content envelope.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Plain(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicContent {
    /// Flatten text blocks into a single string. Non-text blocks
    /// (image, tool_use, tool_result) are dropped in P1.8a; tools land
    /// in P1.8c via a richer translation path.
    pub fn to_plain(&self) -> String {
        match self {
            Self::Plain(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// One Anthropic content block. Tagged on `type`. Unknown variants
/// deserialise as `Other` so the request still parses; the route
/// handler can ignore them or 400 as appropriate.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    /// `image`: P1.8 scope is text-only; deserialised but discarded.
    Image {
        #[serde(default)]
        source: serde_json::Value,
    },
    /// Tool-call invocation (assistant turn). P1.8c maps these into
    /// `tool_calls[]` on a synthetic OpenAI assistant message.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result (user turn). P1.8c maps these into `role="tool"` +
    /// `tool_call_id`.
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
        #[serde(default)]
        is_error: Option<bool>,
    },
    /// Assistant reasoning replayed on a multi-turn request. We do not
    /// validate signatures and drop prior-turn reasoning from the prompt
    /// (matching how OSS stacks handle replayed thinking).
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
}

/// Anthropic /v1/messages response. `content` is always an array.
/// `stop_reason` is mapped from OpenAI's `finish_reason`:
/// `stop` → `end_turn`, `length` → `max_tokens`, `tool_calls` → `tool_use`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub content: Vec<AnthropicResponseBlock>,
    pub model: String,
    pub stop_reason: String,
    /// When `stop_reason="stop_sequence"`, the actual sequence that
    /// matched. `null` otherwise.
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    /// **P1.8c** — model-emitted tool call. `id` is the call id the
    /// caller will echo back on the corresponding `tool_result` block.
    /// `input` is the parsed JSON object the caller passes to its tool.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Extended-thinking reasoning, emitted before the answer text when
    /// thinking is enabled. `signature` is an opaque stub — flambeau does
    /// not cryptographically sign reasoning.
    Thinking {
        thinking: String,
        signature: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Back-compat: clients that POST messages with string `content`
    /// and no tool fields must deserialise unchanged.
    #[test]
    fn v1_8_compat_message_roundtrip() {
        let wire = r#"{"role":"user","content":"hello"}"#;
        let msg: ChatMessage = serde_json::from_str(wire).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content_str(), "hello");
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());

        // Re-serialising must not introduce null fields.
        let out = serde_json::to_string(&msg).unwrap();
        assert_eq!(out, wire);
    }

    /// Assistant message carrying only tool_calls (no text) — OpenAI
    /// sends this with `content: null`. Our `Option<String>` handles both
    /// missing and explicit-null.
    #[test]
    fn assistant_tool_calls_only_deserialises() {
        let wire = r#"{
            "role":"assistant",
            "content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"SF\"}"}}]
        }"#;
        let msg: ChatMessage = serde_json::from_str(wire).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        let calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // Arguments must remain a string on the wire (guards #20198).
        assert_eq!(calls[0].function.arguments, r#"{"loc":"SF"}"#);
    }

    #[test]
    fn tool_message_with_id_deserialises() {
        let wire = r#"{"role":"tool","tool_call_id":"c1","content":"{\"temp\":72}"}"#;
        let msg: ChatMessage = serde_json::from_str(wire).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("c1"));
        assert_eq!(msg.content_str(), r#"{"temp":72}"#);
    }

    #[test]
    fn tool_choice_mode_and_named() {
        let mode: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
        assert!(matches!(mode, ToolChoice::Mode(ref s) if s == "auto"));

        let named: ToolChoice =
            serde_json::from_str(r#"{"type":"function","function":{"name":"get_weather"}}"#)
                .unwrap();
        match named {
            ToolChoice::Named(n) => {
                assert_eq!(n.kind, "function");
                assert_eq!(n.function.name, "get_weather");
            }
            _ => panic!("expected named tool choice"),
        }
    }

    #[test]
    fn tool_choice_forbids_tools_recognises_none() {
        let none: ToolChoice = serde_json::from_str(r#""none""#).unwrap();
        assert!(none.forbids_tools());
        // Case-insensitive (some clients send "None" or "NONE").
        let ucase: ToolChoice = serde_json::from_str(r#""NONE""#).unwrap();
        assert!(ucase.forbids_tools());
        let auto: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
        assert!(!auto.forbids_tools());
        let required: ToolChoice = serde_json::from_str(r#""required""#).unwrap();
        assert!(!required.forbids_tools());
        let named: ToolChoice =
            serde_json::from_str(r#"{"type":"function","function":{"name":"f"}}"#).unwrap();
        assert!(!named.forbids_tools());
    }

    #[test]
    fn anthropic_tool_choice_forbids_tools_recognises_none() {
        let none: AnthropicToolChoice = serde_json::from_str(r#"{"type":"none"}"#).unwrap();
        assert!(none.forbids_tools());
        let auto: AnthropicToolChoice = serde_json::from_str(r#"{"type":"auto"}"#).unwrap();
        assert!(!auto.forbids_tools());
        let any: AnthropicToolChoice = serde_json::from_str(r#"{"type":"any"}"#).unwrap();
        assert!(!any.forbids_tools());
    }

    #[test]
    fn infill_request_llamacpp_shape() {
        let wire = r##"{
            "input_prefix":"def fizzbuzz(n):\n    ",
            "input_suffix":"    return result\n",
            "n_predict":64,
            "stop":["\n\n"],
            "input_extra":[{"filename":"main.py","text":"# entry"}]
        }"##;
        let req: InfillRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.input_prefix, "def fizzbuzz(n):\n    ");
        assert_eq!(req.input_suffix, "    return result\n");
        assert_eq!(req.n_predict, Some(64));
        assert_eq!(req.input_extra.len(), 1);
        assert_eq!(req.input_extra[0].filename, "main.py");
    }

    #[test]
    fn infill_request_max_tokens_alias_works() {
        // OpenAI-style clients send `max_tokens`, not `n_predict`.
        let wire = r#"{
            "input_prefix":"a","input_suffix":"b","max_tokens":32
        }"#;
        let req: InfillRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.n_predict, Some(32));
    }

    #[test]
    fn completion_request_with_suffix_for_fim() {
        let wire = r#"{
            "model":"qwen3-coder",
            "prompt":"def fizzbuzz(n):\n    ",
            "suffix":"    return result\n",
            "max_tokens":64
        }"#;
        let req: CompletionRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.prompt, "def fizzbuzz(n):\n    ");
        assert_eq!(req.suffix.as_deref(), Some("    return result\n"));
        assert_eq!(req.max_tokens, Some(64));
    }

    #[test]
    fn completion_request_without_suffix() {
        let wire = r#"{"prompt":"hello"}"#;
        let req: CompletionRequest = serde_json::from_str(wire).unwrap();
        assert!(req.suffix.is_none());
    }

    #[test]
    fn infill_request_middle_alias_works() {
        let wire = r#"{
            "input_prefix":"a","input_suffix":"b","middle":"x"
        }"#;
        let req: InfillRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.prompt.as_deref(), Some("x"));
    }

    #[test]
    fn anthropic_request_plain_string_content() {
        let wire = r#"{
            "model":"flambeau",
            "messages":[{"role":"user","content":"hi"}],
            "max_tokens":256
        }"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.model, "flambeau");
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content.to_plain(), "hi");
        assert!(req.system.is_none());
    }

    #[test]
    fn anthropic_request_content_blocks_text_only() {
        let wire = r#"{
            "model":"flambeau",
            "max_tokens":64,
            "messages":[
                {"role":"user","content":[
                    {"type":"text","text":"part 1 "},
                    {"type":"text","text":"part 2"}
                ]}
            ]
        }"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.messages[0].content.to_plain(), "part 1 part 2");
    }

    #[test]
    fn anthropic_request_system_string_and_blocks() {
        let plain = r#"{
            "model":"x","max_tokens":10,"messages":[],
            "system":"be terse"
        }"#;
        let r: AnthropicMessagesRequest = serde_json::from_str(plain).unwrap();
        assert_eq!(r.system.as_ref().unwrap().to_plain(), "be terse");

        let blocks = r#"{
            "model":"x","max_tokens":10,"messages":[],
            "system":[
                {"type":"text","text":"sys A "},
                {"type":"text","text":"sys B"}
            ]
        }"#;
        let r: AnthropicMessagesRequest = serde_json::from_str(blocks).unwrap();
        assert_eq!(r.system.as_ref().unwrap().to_plain(), "sys A sys B");
    }

    #[test]
    fn anthropic_request_tool_blocks_parse_but_drop_text() {
        // Tool blocks are parsed (so the request doesn't error) but
        // contribute no plain text in P1.8a.
        let wire = r#"{
            "model":"x","max_tokens":10,
            "messages":[
                {"role":"assistant","content":[
                    {"type":"text","text":"calling tool"},
                    {"type":"tool_use","id":"t1","name":"weather","input":{"loc":"SF"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t1","content":"72"}
                ]}
            ]
        }"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.messages[0].content.to_plain(), "calling tool");
        assert_eq!(req.messages[1].content.to_plain(), ""); // tool_result drops
    }

    #[test]
    fn anthropic_response_serialises_textcontent_block() {
        let r = AnthropicMessagesResponse {
            id: "msg_x".into(),
            kind: "message",
            role: "assistant",
            content: vec![AnthropicResponseBlock::Text { text: "hi".into() }],
            model: "flambeau".into(),
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 1,
            },
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"type\":\"message\""));
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(s.contains("\"type\":\"text\",\"text\":\"hi\""));
        assert!(s.contains("\"input_tokens\":5"));
        assert!(s.contains("\"output_tokens\":1"));
        assert!(s.contains("\"stop_reason\":\"end_turn\""));
    }

    #[test]
    fn anthropic_request_tools_field_parses() {
        let wire = r#"{
            "model":"x","max_tokens":10,
            "messages":[{"role":"user","content":"do something"}],
            "tools":[
                {"name":"get_weather","description":"Get weather",
                 "input_schema":{"type":"object","properties":{"loc":{"type":"string"}}}}
            ],
            "tool_choice":{"type":"auto"}
        }"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(wire).unwrap();
        let tools = req.tools.as_ref().expect("tools field present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description.as_deref(), Some("Get weather"));
        assert!(matches!(req.tool_choice, Some(AnthropicToolChoice::Auto)));
    }

    #[test]
    fn anthropic_tool_choice_variants() {
        let auto: AnthropicToolChoice = serde_json::from_str(r#"{"type":"auto"}"#).unwrap();
        assert!(matches!(auto, AnthropicToolChoice::Auto));
        let any: AnthropicToolChoice = serde_json::from_str(r#"{"type":"any"}"#).unwrap();
        assert!(matches!(any, AnthropicToolChoice::Any));
        let none: AnthropicToolChoice = serde_json::from_str(r#"{"type":"none"}"#).unwrap();
        assert!(matches!(none, AnthropicToolChoice::None));
        let named: AnthropicToolChoice =
            serde_json::from_str(r#"{"type":"tool","name":"weather"}"#).unwrap();
        match named {
            AnthropicToolChoice::Tool { name } => assert_eq!(name, "weather"),
            _ => panic!("expected Tool variant"),
        }
    }

    #[test]
    fn anthropic_response_serialises_tool_use_block() {
        let r = AnthropicMessagesResponse {
            id: "msg_x".into(),
            kind: "message",
            role: "assistant",
            content: vec![
                AnthropicResponseBlock::Text {
                    text: "Calling tool".into(),
                },
                AnthropicResponseBlock::ToolUse {
                    id: "toolu_abc".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"loc": "SF"}),
                },
            ],
            model: "flambeau".into(),
            stop_reason: "tool_use".into(),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 12,
            },
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"type\":\"tool_use\""));
        assert!(s.contains("\"id\":\"toolu_abc\""));
        assert!(s.contains("\"name\":\"get_weather\""));
        assert!(s.contains("\"loc\":\"SF\""));
        assert!(s.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn logprobs_request_fields_parse() {
        let wire = r#"{
            "model":"x","messages":[{"role":"user","content":"hi"}],
            "logprobs":true,"top_logprobs":5
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
    }

    #[test]
    fn logprobs_omitted_means_off() {
        let wire = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.logprobs, None);
        assert_eq!(req.top_logprobs, None);
    }

    #[test]
    fn chat_choice_omits_logprobs_when_none() {
        let choice = ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some("hi".into()),
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            },
            finish_reason: "stop".into(),
            logprobs: None,
        };
        let s = serde_json::to_string(&choice).unwrap();
        assert!(!s.contains("logprobs"));
    }

    #[test]
    fn chat_choice_serialises_logprobs_when_present() {
        let choice = ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some("hi".into()),
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            },
            finish_reason: "stop".into(),
            logprobs: Some(ChatLogProbs {
                content: vec![ChatLogProbContent {
                    token: "h".into(),
                    logprob: -0.5,
                    bytes: vec![104],
                    top_logprobs: vec![TopLogProb {
                        token: "H".into(),
                        logprob: -1.2,
                        bytes: vec![72],
                    }],
                }],
            }),
        };
        let s = serde_json::to_string(&choice).unwrap();
        assert!(s.contains("\"logprobs\""));
        assert!(s.contains("\"token\":\"h\""));
        assert!(s.contains("\"top_logprobs\""));
    }

    #[test]
    fn stream_options_include_usage_roundtrip() {
        let wire = r#"{
            "model":"qwen3.6",
            "messages":[{"role":"user","content":"hi"}],
            "stream":true,
            "stream_options":{"include_usage":true}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(wire).unwrap();
        assert!(req.stream);
        assert_eq!(
            req.stream_options.as_ref().and_then(|o| o.include_usage),
            Some(true)
        );
    }

    #[test]
    fn stream_options_missing_is_none() {
        let wire = r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(wire).unwrap();
        assert!(req.stream_options.is_none());
    }

    #[test]
    fn request_accepts_all_new_fields() {
        let wire = r#"{
            "model":"qwen3.6",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"noop","parameters":{"type":"object","properties":{}}}}],
            "tool_choice":"auto",
            "parallel_tool_calls":true,
            "tool_call_format":"hermes",
            "repetition_penalty":1.1,
            "presence_penalty":1.5,
            "top_k":20,
            "min_p":0.0
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(wire).unwrap();
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert!(matches!(req.tool_choice, Some(ToolChoice::Mode(_))));
        assert_eq!(req.parallel_tool_calls, Some(true));
        assert_eq!(req.tool_call_format.as_deref(), Some("hermes"));
        assert_eq!(req.repetition_penalty, Some(1.1));
        assert_eq!(req.top_k, Some(20));
    }
}

// ============================================================================
// **#231 P2.11b** — `/v1/embeddings` types.
// ============================================================================

/// Inputs for `POST /v1/embeddings`. OpenAI accepts:
/// - a single string,
/// - an array of strings (batch),
/// - a single token id array,
/// - an array of token id arrays.
///   V1 only handles strings; integer-array forms return 400.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

/// Embeddings request body. Mirrors the OpenAI shape; unsupported
/// fields (`encoding_format = base64`, `dimensions`, `user`) are
/// accepted and ignored in V1 — we always return raw F32.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingsInput,
    /// Echoed in the response for compatibility — not actually
    /// dispatched to a different model since V1 ships exactly one
    /// embedding head.
    #[serde(default)]
    pub model: Option<String>,
    /// V1 only emits `"float"`. Reserved for compat.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// V1 returns the model's native hidden size; the OpenAI
    /// truncation-at-`dimensions` feature is V2.
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

/// Single-input response data slot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddingData {
    /// Always `"embedding"`.
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Token usage block for `/v1/embeddings`. Mirrors OpenAI's shape
/// (no `completion_tokens`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Top-level `/v1/embeddings` response.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddingsResponse {
    /// Always `"list"`.
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

// ---- /tokenize, /detokenize -----------------------------------------------

/// **#234 P3.14** — llama.cpp-compatible tokenize endpoint body.
/// Mirrors `llama.cpp` server: `content` is the text to tokenize,
/// `add_special` toggles BOS/EOS injection (default `false` — the
/// chat template handles specials for actual chat turns), and
/// `with_pieces` switches the response from a flat `[id, ...]` to
/// a list of `{id, piece}` objects so clients can render the
/// per-token surface form (token highlighter UIs).
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    pub content: String,
    #[serde(default)]
    pub add_special: bool,
    #[serde(default)]
    pub with_pieces: bool,
}

/// **#234 P3.14** — llama.cpp-compatible detokenize endpoint body.
#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub tokens: Vec<u32>,
}
