//! OpenAI-compatible request/response types.
//!
//! V1.8 shipped chat completions, text completions, models list, health.
//! V2 tool-calling track (T1.1 — ROADMAP-V2-TOOL-CALLING-AND-MCP.md) adds
//! the wire surface for `tools[]`, `tool_choice`, `tool_calls[]`,
//! `role="tool"`, and `finish_reason="tool_calls"`. The types are wired
//! here but not acted upon yet — T1.2/T2.x fill in the behaviour.
//!
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
}

// ---- /v1/chat/completions --------------------------------------------------

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

    // ---- Streaming options (P0.3) ------------------------------------
    /// OpenAI `stream_options`. When `include_usage=true`, the server
    /// emits a final `choices=[]` SSE chunk carrying the `usage` block
    /// after the finish-reason chunk. This is the canonical OpenAI
    /// shape that the openai-python SDK and LangChain expect; without
    /// it, some clients silently drop the `usage` field embedded in
    /// the finish chunk. Default `None` (no extra chunk).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
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
///
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

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// One tool call emitted by the model and echoed back in prior-turn
/// assistant messages.
///
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

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    /// OpenAI `finish_reason`. Valid values today: `"stop"`, `"length"`,
    /// and — with T2.5 wired — `"tool_calls"`.
    pub finish_reason: String,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Back-compat: V1.8 clients that POST messages with string `content`
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

        let named: ToolChoice = serde_json::from_str(
            r#"{"type":"function","function":{"name":"get_weather"}}"#,
        )
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
