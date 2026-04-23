//! Chat-template renderer — V1.8.A piece 3 of 3.
//!
//! GGUF embeds `tokenizer.chat_template` as a Jinja2 string (7816 bytes for
//! Qwen3.6 — macros, whitespace control, conditionals). `minijinja` renders
//! it against a `messages: [{role, content}, ...]` array plus a few flags
//! (`add_generation_prompt`, `enable_thinking`, ...) to produce the raw text
//! prompt the model sees.
//!
//! The HTTP layer (V1.8.B) constructs this from OpenAI `/v1/chat/completions`
//! payloads, renders, tokenizes (V1.8.A.2), runs the forward, and samples
//! (V1.8.A.1). Chat template is the glue between wire format and model input.

use anyhow::{anyhow, Context, Result};
use minijinja::{value::Value as MjValue, Environment};
use serde::{Deserialize, Serialize};

use crate::gguf::GgufFile;

/// One entry in an OpenAI-style `messages` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// "system" / "user" / "assistant" / "tool".
    pub role: String,
    /// Message text. Qwen's template handles strings; multi-modal content
    /// arrays (vision / video) are V2.
    pub content: String,
}

/// Rendered chat template: the raw text prompt to feed the tokenizer.
pub struct ChatTemplate {
    env: Environment<'static>,
    template_name: &'static str,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("template_name", &self.template_name)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Load `tokenizer.chat_template` from a GGUF and pre-compile it.
    ///
    /// # Errors
    /// - `anyhow` wrapping "tokenizer.chat_template missing" if the GGUF
    ///   lacks the metadata key.
    /// - Any error [`from_string`] can return (minijinja parse failure).
    pub fn load_from_gguf(file: &GgufFile) -> Result<Self> {
        let tpl_str = file
            .metadata_str("tokenizer.chat_template")
            .context("tokenizer.chat_template missing from GGUF metadata")?
            .to_owned();
        Self::from_string(tpl_str)
    }

    /// Pre-compile a raw template string. Accepts owned String so we can
    /// stash it in the environment for the `'static` lifetime.
    ///
    /// # Errors
    /// `anyhow` wrapping `minijinja::Error` if the template fails to parse.
    pub fn from_string(tpl_str: String) -> Result<Self> {
        let mut env = Environment::new();
        // Install Python-compat string/list methods. Qwen's template uses
        // `.startswith`, `.split`, `.endswith`, etc. — minijinja proper
        // doesn't ship these; `minijinja_contrib::pycompat` does.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Leak the template string — it lives for the rest of the process
        // and Jinja needs a `'static` reference. One per model load, fine.
        let leaked: &'static str = Box::leak(tpl_str.into_boxed_str());
        env.add_template("chat", leaked)
            .map_err(|e| anyhow!("minijinja add_template: {e}"))?;
        Ok(Self {
            env,
            template_name: "chat",
        })
    }

    /// Render messages into the model prompt. `add_generation_prompt=true`
    /// appends the assistant header so the model completes into a new turn.
    ///
    /// # Errors
    /// `anyhow` wrapping a `minijinja::Error` if the template references an
    /// undefined variable, applies an unknown filter/method, or raises from
    /// inside a `{% raise %}` block.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tpl = self
            .env
            .get_template(self.template_name)
            .map_err(|e| anyhow!("minijinja get_template: {e}"))?;
        let rendered = tpl
            .render(minijinja::context! {
                messages => MjValue::from_serialize(messages),
                add_generation_prompt => add_generation_prompt,
                // Qwen3-family flags. False keeps the template's "non-thinking"
                // path (no <think>...</think> blocks). Flip for reasoning mode.
                enable_thinking => false,
                // Vision/tool calls disabled on the server-level V1 slice.
                tools => MjValue::from_serialize(Vec::<String>::new()),
            })
            .map_err(|e| anyhow!("minijinja render: {e}"))?;
        Ok(rendered)
    }
}
