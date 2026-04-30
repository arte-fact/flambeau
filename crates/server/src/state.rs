//! Shared per-request sampling parameters derived from OpenAI request.

use flambeau_quant::GgufFile;
use flambeau_runtime::Sampling;

/// Decoded sampling config + limits.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub sampling: Sampling,
    pub seed: u64,
    pub max_tokens: u32,
    /// **P0.1** — when `true`, the decode loop applies a per-step JSON
    /// structural mask: candidate tokens are filtered against a small
    /// JSON state machine (object/array/string/number balance). Driven
    /// by the OpenAI `response_format: {"type":"json_object"}` request
    /// field. Default `false` (free-form text).
    pub json_mode: bool,
}

/// Model-side recommended sampling defaults read from GGUF metadata
/// (`general.sampling.{temp,top_p,top_k,min_p}`). When the OpenAI
/// request omits a knob, the corresponding model default fills in. If
/// neither is set, [`SamplingParams::from_parts`] falls back to the
/// OpenAI/unbiased defaults documented on each field.
///
/// Authors of GGUFs (Unsloth, official Qwen, etc.) ship these values
/// because the model is calibrated for them — e.g.
/// Qwen3-Coder-Next ships `temp=1.0, top_p=0.95, top_k=40` and
/// degrades visibly at the OpenAI defaults of `temp=1.0, top_p=1.0,
/// top_k=∞`. Honouring them gives every model its baked-in best-shot
/// without the operator hand-tuning per deployment.
#[derive(Debug, Clone, Default)]
pub struct ModelDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
}

impl ModelDefaults {
    /// Read `general.sampling.*` metadata. Missing keys leave the
    /// corresponding field as `None`.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        Self {
            temperature: gguf.metadata_f32("general.sampling.temp"),
            top_p: gguf.metadata_f32("general.sampling.top_p"),
            top_k: gguf.metadata_u32("general.sampling.top_k"),
            min_p: gguf.metadata_f32("general.sampling.min_p"),
        }
    }
}

impl SamplingParams {
    /// Derive from OpenAI params. Defaults match the OpenAI surface so
    /// that an unconfigured client gets unbiased sampling.
    ///
    /// - `temperature` → `1.0` when omitted. `0.0` means greedy.
    /// - `top_p` → passed through when in `(0, 1)`.
    /// - `top_k` → `None` (disabled). `0` is also treated as disabled.
    /// - `min_p` → `0.0` (off). Passed through when positive.
    /// - `repetition_penalty` → `1.0` (off) unless the client provides one.
    /// - `presence_penalty` → `0.0` (OpenAI default) unless client-set.
    ///   An earlier default of `1.5` was lifted from a community claim
    ///   about Qwen3.5's "stable agent setup"; in practice it is far too
    ///   aggressive for normal chat — every token already in history
    ///   loses 1.5 nats, and on long generations the distribution drifts
    ///   into degenerate synonym-spam (verified live). Keep it 0.0;
    ///   callers doing agent loops can opt into a small positive value.
    /// - `frequency_penalty` → `0.0` unless client-supplied.
    /// - `max_tokens` → `4096` when omitted; capped at `8192` to keep a
    ///   single request from monopolising the server. Earlier 2048 cap
    ///   silently truncated long answers (`finish=length` after exactly
    ///   2048 tokens regardless of the request); earlier 512 default
    ///   was too tight for code-generation requests via curl/clients
    ///   that don't pass `max_tokens` explicitly.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u64>,
        json_mode: bool,
        defaults: &ModelDefaults,
    ) -> Self {
        // Resolution order: explicit OpenAI request → GGUF model
        // default → OpenAI/unbiased fallback. `temperature == 0.0`
        // is a documented greedy override and must NOT fall back to
        // the model default — that would surprise a caller who
        // explicitly asked for greedy.
        let temperature = temperature
            .or(defaults.temperature)
            .unwrap_or(1.0);
        let top_p = top_p.or(defaults.top_p).filter(|p| *p < 1.0 && *p > 0.0);
        let top_k = top_k.or(defaults.top_k).filter(|k| *k > 0);
        let min_p = min_p.or(defaults.min_p).filter(|m| *m > 0.0);
        let sampling = Sampling {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty: repetition_penalty.unwrap_or(1.0),
            presence_penalty: presence_penalty.unwrap_or(0.0),
            frequency_penalty: frequency_penalty.unwrap_or(0.0),
        };
        SamplingParams {
            sampling,
            seed: seed.unwrap_or_else(default_seed),
            max_tokens: max_tokens.unwrap_or(4096).min(8192),
            json_mode,
        }
    }
}

fn default_seed() -> u64 {
    // Non-deterministic if `seed` omitted — uses the current-time nanos
    // to avoid repeating exact outputs per-request.
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
}
