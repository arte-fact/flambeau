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
    /// **P0.2** — caller-provided stop sequences. OpenAI accepts a
    /// string or an array of up to 4 strings; parsed at the request
    /// boundary into a `Vec<String>`. Decode loops detect each on the
    /// running emitted text (string-level, identical mechanism to the
    /// reasoning-marker stop) and cut decoding when matched. The stop
    /// sequence itself is stripped from the final text in `finalise`.
    pub stop_strings: Vec<String>,
    /// **P1.7** — when `Some(n)`, the decode loop collects per-token
    /// log-probabilities + the top-`n` alternatives. `None` means no
    /// collection (the OpenAI default). Range 0..=20, capped at the
    /// boundary. V1 implementation is non-streaming, host-sampler-only;
    /// other paths silently return `None` with a warn-once log.
    pub collect_logprobs: Option<u32>,
    /// **#233 P3.13** — opt-in for Qwen3.6 reasoning mode. `true` ⇒
    /// the chat template is rendered with `enable_thinking=true`, the
    /// `<think>` / `</think>` reasoning-marker stop logic is
    /// disabled in the decode loop, and `finalise` splits the
    /// generated text at the closing tag into
    /// `(reasoning_content, content)` for the response body.
    pub enable_thinking: bool,
    /// **#236 P0.1b** — bytes to feed into the JSON state machine
    /// before the first decoded token. Two callers populate this:
    /// (a) JSON-mode requests where the last `messages[]` entry is
    /// `role: "assistant"` (assistant-prefill / partial-completion
    /// pattern; OpenAI lets clients seed the model's response). The
    /// chat handler renders with `add_generation_prompt=false` in
    /// that case so the model continues from the prefill content,
    /// and `json_prime_bytes` carries the same content so the mask
    /// state matches the model's position. (b) Future template-
    /// emitted JSON prefixes (`{`-injected meta-prompts, V2). Empty
    /// for non-JSON-mode requests and for the common no-prefill
    /// case; the decoder then starts the JSON state machine fresh.
    pub json_prime_bytes: Vec<u8>,
}

/// Model-side recommended sampling defaults read from GGUF metadata
/// (`general.sampling.{temp,top_p,top_k,min_p}`). When the OpenAI
/// request omits a knob, the corresponding model default fills in. If
/// neither is set, [`SamplingParams::from_parts`] falls back to the
/// OpenAI/unbiased defaults documented on each field.
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
    pub repetition_penalty: Option<f32>,
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
            repetition_penalty: gguf
                .metadata_f32("general.sampling.repetition_penalty"),
        }
    }
}

/// Raw OpenAI sampling knobs, all optional. Each `None` falls back to
/// the matching [`ModelDefaults`] entry, then to the unbiased OpenAI
/// default documented on [`SamplingParams::from_parts`].
#[derive(Debug, Clone, Default)]
pub struct SamplingKnobs {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

/// Per-request generation budget + stop conditions.
#[derive(Debug, Clone, Default)]
pub struct GenerationLimits {
    pub max_tokens: Option<u32>,
    pub seed: Option<u64>,
    pub stop_strings: Vec<String>,
}

/// Response-mode flags + structured-output primer.
#[derive(Debug, Clone, Default)]
pub struct ResponseMode {
    pub json_mode: bool,
    pub collect_logprobs: Option<u32>,
    pub enable_thinking: bool,
    pub json_prime_bytes: Vec<u8>,
}

impl SamplingParams {
    /// Derive from OpenAI params. Defaults match the OpenAI surface so
    /// that an unconfigured client gets unbiased sampling.
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
    pub fn from_parts(
        knobs: SamplingKnobs,
        limits: GenerationLimits,
        mode: ResponseMode,
        defaults: &ModelDefaults,
    ) -> Self {
        let SamplingKnobs {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
        } = knobs;
        let GenerationLimits { max_tokens, seed, stop_strings } = limits;
        let ResponseMode {
            json_mode,
            collect_logprobs,
            enable_thinking,
            json_prime_bytes,
        } = mode;
        // Resolution order: explicit OpenAI request → GGUF model
        // default → OpenAI/unbiased fallback. `temperature == 0.0`
        // is a documented greedy override and must NOT fall back to
        // the model default — that would surprise a caller who
        // explicitly asked for greedy.
        // **P0.4** — auto-low-temp for tool-router / structured-output
        // meta-prompts. OpenWebUI's auto-prompts (search-query-gen,
        // follow-ups, title, tags) and Aider/Continue/LangChain JSON
        // modes all set `response_format`. At default temperature
        // these reliably hallucinate (non-existent search hits, made-
        // up file names, fake tool args). Clamp to 0.2 for JSON-mode
        // turns where the caller did NOT set temperature explicitly;
        // an explicit request value always wins so a power user can
        // opt out per-call by passing any temperature.
        let request_set_temp = temperature.is_some();
        let temperature = if json_mode && !request_set_temp {
            0.2
        } else {
            temperature.or(defaults.temperature).unwrap_or(1.0)
        };
        let top_p = top_p.or(defaults.top_p).filter(|p| *p < 1.0 && *p > 0.0);
        let top_k = top_k.or(defaults.top_k).filter(|k| *k > 0);
        let min_p = min_p.or(defaults.min_p).filter(|m| *m > 0.0);
        // Rep-penalty resolution: request → GGUF default → 1.0 (off).
        // Low-temp override: at effectively-greedy temperatures with no
        // caller-supplied penalty and no GGUF default, bump to 1.05 so
        // the model can emit EOS instead of locking on its own argmax.
        // Verified live on Qwen3.6-27B UD-Q3_K_XL where temp=0 looped
        // "Paris.\n\nParis.\n\nParis…" indefinitely until rep_penalty
        // unblocked EOS. Explicit caller value (including 1.0) always
        // wins so a power user can opt out per-call.
        let request_set_rep = repetition_penalty.is_some();
        let resolved_rep = repetition_penalty
            .or(defaults.repetition_penalty)
            .unwrap_or(1.0);
        let repetition_penalty =
            if temperature < 0.1 && !request_set_rep && resolved_rep == 1.0 {
                1.05
            } else {
                resolved_rep
            };
        let sampling = Sampling {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty: presence_penalty.unwrap_or(0.0),
            frequency_penalty: frequency_penalty.unwrap_or(0.0),
        };
        SamplingParams {
            sampling,
            seed: seed.unwrap_or_else(default_seed),
            max_tokens: max_tokens.unwrap_or(4096).min(8192),
            json_mode,
            stop_strings,
            collect_logprobs: collect_logprobs.map(|n| n.min(20)),
            enable_thinking,
            json_prime_bytes,
        }
    }
}

/// Parse the OpenAI `stop` field. Accepts:
/// - `null` / missing → empty
/// - a single string → one-element vec
/// - an array of strings → first 4 non-empty entries
///   OpenAI caps the array at 4 entries; longer arrays are truncated rather
///   than rejected so a misconfigured client gets a usable response.
pub fn parse_stop(stop: Option<&serde_json::Value>) -> Vec<String> {
    let Some(v) = stop else { return Vec::new() };
    match v {
        serde_json::Value::String(s) if !s.is_empty() => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .take(4)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_stop_missing_is_empty() {
        assert!(parse_stop(None).is_empty());
        assert!(parse_stop(Some(&json!(null))).is_empty());
    }

    #[test]
    fn parse_stop_single_string() {
        let v = json!("</done>");
        let expected: Vec<String> = vec!["</done>".to_string()];
        assert_eq!(parse_stop(Some(&v)), expected);
    }

    #[test]
    fn parse_stop_empty_string_dropped() {
        assert!(parse_stop(Some(&json!(""))).is_empty());
    }

    #[test]
    fn parse_stop_array_caps_at_four() {
        let v = json!(["a", "b", "c", "d", "e", "f"]);
        let expected: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_stop(Some(&v)), expected);
    }

    #[test]
    fn parse_stop_array_filters_non_strings_and_empty() {
        let v = json!(["", "x", 42, null, "y"]);
        let expected: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_stop(Some(&v)), expected);
    }

    #[test]
    fn parse_stop_other_kinds_yield_empty() {
        assert!(parse_stop(Some(&json!(42))).is_empty());
        assert!(parse_stop(Some(&json!({"k": "v"}))).is_empty());
    }

    fn defaults_none() -> ModelDefaults {
        ModelDefaults::default()
    }

    fn knobs_with_temp(t: Option<f32>) -> SamplingKnobs {
        SamplingKnobs { temperature: t, ..Default::default() }
    }

    fn mode_json(json_mode: bool) -> ResponseMode {
        ResponseMode { json_mode, ..Default::default() }
    }

    #[test]
    fn p04_json_mode_no_explicit_temp_clamps_to_low() {
        let p = SamplingParams::from_parts(
            knobs_with_temp(None),
            GenerationLimits::default(),
            mode_json(true),
            &defaults_none(),
        );
        assert!((p.sampling.temperature - 0.2).abs() < 1e-6);
    }

    #[test]
    fn p04_json_mode_explicit_temp_wins() {
        let p = SamplingParams::from_parts(
            knobs_with_temp(Some(0.9)),
            GenerationLimits::default(),
            mode_json(true),
            &defaults_none(),
        );
        assert!((p.sampling.temperature - 0.9).abs() < 1e-6);
    }

    #[test]
    fn p04_non_json_mode_keeps_default() {
        let p = SamplingParams::from_parts(
            knobs_with_temp(None),
            GenerationLimits::default(),
            mode_json(false),
            &defaults_none(),
        );
        assert!((p.sampling.temperature - 1.0).abs() < 1e-6);
    }

    #[test]
    fn p04_json_mode_explicit_zero_temp_stays_greedy() {
        let p = SamplingParams::from_parts(
            knobs_with_temp(Some(0.0)),
            GenerationLimits::default(),
            mode_json(true),
            &defaults_none(),
        );
        assert!(p.sampling.temperature.abs() < 1e-6);
    }
}
