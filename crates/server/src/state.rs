//! Shared per-request sampling parameters derived from OpenAI request.

use flambeau_runtime::Sampling;

/// Decoded sampling config + limits.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub sampling: Sampling,
    pub seed: u64,
    pub max_tokens: u32,
}

impl SamplingParams {
    /// Derive from OpenAI params. Missing fields default per Qwen3.5's
    /// published `generation_config.json` — the numbers that make
    /// multi-turn agent loops stable (T4.b, ROADMAP-V2 §T4.b).
    ///
    /// - `temperature` → `1.0` when omitted. `0.0` means greedy.
    /// - `top_p` → passed through when in `(0, 1)`.
    /// - `top_k` → `20` (Qwen recommended). `0` or `None` both disable it.
    /// - `min_p` → `0.0` (off). Passed through when positive.
    /// - `repetition_penalty` → `1.0` (off) unless the client provides one.
    /// - `presence_penalty` → `1.5` (Qwen3.5 default) when omitted — the
    ///   value cited as the "load-bearing" fix for agent loops on
    ///   r/LocalLLaMA 2026-03-03.
    /// - `frequency_penalty` → `0.0` unless client-supplied.
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
    ) -> Self {
        let sampling = Sampling {
            temperature: temperature.unwrap_or(1.0),
            top_p: top_p.filter(|p| *p < 1.0 && *p > 0.0),
            // `top_k=0` in OpenAI-compat land means "disabled". Map it to
            // None; any positive value passes through.
            top_k: top_k.filter(|k| *k > 0),
            min_p: min_p.filter(|m| *m > 0.0),
            repetition_penalty: repetition_penalty.unwrap_or(1.0),
            presence_penalty: presence_penalty.unwrap_or(1.5),
            frequency_penalty: frequency_penalty.unwrap_or(0.0),
        };
        SamplingParams {
            sampling,
            seed: seed.unwrap_or_else(default_seed),
            max_tokens: max_tokens.unwrap_or(128).min(2048),
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
