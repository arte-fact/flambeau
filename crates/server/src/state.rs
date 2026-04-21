//! Shared per-request sampling parameters derived from OpenAI request.

use flambeau_runtime::Sampling;

/// Decoded sampling config + limits.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub sampling: Sampling,
    pub seed: u64,
    pub max_tokens: u32,
}

impl SamplingParams {
    /// Derive from OpenAI params. Missing fields map to sensible defaults:
    ///   - temperature=0 or omitted → greedy
    ///   - temperature>0 + top_p<1 → top-p
    ///   - temperature>0 alone → plain temperature
    pub fn from_parts(
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u64>,
    ) -> Self {
        let temp = temperature.unwrap_or(1.0);
        let max_tokens = max_tokens.unwrap_or(128).min(2048);
        let seed = seed.unwrap_or_else(default_seed);
        let sampling = if temp <= 0.0 {
            Sampling::Greedy
        } else if let Some(p) = top_p.filter(|p| *p < 1.0 && *p > 0.0) {
            Sampling::TopP { temp, p }
        } else {
            Sampling::Temperature { temp }
        };
        SamplingParams {
            sampling,
            seed,
            max_tokens,
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
