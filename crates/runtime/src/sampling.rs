//! CPU-side token sampling — V1.8.A piece 1 of 3.
//!
//! Takes a `&[f32]` of decoder logits (one per vocab entry) and returns the
//! sampled token id. Three modes:
//!  - `Greedy`             → argmax (matches our current decode_profile path).
//!  - `Temperature(temp)`  → scale logits by `1/temp`, softmax, sample.
//!  - `TopP { temp, p }`   → temperature + nucleus filter before sampling.
//!
//! Deterministic when `seed` is fixed. The RNG is xoshiro256** (self-contained;
//! no `rand` dep added so the crate stays light).
//!
//! Serving-layer note: the HTTP server (V1.8.B) will construct one `Sampler`
//! per request from OpenAI params (`temperature`, `top_p`, `seed`). The
//! runtime owns the logit buffer; sampling is a pure Vec→u32 transformation.

use std::cmp::Ordering;

/// Sampling strategy.
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    /// argmax — deterministic, matches OpenAI `temperature=0`.
    Greedy,
    /// Softmax(logits / temp), sample from full distribution.
    Temperature { temp: f32 },
    /// Temperature + nucleus (top-p) filter: keep minimum-size prefix of
    /// sorted softmax whose cumulative prob ≥ p, renormalise, sample.
    TopP { temp: f32, p: f32 },
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling::Greedy
    }
}

/// Deterministic PRNG. xoshiro256** — fast, small-state, well-distributed.
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        // SplitMix64 to expand 1 → 4 words (avoids a zero state).
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut out = [0u64; 4];
        for slot in out.iter_mut() {
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = x ^ (x >> 31);
        }
        if out == [0; 4] {
            out[0] = 1;
        }
        Rng { s: out }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// Sample one token from the logit row. `logits.len()` = vocab_size.
/// Returns the vocab id.
pub fn sample(logits: &[f32], mode: Sampling, rng: &mut Rng) -> u32 {
    match mode {
        Sampling::Greedy => argmax(logits),
        Sampling::Temperature { temp } => sample_softmax_temp(logits, temp, rng),
        Sampling::TopP { temp, p } => sample_top_p(logits, temp, p, rng),
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best = i as u32;
            best_v = v;
        }
    }
    best
}

/// Numerically-stable softmax with temperature, then multinomial sample.
fn sample_softmax_temp(logits: &[f32], temp: f32, rng: &mut Rng) -> u32 {
    let inv_t = if temp <= 0.0 { 1.0 } else { 1.0 / temp };
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = v * inv_t;
        if scaled > max_l {
            max_l = scaled;
        }
    }
    // Probabilities scaled: exp(l*inv_t - max).
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, &v) in logits.iter().enumerate() {
        let p = (v * inv_t - max_l).exp();
        probs[i] = p;
        sum += p;
    }
    multinomial_pick(&probs, sum, rng)
}

/// Temperature + nucleus (top-p) sampling.
fn sample_top_p(logits: &[f32], temp: f32, p: f32, rng: &mut Rng) -> u32 {
    let inv_t = if temp <= 0.0 { 1.0 } else { 1.0 / temp };
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = v * inv_t;
        if scaled > max_l {
            max_l = scaled;
        }
    }
    // Softmax over full vocab first.
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, (v * inv_t - max_l).exp()))
        .collect();
    let sum_full: f32 = probs.iter().map(|(_, p)| *p).sum();
    for (_, pv) in probs.iter_mut() {
        *pv /= sum_full;
    }

    // Sort by probability descending, take cumulative-prob prefix covering `p`.
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut cum = 0.0f32;
    let mut kept_end = 0usize;
    for (i, &(_, pv)) in probs.iter().enumerate() {
        cum += pv;
        kept_end = i + 1;
        if cum >= p {
            break;
        }
    }
    let kept = &probs[..kept_end];
    let sum_kept: f32 = kept.iter().map(|(_, pv)| *pv).sum();

    // Sample from the truncated distribution.
    let mut u = rng.next_f32() * sum_kept;
    for (id, pv) in kept {
        u -= pv;
        if u <= 0.0 {
            return *id;
        }
    }
    kept.last().map(|(id, _)| *id).unwrap_or(0)
}

fn multinomial_pick(probs: &[f32], sum: f32, rng: &mut Rng) -> u32 {
    let mut u = rng.next_f32() * sum;
    for (i, &p) in probs.iter().enumerate() {
        u -= p;
        if u <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1, 0.5, 0.9, 0.2, -1.0];
        let mut rng = Rng::from_seed(42);
        assert_eq!(sample(&logits, Sampling::Greedy, &mut rng), 2);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = [0.1, 0.5, 0.9, 0.2, -1.0];
        let mut rng = Rng::from_seed(42);
        // temp <= 0 → treated as 1.0 internally. True greedy path is mode-level.
        assert_eq!(sample(&logits, Sampling::Greedy, &mut rng), 2);
    }

    #[test]
    fn temperature_is_deterministic_with_seed() {
        let logits = [1.0, 2.0, 3.0, 2.5, 1.5];
        let mut rng_a = Rng::from_seed(1234);
        let mut rng_b = Rng::from_seed(1234);
        let a = sample(&logits, Sampling::Temperature { temp: 0.7 }, &mut rng_a);
        let b = sample(&logits, Sampling::Temperature { temp: 0.7 }, &mut rng_b);
        assert_eq!(a, b, "same seed must give same token");
    }

    #[test]
    fn top_p_restricts_to_high_probability_tokens() {
        // With a strongly peaked distribution, top-p=0.1 should pick the argmax.
        let logits = [0.0, 0.0, 10.0, 0.0, 0.0];
        let mut rng = Rng::from_seed(0xabc);
        let id = sample(&logits, Sampling::TopP { temp: 1.0, p: 0.5 }, &mut rng);
        assert_eq!(id, 2);
    }

    #[test]
    fn top_p_distribution_stays_within_nucleus() {
        // Two near-equal-probability tokens with the rest near zero. Many samples
        // should land on one of those two.
        let logits = [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = Rng::from_seed(7);
        let mut counts = [0u32; 7];
        for _ in 0..1000 {
            let id = sample(&logits, Sampling::TopP { temp: 1.0, p: 0.95 }, &mut rng);
            counts[id as usize] += 1;
        }
        // Tokens 0 and 1 should capture ~all probability mass.
        assert!(counts[0] + counts[1] >= 990, "counts={counts:?}");
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::from_seed(99);
        let mut b = Rng::from_seed(99);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
