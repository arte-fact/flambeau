//! CPU-side token sampling.
//! Before 2 the config was a three-variant enum (`Greedy | Temperature |
//! TopP`). 2 (T4.b — ROADMAP-V2-TOOL-CALLING-AND-MCP §T4.b) extends it
//! to a single struct carrying the full OpenAI sampler surface —
//! `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
//! `frequency_penalty`, `repetition_penalty`. Without the penalties,
//! multi-turn agent loops on Qwen3.5/3.6 degrade to the "long CoT /
//! garbage output" failure mode community-reported on Ollama.
//! Takes a `&[f32]` of decoder logits (one per vocab entry) and an
//! optional token-history slice (prior-turn tokens — used by the
//! penalties) and returns the sampled token id.
//! Deterministic when `seed` is fixed. The RNG is xoshiro256** (self-contained;
//! no `rand` dep added so the crate stays light).
//! Serving-layer note: the HTTP server constructs one [`Sampler`] per
//! request from OpenAI params and calls [`Sampler::sample`] once per
//! token, reusing its scratch buffers across the decode loop.

use std::cmp::Ordering;

/// Full sampler config for one decode step.
/// Field semantics mirror OpenAI / vLLM:
/// - `temperature == 0.0` → greedy (skip softmax, argmax). Historical
/// `Sampling::Greedy` variant maps to `temperature = 0.0` here.
/// - `top_p = None` → no top-p filter.
/// - `top_k = None` → no top-k filter.
/// - `min_p = None` → no min-p filter.
/// - `repetition_penalty = 1.0` → disabled. Non-1 values scale the
/// logit of previously-seen tokens (`logit /= penalty` when
/// `logit > 0`, `logit *= penalty` otherwise) — the llama.cpp
/// convention, shared by Qwen's `generation_config.json`.
/// - `presence_penalty = 0.0` → disabled. Subtracts `penalty` from the
/// logit of any token appearing at least once in history.
/// - `frequency_penalty = 0.0` → disabled. Subtracts
/// `penalty * count_in_history` from the logit of each token.
/// Penalty semantics match OpenAI's reference: penalties are applied
/// IN-PLACE to logits before any filter or softmax. History is the
/// caller's per-turn generated-tokens slice.
#[derive(Debug, Clone)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self::greedy()
    }
}

impl Sampling {
    /// Greedy sampler — `temperature = 0.0`, no filters, no penalties.
    /// Deterministic argmax.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            min_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }

    /// Temperature-only sampler. Equivalent to the pre-2
    /// `Sampling::Temperature { temp }` variant.
    pub fn temperature(temp: f32) -> Self {
        Self {
            temperature: temp,
            ..Self::greedy()
        }
    }

    /// Temperature + top-p. Equivalent to the pre-2
    /// `Sampling::TopP { temp, p }` variant.
    pub fn top_p(temp: f32, p: f32) -> Self {
        Self {
            temperature: temp,
            top_p: Some(p),
            ..Self::greedy()
        }
    }

    /// `true` iff this config will behave as argmax (no stochastic
    /// branch will be taken). Used by the server to short-circuit the
    /// stop-token-mask softening that only makes sense with
    /// non-deterministic sampling.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// True iff any penalty field is active. Caller can skip the
    /// history-walk when both we and the model don't need it.
    pub fn has_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
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
        for slot in &mut out {
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

/// One-shot sample: allocates per call. Prefer [`Sampler::sample`] on
/// the decode hot path where the scratch buffers can be reused.
pub fn sample(
    logits: &[f32],
    mode: &Sampling,
    history: &[u32],
    rng: &mut Rng,
) -> u32 {
    // Clone logits into scratch so penalties can mutate without
    // touching the caller's buffer.
    let mut scratch: Vec<f32> = logits.to_vec();
    apply_penalties(&mut scratch, history, mode);
    if mode.is_greedy() {
        return argmax(&scratch);
    }
    let mut pair_scratch: Vec<(u32, f32)> = Vec::new();
    sample_stochastic(&scratch, mode, rng, &mut pair_scratch)
}

/// Per-session sampler that reuses scratch across tokens. On Qwen3.6
/// (vocab=151 936) each non-greedy call would otherwise allocate ~593 KiB
/// per token; reusing the buffers turns the alloc churn into a one-time
/// cost per session — see C2 in `RUST-PERF-CORRECTIONS.md`.
#[derive(Debug)]
pub struct Sampler {
    rng: Rng,
    /// Reused as the penalty-adjusted logit buffer.
    logit_scratch: Vec<f32>,
    /// Reused for top-k / top-p / min-p filtering.
    pair_scratch: Vec<(u32, f32)>,
    /// Sampler-F (#208) — reused sorted+dedup history snapshot for the
    /// penalty path. Replaces the per-call `HashMap<u32, u32>` with a
    /// `Vec<(u32, u32)>` (token id, count) sorted by id. Build via
    /// sort_unstable on a copy of `history`, then dedup-with-counter.
    /// At typical history lengths (≤2k), this beats hashing on cache
    /// behaviour and avoids the per-call HashMap allocation.
    history_counts: Vec<(u32, u32)>,
    /// Sampler-F sort-staging buffer (cleared + filled with history
    /// each call when penalties active).
    history_sorted: Vec<u32>,
}

impl Sampler {
    /// Build a sampler seeded by `seed`. Scratch buffers are empty and
    /// grow on the first call to `vocab_size`, after which they stay at
    /// peak capacity for the lifetime of this `Sampler`.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: Rng::from_seed(seed),
            logit_scratch: Vec::new(),
            pair_scratch: Vec::new(),
            history_counts: Vec::new(),
            history_sorted: Vec::new(),
        }
    }

    /// Pre-reserve scratch for a known `vocab_size`.
    pub fn reserve(&mut self, vocab_size: usize) {
        self.logit_scratch.reserve(vocab_size);
        self.pair_scratch.reserve(vocab_size);
    }

    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// **Sampler-D3 (#211)** — sample one token from a pre-computed top-K
    /// `(id, prob)` distribution. The caller is responsible for
    /// providing a top-K already softmax-normalised so the K probs sum
    /// to ≈1 (the GPU `topk_softmax_f32` kernel does exactly this).
    /// `top_p` and `min_p` filters are applied across the K-tuple here;
    /// `top_k` is implicit (already the size of the input). Penalties
    /// are NOT applied — the GPU sampler path requires `!mode.has_penalties()`
    /// because penalties need full-vocab access on host.
    /// `temperature` is also implicit: the GPU kernel applied it before
    /// emitting the top-K probs, so the input distribution already
    /// reflects the chosen temperature.
    /// Returns the multinomial-sampled token id, or `topk_ids[0]` (the
    /// argmax) when `mode.is_greedy()`.
    pub fn sample_from_topk(
        &mut self,
        topk_ids: &[u32],
        topk_probs: &[f32],
        mode: &Sampling,
    ) -> u32 {
        debug_assert_eq!(
            topk_ids.len(),
            topk_probs.len(),
            "sample_from_topk: ids/probs length mismatch"
        );
        if topk_ids.is_empty() {
            return 0;
        }
        if mode.is_greedy() {
            return topk_ids[0];
        }
        // Reuse pair_scratch for the small K-tuple. We don't need a
        // sort — the input is already sorted descending by prob from
        // the GPU kernel.
        let mut end = topk_ids.len();
        // Apply top-p over the prefix.
        if let Some(p) = mode.top_p {
            if p < 1.0 && p > 0.0 {
                let mut cum = 0.0f32;
                let mut prefix_end = 0usize;
                for &pv in topk_probs.iter().take(end) {
                    cum += pv;
                    prefix_end += 1;
                    if cum >= p {
                        break;
                    }
                }
                end = prefix_end;
            }
        }
        // Apply min-p (drop entries whose prob < min_p * top_prob).
        if let Some(min_p) = mode.min_p {
            if min_p > 0.0 && !topk_probs.is_empty() {
                let threshold = topk_probs[0] * min_p;
                for (i, &pv) in topk_probs.iter().take(end).enumerate() {
                    if pv < threshold {
                        end = i.max(1);
                        break;
                    }
                }
            }
        }
        // Multinomial pick over [0..end].
        let kept_probs = &topk_probs[..end];
        let kept_ids = &topk_ids[..end];
        let sum: f32 = kept_probs.iter().sum();
        let mut u = self.rng.next_f32() * sum;
        for (id, p) in kept_ids.iter().zip(kept_probs.iter()) {
            u -= *p;
            if u <= 0.0 {
                return *id;
            }
        }
        // Numerical-noise fallback.
        *kept_ids.last().unwrap_or(&topk_ids[0])
    }

    /// Sample one token. `history` is the per-turn generated-token
    /// slice (empty `&[]` is fine for the first token, or when the
    /// request disables penalties). Penalties and filters are applied
    /// to a clone of `logits` only when needed; otherwise the caller's
    /// buffer is read directly.
    pub fn sample(
        &mut self,
        logits: &[f32],
        mode: &Sampling,
        history: &[u32],
    ) -> u32 {
        // **Sampler-E (#207)** — when no penalty is active we don't
        // need a writeable copy of `logits`, so skip the 600 KB
        // `extend_from_slice` and read the caller's buffer directly.
        // Saves ~150 µs/token on Qwen3.6's V=151424 vocab at default
        // penalties (the chat-temp+top_p case the user hit).
        let needs_penalties = mode.has_penalties() && !history.is_empty();
        let logits_view: &[f32] = if needs_penalties {
            self.logit_scratch.clear();
            self.logit_scratch.extend_from_slice(logits);
            apply_penalties_with_scratch(
                &mut self.logit_scratch,
                history,
                mode,
                &mut self.history_sorted,
                &mut self.history_counts,
            );
            &self.logit_scratch
        } else {
            logits
        };
        if mode.is_greedy() {
            return argmax(logits_view);
        }
        sample_stochastic(
            logits_view,
            mode,
            &mut self.rng,
            &mut self.pair_scratch,
        )
    }
}

/// build the normalized (id, prob) distribution that
/// [`Sampler::sample`] would draw from, given `logits`, `mode`, and
/// `history`. Used by the rejection-sampling spec-decode driver to
/// compare MTP draft `q(·)` against base verify `p(·)`.
/// Mirrors `sample_stochastic` exactly except the multinomial pick
/// at the end — temperature, top-k, top-p, min-p, repetition /
/// presence / frequency penalties (the last three driven by `history`)
/// are all applied.
/// For a `mode` whose [`is_greedy`](Sampling::is_greedy) returns
/// `true`, the returned distribution is one-hot at the argmax of the
/// **penalty-adjusted** logits (history is still applied so penalty
/// effects on the argmax are respected).
pub fn build_distribution(logits: &[f32], mode: &Sampling, history: &[u32]) -> Vec<(u32, f32)> {
    // /h penalty-aware path. If no penalties active, skip the
    // O(V)-byte clone and operate on the input slice directly.
    let needs_penalties = mode.has_penalties() && !history.is_empty();
    let logits_owned: Vec<f32>;
    let logits_view: &[f32] = if needs_penalties {
        let mut scratch: Vec<f32> = logits.to_vec();
        apply_penalties(&mut scratch, history, mode);
        logits_owned = scratch;
        &logits_owned
    } else {
        logits
    };
    if mode.is_greedy() {
        let id = argmax(logits_view);
        return vec![(id, 1.0)];
    }
    let inv_t = if mode.temperature <= 0.0 {
        1.0
    } else {
        1.0 / mode.temperature
    };
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits_view {
        let scaled = v * inv_t;
        if scaled > max_l {
            max_l = scaled;
        }
    }
    let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(logits_view.len());
    let mut sum = 0.0f32;
    for (i, &v) in logits_view.iter().enumerate() {
        let p = (v * inv_t - max_l).exp();
        pairs.push((i as u32, p));
        sum += p;
    }
    if sum > 0.0 {
        for (_, p) in pairs.iter_mut() {
            *p /= sum;
        }
    }

    let any_filter = mode.top_k.is_some() || mode.top_p.is_some() || mode.min_p.is_some();
    if !any_filter {
        return pairs;
    }
    // /h — partial-sort optimisation: when `top_k` is active, use
    // `select_nth_unstable_by` to partition the top-k to the front in
    // O(V) instead of an O(V log V) full sort, then sort only those k
    // entries. On Qwen3.6 vocab=151 936 with top_k=40, this drops
    // build_distribution from ~10 ms to ~1 ms per call.
    if let Some(k) = mode.top_k {
        let k = (k as usize).min(pairs.len());
        if k > 0 && k < pairs.len() {
            pairs.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            pairs.truncate(k);
        }
    }
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut end = pairs.len();
    if let Some(k) = mode.top_k {
        end = end.min(k as usize);
    }
    if let Some(p) = mode.top_p {
        if p < 1.0 && p > 0.0 {
            let mut cum = 0.0f32;
            let mut prefix_end = 0usize;
            for (i, &(_, pv)) in pairs.iter().take(end).enumerate() {
                cum += pv;
                prefix_end = i + 1;
                if cum >= p {
                    break;
                }
            }
            end = prefix_end;
        }
    }
    if let Some(min_p) = mode.min_p {
        if min_p > 0.0 {
            let threshold = pairs[0].1 * min_p;
            let mut drop_from = end;
            for (i, &(_, pv)) in pairs.iter().take(end).enumerate() {
                if pv < threshold {
                    drop_from = i;
                    break;
                }
            }
            end = drop_from.max(1);
        }
    }
    pairs.truncate(end);
    let kept_sum: f32 = pairs.iter().map(|(_, p)| *p).sum();
    if kept_sum > 0.0 {
        for (_, p) in pairs.iter_mut() {
            *p /= kept_sum;
        }
    }
    pairs
}

/// Multinomial pick from a pre-normalized `[(id, prob)]` distribution.
/// `dist`'s probs must sum to ≈1; the function tolerates small numeric
/// drift but not arbitrary rescaling.
pub fn sample_from_distribution(dist: &[(u32, f32)], rng: &mut Rng) -> u32 {
    if dist.is_empty() {
        return 0;
    }
    if dist.len() == 1 {
        return dist[0].0;
    }
    let sum: f32 = dist.iter().map(|(_, p)| *p).sum();
    let mut u = rng.next_f32() * sum;
    for &(id, p) in dist {
        u -= p;
        if u <= 0.0 {
            return id;
        }
    }
    dist.last().map_or(0, |&(id, _)| id)
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

/// Apply repetition / presence / frequency penalties in place on
/// `logits`. All three are no-ops at their default values, so this
/// returns early when no penalty is active — no history walk at all
/// on a default-config greedy call.
/// Allocates a fresh HashMap per call. Prefer
/// [`apply_penalties_with_scratch`] from the per-session [`Sampler`]
/// which reuses sort+dedup buffers.
fn apply_penalties(logits: &mut [f32], history: &[u32], mode: &Sampling) {
    if !mode.has_penalties() || history.is_empty() {
        return;
    }
    // Single-pass build: counts doubles as the "seen" set (any key with
    // count >= 1 was seen). Saves the second hash pass at ctx ~2K+ where
    // the prior code did seen-build + counts-build separately.
    let mut counts: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::with_capacity(history.len().min(8192));
    for &tok in history {
        *counts.entry(tok).or_insert(0) += 1;
    }
    apply_penalty_kernel(logits, mode, counts.iter().map(|(&t, &c)| (t, c)));
}

/// **Sampler-F / Sampler-D4** — sort+dedup `history` into `(tok, count)`
/// pairs in `counts_out`. `sorted_scratch` is a reused per-session
/// buffer (typically `Sampler::history_sorted`). Public so the GPU
/// sampler path (`crates/server/src/gpu_sampler.rs`) can build the
/// same dedup view before uploading to device.
pub fn build_history_counts(
    history: &[u32],
    sorted_scratch: &mut Vec<u32>,
    counts_out: &mut Vec<(u32, u32)>,
) {
    counts_out.clear();
    if history.is_empty() {
        return;
    }
    sorted_scratch.clear();
    sorted_scratch.extend_from_slice(history);
    sorted_scratch.sort_unstable();
    let mut prev = sorted_scratch[0];
    let mut run = 1u32;
    for &tok in &sorted_scratch[1..] {
        if tok == prev {
            run += 1;
        } else {
            counts_out.push((prev, run));
            prev = tok;
            run = 1;
        }
    }
    counts_out.push((prev, run));
}

/// **Sampler-F (#208)** — penalty path with caller-owned scratch
/// buffers. Replaces the per-call `HashMap<u32, u32>` build with a
/// sort+dedup over a reused `Vec<u32>`. Faster than hashing for
/// `history.len() ≤ ~2k` (typical chat-decode history at the time
/// the penalty path runs) because the sort+dedup is O(N log N) but
/// with much better cache behaviour and no allocator traffic.
fn apply_penalties_with_scratch(
    logits: &mut [f32],
    history: &[u32],
    mode: &Sampling,
    sorted: &mut Vec<u32>,
    counts: &mut Vec<(u32, u32)>,
) {
    if !mode.has_penalties() || history.is_empty() {
        return;
    }
    build_history_counts(history, sorted, counts);
    apply_penalty_kernel(logits, mode, counts.iter().copied());
}

/// Inner penalty-application kernel — shared between the one-shot
/// (`apply_penalties`) and per-session (`apply_penalties_with_scratch`)
/// entry points so the actual logit math lives in one place.
#[inline]
fn apply_penalty_kernel<I>(logits: &mut [f32], mode: &Sampling, counts: I)
where
    I: IntoIterator<Item = (u32, u32)>,
{
    let needs_count = mode.frequency_penalty != 0.0;
    let logits_len = logits.len();
    for (tok, c) in counts {
        let idx = tok as usize;
        if idx >= logits_len {
            continue;
        }
        let l = &mut logits[idx];
        // Repetition penalty (llama.cpp / HF convention). `penalty` is
        // divisive when logit is positive, multiplicative otherwise —
        // this preserves sign and is what Qwen's config expects.
        if mode.repetition_penalty != 1.0 && mode.repetition_penalty > 0.0 {
            if *l > 0.0 {
                *l /= mode.repetition_penalty;
            } else {
                *l *= mode.repetition_penalty;
            }
        }
        // Presence penalty (OpenAI: subtract penalty from logit).
        if mode.presence_penalty != 0.0 {
            *l -= mode.presence_penalty;
        }
        // Frequency penalty (OpenAI: subtract penalty * count).
        if needs_count {
            *l -= mode.frequency_penalty * c as f32;
        }
    }
}

/// Stochastic path: temperature + optional top-k / top-p / min-p.
/// `pair_scratch` is caller-owned reusable storage.
fn sample_stochastic(
    logits: &[f32],
    mode: &Sampling,
    rng: &mut Rng,
    pair_scratch: &mut Vec<(u32, f32)>,
) -> u32 {
    let inv_t = if mode.temperature <= 0.0 {
        1.0
    } else {
        1.0 / mode.temperature
    };
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = v * inv_t;
        if scaled > max_l {
            max_l = scaled;
        }
    }
    // Build (id, prob) pairs via numerically-stable softmax.
    pair_scratch.clear();
    pair_scratch.reserve(logits.len());
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let p = (v * inv_t - max_l).exp();
        pair_scratch.push((i as u32, p));
        sum += p;
    }
    // Normalise.
    if sum > 0.0 {
        for (_, p) in pair_scratch.iter_mut() {
            *p /= sum;
        }
    }

    // Sort by descending prob — needed for top-k / top-p. We always
    // sort when any filter is active; for plain-temperature there's no
    // sort (we go straight to multinomial over the full distribution).
    // **Sampler-A** (#206) — when `top_p` (or `min_p`) is set without an
    // explicit `top_k`, default to `top_k = TOP_K_AUTO_CAP` so the
    // partial-sort path activates instead of an O(V log V) full sort.
    // The cap is chosen well above the largest plausible top-p prefix
    // at any sane temperature × top_p (a flat-ish distribution at
    // temp=2.0, top_p=0.99 picks ~hundreds of candidates). If a
    // pathological prompt exceeds this cap, we log + truncate — it's
    // a faithful approximation of the top-p prefix anyway.
    const TOP_K_AUTO_CAP: u32 = 2048;
    let any_filter = mode.top_k.is_some() || mode.top_p.is_some() || mode.min_p.is_some();
    let effective_top_k: Option<u32> = match mode.top_k {
        Some(k) => Some(k),
        None if mode.top_p.is_some() || mode.min_p.is_some() => Some(TOP_K_AUTO_CAP),
        None => None,
    };
    let kept_end: usize = if any_filter {
        // Partial-sort to top-K when K < V; full sort otherwise. At
        // V=151424 vocab this turns the hot path from O(V log V) ≈
        // 2.6M ops to O(V) + O(K log K) ≈ 150k + 22k = ~7× faster.
        if let Some(k) = effective_top_k {
            let k = (k as usize).min(pair_scratch.len());
            if k > 0 && k < pair_scratch.len() {
                pair_scratch.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                pair_scratch.truncate(k);
            }
        }
        pair_scratch.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        // Apply top-k: truncate to first `k` entries.
        let mut end = pair_scratch.len();
        if let Some(k) = mode.top_k {
            end = end.min(k as usize);
        }
        // Apply top-p: keep the smallest prefix whose cumulative prob ≥ p.
        if let Some(p) = mode.top_p {
            if p < 1.0 && p > 0.0 {
                let mut cum = 0.0f32;
                let mut prefix_end = 0usize;
                let mut hit_p = false;
                for (i, &(_, pv)) in pair_scratch.iter().take(end).enumerate() {
                    cum += pv;
                    prefix_end = i + 1;
                    if cum >= p {
                        hit_p = true;
                        break;
                    }
                }
                end = prefix_end;
                // Sampler-A diagnostic: if the user didn't pass top_k
                // but the auto-cap truncated the top_p prefix, the
                // distribution was flatter than expected. The result is
                // a faithful but not-quite-exact top_p (the missing tail
                // would carry probability < (1 - cum) · 1/V_tail). Warn
                // once per call so misconfigured prompts surface.
                if !hit_p && mode.top_k.is_none() {
                    tracing::warn!(
                        target: "flambeau_runtime::sampling",
                        kept = end,
                        cap = TOP_K_AUTO_CAP,
                        top_p = p,
                        cum_p = cum,
                        "top_p prefix exceeded TOP_K_AUTO_CAP; distribution flatter than expected — \
                         increase top_k explicitly if exact top_p mass matters here"
                    );
                }
            }
        }
        // Apply min-p: drop entries whose prob < min_p * max_prob. The
        // sort guarantees the first entry is max_prob.
        if let Some(min_p) = mode.min_p {
            if min_p > 0.0 {
                let threshold = pair_scratch[0].1 * min_p;
                let mut drop_from = end;
                for (i, &(_, pv)) in pair_scratch.iter().take(end).enumerate() {
                    if pv < threshold {
                        drop_from = i;
                        break;
                    }
                }
                end = drop_from.max(1); // always keep at least the argmax.
            }
        }
        end
    } else {
        pair_scratch.len()
    };

    // Multinomial pick over [0..kept_end] using the retained distribution.
    let kept = &pair_scratch[..kept_end];
    let sum_kept: f32 = kept.iter().map(|(_, p)| *p).sum();
    let mut u = rng.next_f32() * sum_kept;
    for (id, pv) in kept {
        u -= pv;
        if u <= 0.0 {
            return *id;
        }
    }
    kept.last().map_or(0, |(id, _)| *id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1, 0.5, 0.9, 0.2, -1.0];
        let mut rng = Rng::from_seed(42);
        assert_eq!(sample(&logits, &Sampling::greedy(), &[], &mut rng), 2);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = [0.1, 0.5, 0.9, 0.2, -1.0];
        let mut rng = Rng::from_seed(42);
        let s = Sampling::temperature(0.0);
        assert_eq!(sample(&logits, &s, &[], &mut rng), 2);
    }

    #[test]
    fn temperature_is_deterministic_with_seed() {
        let logits = [1.0, 2.0, 3.0, 2.5, 1.5];
        let mut rng_a = Rng::from_seed(1234);
        let mut rng_b = Rng::from_seed(1234);
        let s = Sampling::temperature(0.7);
        let a = sample(&logits, &s, &[], &mut rng_a);
        let b = sample(&logits, &s, &[], &mut rng_b);
        assert_eq!(a, b, "same seed must give same token");
    }

    #[test]
    fn top_p_restricts_to_high_probability_tokens() {
        let logits = [0.0, 0.0, 10.0, 0.0, 0.0];
        let mut rng = Rng::from_seed(0xabc);
        let s = Sampling::top_p(1.0, 0.5);
        assert_eq!(sample(&logits, &s, &[], &mut rng), 2);
    }

    #[test]
    fn top_p_distribution_stays_within_nucleus() {
        let logits = [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = Rng::from_seed(7);
        let s = Sampling::top_p(1.0, 0.95);
        let mut counts = [0u32; 7];
        for _ in 0..1000 {
            let id = sample(&logits, &s, &[], &mut rng);
            counts[id as usize] += 1;
        }
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

    #[test]
    fn sampler_matches_free_fn_on_equal_seed() {
        let logits = [1.0, 2.0, 3.0, 2.5, 1.5];
        let modes = [
            Sampling::greedy(),
            Sampling::temperature(0.7),
            Sampling::top_p(1.0, 0.9),
        ];
        for mode in modes {
            let mut rng = Rng::from_seed(4242);
            let free = sample(&logits, &mode, &[], &mut rng);
            let mut sampler = Sampler::from_seed(4242);
            let owned = sampler.sample(&logits, &mode, &[]);
            assert_eq!(free, owned, "mismatch on mode {mode:?}");
        }
    }

    #[test]
    fn sampler_reuse_across_calls_is_deterministic() {
        let logits = [0.5, 0.3, 1.2, 0.9, 0.1, 2.0, 0.0];
        let mode = Sampling::temperature(0.9);
        let mut a = Sampler::from_seed(777);
        let mut b = Sampler::from_seed(777);
        for _ in 0..500 {
            assert_eq!(a.sample(&logits, &mode, &[]), b.sample(&logits, &mode, &[]));
        }
    }

    // ---- T4.b.1: new sampler knobs ----

    #[test]
    fn top_k_restricts_to_k_tokens() {
        // Five tokens, only the top 2 should be reachable with top_k=2.
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mode = Sampling {
            temperature: 1.0,
            top_k: Some(2),
            ..Sampling::greedy()
        };
        let mut rng = Rng::from_seed(0xdead);
        let mut seen = std::collections::HashSet::<u32>::new();
        for _ in 0..1000 {
            seen.insert(sample(&logits, &mode, &[], &mut rng));
        }
        // Only tokens 3, 4 (the top two) should ever be selected.
        assert!(seen.is_subset(&[3, 4].into_iter().collect()));
    }

    #[test]
    fn min_p_drops_low_prob_tokens() {
        let logits = [0.0, 5.0, 0.0, 5.0, 0.0];
        // With min_p = 0.5, tokens with prob < 0.5 * max_prob are excluded.
        // Tokens 1 and 3 are the peaks; 0, 2, 4 are near-zero prob.
        let mode = Sampling {
            temperature: 1.0,
            min_p: Some(0.5),
            ..Sampling::greedy()
        };
        let mut rng = Rng::from_seed(5);
        let mut seen = std::collections::HashSet::<u32>::new();
        for _ in 0..1000 {
            seen.insert(sample(&logits, &mode, &[], &mut rng));
        }
        assert!(seen.is_subset(&[1, 3].into_iter().collect()), "{seen:?}");
    }

    #[test]
    fn repetition_penalty_suppresses_repeats() {
        // Token 2 is the clear argmax. With repetition_penalty > 1 and
        // history = [2], its logit should be divided by the penalty,
        // letting token 1 (second-best) overtake it in greedy mode.
        let logits = [1.0, 2.5, 3.0];
        let mut no_pen = Sampling::greedy();
        no_pen.repetition_penalty = 1.0;
        let with_pen = Sampling {
            repetition_penalty: 2.0,
            ..Sampling::greedy()
        };
        let mut rng = Rng::from_seed(1);
        assert_eq!(sample(&logits, &no_pen, &[], &mut rng), 2);
        assert_eq!(
            sample(&logits, &with_pen, &[2], &mut rng),
            1,
            "token 2 (already seen) should lose argmax to token 1 under penalty"
        );
    }

    #[test]
    fn presence_penalty_subtracts_from_seen_token_logit() {
        // Subtract 2.0 from the logit of any token in history.
        let logits = [1.0, 2.5, 3.0];
        let mode = Sampling {
            presence_penalty: 2.0,
            ..Sampling::greedy()
        };
        let mut rng = Rng::from_seed(1);
        // Without history, greedy picks 2 (argmax).
        assert_eq!(sample(&logits, &mode, &[], &mut rng), 2);
        // With history [2], logit 3.0 → 1.0; argmax becomes token 1 (2.5).
        assert_eq!(sample(&logits, &mode, &[2], &mut rng), 1);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let logits = [1.0, 5.0, 3.0];
        // Penalty 1.0 per occurrence. Token 1 seen twice → logit 5-2=3.
        // Token 2 seen once → logit 3-1=2. Argmax then = token 1 (tied
        // between 0 and 1, token 1 is first in iteration order when
        // tied; actually token 0 = 1.0 < 3.0 = token 1).
        let mode = Sampling {
            frequency_penalty: 1.0,
            ..Sampling::greedy()
        };
        let mut rng = Rng::from_seed(1);
        assert_eq!(sample(&logits, &mode, &[1, 1, 2], &mut rng), 1);
    }

    #[test]
    fn no_penalty_no_allocation_path() {
        // History is non-empty but all penalties are default — the
        // apply_penalties fast-path should skip the history walk. We
        // can't observe allocation directly, but we CAN assert the
        // output equals the no-history output.
        let logits = [0.1, 0.5, 0.9];
        let mode = Sampling::temperature(0.5);
        let mut rng_a = Rng::from_seed(0xaaaa);
        let mut rng_b = Rng::from_seed(0xaaaa);
        let a = sample(&logits, &mode, &[0, 1, 2, 2, 1], &mut rng_a);
        let b = sample(&logits, &mode, &[], &mut rng_b);
        assert_eq!(a, b, "default penalties must be history-independent");
    }
}
