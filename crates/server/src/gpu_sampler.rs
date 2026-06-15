//! Sampler helpers used by the decode loop.
//!
//! After the legacy GPU-side topk-with-penalties path was dropped
//! (it was qwen3-moe-typed end-to-end), only the JSON grammar mask
//! helper remains. The decode loop runs the host sampler.

#![cfg(feature = "hip")]

/// Stub kept so `decode_loop.rs` can carry an `Option<GpuSamplerScratch>`
/// without changing its shape. The GPU sampler is disabled on v2 until
/// it's reimplemented over `Session<A>`.
pub struct GpuSamplerScratch;

/// Mask non-JSON-grammar-feasible vocab tokens in-place by setting
/// their logit to `-inf`. Pure host-side helper — no GPU involvement
/// despite the module name.
///
/// `max_candidates` bounds the per-call cost. Tokens outside the top-K
/// by logit value are left untouched (below the threshold any sane
/// sampler picks from).
pub fn apply_json_mask_to_logits(
    state: &flambeau_runtime::json_schema::JsonConstraint,
    tokenizer: &flambeau_quant::GgufTokenizer,
    logits: &mut [f32],
    max_candidates: usize,
) {
    if logits.is_empty() {
        return;
    }
    let k = max_candidates.min(logits.len());
    if k == 0 {
        return;
    }

    let mut pairs: Vec<(u32, f32)> = (0..logits.len() as u32)
        .map(|i| (i, logits[i as usize]))
        .collect();
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(k);
    }

    let complete = state.is_complete();
    let already_started = state.has_started();
    for &(id, _) in pairs.iter() {
        let bytes = match tokenizer.decode(&[id]) {
            Ok(s) => s,
            Err(_) => {
                logits[id as usize] = f32::NEG_INFINITY;
                continue;
            }
        };
        if bytes.is_empty() {
            // Specials / EOS — only allowed once the JSON is
            // structurally closed; otherwise the model would emit EOS
            // mid-value and the response would be invalid.
            if !complete {
                logits[id as usize] = f32::NEG_INFINITY;
            }
            continue;
        }
        let mut probe = state.clone();
        if !probe.feed_slice(bytes.as_bytes()) {
            logits[id as usize] = f32::NEG_INFINITY;
            continue;
        }
        // Forbid the "infinite leading whitespace" failure mode: a
        // candidate that neither starts the value nor closes it is rejected.
        if !already_started && !probe.has_started() {
            logits[id as usize] = f32::NEG_INFINITY;
            continue;
        }
        // OpenAI `response_format: json_object` requires the top-level value
        // to be an object — forbid non-`{` openers. Schemas gate their own
        // top-level type in the validator, so this extra rule is skipped.
        if state.top_must_be_object()
            && !already_started
            && probe.has_started()
            && bytes.as_bytes().iter().all(|&b| b != b'{')
        {
            logits[id as usize] = f32::NEG_INFINITY;
        }
    }
}
