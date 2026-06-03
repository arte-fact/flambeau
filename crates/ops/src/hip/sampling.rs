//! GPU-side sampler primitives — Solution D in the sampler-perf plan.
//! `sampler_topk_softmax_f32` runs on the head rank after the LM head
//! produces F32 logits and replaces the host-side
//! `Sampler::sample_stochastic` softmax+sort hot path.
//! Output is the top-K (id, prob) pairs, sorted descending by prob,
//! with probs renormalised over the kept K. Caller does host-side
//! `top_p` filter + multinomial draw on the small K-tuple (DtoH cost
//! is `K * 8` bytes versus `V * 4` bytes for the full logit slab —
//! ~37× less PCIe traffic at V=151424, K=256).

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "op wrapper — every unsafe block is a `kernel.launch` over caller-validated \
              `DevicePtr`s; the kernel stem + entry are resolved through the validated \
              registry and the ABI matches the kernel's extern-C signature."
)]

use anyhow::{bail, Result};
use flambeau_backend_hip::{KernelArgs, LaunchCfg};


/// Caller-visible upper bound on `K`. Must match `SAMPLER_K_OUT_MAX` in
/// `kernels-hip/src/kernels/sampler_topk_softmax_f32.cu`. Bumped from
/// 256 to 2048 to match Sampler-A's host-side
/// `effective_top_k = mode.top_k.unwrap_or(2048)`; smaller K caused
/// the GPU sampler to bias multinomial toward EOS at natural-endpoint
/// positions (one-sentence-then-stop chat-truncation bug).
pub const SAMPLER_K_OUT_MAX: usize = 2048;

/// **Sampler-D4 (#212)** — apply repetition / presence / frequency
/// penalties in place on `[V]` F32 logits. `token_counts` is a flat
/// device buffer of `n_pairs * 2` u32s laid out as
/// `[tok0, count0, tok1, count1, ...]`. Caller must dedup the
/// history before passing — the kernel assumes unique `tok` per
/// pair (no atomic-add).
/// # Errors
/// - `n_pairs == 0` (penalty path should short-circuit on host).
/// - kernel-launch dispatch failure.
pub fn apply_penalties_f32(
    ctx: crate::OpCtx<'_>,
    bufs: crate::PenaltyBuffers,
    n_pairs: usize,
    vocab: usize,
    knobs: crate::PenaltyKnobs,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::PenaltyBuffers { logits, token_counts } = bufs;
    let crate::PenaltyKnobs {
        repetition: repetition_penalty,
        presence: presence_penalty,
        frequency: frequency_penalty,
    } = knobs;
    if n_pairs == 0 {
        bail!("sampler::apply_penalties_f32: n_pairs must be >= 1 — caller should short-circuit");
    }
    let module = reg.expect_module("sampler_apply_penalties_f32")?;
    let kernel = module.kernel("flambeau_sampler_apply_penalties_f32")?;

    let n_pairs_i = n_pairs as i32;
    let vocab_i = vocab as i32;
    let l_ptr: u64 = logits.as_usize() as u64;
    let c_ptr: u64 = token_counts.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&l_ptr);
    args.push(&c_ptr);
    args.push(&n_pairs_i);
    args.push(&vocab_i);
    args.push(&repetition_penalty);
    args.push(&presence_penalty);
    args.push(&frequency_penalty);
    let block: u32 = 256;
    let grid: u32 = (n_pairs as u32).div_ceil(block);
    let cfg = LaunchCfg::one_d(grid, block);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}

/// Block-level top-K + softmax-normalize over a length-`vocab` F32 logit
/// vector. Single-block kernel: launches one 256-thread block on
/// `stream` regardless of `vocab` (the kernel grid-strides over the
/// logit reads internally).
/// Shapes:
/// - `logits` `[vocab]` F32 — the LM-head output for one token, in
///   place where the forward kernel left it.
/// - `out_ids` `[k]` i32 — sorted-descending-by-prob token ids.
/// - `out_probs` `[k]` F32 — softmax-normalised probabilities,
///   renormalised so the kept K sum to 1.
///   Args:
/// - `inv_temp` — `1.0 / temperature` (caller passes `1.0` when
///   `temperature <= 0`; the kernel applies the multiply uniformly).
/// - `k` ≤ [`SAMPLER_K_OUT_MAX`].
/// # Errors
/// - `k <= 0` or `k > SAMPLER_K_OUT_MAX`.
/// - kernel-launch dispatch failure.
pub fn topk_softmax_f32(
    ctx: crate::OpCtx<'_>,
    bufs: crate::SamplerTopkSoftmaxBuffers,
    knobs: crate::SamplerTopkSoftmaxKnobs,
) -> Result<()> {
    let crate::OpCtx { reg, stream } = ctx;
    let crate::SamplerTopkSoftmaxBuffers { logits, out_ids, out_probs } = bufs;
    let crate::SamplerTopkSoftmaxKnobs { vocab, k, inv_temp } = knobs;
    if k == 0 {
        bail!("sampler::topk_softmax_f32: k must be >= 1");
    }
    if k > SAMPLER_K_OUT_MAX {
        bail!(
            "sampler::topk_softmax_f32: k {k} > SAMPLER_K_OUT_MAX {SAMPLER_K_OUT_MAX} \
             (bump SAMPLER_K_OUT_MAX in the .cu and the wrapper if a higher K is \
             genuinely needed; the single-block layout caps at 4096 candidates total)"
        );
    }
    let module = reg.expect_module("sampler_topk_softmax_f32")?;
    let kernel = module.kernel("flambeau_sampler_topk_softmax_f32")?;

    let vocab_i = vocab as i32;
    let k_i = k as i32;
    let l_ptr: u64 = logits.as_usize() as u64;
    let i_ptr: u64 = out_ids.as_usize() as u64;
    let p_ptr: u64 = out_probs.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&l_ptr);
    args.push(&i_ptr);
    args.push(&p_ptr);
    args.push(&vocab_i);
    args.push(&k_i);
    args.push(&inv_temp);
    let cfg = LaunchCfg::one_d(1, 256);
    unsafe { kernel.launch(stream, cfg, args)? };
    Ok(())
}
