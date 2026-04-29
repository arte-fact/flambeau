//! MTP-4 — passive acceptance-rate measurement on Qwen3.6-27B.
//!
//! At each decode step we:
//!   1. Run the base model: forward_one_token_pp(token_t, position_t)
//!      → token_{t+1} (sampled greedy by argmax).
//!   2. Capture h_t (the pre-output_norm hidden on the last rank).
//!   3. Run MTP probe: forward_mtp_step_with_lm_head(h_t, embed(token_{t+1}),
//!      position_{t+1}) → predicted_{t+2}.
//!   4. Save the prediction. The next loop iteration produces
//!      actual_{t+2}; compare predicted vs actual to count an
//!      acceptance.
//!
//! HARD GATE per CLAUDE.md / MTP-4 task: ≥75% acceptance proceeds to
//! active spec-decode (KV rollback + verify in MTP-4 session 2);
//! <75% files null and we pivot.
//!
//! Skipped when:
//!   - no Qwen3.6-27B-Q4_0.gguf
//!   - no Qwen3.6-27B-mtp.gguf
//!   - fewer than 4 HIP devices (model is 16 GB → needs pp4)
//!
//! `FLAMBEAU_MTP_ACCEPT_STEPS=N` overrides the default 8-step decode
//! window.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel \
              launch over host/device buffers that live for the bounded \
              synchronize that follows."
)]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_one_token_pp_logits, forward_prefill_pp,
    ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::mtp::{
    forward_mtp_step_with_lm_head, load_mtp_head, load_mtp_head_bf16,
    MtpForwardScratch, MtpKvCache,
};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

// Switchable via FLAMBEAU_MTP_BASE env. Default Q4_0 (smaller, faster
// load). Set FLAMBEAU_MTP_BASE=ud_q8_k_xl to test the precision lever.
fn base_path() -> std::path::PathBuf {
    let v = std::env::var("FLAMBEAU_MTP_BASE").unwrap_or_default();
    let p = match v.as_str() {
        "ud_q8_k_xl" => "/artefact/models/Qwen3.6-27B-UD-Q8_K_XL.gguf",
        "q8_0"       => "/artefact/models/Qwen3.6-27B-Q8_0.gguf",
        "q4_1"       => "/artefact/models/Qwen3.6-27B-Q4_1.gguf",
        _            => "/artefact/models/Qwen3.6-27B-Q4_0.gguf",
    };
    PathBuf::from(p)
}
const _UNUSED_BASE_PATH: &str = "/artefact/models/Qwen3.6-27B-Q4_0.gguf";
/// Default MTP head (F16 linears). Override via FLAMBEAU_MTP_HEAD=path.
fn mtp_path() -> std::path::PathBuf {
    std::env::var("FLAMBEAU_MTP_HEAD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf"))
}
const DEFAULT_STEPS: usize = 8;
// Default prose prompt (V1.7.4 canonical parity prompt):
// "The capital of France is" → [760, 6511, 314, 9338, 369].
// Other prompts selected via FLAMBEAU_MTP_PROMPT={prose|code|json|math}
// and tokenized at runtime via the GGUF-embedded BPE tokenizer.
const PROMPT_PROSE_IDS: &[u32] = &[760, 6511, 314, 9338, 369];

/// Source text for non-default prompts. Tokenized at runtime by the
/// base GGUF's tokenizer.
fn prompt_text() -> Option<&'static str> {
    let v = std::env::var("FLAMBEAU_MTP_PROMPT").ok()?;
    Some(match v.to_lowercase().as_str() {
        "code" => {
            // Dense Python class — patrickbdevaney's "dense code" content
            // type measured 0.94-0.96 acceptance on the same MTP arch.
            "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):\n        super().__init__()\n        self.dim = dim\n        self.n_heads = n_heads\n        self.head_dim = dim // n_heads\n        self.q_proj = nn.Linear(dim, dim, bias=False)\n        self.k_proj = nn.Linear(dim, dim, bias=False)\n        self.v_proj"
        }
        "json" => {
            // Structured output continuation.
            "{\n  \"name\": \"flambeau\",\n  \"version\": \"0.1.0\",\n  \"description\": \"Inference framework for HIP and CUDA\",\n  \"authors\": [\n    {\n      \"name\": \"arte-fact\",\n      \"email\""
        }
        "math" => {
            // Mathematical reasoning prose — patrickbdevaney measured
            // 0.82-0.88 (still > our prose).
            "To prove that the square root of 2 is irrational, we proceed by contradiction. Suppose sqrt(2) is rational. Then sqrt(2) = a/b where a and b are integers with no common factor and b is non-zero. Squaring both sides yields 2 = a^2/b^2, so a^2 = 2"
        }
        "prose" | _ => "The capital of France is",
    })
}

#[test]
fn mtp_acceptance_passive_qwen36_27b() -> Result<()> {
    let base_path = base_path();
    let mtp_path = mtp_path();
    if !base_path.exists() {
        eprintln!("skip: {} not present", base_path.display());
        return Ok(());
    }
    if !mtp_path.exists() {
        eprintln!("skip: {} not present (run convert_qwen36_mtp.py)", mtp_path.display());
        return Ok(());
    }
    let n_gpus = device_count().unwrap_or(0);
    if n_gpus < 4 {
        eprintln!("skip: need ≥4 HIP devices for pp4 (got {n_gpus})");
        return Ok(());
    }

    let n_steps: usize = std::env::var("FLAMBEAU_MTP_ACCEPT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_STEPS);

    eprintln!("=== MTP-4 passive acceptance ===");
    eprintln!("base: {}", base_path.display());
    eprintln!("mtp:  {}", mtp_path.display());

    // ── Load base model on pp{n_gpus}
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;

    // ── Resolve prompt IDs.
    //   FLAMBEAU_MTP_PROMPT={prose|code|json|math} → tokenize via GGUF
    //   tokenizer; default unset → V1.7.4 canonical 5-token prose IDs.
    let prompt_ids: Vec<u32> = if let Some(text) = prompt_text() {
        let tok = flambeau_quant::load_from_gguf(&base_file)?;
        let ids = tok.encode(text)?;
        eprintln!("prompt: \"{}…\" ({} tokens)",
            &text.chars().take(60).collect::<String>(), ids.len());
        ids
    } else {
        eprintln!("prompt: default 5-token prose (\"The capital of France is\")");
        PROMPT_PROSE_IDS.to_vec()
    };
    eprintln!("decode steps: {n_steps}, prompt prefill: {} tokens", prompt_ids.len());
    let cluster = HipCluster::new(&(0..n_gpus).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!("loading base across {} ranks ({} layers)…", cluster.ranks(), cfg.num_layers);
    let model = Qwen3MoEShardedModel::load(&base_file, &cluster, &assignment)?;
    eprintln!("base loaded ({:.2} GiB across shards)",
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0));

    let last_rank = (cluster.ranks() - 1) as usize;
    let last_device = cluster.device(last_rank);
    let last_ops = &model.shards[last_rank].ops;

    // ── Load MTP on the last rank (where output_norm + lm_head live)
    eprintln!("loading MTP on rank {last_rank}…");
    let mtp_file = GgufFile::open(&mtp_path)?;
    let use_bf16 = std::env::var("FLAMBEAU_MTP_BF16")
        .map(|v| !matches!(v.as_str(), "" | "0" | "off" | "false"))
        .unwrap_or(false);
    eprintln!("MTP forward dtype: {}", if use_bf16 { "BF16" } else { "F16/Q8_1" });
    let mtp = if use_bf16 {
        load_mtp_head_bf16(&mtp_file, last_device)?
    } else {
        load_mtp_head(&mtp_file, last_device)?
    };

    // The base loader put output_norm + output (lm_head) on the last
    // rank. Find them on the shard.
    let last_shard = &model.shards[last_rank];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .ok_or_else(|| anyhow!("last rank missing output_norm"))?;
    let lm_head = last_shard
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("last rank missing output (lm_head)"))?;
    // vLLM convention: base model applies self.norm before returning
    // hidden_states; MTP gets the post-norm value. flambeau's
    // forward_one_token_pp writes pre-norm to scratch.hidden_a, so we
    // re-apply output_norm in forward_mtp_step_with_lm_head.

    // ── Allocate session + scratch
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    let hidden = cfg.hidden_size;
    let h_t_saved = last_device.alloc(hidden * 2)?;        // F16 [hidden]
    let e_token_dev = last_device.alloc(hidden * 2)?;      // F16 [hidden]

    // MTP forward scratch — one alloc per session (MTP-4-CLEAN).
    last_device.bind()?;
    let mtp_scratch = MtpForwardScratch::new(last_device, &cfg)?;

    // Persistent MTP KV cache (accumulates across decode steps).
    const MTP_KV_MAX: usize = 256;
    let kv_row_bytes = cfg.num_kv_heads * cfg.head_dim * 2; // F16
    let mtp_kcache = last_device.alloc(MTP_KV_MAX * kv_row_bytes)?;
    let mtp_vcache = last_device.alloc(MTP_KV_MAX * kv_row_bytes)?;
    let mut mtp_cache_pos: usize = 0;
    let use_kv_accum = std::env::var("FLAMBEAU_MTP_KV_ACCUM")
        .as_deref()
        .map(|s| s != "0" && s != "off")
        .unwrap_or(true);
    // MTP-INV-2: prefill priming. Runs MTP forward over each prompt
    // position to populate MTP attention KV with the prompt prefix
    // (mirrors vLLM's spec_info.hidden_states flow). Requires KV accum.
    let use_prefill_prime = use_kv_accum
        && std::env::var("FLAMBEAU_MTP_PREFILL_PRIME")
            .as_deref()
            .map(|s| s != "0" && s != "off")
            .unwrap_or(false);
    if use_kv_accum {
        eprintln!("MTP KV accumulation: ON (persistent cache, {} slots)", MTP_KV_MAX);
    } else {
        eprintln!("MTP KV accumulation: OFF (transient 1-slot per call)");
    }
    eprintln!("MTP prefill priming: {}", if use_prefill_prime { "ON" } else { "OFF" });

    // Rank 0 helper buffer for embedding lookup.
    let rank0_device = cluster.device(0);
    let token_embd = model.shards[0]
        .token_embd
        .as_ref()
        .ok_or_else(|| anyhow!("rank 0 missing token_embd"))?;
    let rank0_embed_buf = rank0_device.alloc(hidden * 2)?;

    // Helper: fetch embedding for token id on rank 0, copy to e_token_dev on last rank.
    // Uses staged D2H + H2D since the cluster lacks direct peer DMA in test infra.
    let embed_into_e_token = |token_id: u32| -> Result<()> {
        rank0_device.bind()?;
        flambeau_qwen3_moe::forward::forward_embed_decode_host(
            rank0_device, rank0_device.default_stream(),
            token_embd, token_id, rank0_embed_buf, hidden,
        )?;
        let mut row_host = vec![half::f16::from_f32(0.0); hidden];
        unsafe {
            rank0_device.memcpy_async(
                rank0_device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(row_host.as_mut_ptr() as usize),
                rank0_embed_buf,
                hidden * 2,
            )?;
        }
        rank0_device.default_stream().synchronize()?;
        last_device.bind()?;
        unsafe {
            last_device.memcpy_async(
                last_device.default_stream(),
                CopyDirection::HostToDevice,
                e_token_dev,
                DevicePtr(row_host.as_ptr() as usize),
                hidden * 2,
            )?;
        }
        last_device.default_stream().synchronize()?;
        Ok(())
    };

    // ── Prefill the prompt.
    //
    // Two paths:
    //  (default)   forward_prefill_pp — efficient parallel prefill, no
    //              per-position hidden capture, MTP KV starts empty.
    //  (prime=on)  sequential base decodes — captures h_p at each prompt
    //              position p, runs MTP forward at position p+1 with
    //              (h_p, embed(prompt[p+1])) to seed MTP KV slot p.
    //              Mirrors vLLM's spec_info.hidden_states behaviour.
    let mut last_token: u32 = 0;
    if use_prefill_prime {
        let prompt_len = prompt_ids.len();
        for (i, &token_i) in prompt_ids.iter().enumerate() {
            let next_i = forward_one_token_pp(
                &model, &mut session, &cluster, &mut decode_scratch, token_i, i,
            )?;
            // hidden_a holds h_i — base hidden state at position i.
            let hidden_a = decode_scratch.per_rank[last_rank].hidden_a;

            // e_token semantics: matches our decode-loop convention —
            // embed of the base's just-produced next token (look-ahead
            // by 1). MTP-INV-2 measured both this and the vLLM-style
            // no-lookahead variant (`embed(prompt[i])`); lookahead = 56.2%
            // (baseline), no-lookahead = 50.0%. Lookahead is the default.
            let next_token = if i + 1 < prompt_len { prompt_ids[i + 1] } else { next_i };
            embed_into_e_token(next_token)?;

            // MTP forward at position i+1 with (h_i, embed(next_token)),
            // appending K/V to slot i. KV side-effect is the point —
            // we discard the prediction.
            last_device.bind()?;
            flambeau_ops::hip::norm::rmsnorm_f16(
                last_ops, last_device.default_stream(),
                hidden_a, output_norm.ptr, mtp_scratch.h_t_post_norm,
                1, hidden, cfg.rms_norm_eps,
            )?;
            let kv = MtpKvCache {
                kcache: mtp_kcache, vcache: mtp_vcache,
                cache_position: i, n_tokens_kv: i + 1,
            };
            flambeau_qwen3_moe::mtp::forward_mtp_step(
                last_ops, last_device.default_stream(), last_device, &cfg, &mtp,
                &mtp_scratch,
                mtp_scratch.h_t_post_norm, e_token_dev, i + 1,
                mtp_scratch.mtp_h_final, Some(kv),
            )?;
            mtp_cache_pos = i + 1;

            // Last iteration's sample is what `forward_prefill_pp` would
            // have returned. Don't re-call base — it would re-insert at
            // the same position and corrupt the KV.
            if i == prompt_len - 1 {
                last_token = next_i;
            }
        }
        eprintln!(
            "prefill primed via {} sequential decodes; MTP cache_pos = {}; first sampled token = {}",
            prompt_len, mtp_cache_pos, last_token,
        );
    } else {
        let mut prefill_scratch =
            ShardedForwardPrefillScratch::new(&model, &cluster, prompt_ids.len())?;
        last_token = forward_prefill_pp(
            &model, &mut session, &cluster, &mut prefill_scratch, &prompt_ids, 0,
        )?;
        prefill_scratch.dispose(&cluster).ok();
        eprintln!("prefill done; first sampled token = {last_token}");
    }

    // Tokens generated, including the prefill's last-emitted one.
    let mut tokens: Vec<u32> = prompt_ids.to_vec();
    tokens.push(last_token);

    // Diagnostic: dump embedding for token 11751 (ĠParis) and compare to
    // Python ref to verify Q4_0 dequant of token_embd is bit-exact.
    // Confirmed match (2026-04-28): flambeau's dequant produces identical
    // F16 values to gguf python's dequant. Embedding lookup ruled out as
    // the source of the 0% acceptance.
    // Reference row[:8]:
    //   [ 0.01465  0.02441  0.00488 -0.01465  0.00488 -0.00488  0.00488  0.0]
    {
        rank0_device.bind()?;
        flambeau_qwen3_moe::forward::forward_embed_decode_host(
            rank0_device, rank0_device.default_stream(),
            token_embd, 11751, rank0_embed_buf, hidden,
        )?;
        let mut buf = vec![half::f16::from_f32(0.0); hidden];
        unsafe {
            rank0_device.memcpy_async(
                rank0_device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(buf.as_mut_ptr() as usize),
                rank0_embed_buf,
                hidden * 2,
            )?;
        }
        rank0_device.default_stream().synchronize()?;
        let f32_first8: Vec<f32> = buf[..8].iter().map(|x| x.to_f32()).collect();
        eprintln!("[diag] embed(11751) flambeau first 8: {f32_first8:?}");
        eprintln!("[diag] embed(11751) python    first 8: [0.01465, 0.02441, 0.00488, -0.01465, 0.00488, -0.00488, 0.00488, 0.0]");
    }

    let mut predicted: Vec<u32> = Vec::with_capacity(n_steps);
    let mut have_h_t = false;
    let mut accepted = 0usize;
    let mut compared = 0usize;
    // MTP-INV-3: sampling-aware verify. For each scored step, sum
    // accept_prob = min(1, P_target(pred) / P_draft(pred)). The
    // expected acceptance under vLLM's rejection-sampling policy is
    // the average of these probabilities across compared steps.
    let mut accept_prob_sum: f64 = 0.0;
    let vocab = cfg.vocab_size;
    let mut base_logits: Vec<f32> = Vec::with_capacity(vocab);
    let mut mtp_logits_host: Vec<f32> = Vec::with_capacity(vocab);
    let mut prev_mtp_logits: Option<Vec<f32>> = None;

    fn argmax_u32(xs: &[f32]) -> u32 {
        let mut best_i = 0usize;
        let mut best_v = xs[0];
        for (i, &v) in xs.iter().enumerate().skip(1) {
            if v > best_v { best_v = v; best_i = i; }
        }
        best_i as u32
    }
    /// Numerically-stable softmax probability at index `idx` over `logits`.
    fn softmax_at(logits: &[f32], idx: u32) -> f32 {
        let mut m = f32::NEG_INFINITY;
        for &v in logits { if v > m { m = v; } }
        let mut sum = 0.0_f64;
        for &v in logits { sum += ((v - m) as f64).exp(); }
        let target = (logits[idx as usize] - m) as f64;
        (target.exp() / sum) as f32
    }

    let t0 = std::time::Instant::now();
    for step in 0..=n_steps {
        let position = prompt_ids.len() + step;

        // ── Run base step. Download F32 logits (vocab*4 ≈ 1 MB) so we
        //    can compute P_target(pred) for sampling-aware verify; argmax
        //    host-side to recover the sampled token.
        forward_one_token_pp_logits(
            &model, &mut session, &cluster, &mut decode_scratch, last_token, position,
            &mut base_logits,
        )?;
        let next = argmax_u32(&base_logits);
        // After this call, `decode_scratch.per_rank[last_rank].hidden_a`
        // holds the pre-output_norm hidden at position `position`.
        let hidden_a_after = decode_scratch.per_rank[last_rank].hidden_a;

        // ── If we have a prediction from the previous step, score it.
        //    Two metrics:
        //      strict_greedy:    pred == next (legacy)
        //      sampling-aware:   accept_prob = min(1, P_t(pred) / P_d(pred))
        if let Some(&pred) = predicted.last() {
            compared += 1;
            let strict_match = pred == next;
            if strict_match { accepted += 1; }

            // Sampling-aware: needs P_target from *this* step's base
            // logits and P_draft from the *previous* step's MTP logits
            // (saved in `prev_mtp_logits`).
            let accept_prob = if let Some(ref mtp_logits) = prev_mtp_logits {
                let p_target = softmax_at(&base_logits, pred);
                let p_draft = softmax_at(mtp_logits, pred);
                if p_draft > 0.0 {
                    (p_target / p_draft).min(1.0) as f64
                } else {
                    1.0
                }
            } else {
                f64::NAN
            };
            if !accept_prob.is_nan() {
                accept_prob_sum += accept_prob;
            }
            eprintln!(
                "  step {step}: predicted={pred} actual={next} {}  P_t(pred)={:.4}  accept_prob={:.4}",
                if strict_match { "✓" } else { "✗" },
                softmax_at(&base_logits, pred),
                accept_prob,
            );
        }

        // ── Compute MTP draft for token at position `position+1` (i.e.
        //    the successor of `next`). Inputs: h_t = previously-saved
        //    pre-norm hidden (from the prior base step that PRODUCED
        //    `last_token`). At step 0 we have to use the just-computed
        //    hidden_a (h_position) and skip the comparison until step 1.
        if step < n_steps {
            // Embed `next` onto the last rank.
            // 1) Look up the row on rank 0 → rank0_embed_buf (F16).
            rank0_device.bind()?;
            flambeau_qwen3_moe::forward::forward_embed_decode_host(
                rank0_device,
                rank0_device.default_stream(),
                token_embd,
                next,
                rank0_embed_buf,
                hidden,
            )?;
            // 2) D2H rank 0 → host → D2H rank last (cluster lacks
            //    direct peer-DMA in test infra; staged via host is
            //    fine for the smoke).
            let mut row_host = vec![half::f16::from_f32(0.0); hidden];
            unsafe {
                rank0_device.memcpy_async(
                    rank0_device.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(row_host.as_mut_ptr() as usize),
                    rank0_embed_buf,
                    hidden * 2,
                )?;
            }
            rank0_device.default_stream().synchronize()?;
            last_device.bind()?;
            unsafe {
                last_device.memcpy_async(
                    last_device.default_stream(),
                    CopyDirection::HostToDevice,
                    e_token_dev,
                    DevicePtr(row_host.as_ptr() as usize),
                    hidden * 2,
                )?;
            }
            last_device.default_stream().synchronize()?;

            // For the FIRST step, snapshot h_t from THIS step's
            // hidden_a and continue without scoring (we don't have a
            // prediction yet). For subsequent steps, use the
            // PREVIOUSLY saved h_t (which is from the step that
            // produced `last_token` — that's what we need).
            let h_t_for_mtp = if have_h_t {
                h_t_saved
            } else {
                // Step 0: no previous h. Use this step's hidden_a as
                // a stand-in to keep the chain rolling; we won't
                // score this prediction (it's "predict the token AFTER
                // next, given (h_at_position, embed(next))" which is
                // valid; we just align comparisons one step later).
                hidden_a_after
            };

            let kv_arg = if use_kv_accum {
                let kv = MtpKvCache {
                    kcache: mtp_kcache,
                    vcache: mtp_vcache,
                    cache_position: mtp_cache_pos,
                    n_tokens_kv: mtp_cache_pos + 1,
                };
                Some(kv)
            } else {
                None
            };
            let pred_next_next = forward_mtp_step_with_lm_head(
                last_ops,
                last_device.default_stream(),
                last_device,
                &cfg,
                &mtp,
                &mtp_scratch,
                output_norm,
                lm_head,
                h_t_for_mtp,
                e_token_dev,
                position + 1,
                kv_arg,
            )?;
            if use_kv_accum {
                mtp_cache_pos += 1;
            }
            predicted.push(pred_next_next);

            // MTP-INV-3: capture this step's MTP logits for the next
            // iteration's P_draft calculation. They live in
            // `mtp_scratch.logits_f32` (vocab × F32, ≈ 1 MB).
            mtp_logits_host.resize(vocab, 0.0);
            unsafe {
                last_device.memcpy_async(
                    last_device.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(mtp_logits_host.as_mut_ptr() as usize),
                    mtp_scratch.logits_f32,
                    vocab * 4,
                )?;
            }
            last_device.default_stream().synchronize()?;
            prev_mtp_logits = Some(mtp_logits_host.clone());

            // ── Snapshot h_t = current hidden_a for next iteration.
            unsafe {
                last_device.memcpy_async(
                    last_device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    h_t_saved,
                    hidden_a_after,
                    hidden * 2,
                )?;
            }
            last_device.default_stream().synchronize()?;
            have_h_t = true;
        }

        tokens.push(next);
        last_token = next;
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let acceptance = if compared == 0 {
        0.0
    } else {
        accepted as f64 / compared as f64
    };
    // Sampling-aware (vLLM rejection-sampling) expected acceptance.
    // First compared step has no prev_mtp_logits (NaN'd above), so the
    // sum is over compared - 1 steps once we get going. To keep things
    // simple, divide by `compared` for a slight under-estimate at the
    // first step, or by `compared - 1` for the strict mean from when
    // we actually have logits.
    let n_sampling = compared.saturating_sub(0); // accept_prob_sum already skips NaN entries
    let sampling_acceptance = if n_sampling == 0 {
        0.0
    } else {
        // accept_prob_sum was only added for non-NaN entries; track that count
        // implicitly: the first scored step has prev_mtp_logits=None so its
        // contribution is 0. Divide by compared to give the harness-wide mean.
        accept_prob_sum / compared as f64
    };
    eprintln!("\n=== MTP-4 acceptance result ===");
    eprintln!("  compared: {compared} steps");
    eprintln!("  accepted (strict greedy):  {accepted} / {compared} = {:.1}%",
        acceptance * 100.0);
    eprintln!("  E[accept] (rejection-sampling, vLLM): sum={:.4} / {} = {:.1}%",
        accept_prob_sum, compared, sampling_acceptance * 100.0);
    eprintln!("  wall: {wall_ms:.0} ms ({:.1} ms/step base+probe)",
        wall_ms / (n_steps + 1) as f64);
    eprintln!("  GATE (sampling ≥75%): {}",
        if sampling_acceptance >= 0.75 { "PASS" } else { "FAIL" });

    // Free
    unsafe {
        last_device.dealloc(h_t_saved, hidden * 2)?;
        last_device.dealloc(e_token_dev, hidden * 2)?;
        last_device.dealloc(mtp_kcache, MTP_KV_MAX * kv_row_bytes)?;
        last_device.dealloc(mtp_vcache, MTP_KV_MAX * kv_row_bytes)?;
        rank0_device.dealloc(rank0_embed_buf, hidden * 2)?;
    }
    last_device.bind()?;
    mtp_scratch.dispose(last_device).ok();
    decode_scratch.dispose(&cluster).ok();
    session.dispose(&cluster).ok();
    model.dispose(&cluster)?;
    cluster.dispose()?;

    // Don't FAIL the test on low acceptance — this is a measurement,
    // not a correctness assertion. The cert documents the result;
    // pass/fail decision goes in MTP-4-decision.
    Ok(())
}

#[allow(dead_code)]
fn _ops_registry_alive(reg: &OpsRegistry) {
    let _ = reg;
}
