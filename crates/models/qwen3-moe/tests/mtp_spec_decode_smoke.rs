//! MTP-5c smoke — drive `forward_speculative_pp_step` for N macro
//! steps on Qwen3.6-27B + verify (a) coherent output, (b) per-step
//! wall-clock vs baseline.
//!
//! Reports: committed tokens, accept rate, per-stage timings.
//! Skipped without the GGUF + 4 HIP devices.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel \
              launch over host/device buffers that live for the bounded \
              synchronize that follows."
)]

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp_logits, forward_speculative_pp_step,
    ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::mtp::{load_mtp_head, MtpForwardScratch};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];

#[test]
fn mtp_spec_decode_smoke() -> Result<()> {
    let base_path = PathBuf::from("/artefact/models/Qwen3.6-27B-Q4_0.gguf");
    let mtp_path = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    if !base_path.exists() || !mtp_path.exists() {
        eprintln!("skip: GGUFs not present");
        return Ok(());
    }
    let n_gpus = device_count().unwrap_or(0);
    if n_gpus < 4 {
        eprintln!("skip: need ≥4 HIP devices (got {n_gpus})");
        return Ok(());
    }
    let n_macro: usize = std::env::var("FLAMBEAU_SPEC_MACRO_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    eprintln!("=== MTP-5c spec-decode smoke ===");
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    let cluster = HipCluster::new(&(0..n_gpus).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!("loading base across {} ranks…", cluster.ranks());
    let model = Qwen3MoEShardedModel::load(&base_file, &cluster, &assignment)?;

    let last_rank = (cluster.ranks() - 1) as usize;
    let last_device = cluster.device(last_rank);
    let last_shard = &model.shards[last_rank];
    let output_norm = last_shard.output_norm.as_ref().ok_or_else(|| anyhow!("no output_norm"))?;
    let lm_head = last_shard.output.as_ref().ok_or_else(|| anyhow!("no lm_head"))?;

    eprintln!("loading MTP on rank {last_rank}…");
    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, last_device)?;
    last_device.bind()?;
    let mtp_scratch = MtpForwardScratch::new(last_device, &cfg)?;

    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
    // L=2 prefill scratch — sized for two-token batched verify.
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 2)?;

    // ── Prime base via sequential decodes over the prompt (so KV is populated
    //    AND we hold a fresh hidden_a on the last rank for the first MTP draft).
    let mut last_token = 0u32;
    for (i, &tok) in PROMPT_IDS.iter().enumerate() {
        let mut logits = Vec::new();
        forward_one_token_pp_logits(
            &model, &mut session, &cluster, &mut decode_scratch, tok, i,
            &mut logits,
        )?;
        // argmax for last_token (first sample at position prompt_len)
        if i == PROMPT_IDS.len() - 1 {
            let mut best_i = 0usize;
            let mut best_v = logits[0];
            for (j, &v) in logits.iter().enumerate().skip(1) {
                if v > best_v { best_v = v; best_i = j; }
            }
            last_token = best_i as u32;
        }
    }
    eprintln!("prefill done; first sampled token = {last_token}");

    // After the last prefill decode, decode_scratch.per_rank[last_rank].hidden_a
    // holds the hidden that produced last_token. Use it for MTP-1 input.
    let mut h_for_mtp = decode_scratch.per_rank[last_rank].hidden_a;
    let mut position = PROMPT_IDS.len();
    let mut all_tokens: Vec<u32> = vec![last_token];

    let mut accept_count = 0usize;
    let mut macro_count = 0usize;
    let mut total_tokens = 0usize;
    let mut total_t = flambeau_qwen3_moe::forward::SpecTimings::default();

    let t0 = std::time::Instant::now();
    for step in 0..n_macro {
        let (s, h_next) = forward_speculative_pp_step(
            &model, &mut session, &cluster, &mut decode_scratch, &mut prefill_scratch,
            &mtp, &mtp_scratch,
            output_norm, lm_head,
            last_token, h_for_mtp, position,
        )?;
        eprintln!(
            "  macro {step}: pos {position}→{} draft={} verify={} {} committed={:?} \
             timings(ms): snap={:.2} mtp={:.2} l2={:.2} restore={:.2} redo={:.2}",
            s.new_position, s.draft, s.verify,
            if s.accepted { "✓" } else { "✗" },
            s.committed,
            s.timings_ms.gdn_snapshot, s.timings_ms.mtp_draft,
            s.timings_ms.base_l2, s.timings_ms.gdn_restore, s.timings_ms.base_l1_redo,
        );
        total_t.gdn_snapshot += s.timings_ms.gdn_snapshot;
        total_t.mtp_draft    += s.timings_ms.mtp_draft;
        total_t.base_l2      += s.timings_ms.base_l2;
        total_t.gdn_restore  += s.timings_ms.gdn_restore;
        total_t.base_l1_redo += s.timings_ms.base_l1_redo;

        if s.accepted { accept_count += 1; }
        for &t in &s.committed { all_tokens.push(t); }
        last_token = *s.committed.last().unwrap();
        position = s.new_position;
        h_for_mtp = h_next;
        macro_count += 1;
        total_tokens += s.committed.len();
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("\n=== spec-decode result ===");
    eprintln!(
        "  macro steps: {macro_count}, committed tokens: {total_tokens}, accept rate: {:.1}%",
        100.0 * accept_count as f64 / macro_count.max(1) as f64
    );
    eprintln!(
        "  wall: {total_ms:.0} ms, {:.1} ms/token (vs baseline ~50-60 ms/token)",
        total_ms / total_tokens.max(1) as f64
    );
    eprintln!(
        "  cumulative timings: snap={:.0} mtp={:.0} l2={:.0} restore={:.0} redo={:.0} (ms)",
        total_t.gdn_snapshot, total_t.mtp_draft, total_t.base_l2,
        total_t.gdn_restore, total_t.base_l1_redo,
    );
    eprintln!("  tokens: {all_tokens:?}");

    decode_scratch.dispose(&cluster).ok();
    prefill_scratch.dispose(&cluster).ok();
    last_device.bind()?;
    mtp_scratch.dispose(last_device).ok();
    session.dispose(&cluster).ok();
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
