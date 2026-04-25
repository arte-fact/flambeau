//! Same diagnostic as prefill_vs_decode_parity, but on Qwen3.5-27B-Q4_1
//! (arch=qwen35 gated full-attn) instead of Coder (arch=qwen3moe dense).
//! Isolates whether the per-position prefill bug is universal or
//! specific to V2.28.b-i1 dense-attn prefill.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::{Device, Stream};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_output_head_decode, forward_prefill_pp,
    argmax_token_host, ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;

#[test]
fn prefill_vs_decode_27b() -> Result<()> {
    let path = std::path::Path::new("/artefact/models/Qwen3.5-27B-Q4_1.gguf");
    if !path.exists() {
        eprintln!("skip");
        return Ok(());
    }
    let n_available: i32 = device_count().unwrap_or(0);
    if n_available < 4 { return Ok(()); }

    let file = GgufFile::open(path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&[0, 1, 2, 3])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Decode 8 tokens.
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
    let seed = forward_prefill_pp(&model, &mut session, &cluster, &mut prefill_scratch, &[9419], 0)?;
    let mut decode_history = vec![9419u32, seed];
    let mut cur = seed;
    for step in 0..7 {
        cur = forward_one_token_pp(&model, &mut session, &cluster, &mut decode_scratch, cur, 1 + step)?;
        decode_history.push(cur);
    }
    eprintln!("decode:  {:?}", decode_history);
    decode_scratch.dispose(&cluster)?;
    prefill_scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;

    // Prefill on the 8-token sequence.
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 8)?;
    let last_argmax = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &decode_history[..8], 0)?;
    eprintln!("forward_prefill_pp returned last_argmax = {last_argmax} (expected {})", decode_history[8]);

    let last_idx = cluster.ranks() as usize - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard.output_norm.as_ref().unwrap();
    let lm_head = last_shard.output.as_ref().or(last_shard.token_embd.as_ref()).unwrap();
    let output_head_scratch = last_scratch.output_head.as_mut().unwrap();
    let row_bytes = cfg.hidden_size * 2;

    let mut argmaxes = Vec::new();
    for pos in 0..8 {
        let x = last_scratch.hidden_a.offset_bytes(pos * row_bytes);
        forward_output_head_decode(&last_shard.ops, last_device.default_stream(), &cfg,
            output_norm, lm_head, output_head_scratch, x)?;
        let tok = argmax_token_host(last_device, last_device.default_stream(),
            output_head_scratch.logits_f32, cfg.vocab_size)?;
        argmaxes.push(tok);
    }
    eprintln!("prefill per-pos from hidden_a: {:?}", argmaxes);
    let mut matches = 0;
    for i in 0..8 {
        if argmaxes[i] == decode_history[i + 1] { matches += 1; }
    }
    eprintln!("parity: {matches}/8");

    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
