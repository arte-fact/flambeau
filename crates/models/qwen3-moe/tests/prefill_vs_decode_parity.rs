//! V2.33.d diagnostic — does forward_prefill_pp at position i produce
//! the same argmax as forward_one_token_pp at position i, given the
//! same preceding tokens?
//!
//! If NO → the two kernels diverge at F32-accumulation-order precision,
//! and spec decoding's acceptance rate will be capped by this drift.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::Device;
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_output_head_decode, forward_prefill_pp,
    argmax_token_host, ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;

#[test]
fn prefill_vs_decode_per_position() -> Result<()> {
    let path = std::path::Path::new("/artefact/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf");
    if !path.exists() {
        eprintln!("skip");
        return Ok(());
    }
    let n_available: i32 = device_count().unwrap_or(0);
    if n_available < 4 {
        eprintln!("need 4 GPUs");
        return Ok(());
    }

    let file = GgufFile::open(path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&[0, 1, 2, 3])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Ground-truth: greedy decode 8 tokens from seed 9419.
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
    eprintln!("decode path:  {:?}", decode_history);
    decode_scratch.dispose(&cluster)?;
    prefill_scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;

    // Now prefill the full 8-token sequence in ONE call, extract argmax at each pos.
    // FRESH session so KV cache is empty — start_position=0 matches pos 0.
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 8)?;
    let last_argmax = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &decode_history[..8], 0)?;
    eprintln!("forward_prefill_pp's return (pos-7 argmax): {last_argmax} (decode says should be {})", decode_history[8]);

    // Read argmax at each of 8 positions from last rank's hidden_a.
    let last_idx = cluster.ranks() as usize - 1;
    let last_shard = &model.shards[last_idx];
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_norm = last_shard.output_norm.as_ref().unwrap();
    let lm_head = last_shard.output.as_ref().or(last_shard.token_embd.as_ref()).unwrap();
    let output_head_scratch = last_scratch.output_head.as_mut().unwrap();
    let row_bytes = cfg.hidden_size * 2;

    let mut prefill_argmaxes_a = Vec::new();
    let mut prefill_argmaxes_b = Vec::new();
    for pos in 0..8 {
        // Read from hidden_a
        let x = last_scratch.hidden_a.offset_bytes(pos * row_bytes);
        forward_output_head_decode(&last_shard.ops, last_device.default_stream(), &cfg,
            output_norm, lm_head, output_head_scratch, x)?;
        let tok_a = argmax_token_host(last_device, last_device.default_stream(),
            output_head_scratch.logits_f32, cfg.vocab_size)?;
        prefill_argmaxes_a.push(tok_a);
        // Read from hidden_b
        let xb = last_scratch.hidden_b.offset_bytes(pos * row_bytes);
        forward_output_head_decode(&last_shard.ops, last_device.default_stream(), &cfg,
            output_norm, lm_head, output_head_scratch, xb)?;
        let tok_b = argmax_token_host(last_device, last_device.default_stream(),
            output_head_scratch.logits_f32, cfg.vocab_size)?;
        prefill_argmaxes_b.push(tok_b);
    }
    eprintln!("prefill argmaxes from hidden_a: {:?}", prefill_argmaxes_a);
    eprintln!("prefill argmaxes from hidden_b: {:?}", prefill_argmaxes_b);
    let prefill_argmaxes = &prefill_argmaxes_a;
    eprintln!();
    eprintln!("comparison (prefill argmax at pos i vs decode token at pos i+1):");
    let mut matches = 0;
    for i in 0..8 {
        let match_ok = prefill_argmaxes[i] == decode_history[i + 1];
        if match_ok { matches += 1; }
        eprintln!("  pos {i}: prefill={:<6} decode_next={:<6} {}",
            prefill_argmaxes[i], decode_history[i + 1],
            if match_ok { "MATCH" } else { "MISMATCH" });
    }
    eprintln!("\nparity: {matches}/8 positions match");

    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
