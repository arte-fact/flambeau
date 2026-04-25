//! V2.34 bisect — at which L does forward_prefill_pp start producing
//! wrong output? Compares prefill's returned last-position argmax
//! against the decode stream's corresponding token.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;

#[test]
fn prefill_L_sweep() -> Result<()> {
    let path = std::path::Path::new("/artefact/models/Qwen3.5-27B-Q4_1.gguf");
    if !path.exists() { return Ok(()); }
    if device_count().unwrap_or(0) < 4 { return Ok(()); }

    let file = GgufFile::open(path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&[0, 1, 2, 3])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Decode ground truth.
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
    let seed = forward_prefill_pp(&model, &mut session, &cluster, &mut prefill_scratch, &[9419], 0)?;
    let mut decode_history = vec![9419u32, seed];
    let mut cur = seed;
    for step in 0..15 {
        cur = forward_one_token_pp(&model, &mut session, &cluster, &mut decode_scratch, cur, 1 + step)?;
        decode_history.push(cur);
    }
    eprintln!("decode: {:?}", decode_history);
    decode_scratch.dispose(&cluster)?;
    prefill_scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;

    // For each L: fresh session, prefill [0..L], verify return = decode[L].
    for &l in &[1, 2, 3, 4, 5, 6, 7, 8, 12, 16] {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, l)?;
        let ret = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &decode_history[..l], 0)?;
        let expect = decode_history[l];
        let ok = if ret == expect { "✓" } else { "✗" };
        eprintln!("  L={l:<2}: prefill returned {ret:<7} expected {expect:<7} {ok}");
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
