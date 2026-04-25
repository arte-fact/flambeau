//! V2.28.b-i2 — forward_one_token_pp + short-L forward_prefill_pp on real
//! Qwen3-Coder-30B-A3B (arch=qwen3moe).

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

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_CODER_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            Some(std::path::PathBuf::from(
                "/artefact/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
            ))
        })
        .filter(|p| p.exists())
}

#[test]
fn forward_smoke_qwen3_coder() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3-Coder GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let n_requested: i32 = std::env::var("FLAMBEAU_MESH_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    if n_requested <= 0 || n_requested > n_available {
        eprintln!("mesh {n_requested} unavailable — skip");
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen3moe");

    let cluster = HipCluster::new(&(0..n_requested).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // One-token decode at seed 9419 — same seed we use across other models.
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    let seed_token = 9419u32;
    let seeded = forward_prefill_pp(&model, &mut session, &cluster, &mut prefill_scratch, &[seed_token], 0)?;
    eprintln!("L=1 prefill: last_id={seeded}");

    // Continue greedy for 7 more tokens → 8 tokens total for llama.cpp parity.
    let mut cur = seeded;
    let mut sequence = vec![seeded];
    for step in 0..7 {
        cur = forward_one_token_pp(
            &model,
            &mut session,
            &cluster,
            &mut decode_scratch,
            cur,
            1 + step,
        )?;
        sequence.push(cur);
    }
    eprintln!("greedy 8 tokens after seed 9419: {:?}", sequence);
    // V2.28.b-i3 parity check against llama.cpp (ROCm 6.4, Mesh<2>):
    // llama.cpp produces [25, 330, 488, 9419, 488, 330, 323, 330].
    let llama_ref: [u32; 8] = [25, 330, 488, 9419, 488, 330, 323, 330];
    let matches = sequence.iter().zip(llama_ref.iter())
        .take_while(|(a, b)| a == b).count();
    eprintln!("parity vs llama.cpp: {matches}/8 bit-exact");

    decode_scratch.dispose(&cluster)?;
    prefill_scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;

    // Short-L prefill smoke at L ∈ {2, 4, 16}.
    for &l in &[2usize, 4, 16] {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, l)?;
        let tokens: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();
        let last_id = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &tokens, 0)?;
        eprintln!("L={l} prefill: last_id={last_id}");
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
