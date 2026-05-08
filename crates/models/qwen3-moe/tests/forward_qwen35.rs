//! V2.2.c smoke test — real-weight Qwen3.5-9B-Q4_1 on Mesh<1>.
//!
//! First end-to-end exercise of the dense-FFN forward path + Q4_1 MMVQ
//! kernel. Success = `forward_one_token_pp` returns a valid token id
//! without crashing. Parity vs llama.cpp is a separate cert.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{forward_one_token_pp, ShardedForwardOneTokenScratch};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from("/artefact/models/Qwen3.5-9B-Q4_1.gguf")))
        .filter(|p| p.exists())
}

#[test]
fn qwen35_forward_one_token_mesh1_smoke() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("skip — no HIP device");
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");
    assert!(cfg.is_dense_ffn(), "qwen35 must route dense FFN path");

    let cluster = HipCluster::new(&[0])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, 1);

    eprintln!(
        "loading Qwen3.5-9B-Q4_1 on Mesh<1> ({} layers)…",
        cfg.num_layers,
    );
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    eprintln!(
        "load ok: {:.2} GiB on device 0",
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    let mut session = Qwen3MoEShardedSession::new(&model, &cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    // Token id 9419 = "Hello" in the rest of the harness. Any valid id
    // exercises the same kernel stack.
    let token_id = 9419u32;
    let next = forward_one_token_pp(
        &model,
        &mut session,
        &cluster,
        &mut scratch,
        token_id,
        /*position=*/ 0,
    )?;

    assert!(
        (next as usize) < cfg.vocab_size,
        "next token {next} out of vocab range {}",
        cfg.vocab_size
    );
    eprintln!("qwen35 Mesh<1> forward ok → argmax next = {next}");

    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
