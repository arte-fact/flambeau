//! V2.30.b — single-point profiling harness for rocprofv3.
//!
//! Runs exactly one prefill-L + one decode-tg cycle. All selection via env
//! vars so a single test binary can profile every model.
//!
//! Env:
//!   FLAMBEAU_PROFILE_GGUF   — path to GGUF
//!   FLAMBEAU_PROFILE_MESH   — mesh ranks (default 4)
//!   FLAMBEAU_PROFILE_L      — prefill L (default 512, 0 = skip prefill)
//!   FLAMBEAU_PROFILE_TG     — decode steps (default 64, 0 = skip decode)
//!   FLAMBEAU_ASYNC_UBATCH / FLAMBEAU_UBATCH / FLAMBEAU_U_LANES — usual

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
use std::time::Instant;

#[test]
fn profile_point() -> Result<()> {
    let Some(path) = std::env::var("FLAMBEAU_PROFILE_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
    else {
        eprintln!("skip — set FLAMBEAU_PROFILE_GGUF to a valid path");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let n_requested: i32 = std::env::var("FLAMBEAU_PROFILE_MESH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    if n_requested <= 0 || n_requested > n_available {
        eprintln!("FLAMBEAU_PROFILE_MESH={n_requested} unavailable");
        return Ok(());
    }

    let prefill_l: usize = std::env::var("FLAMBEAU_PROFILE_L")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let decode_tg: usize = std::env::var("FLAMBEAU_PROFILE_TG")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    let ubatch: Option<usize> = std::env::var("FLAMBEAU_UBATCH")
        .ok().and_then(|s| s.parse().ok()).filter(|&u| u > 0);
    let async_enabled = std::env::var("FLAMBEAU_ASYNC_UBATCH").is_ok();
    let u_lanes: usize = std::env::var("FLAMBEAU_U_LANES")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(if async_enabled { 2 } else { 1 });

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let cluster = HipCluster::new(&(0..n_requested).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Warmup
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let _ = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &[9419], 0)?;
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    if prefill_l > 0 {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let scratch_size = ubatch.map(|u| u.min(prefill_l)).unwrap_or(prefill_l);
        let mut scratch = ShardedForwardPrefillScratch::new_with_lanes(
            &model, &cluster, scratch_size, u_lanes,
        )?;
        let tokens: Vec<u32> = (0..prefill_l as u32).map(|i| (1 + i * 37) % 151000).collect();
        let t0 = Instant::now();
        let _ = if async_enabled && u_lanes >= 2 {
            forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &tokens, 0)?
        } else if let Some(u) = ubatch {
            let mut pos = 0;
            let mut id = 0;
            for chunk in tokens.chunks(u) {
                id = forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, chunk, pos)?;
                pos += chunk.len();
            }
            id
        } else {
            forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &tokens, 0)?
        };
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("prefill L={prefill_l}: {:.2} tok/s ({:.2} ms)", prefill_l as f64 / dt, dt * 1000.0);
        scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    if decode_tg > 0 {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
        let seed = forward_prefill_pp(&model, &mut session, &cluster, &mut prefill_scratch, &[9419u32], 0)?;
        let mut next = seed;
        let t0 = Instant::now();
        for step in 0..decode_tg {
            next = forward_one_token_pp(&model, &mut session, &cluster, &mut decode_scratch, next, 1 + step)?;
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("decode tg={decode_tg}: {:.2} tok/s", decode_tg as f64 / dt);
        decode_scratch.dispose(&cluster)?;
        prefill_scratch.dispose(&cluster)?;
        session.dispose(&cluster)?;
    }

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
