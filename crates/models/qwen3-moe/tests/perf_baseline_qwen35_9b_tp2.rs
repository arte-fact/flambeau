//! TP world bisect — Qwen3.5-9B-Q4_1, runs the TP forward at world=1
//! (degenerate: AR replaced by add_f16) AND world=2 (real BarP2pAllReduce
//! across GPU 0,1) on the same model, dumping logits stats. Lets us tell
//! whether the all-NaN logits at world=2 come from the multi-rank AR path
//! or from the per-rank TP forward kernels themselves.
//!
//! Skips cleanly when fewer than 2 HIP devices, no GGUF, or peer access
//! between {0, 1} is missing. Drop guards prevent ~5 GiB leaks on `?`.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test harness — load + forward + dispose; per-call unsafety \
              is documented in TP-1c / TP-2a / TP-2d."
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_qwen3_moe::forward::{forward_one_token_tp, forward_one_token_tp_logits, ShardedForwardOneTokenScratchTp};
use flambeau_qwen3_moe::{
    Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession,
};
use flambeau_quant::GgufFile;

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_PATH);
            p.exists().then_some(p)
        })
}

struct Cleanup<'c> {
    cluster: &'c HipCluster,
    session: Option<Qwen3MoETpSession>,
    scratch: Option<ShardedForwardOneTokenScratchTp>,
    model: Option<Qwen3MoETpModel>,
}

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() {
            let _ = s.dispose(self.cluster);
        }
        if let Some(s) = self.scratch.take() {
            let _ = s.dispose(self.cluster);
        }
        if let Some(m) = self.model.take() {
            let _ = m.dispose(self.cluster);
        }
    }
}

fn summarize(label: &str, logits: &[f32]) -> u32 {
    let mut zero = 0usize;
    let mut nan = 0usize;
    let mut neg_inf = 0usize;
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    let mut argmax = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v == 0.0 {
            zero += 1;
        }
        if v.is_nan() {
            nan += 1;
        }
        if v == f32::NEG_INFINITY {
            neg_inf += 1;
        }
        if v > max {
            max = v;
            argmax = i;
        }
        if v < min {
            min = v;
        }
    }
    eprintln!(
        "  [{label}] len={}  zero={}  nan={}  -inf={}  min={:.4}  max={:.4}  argmax={}",
        logits.len(),
        zero,
        nan,
        neg_inf,
        min,
        max,
        argmax
    );
    eprintln!(
        "  [{label}] first 8 logits = {:?}",
        &logits[..logits.len().min(8)]
    );
    let mut idx_sorted: Vec<usize> = (0..logits.len()).collect();
    idx_sorted
        .sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "  [{label}] top5 = {:?}",
        idx_sorted[..5]
            .iter()
            .map(|&i| (i, logits[i]))
            .collect::<Vec<_>>()
    );
    argmax as u32
}

fn run_forward_at(devices: &[i32]) -> Result<u32> {
    let world = devices.len() as u32;
    let label = format!("world={world}");
    eprintln!("\n=== {label} on devices {devices:?} ===");

    let path = gguf_path().expect("gguf already validated");
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let tp = Qwen35DenseTpLayout::new(&cfg, world)?;
    let cluster = Arc::new(HipCluster::new(devices)?);
    if world > 1 && !cluster.peer_access_full() {
        eprintln!("  [skip] peer-access between {devices:?} not fully connected");
        return Ok(0);
    }

    let mut model = Qwen3MoETpModel::load(&file, &cluster, tp)?;
    let ctx_cap: usize = std::env::var("FLAMBEAU_CTX_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    if model.config.context_length > ctx_cap {
        model.config.context_length = ctx_cap;
    }
    eprintln!(
        "  load: {:.2} GiB total ({:.2} GiB/rank avg)",
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0) / world as f64
    );

    let mut guard = Cleanup {
        cluster: &cluster,
        session: None,
        scratch: None,
        model: Some(model),
    };
    let scratch =
        ShardedForwardOneTokenScratchTp::new(&guard.model.as_ref().unwrap().config, &cluster)?;
    guard.scratch = Some(scratch);
    let session = Qwen3MoETpSession::new(guard.model.as_ref().unwrap(), &cluster)?;
    guard.session = Some(session);
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))?;

    let warm_token: u32 = 9419;
    let mut logits = Vec::<f32>::new();
    let model_ref = guard.model.as_ref().unwrap();
    let scratch_ref = guard.scratch.as_mut().unwrap();
    let session_ref = guard.session.as_mut().unwrap();

    forward_one_token_tp_logits(
        model_ref,
        scratch_ref,
        &cluster,
        &ar,
        &mut session_ref.caches,
        warm_token,
        0,
        &mut logits,
    )?;
    let argmax = summarize(&label, &logits);

    // Timed decode loop to compare per-token cost across worlds.
    let tg: usize = 4;
    let mut next = argmax;
    let t0 = Instant::now();
    for step in 0..tg {
        next = forward_one_token_tp(
            model_ref,
            scratch_ref,
            &cluster,
            &ar,
            &mut session_ref.caches,
            next,
            1 + step,
        )?;
    }
    let dt = t0.elapsed().as_secs_f64();
    let tps = tg as f64 / dt;
    let n_layers = model_ref.config.num_layers;
    eprintln!(
        "  [{label}] decode tg={tg} → {tps:.2} tok/s  ({:.1} ms total, {:.2} ms/layer/token, last_id={next})",
        dt * 1000.0,
        dt * 1000.0 / (tg as f64) / (n_layers as f64),
    );
    drop(guard);
    Ok(argmax)
}

#[test]
fn tp_world1_vs_world2_logits_bisect() -> Result<()> {
    let n_available = device_count().unwrap_or(0);
    if n_available < 2 {
        eprintln!("[skip] need >= 2 HIP devices (have {n_available})");
        return Ok(());
    }
    if gguf_path().is_none() {
        eprintln!("[skip] no GGUF at {DEFAULT_PATH}");
        return Ok(());
    }

    let argmax_w1 = run_forward_at(&[0])?;
    let argmax_w2 = run_forward_at(&[0, 1])?;

    eprintln!("\n=== bisect summary ===");
    eprintln!("  world=1 argmax = {argmax_w1}");
    eprintln!("  world=2 argmax = {argmax_w2}");
    eprintln!(
        "  reference (llama.cpp + candle for seed 9419 on Qwen3.x with this tokenizer) = 11"
    );
    Ok(())
}
