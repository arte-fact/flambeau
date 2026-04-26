//! Generic TP world=2 bench that takes the GGUF path from
//! `FLAMBEAU_BENCH_GGUF` and auto-dispatches on arch (qwen35,
//! qwen35moe, qwen3moe). Used by the cross-model sweep — same forward
//! driver / same TP w=2 GPU 0,1 setup as the per-model perf tests.
//!
//! Set `FLAMBEAU_BENCH_GGUF=/path/to/model.gguf` and
//! optionally `FLAMBEAU_CTX_CAP=N` (default 4096) to clamp KV-cache
//! provisioning. Writes one tok/s line per run; the harness collects
//! across many models.

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
use flambeau_qwen3_moe::forward::{
    forward_one_token_tp, forward_one_token_tp_logits, ShardedForwardOneTokenScratchTp,
};
use flambeau_qwen3_moe::{
    Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoETpModel, Qwen3MoETpSession,
};
use flambeau_quant::GgufFile;

const DEVICES: [i32; 2] = [0, 1];
const WORLD: u32 = 2;
const TG: usize = 64;
const WARM: u32 = 9419;

struct Cleanup<'c> {
    cluster: &'c HipCluster,
    session: Option<Qwen3MoETpSession>,
    scratch: Option<ShardedForwardOneTokenScratchTp>,
    model: Option<Qwen3MoETpModel>,
}
impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() { let _ = s.dispose(self.cluster); }
        if let Some(s) = self.scratch.take() { let _ = s.dispose(self.cluster); }
        if let Some(m) = self.model.take() { let _ = m.dispose(self.cluster); }
    }
}

#[test]
fn bench_tp2_any_model() -> Result<()> {
    let Ok(path_s) = std::env::var("FLAMBEAU_BENCH_GGUF") else {
        eprintln!("[skip] FLAMBEAU_BENCH_GGUF unset");
        return Ok(());
    };
    let path = PathBuf::from(&path_s);
    if !path.exists() {
        eprintln!("[skip] not found: {path_s}");
        return Ok(());
    }
    let n_avail = device_count().unwrap_or(0);
    if n_avail < WORLD as i32 {
        eprintln!("[skip] need {WORLD} HIP devices (have {n_avail})");
        return Ok(());
    }

    let model_label = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("(unknown)");
    eprintln!("BENCH_BEGIN model={model_label}");

    let file = match GgufFile::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("BENCH_RESULT model={model_label} status=open_failed err={e}");
            return Ok(());
        }
    };
    let cfg = match Qwen3MoEConfig::from_gguf(&file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("BENCH_RESULT model={model_label} status=cfg_failed err={e}");
            return Ok(());
        }
    };
    let arch = cfg.arch.clone();
    if !["qwen35", "qwen35moe", "qwen3moe"].contains(&arch.as_str()) {
        eprintln!(
            "BENCH_RESULT model={model_label} status=unsupported_arch arch={arch}"
        );
        return Ok(());
    }
    eprintln!("  arch={arch} layers={} hidden={}", cfg.num_layers, cfg.hidden_size);

    let tp = match Qwen35DenseTpLayout::new(&cfg, WORLD) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "BENCH_RESULT model={model_label} arch={arch} status=layout_failed err={e}"
            );
            return Ok(());
        }
    };
    let cluster = Arc::new(HipCluster::new(&DEVICES)?);
    if !cluster.peer_access_full() {
        eprintln!("[skip] peer-access not full on {DEVICES:?}");
        return Ok(());
    }

    let ctx_cap: usize = std::env::var("FLAMBEAU_CTX_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let load_start = Instant::now();
    let mut model = match Qwen3MoETpModel::load(&file, &cluster, tp) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "BENCH_RESULT model={model_label} arch={arch} status=load_failed err={e}"
            );
            return Ok(());
        }
    };
    let load_dt = load_start.elapsed().as_secs_f64();
    let weight_gib = model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    if model.config.context_length > ctx_cap {
        model.config.context_length = ctx_cap;
    }

    let mut guard = Cleanup { cluster: &cluster, session: None, scratch: None, model: Some(model) };
    let scratch = match ShardedForwardOneTokenScratchTp::new(
        &guard.model.as_ref().unwrap().config,
        &cluster,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "BENCH_RESULT model={model_label} arch={arch} weight_gib={weight_gib:.2} \
                 status=scratch_alloc_failed err={e}"
            );
            return Ok(());
        }
    };
    guard.scratch = Some(scratch);
    let session = match Qwen3MoETpSession::new(guard.model.as_ref().unwrap(), &cluster) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "BENCH_RESULT model={model_label} arch={arch} weight_gib={weight_gib:.2} \
                 status=session_alloc_failed err={e}"
            );
            return Ok(());
        }
    };
    guard.session = Some(session);
    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))?;

    let model_ref = guard.model.as_ref().unwrap();
    let scratch_ref = guard.scratch.as_mut().unwrap();
    let session_ref = guard.session.as_mut().unwrap();

    let mut logits = Vec::<f32>::new();
    if let Err(e) = forward_one_token_tp_logits(
        model_ref, scratch_ref, &cluster, &ar, &mut session_ref.caches, WARM, 0, &mut logits,
    ) {
        eprintln!(
            "BENCH_RESULT model={model_label} arch={arch} weight_gib={weight_gib:.2} \
             status=warm_forward_failed err={e}"
        );
        return Ok(());
    }
    let mut argmax = 0usize;
    let mut max = f32::NEG_INFINITY;
    let mut nan = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() { nan += 1; }
        if v > max { max = v; argmax = i; }
    }

    let mut next = argmax as u32;
    let t0 = Instant::now();
    let mut decode_err: Option<String> = None;
    for step in 0..TG {
        match forward_one_token_tp(
            model_ref, scratch_ref, &cluster, &ar, &mut session_ref.caches, next, 1 + step,
        ) {
            Ok(t) => next = t,
            Err(e) => {
                decode_err = Some(format!("{e}"));
                break;
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let tps = (TG as f64) / dt;

    if let Some(e) = decode_err {
        eprintln!(
            "BENCH_RESULT model={model_label} arch={arch} weight_gib={weight_gib:.2} \
             load_s={load_dt:.2} ctx_cap={ctx_cap} \
             status=decode_failed err={e}"
        );
    } else {
        eprintln!(
            "BENCH_RESULT model={model_label} arch={arch} layers={} hidden={} \
             weight_gib={weight_gib:.2} load_s={load_dt:.2} ctx_cap={ctx_cap} \
             warm_argmax={argmax} nan={nan} \
             tg={TG} decode_tok_s={tps:.2} last_id={next} status=ok",
            model_ref.config.num_layers, model_ref.config.hidden_size,
        );
    }

    drop(guard);
    Ok(())
}
