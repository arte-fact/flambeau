//! TP world=2 sanity + perf — Qwen3.6-27B-Q4_0 on GPU {0, 1}.
//!
//! Qwen3.6-27B is arch=qwen35 (dense FFN + hybrid GDN/full-attn),
//! 64 layers, 14.7 GiB on disk → ~7.4 GiB/rank weights at world=2.
//! Same code path as the Qwen3.5 dense-hybrid models, but exercises
//! the F32→F16 / F32→Q8_0 TP-loader conversions added in this session
//! (Qwen3.6 ships F32 norms + F32 ssm_alpha/beta, both of which need
//! conversion before the F16-rmsnorm + mmvq_q8_0 kernels can consume them).
//!
//! Writes `certs/perf/qwen36_27b_q4_0_tp2_decode.json`.
//!
//! Skips cleanly when fewer than 2 HIP devices, no GGUF, or peer
//! access between {0, 1} is missing. Drop guards prevent ~7 GiB
//! leaks on `?`.

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

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.6-27B-Q4_0.gguf";
const DEVICES: [i32; 2] = [0, 1];
const WORLD: u32 = 2;
const TG: usize = 64;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN36_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_PATH);
            p.exists().then_some(p)
        })
}

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
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
    let mut nan = 0usize;
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    let mut argmax = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            nan += 1;
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
        "  [{label}] len={}  nan={}  min={:.4}  max={:.4}  argmax={argmax}",
        logits.len(),
        nan,
        min,
        max
    );
    let mut idx_sorted: Vec<usize> = (0..logits.len()).collect();
    idx_sorted.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "  [{label}] top5 = {:?}",
        idx_sorted[..5]
            .iter()
            .map(|&i| (i, logits[i]))
            .collect::<Vec<_>>()
    );
    argmax as u32
}

#[test]
fn perf_baseline_qwen36_27b_tp2_gpu_0_1() -> Result<()> {
    let n_available = device_count().unwrap_or(0);
    if n_available < WORLD as i32 {
        eprintln!("[skip] need {WORLD} HIP devices (have {n_available})");
        return Ok(());
    }
    let Some(path) = gguf_path() else {
        eprintln!(
            "[skip] no GGUF: set FLAMBEAU_QWEN36_GGUF or place Qwen3.6-27B-Q4_0.gguf at {DEFAULT_PATH}"
        );
        return Ok(());
    };

    eprintln!("loading {} on devices {DEVICES:?}", path.display());
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    if cfg.arch != "qwen35" {
        eprintln!("[skip] expected arch=qwen35, got {}", cfg.arch);
        return Ok(());
    }
    let ctx_cap: usize = std::env::var("FLAMBEAU_CTX_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let tp = Qwen35DenseTpLayout::new(&cfg, WORLD)?;
    let cluster = Arc::new(HipCluster::new(&DEVICES)?);
    if !cluster.peer_access_full() {
        eprintln!("[skip] BAR1 peer-access between {DEVICES:?} not fully connected");
        return Ok(());
    }

    let load_start = Instant::now();
    let mut model = Qwen3MoETpModel::load(&file, &cluster, tp)?;
    let load_dt = load_start.elapsed();
    if model.config.context_length > ctx_cap {
        eprintln!(
            "  shrinking model.config.context_length {} -> {ctx_cap} (KV-cache fit)",
            model.config.context_length
        );
        model.config.context_length = ctx_cap;
    }
    eprintln!(
        "  load: {:.2}s ({:.2} GiB total)",
        load_dt.as_secs_f64(),
        model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
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
    let model_ref = guard.model.as_ref().unwrap();
    let scratch_ref = guard.scratch.as_mut().unwrap();
    let session_ref = guard.session.as_mut().unwrap();

    let mut logits = Vec::<f32>::new();
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
    let argmax = summarize("warm world=2", &logits);

    let mut next: u32 = argmax;
    let t0 = Instant::now();
    for step in 0..TG {
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
    let tps = TG as f64 / dt;
    let n_layers = model_ref.config.num_layers;
    eprintln!(
        "  decode tg={TG} → {tps:.2} tok/s ({:.1} ms total, last_id={next})",
        dt * 1000.0
    );

    let cert_dir = workspace_root().join("certs/perf");
    std::fs::create_dir_all(&cert_dir).ok();
    let cert_path = cert_dir.join("qwen36_27b_q4_0_tp2_decode.json");
    let cert = format!(
        "{{\n  \"schema_version\": 1,\n  \"kind\": \"tp_decode_perf\",\n  \
         \"model\": \"Qwen3.6-27B-Q4_0\",\n  \"arch\": \"qwen35\",\n  \"world\": {WORLD},\n  \
         \"devices\": [{}, {}],\n  \"topology\": \"Tp\",\n  \
         \"layers\": {n_layers},\n  \
         \"context_length\": {ctx_cap},\n  \
         \"tg\": {TG},\n  \"decode_tok_per_s\": {tps:.4},\n  \
         \"decode_total_ms\": {:.4},\n  \
         \"warm_token_id\": {warm_token},\n  \
         \"warm_argmax\": {argmax},\n  \
         \"last_token_id\": {next},\n  \
         \"captured_via\": \"cargo test --release -p flambeau-qwen3-moe --features hip --test perf_baseline_qwen36_27b_tp2_gpu_0_1 -- --nocapture\",\n  \
         \"captured_at\": \"2026-04-26\",\n  \
         \"notes\": \"Qwen3.6-27B-Q4_0 (qwen35 arch, dense+hybrid). World=2 TP on physical GPUs {{0, 1}}. Exercises this session's TP-loader F32 conversions for norms (F32→F16) and ssm_alpha/beta (F32→Q8_0).\"\n}}\n",
        DEVICES[0],
        DEVICES[1],
        dt * 1000.0,
    );
    std::fs::write(&cert_path, cert)?;
    eprintln!("  wrote {}", cert_path.display());

    drop(guard);
    Ok(())
}
