//! AUTO-5 — topology comparison bench for Qwen3.5-9B-Q4_1.
//!
//! Loads the same model in five configurations on the same rig and
//! reports prefill + decode throughput for each. The pure-TP world=4
//! configuration is intentionally **omitted**: this rig has a known
//! BAR1 fault on the `{2,3}` peer pair (`project_rig_gpu23_link_fault`)
//! that makes tp4 either silently corrupt or fail to engage; pp+tp
//! with stages chosen to avoid co-locating `{2,3}` is the workaround.
//!
//! Configurations:
//!
//! | Tag       | mesh-mode  | pp_size | tp_size | devices       |
//! |-----------|------------|---------|---------|---------------|
//! | `mesh1`   | pp         | 1       | n/a     | `[0]`         |
//! | `pp2`     | pp         | 2       | n/a     | `[0, 1]`      |
//! | `tp2`     | tp         | n/a     | 2       | `[0, 1]`      |
//! | `pp4`     | pp         | 4       | n/a     | `[0, 1, 2, 3]`|
//! | `pp2tp2`  | pp+tp      | 2       | 2       | `[0, 2, 1, 3]`|
//!
//! `pp2tp2` lays out devices stage-major so stage 0 = `{0, 2}` and
//! stage 1 = `{1, 3}` — both intra-stage TP groups are healthy ascending
//! pairs that pass the BAR1 probe in `tests/peer_access_pairs.rs`.
//!
//! Output: `certs/perf/topology_compare/qwen35_9b_q4_1.json` with one
//! entry per topology, each containing prefill (L=8/64/128/512) and
//! decode (tg=64) timings.

#![cfg(feature = "hip")]
#![expect(clippy::undocumented_unsafe_blocks, reason = "test harness — load + bench + dispose")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_hybrid, forward_one_token_pp, forward_one_token_tp,
    forward_prefill_hybrid_logits, forward_prefill_pp, forward_prefill_tp_logits,
    ShardedForwardOneTokenScratch, ShardedForwardOneTokenScratchHybrid,
    ShardedForwardOneTokenScratchTp, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoEHybridModel,
    Qwen3MoEHybridSession, Qwen3MoEShardedModel, Qwen3MoEShardedSession, Qwen3MoETpModel,
    Qwen3MoETpSession,
};
use flambeau_runtime::LayerAssignment;
use serde::{Deserialize, Serialize};

const DEFAULT_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";
const PREFILL_LENGTHS: &[usize] = &[8, 64, 128, 512];
const DECODE_LEN: usize = 64;
const SEED_TOKEN: u32 = 9419;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(DEFAULT_PATH)))
        .filter(|p| p.exists())
}

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhaseRun {
    phase: String,
    n_tokens: usize,
    wall_secs: f64,
    tok_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TopologyResult {
    tag: String,
    mesh_mode: String,
    pp_size: u32,
    tp_size: u32,
    devices: Vec<i32>,
    load_secs: f64,
    runs: Vec<PhaseRun>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComparisonCert {
    model_tag: String,
    rig_notes: String,
    topologies: Vec<TopologyResult>,
}

fn build_prompt(len: usize) -> Vec<u32> {
    // Synthetic seed-token-prompt — the same scheme the existing
    // perf_baseline tests use. Token id `9419` is the V2.2.c.1 / V2.2.b
    // seed; pad with sequential IDs to reach `len` distinct positions
    // (avoids hammering the same KV cache row).
    (0..len as u32)
        .map(|i| if i == 0 { SEED_TOKEN } else { 1 + i })
        .collect()
}

fn pretty_table(results: &[TopologyResult]) -> String {
    let mut out = String::new();
    out.push_str("topology      pp tp  load_s  ");
    for &l in PREFILL_LENGTHS {
        out.push_str(&format!("pp{l:<4} "));
    }
    out.push_str(&format!("tg{DECODE_LEN}\n"));
    for r in results {
        out.push_str(&format!(
            "{:<13} {:>2} {:>2}  {:>6.2}  ",
            r.tag, r.pp_size, r.tp_size, r.load_secs
        ));
        for &l in PREFILL_LENGTHS {
            let v = r
                .runs
                .iter()
                .find(|p| p.phase == "prefill" && p.n_tokens == l)
                .map(|p| p.tok_per_sec)
                .unwrap_or(0.0);
            out.push_str(&format!("{v:>6.1} "));
        }
        let dec = r
            .runs
            .iter()
            .find(|p| p.phase == "decode" && p.n_tokens == DECODE_LEN)
            .map(|p| p.tok_per_sec)
            .unwrap_or(0.0);
        out.push_str(&format!("{dec:>6.1}\n"));
        // **AUTO-6d** — emit a second row when this topology has
        // batched prefill samples (TP-only paths today).
        let has_batched = r.runs.iter().any(|p| p.phase == "prefill_batched");
        if has_batched {
            out.push_str(&format!(
                "{:<13} {:>2} {:>2}  {:>6}  ",
                format!("{}+batched", r.tag),
                r.pp_size,
                r.tp_size,
                "—"
            ));
            for &l in PREFILL_LENGTHS {
                let v = r
                    .runs
                    .iter()
                    .find(|p| p.phase == "prefill_batched" && p.n_tokens == l)
                    .map(|p| p.tok_per_sec)
                    .unwrap_or(0.0);
                out.push_str(&format!("{v:>6.1} "));
            }
            out.push_str(&format!("{:>6}\n", "—"));
        }
    }
    out
}

// ───────────────────────── PP path ──────────────────────────────────

fn bench_pp(
    tag: &'static str,
    pp_size: u32,
    devices: &[i32],
    file: &GgufFile,
) -> Result<TopologyResult> {
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(devices)?);
    let cfg = Qwen3MoEConfig::from_gguf(file)?;

    let load_start = Instant::now();
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(file, &cluster, &assignment)
        .with_context(|| format!("{tag} sharded load"))?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let mut runs: Vec<PhaseRun> = Vec::new();

    for &l in PREFILL_LENGTHS {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, l)?;
        let prompt = build_prompt(l);
        let t = Instant::now();
        forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &prompt, 0)?;
        let secs = t.elapsed().as_secs_f64();
        runs.push(PhaseRun {
            phase: "prefill".to_string(),
            n_tokens: l,
            wall_secs: secs,
            tok_per_sec: l as f64 / secs,
        });
        scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    // Decode tg=64 from a 1-token prefill.
    {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let mut prefill_scratch = ShardedForwardPrefillScratch::new(&model, &cluster, 1)?;
        let mut decode_scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;
        forward_prefill_pp(
            &model,
            &mut session,
            &cluster,
            &mut prefill_scratch,
            &[SEED_TOKEN],
            0,
        )?;
        prefill_scratch.dispose(&cluster).ok();
        let mut last = SEED_TOKEN;
        let t = Instant::now();
        for pos in 0..DECODE_LEN {
            last = forward_one_token_pp(
                &model,
                &mut session,
                &cluster,
                &mut decode_scratch,
                last,
                1 + pos,
            )?;
        }
        let secs = t.elapsed().as_secs_f64();
        runs.push(PhaseRun {
            phase: "decode".to_string(),
            n_tokens: DECODE_LEN,
            wall_secs: secs,
            tok_per_sec: DECODE_LEN as f64 / secs,
        });
        decode_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    model.dispose(&cluster).ok();
    drop(cluster);
    let _ = pp_size;

    Ok(TopologyResult {
        tag: tag.to_string(),
        mesh_mode: "pp".to_string(),
        pp_size,
        tp_size: 1,
        devices: devices.to_vec(),
        load_secs,
        runs,
    })
}

// ───────────────────────── TP path ──────────────────────────────────

fn bench_tp(tag: &'static str, devices: &[i32], file: &GgufFile) -> Result<TopologyResult> {
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(devices)?);
    let world = cluster.ranks() as u32;
    let cfg = Qwen3MoEConfig::from_gguf(file)?;
    let layout = Qwen35DenseTpLayout::new(&cfg, world)?;

    let load_start = Instant::now();
    let model = Qwen3MoETpModel::load(file, &cluster, layout)
        .with_context(|| format!("{tag} tp load"))?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let ar = BarP2pAllReduce::new(Arc::clone(&cluster))
        .with_context(|| format!("{tag} BarP2pAllReduce"))?;

    let mut runs: Vec<PhaseRun> = Vec::new();

    // **AUTO-6d** — record two prefill phases per L:
    //   * `prefill`         — per-token loop (AUTO-6a wrapper, baseline)
    //   * `prefill_batched` — L-batched (AUTO-6b/6c, opt-in via
    //                         FLAMBEAU_TP_BATCHED=1)
    // Both routes go through `forward_prefill_tp_logits`; the env flag
    // toggles which inner driver runs. Logits are downloaded into a
    // shared host buffer so the two paths are comparable wall-clock.
    let mut logits = vec![0.0f32; cfg.vocab_size];
    for &l in PREFILL_LENGTHS {
        let prompt = build_prompt(l);

        // Per-token baseline.
        std::env::remove_var("FLAMBEAU_TP_BATCHED");
        {
            let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
            let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
            let t = Instant::now();
            forward_prefill_tp_logits(
                &model,
                &mut scratch,
                &cluster,
                &ar,
                &mut session.caches,
                &prompt,
                0,
                &mut logits,
            )?;
            let secs = t.elapsed().as_secs_f64();
            runs.push(PhaseRun {
                phase: "prefill".to_string(),
                n_tokens: l,
                wall_secs: secs,
                tok_per_sec: l as f64 / secs,
            });
            scratch.dispose(&cluster).ok();
            session.dispose(&cluster).ok();
        }

        // L-batched (AUTO-6b/6c).
        std::env::set_var("FLAMBEAU_TP_BATCHED", "1");
        {
            let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
            let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
            let t = Instant::now();
            forward_prefill_tp_logits(
                &model,
                &mut scratch,
                &cluster,
                &ar,
                &mut session.caches,
                &prompt,
                0,
                &mut logits,
            )?;
            let secs = t.elapsed().as_secs_f64();
            runs.push(PhaseRun {
                phase: "prefill_batched".to_string(),
                n_tokens: l,
                wall_secs: secs,
                tok_per_sec: l as f64 / secs,
            });
            scratch.dispose(&cluster).ok();
            session.dispose(&cluster).ok();
        }
        std::env::remove_var("FLAMBEAU_TP_BATCHED");
    }

    {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
        // 1-token "prefill"
        forward_one_token_tp(
            &model,
            &mut scratch,
            &cluster,
            &ar,
            &mut session.caches,
            SEED_TOKEN,
            0,
        )?;
        let mut last = SEED_TOKEN;
        let t = Instant::now();
        for pos in 0..DECODE_LEN {
            last = forward_one_token_tp(
                &model,
                &mut scratch,
                &cluster,
                &ar,
                &mut session.caches,
                last,
                1 + pos,
            )?;
        }
        let secs = t.elapsed().as_secs_f64();
        runs.push(PhaseRun {
            phase: "decode".to_string(),
            n_tokens: DECODE_LEN,
            wall_secs: secs,
            tok_per_sec: DECODE_LEN as f64 / secs,
        });
        scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
    }

    model.dispose(&cluster).ok();
    drop(ar);
    drop(cluster);

    Ok(TopologyResult {
        tag: tag.to_string(),
        mesh_mode: "tp".to_string(),
        pp_size: 1,
        tp_size: world,
        devices: devices.to_vec(),
        load_secs,
        runs,
    })
}

// ─────────────────────── Hybrid path ────────────────────────────────

fn bench_hybrid(
    tag: &'static str,
    spec: HybridMeshSpec,
    devices: &[i32],
    file: &GgufFile,
) -> Result<TopologyResult> {
    let load_start = Instant::now();
    // 1. Sub-clusters via the hybrid loader (per project_hybrid_cluster_order).
    let model = Qwen3MoEHybridModel::load(file, devices, spec)
        .with_context(|| format!("{tag} hybrid load"))?;
    let load_secs = load_start.elapsed().as_secs_f64();
    // 2. Per-stage AllReduce.
    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        stage_ars.push(
            BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))
                .with_context(|| format!("{tag} stage {} AR", stage.stage_idx))?,
        );
    }
    // 3. Global cluster LAST.
    let global_cluster: Arc<HipCluster> =
        Arc::new(HipCluster::new(devices).with_context(|| format!("{tag} global cluster"))?);

    let mut runs: Vec<PhaseRun> = Vec::new();

    // **AUTO-6e4** — record both per-token and batched hybrid prefill.
    // FLAMBEAU_TP_BATCHED=1 toggles forward_prefill_hybrid_logits onto
    // the L-batched path internally (AUTO-6e3).
    let mut logits = vec![0.0f32; model.config.vocab_size];
    for &l in PREFILL_LENGTHS {
        let prompt = build_prompt(l);

        // Per-token baseline.
        std::env::remove_var("FLAMBEAU_TP_BATCHED");
        {
            let mut session = Qwen3MoEHybridSession::new(&model)?;
            let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
            let t = Instant::now();
            forward_prefill_hybrid_logits(
                &model,
                &mut scratch,
                &global_cluster,
                &stage_ars,
                &mut session,
                &prompt,
                0,
                &mut logits,
            )?;
            let secs = t.elapsed().as_secs_f64();
            runs.push(PhaseRun {
                phase: "prefill".to_string(),
                n_tokens: l,
                wall_secs: secs,
                tok_per_sec: l as f64 / secs,
            });
            scratch.dispose(&model).ok();
            session.dispose(&model).ok();
        }

        // L-batched (AUTO-6e3).
        std::env::set_var("FLAMBEAU_TP_BATCHED", "1");
        {
            let mut session = Qwen3MoEHybridSession::new(&model)?;
            let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
            let t = Instant::now();
            forward_prefill_hybrid_logits(
                &model,
                &mut scratch,
                &global_cluster,
                &stage_ars,
                &mut session,
                &prompt,
                0,
                &mut logits,
            )?;
            let secs = t.elapsed().as_secs_f64();
            runs.push(PhaseRun {
                phase: "prefill_batched".to_string(),
                n_tokens: l,
                wall_secs: secs,
                tok_per_sec: l as f64 / secs,
            });
            scratch.dispose(&model).ok();
            session.dispose(&model).ok();
        }
        std::env::remove_var("FLAMBEAU_TP_BATCHED");
    }

    {
        let mut session = Qwen3MoEHybridSession::new(&model)?;
        let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
        // 1-token "prefill"
        forward_one_token_hybrid(
            &model,
            &mut scratch,
            &global_cluster,
            &stage_ars,
            &mut session,
            SEED_TOKEN,
            0,
        )?;
        let mut last = SEED_TOKEN;
        let t = Instant::now();
        for pos in 0..DECODE_LEN {
            last = forward_one_token_hybrid(
                &model,
                &mut scratch,
                &global_cluster,
                &stage_ars,
                &mut session,
                last,
                1 + pos,
            )?;
        }
        let secs = t.elapsed().as_secs_f64();
        runs.push(PhaseRun {
            phase: "decode".to_string(),
            n_tokens: DECODE_LEN,
            wall_secs: secs,
            tok_per_sec: DECODE_LEN as f64 / secs,
        });
        scratch.dispose(&model).ok();
        session.dispose(&model).ok();
    }

    model.dispose().ok();
    drop(stage_ars);
    drop(global_cluster);

    Ok(TopologyResult {
        tag: tag.to_string(),
        mesh_mode: "pp+tp".to_string(),
        pp_size: spec.pp_size,
        tp_size: spec.tp_size,
        devices: devices.to_vec(),
        load_secs,
        runs,
    })
}

// ───────────────────────── runner ───────────────────────────────────

#[test]
#[ignore = "AUTO-5 — multi-topology bench (FLAMBEAU_TOPOLOGY_COMPARE=1, GGUF + ≥4 HIP devices). \
             Skips configurations that don't fit the rig (tp4 is omitted by design — see \
             project_rig_gpu23_link_fault). Writes certs/perf/topology_compare/qwen35_9b_q4_1.json."]
fn topology_compare_qwen35_9b_q4_1() -> Result<()> {
    if std::env::var("FLAMBEAU_TOPOLOGY_COMPARE").ok().as_deref() != Some("1") {
        eprintln!("skip — FLAMBEAU_TOPOLOGY_COMPARE=1 not set");
        return Ok(());
    }
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5-9B-Q4_1 GGUF not present");
        return Ok(());
    };
    let n_dev = device_count().unwrap_or(0);
    if n_dev < 4 {
        eprintln!("skip — need ≥ 4 HIP devices (have {n_dev})");
        return Ok(());
    }
    let file = GgufFile::open(&path)?;

    // Topologies — mesh1 / pp2 / tp2 / pp4 / pp2tp2. tp4 OMITTED
    // because the rig's {2,3} pair fails BAR1 AR (memory:
    // project_rig_gpu23_link_fault).
    let plan: &[(&'static str, &'static str)] = &[
        ("mesh1", "pp"),
        ("pp2", "pp"),
        ("tp2", "tp"),
        ("pp4", "pp"),
        ("pp2tp2", "pp+tp"),
    ];

    // Cross-topology cleanup in a single process can SIGSEGV inside the
    // HIP runtime (observed 2026-04-27 between mesh1 → pp2). Run with
    // `FLAMBEAU_TOPOLOGY_TAG=<tag>` to bench one tag at a time and
    // accumulate; each invocation merges its result into the shared
    // JSON. With no tag set, the loop tries all five and stops at the
    // first failure (still useful when only one config needs a re-run).
    let only: Option<String> = std::env::var("FLAMBEAU_TOPOLOGY_TAG").ok();

    let out_dir = workspace_root().join("certs/perf/topology_compare");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("qwen35_9b_q4_1.json");

    // Load any previous run's results so a per-tag invocation merges
    // rather than overwrites.
    let mut results: Vec<TopologyResult> = if out_path.exists() {
        match std::fs::read(&out_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<ComparisonCert>(&b).ok())
        {
            Some(prev) => prev.topologies,
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };

    for &(tag, _mode) in plan {
        if let Some(ref filter) = only {
            if filter != tag {
                continue;
            }
        }
        eprintln!("\n=== bench {tag} ===");
        let r: Result<TopologyResult> = match tag {
            "mesh1" => bench_pp(tag, 1, &[0], &file),
            "pp2" => bench_pp(tag, 2, &[0, 1], &file),
            "pp4" => bench_pp(tag, 4, &[0, 1, 2, 3], &file),
            "tp2" => bench_tp(tag, &[0, 1], &file),
            "pp2tp2" => bench_hybrid(
                tag,
                HybridMeshSpec {
                    pp_size: 2,
                    tp_size: 2,
                },
                // stage-major: stage0={0,2}, stage1={1,3} (avoids
                // {2,3} co-location).
                &[0, 2, 1, 3],
                &file,
            ),
            other => Err(anyhow!("unknown topology tag {other}")),
        };
        match r {
            Ok(res) => {
                eprintln!(
                    "  {tag} load={:.2}s; {} runs",
                    res.load_secs,
                    res.runs.len()
                );
                // Replace any prior entry for this tag, otherwise append.
                if let Some(slot) = results.iter_mut().find(|t| t.tag == tag) {
                    *slot = res;
                } else {
                    results.push(res);
                }
            }
            Err(e) => {
                eprintln!("  {tag} FAILED: {e:#}");
            }
        }
    }

    if results.is_empty() {
        bail!("no topology benches succeeded");
    }

    let cert = ComparisonCert {
        model_tag: "Qwen3.5-9B-Q4_1".to_string(),
        rig_notes: "4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} \
                    BAR1 fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses \
                    devices 0,2,1,3 to keep {2,3} out of any TP group."
            .to_string(),
        topologies: results.clone(),
    };

    let json = serde_json::to_string_pretty(&cert)?;
    std::fs::write(&out_path, json)?;
    eprintln!("\nWrote {}\n", out_path.display());

    eprintln!("\n{}", pretty_table(&results));

    Ok(())
}
