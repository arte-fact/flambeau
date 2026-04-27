//! V1-BENCH-S1 — generic multi-model topology + sweep bench harness.
//!
//! Sister of `topology_compare_qwen35_9b.rs` but parametrized: takes a
//! GGUF path + tag from env and writes a per-model cert. The same five
//! topologies get exercised (mesh1 / pp2 / tp2 / pp4 / pp2tp2; tp4 is
//! intentionally omitted per the rig {2,3} BAR1 fault), and each
//! topology runs:
//!
//!   * **Prefill at multiple L** ([`PREFILL_LENGTHS`]) — both per-token
//!     baseline and AUTO-6 batched (TP-based topologies only).
//!   * **Decode at multiple tg lengths** ([`DECODE_LENGTHS`]).
//!
//! Defaults sweep `[8, 32, 128, 512, 1024, 2048, 4096]` for prefill and
//! `[16, 64, 256]` for decode. Override via env:
//!
//!   * `FLAMBEAU_BENCH_GGUF=<path>`            — required
//!   * `FLAMBEAU_BENCH_TAG=<stem>`             — required (cert filename)
//!   * `FLAMBEAU_BENCH_PREFILL_LENGTHS=8,64,…` — comma-separated (optional)
//!   * `FLAMBEAU_BENCH_DECODE_LENGTHS=16,64,…` — comma-separated (optional)
//!   * `FLAMBEAU_BENCH_TOPOLOGY_TAG=<tag>`     — restrict to one topology
//!   * `FLAMBEAU_BENCH_SKIP_TP=1`              — skip TP+batched (parity-fragile arches)
//!
//! Cert output:
//!   * `certs/perf/v1_bench_matrix/<tag>.json`
//!   * `certs/perf/v1_bench_matrix/<tag>_summary.md`

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

const DEFAULT_PREFILL: &[usize] = &[8, 32, 128, 512, 1024, 2048, 4096];
const DEFAULT_DECODE: &[usize] = &[16, 64, 256];
const SEED_TOKEN: u32 = 9419;

fn parse_lengths(env: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(env) {
        Ok(s) => s
            .split(',')
            .filter_map(|p| p.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .collect(),
        Err(_) => default.to_vec(),
    }
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
struct BenchCert {
    model_tag: String,
    gguf_path: String,
    rig_notes: String,
    prefill_lengths: Vec<usize>,
    decode_lengths: Vec<usize>,
    topologies: Vec<TopologyResult>,
}

fn build_prompt(len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| if i == 0 { SEED_TOKEN } else { 1 + i })
        .collect()
}

fn pretty_table(
    results: &[TopologyResult],
    prefill: &[usize],
    decode: &[usize],
) -> String {
    let mut out = String::new();
    out.push_str("topology         pp tp  load_s  ");
    for &l in prefill {
        out.push_str(&format!("pp{l:<5} "));
    }
    for &l in decode {
        out.push_str(&format!("tg{l:<5} "));
    }
    out.push('\n');
    for r in results {
        out.push_str(&format!(
            "{:<16} {:>2} {:>2}  {:>6.2}  ",
            r.tag, r.pp_size, r.tp_size, r.load_secs
        ));
        for &l in prefill {
            let v = r
                .runs
                .iter()
                .find(|p| p.phase == "prefill" && p.n_tokens == l)
                .map(|p| p.tok_per_sec)
                .unwrap_or(0.0);
            out.push_str(&format!("{v:>7.1} "));
        }
        for &l in decode {
            let v = r
                .runs
                .iter()
                .find(|p| p.phase == "decode" && p.n_tokens == l)
                .map(|p| p.tok_per_sec)
                .unwrap_or(0.0);
            out.push_str(&format!("{v:>7.1} "));
        }
        out.push('\n');

        // Legacy: prior runs recorded both `prefill` (per-token) and
        // `prefill_batched` for TP-based topologies. Batched is now the
        // default so we no longer emit a second row.
    }
    out
}

// ───────────────────────── PP path ──────────────────────────────────

fn bench_pp(
    tag: &'static str,
    pp_size: u32,
    devices: &[i32],
    file: &GgufFile,
    prefill_lengths: &[usize],
    decode_lengths: &[usize],
) -> Result<TopologyResult> {
    let cluster: Arc<HipCluster> = Arc::new(HipCluster::new(devices)?);
    let cfg = Qwen3MoEConfig::from_gguf(file)?;

    let load_start = Instant::now();
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(file, &cluster, &assignment)
        .with_context(|| format!("{tag} sharded load"))?;
    let load_secs = load_start.elapsed().as_secs_f64();

    // V1-BENCH-#111 — opt-in scratch sizing. By default sized for the
    // current L (no chunking). Set FLAMBEAU_BENCH_SCRATCH_TOKENS to a
    // smaller value to force `forward_prefill_pp` into its
    // chunked-ubatch path (tight-VRAM regime).
    let scratch_cap_override: Option<usize> = std::env::var("FLAMBEAU_BENCH_SCRATCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n >= 1);

    let mut runs: Vec<PhaseRun> = Vec::new();
    for &l in prefill_lengths {
        let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
        let scratch_cap = scratch_cap_override.unwrap_or(l).min(l);
        let mut scratch = ShardedForwardPrefillScratch::new(&model, &cluster, scratch_cap)?;
        let prompt = build_prompt(l);
        let t = Instant::now();
        forward_prefill_pp(&model, &mut session, &cluster, &mut scratch, &prompt, 0)
            .with_context(|| format!("{tag} pp prefill L={l}"))?;
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

    for &tg in decode_lengths {
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
        for pos in 0..tg {
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
            n_tokens: tg,
            wall_secs: secs,
            tok_per_sec: tg as f64 / secs,
        });
        decode_scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
        let _ = last;
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

fn bench_tp(
    tag: &'static str,
    devices: &[i32],
    file: &GgufFile,
    prefill_lengths: &[usize],
    decode_lengths: &[usize],
) -> Result<TopologyResult> {
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
    let mut logits = vec![0.0f32; cfg.vocab_size];

    // Batched prefill is always-on for TP topologies (AUTO-6 default).
    std::env::set_var("FLAMBEAU_TP_BATCHED", "1");
    for &l in prefill_lengths {
        let prompt = build_prompt(l);
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

    for &tg in decode_lengths {
        let mut session = Qwen3MoETpSession::new(&model, &cluster)?;
        let mut scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;
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
        for pos in 0..tg {
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
            n_tokens: tg,
            wall_secs: secs,
            tok_per_sec: tg as f64 / secs,
        });
        scratch.dispose(&cluster).ok();
        session.dispose(&cluster).ok();
        let _ = last;
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

// ───────────────────────── Hybrid path ──────────────────────────────

fn bench_hybrid(
    tag: &'static str,
    spec: HybridMeshSpec,
    devices: &[i32],
    file: &GgufFile,
    prefill_lengths: &[usize],
    decode_lengths: &[usize],
) -> Result<TopologyResult> {
    let load_start = Instant::now();
    let model = Qwen3MoEHybridModel::load(file, devices, spec)
        .with_context(|| format!("{tag} hybrid load"))?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(model.stages.len());
    for stage in &model.stages {
        stage_ars.push(
            BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))
                .with_context(|| format!("{tag} stage {} AR", stage.stage_idx))?,
        );
    }
    let global_cluster: Arc<HipCluster> =
        Arc::new(HipCluster::new(devices).with_context(|| format!("{tag} global cluster"))?);

    let mut runs: Vec<PhaseRun> = Vec::new();
    let mut logits = vec![0.0f32; model.config.vocab_size];

    // Batched hybrid prefill is the default (AUTO-6e).
    std::env::set_var("FLAMBEAU_TP_BATCHED", "1");
    for &l in prefill_lengths {
        let prompt = build_prompt(l);
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

    for &tg in decode_lengths {
        let mut session = Qwen3MoEHybridSession::new(&model)?;
        let mut scratch = ShardedForwardOneTokenScratchHybrid::new(&model)?;
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
        for pos in 0..tg {
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
            n_tokens: tg,
            wall_secs: secs,
            tok_per_sec: tg as f64 / secs,
        });
        scratch.dispose(&model).ok();
        session.dispose(&model).ok();
        let _ = last;
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
#[ignore = "V1-BENCH-S1 — generic multi-model topology + sweep bench. Run with \
             FLAMBEAU_BENCH_GGUF=<path> FLAMBEAU_BENCH_TAG=<stem> + ≥4 HIP devices. \
             tp4 omitted (rig {2,3} BAR1 fault). Writes \
             certs/perf/v1_bench_matrix/<tag>.{json,_summary.md}."]
fn v1_bench_matrix() -> Result<()> {
    let path = match std::env::var("FLAMBEAU_BENCH_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skip — FLAMBEAU_BENCH_GGUF not set");
            return Ok(());
        }
    };
    if !path.exists() {
        eprintln!("skip — GGUF not present at {}", path.display());
        return Ok(());
    }
    let tag = std::env::var("FLAMBEAU_BENCH_TAG")
        .map_err(|_| anyhow!("FLAMBEAU_BENCH_TAG required"))?;
    let n_dev = device_count().unwrap_or(0);
    if n_dev < 4 {
        eprintln!("skip — need ≥4 HIP devices, found {n_dev}");
        return Ok(());
    }

    let prefill_lengths = parse_lengths("FLAMBEAU_BENCH_PREFILL_LENGTHS", DEFAULT_PREFILL);
    let decode_lengths = parse_lengths("FLAMBEAU_BENCH_DECODE_LENGTHS", DEFAULT_DECODE);
    let only: Option<String> = std::env::var("FLAMBEAU_BENCH_TOPOLOGY_TAG").ok();
    let skip_tp = std::env::var("FLAMBEAU_BENCH_SKIP_TP").is_ok();

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    eprintln!(
        "=== v1_bench_matrix === tag={tag} arch={} hidden={} layers={} vocab={} \
         path={}",
        cfg.arch,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.vocab_size,
        path.display()
    );
    eprintln!(
        "  prefill_lengths={:?}  decode_lengths={:?}  skip_tp={skip_tp}",
        prefill_lengths, decode_lengths
    );

    let topologies: Vec<(&'static str, &'static str)> = vec![
        ("mesh1", "pp"),
        ("pp2", "pp"),
        ("tp2", "tp"),
        ("pp4", "pp"),
        ("pp2tp2", "pp+tp"),
    ];

    let mut results: Vec<TopologyResult> = Vec::new();
    for (top_tag, mesh_mode) in topologies {
        if let Some(ref o) = only {
            if o != top_tag {
                continue;
            }
        }
        if skip_tp && (mesh_mode == "tp" || mesh_mode == "pp+tp") {
            eprintln!("skip {top_tag} (FLAMBEAU_BENCH_SKIP_TP)");
            continue;
        }
        eprintln!("\n=== bench {top_tag} ===");
        let r = match top_tag {
            "mesh1" => bench_pp(top_tag, 1, &[0], &file, &prefill_lengths, &decode_lengths),
            "pp2" => bench_pp(top_tag, 2, &[0, 1], &file, &prefill_lengths, &decode_lengths),
            "pp4" => bench_pp(
                top_tag,
                4,
                &[0, 1, 2, 3],
                &file,
                &prefill_lengths,
                &decode_lengths,
            ),
            "tp2" => bench_tp(top_tag, &[0, 1], &file, &prefill_lengths, &decode_lengths),
            "pp2tp2" => bench_hybrid(
                top_tag,
                HybridMeshSpec {
                    pp_size: 2,
                    tp_size: 2,
                },
                &[0, 2, 1, 3],
                &file,
                &prefill_lengths,
                &decode_lengths,
            ),
            other => bail!("unknown topology {other}"),
        };
        match r {
            Ok(r) => {
                eprintln!("  {} load={:.2}s; {} runs", r.tag, r.load_secs, r.runs.len());
                results.push(r);
            }
            Err(e) => {
                eprintln!("  {} FAILED: {e:#}", top_tag);
                // Continue — let the other topologies run; OOM on big
                // models is expected.
            }
        }
    }

    // ── Cert write ───────────────────────────────────────────────────
    let out_dir = workspace_root().join("certs/perf/v1_bench_matrix");
    std::fs::create_dir_all(&out_dir)?;
    let out_json = out_dir.join(format!("{tag}.json"));
    let out_md = out_dir.join(format!("{tag}_summary.md"));

    // Merge with existing cert if present (so per-topology runs accumulate).
    let mut cert: BenchCert = if out_json.exists() {
        let bytes = std::fs::read(&out_json)?;
        match serde_json::from_slice(&bytes) {
            Ok(c) => c,
            Err(_) => BenchCert {
                model_tag: tag.clone(),
                gguf_path: path.display().to_string(),
                rig_notes: rig_notes_default(),
                prefill_lengths: prefill_lengths.clone(),
                decode_lengths: decode_lengths.clone(),
                topologies: Vec::new(),
            },
        }
    } else {
        BenchCert {
            model_tag: tag.clone(),
            gguf_path: path.display().to_string(),
            rig_notes: rig_notes_default(),
            prefill_lengths: prefill_lengths.clone(),
            decode_lengths: decode_lengths.clone(),
            topologies: Vec::new(),
        }
    };
    // Replace any pre-existing entries for the topologies we just ran.
    let new_tags: std::collections::HashSet<String> =
        results.iter().map(|r| r.tag.clone()).collect();
    cert.topologies.retain(|t| !new_tags.contains(&t.tag));
    cert.topologies.extend(results.iter().cloned());
    cert.topologies.sort_by_key(|t| match t.tag.as_str() {
        "mesh1" => 0,
        "pp2" => 1,
        "tp2" => 2,
        "pp4" => 3,
        "pp2tp2" => 4,
        _ => 99,
    });
    cert.prefill_lengths = prefill_lengths.clone();
    cert.decode_lengths = decode_lengths.clone();
    let json = serde_json::to_string_pretty(&cert)?;
    std::fs::write(&out_json, json)?;
    let table = pretty_table(&cert.topologies, &prefill_lengths, &decode_lengths);
    eprintln!("\nWrote {}\n\n\n{}", out_json.display(), table);
    let md = format!(
        "# {} — V1-BENCH-S1 sweep\n\n\
         GGUF: `{}`  \n\
         Rig: {}\n\n\
         Prefill lengths: {:?}  \n\
         Decode lengths: {:?}\n\n\
         ```\n{}```\n",
        tag,
        path.display(),
        rig_notes_default(),
        prefill_lengths,
        decode_lengths,
        table
    );
    std::fs::write(&out_md, md)?;
    Ok(())
}

fn rig_notes_default() -> String {
    "4× MI50 PCIe 3.0 x16, 100 W cap, ROCm 7.1.1; tp4 omitted (rig {2,3} BAR1 \
     fault per project_rig_gpu23_link_fault). Hybrid pp2tp2 uses devices \
     0,2,1,3 to keep {2,3} out of any TP group."
        .to_string()
}
