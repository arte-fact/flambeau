//! **Phase A2** — chunked-vs-single-shot KV parity test.
//!
//! Runs the same prompt twice on a freshly-built Qwen3.5-9B-Q4_1 PP4
//! session: once with scratch sized to the full prompt (single-shot
//! prefill — the production path) and once with scratch sized to half
//! the prompt (forces `forward_prefill_pp`'s recursive chunking at
//! pp.rs:1198). Snapshots per-layer KV state to host after each run
//! and asserts byte-equal.
//!
//! When the test fails, it reports the FIRST layer whose state
//! differs and whether it's the K/V buffers (full-attn) or the
//! GDN state/conv_history. That localises the cross-chunk-state bug
//! without needing to read kernel source.
//!
//! Skipped when the GGUF is missing or fewer than 4 HIP devices are
//! available. Runs with FLAMBEAU_MAX_CTX=2048 so the per-layer KV
//! buffers stay small enough to snapshot quickly.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_prefill_pp, forward_prefill_pp_logits, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{
    snapshot_layer_caches_to_host, LayerCacheSnapshot, Qwen3MoEConfig, Qwen3MoEShardedModel,
    Qwen3MoEShardedSession,
};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/artefact/models/Qwen3.5-9B-Q4_1.gguf")))
        .filter(|p| p.exists())
}

fn snapshot_pp_session(
    session: &Qwen3MoEShardedSession,
    cluster: &HipCluster,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    let mut per_rank = Vec::with_capacity(cluster.ranks());
    for rank in 0..cluster.ranks() {
        let device = cluster.device(rank);
        let layers = snapshot_layer_caches_to_host(&session.per_rank[rank].caches, device)?;
        per_rank.push(layers);
    }
    Ok(per_rank)
}

fn describe_diff(a: &LayerCacheSnapshot, b: &LayerCacheSnapshot) -> Option<String> {
    match (a, b) {
        (
            LayerCacheSnapshot::FullAttn {
                k: ka,
                v: va,
                current_tokens: cta,
            },
            LayerCacheSnapshot::FullAttn {
                k: kb,
                v: vb,
                current_tokens: ctb,
            },
        ) => {
            if cta != ctb {
                return Some(format!(
                    "FullAttn current_tokens differ: single={cta} chunked={ctb}"
                ));
            }
            if ka != kb {
                let first = (0..ka.len()).find(|i| ka[*i] != kb[*i]).unwrap_or(0);
                return Some(format!(
                    "FullAttn K differs ({} bytes); first byte differing at offset {first}",
                    ka.len()
                ));
            }
            if va != vb {
                let first = (0..va.len()).find(|i| va[*i] != vb[*i]).unwrap_or(0);
                return Some(format!(
                    "FullAttn V differs ({} bytes); first byte differing at offset {first}",
                    va.len()
                ));
            }
            None
        }
        (
            LayerCacheSnapshot::Gdn {
                state: sa,
                conv_history: ca,
            },
            LayerCacheSnapshot::Gdn {
                state: sb,
                conv_history: cb,
            },
        ) => {
            if sa != sb {
                let first = (0..sa.len()).find(|i| sa[*i] != sb[*i]).unwrap_or(0);
                return Some(format!(
                    "GDN state differs ({} bytes); first byte differing at offset {first}",
                    sa.len()
                ));
            }
            if ca != cb {
                let first = (0..ca.len()).find(|i| ca[*i] != cb[*i]).unwrap_or(0);
                return Some(format!(
                    "GDN conv_history differs ({} bytes); first byte differing at offset {first}",
                    ca.len()
                ));
            }
            None
        }
        _ => Some("layer-type mismatch (snapshots disagree on which variant)".into()),
    }
}

#[test]
fn chunked_prefill_kv_parity_pp4() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    if n_available < 4 {
        eprintln!("skip — need 4 HIP devices for PP4 test, have {n_available}");
        return Ok(());
    }

    // Cap the model's declared context so per-layer KV stays small
    // enough to snapshot quickly. 2048 is plenty for an L=128 prompt.
    std::env::set_var("FLAMBEAU_MAX_CTX", "2048");

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let cluster = HipCluster::new(&[0, 1, 2, 3])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // Choose L and chunk_size so BOTH single-shot (m=L) and chunked (m=chunk)
    // stay in the SAME MMQ dispatch bucket (m>=128 on Q4_1 → 4warp_lds kernel).
    // m<128 routes to the MMVQ kernel which differs numerically.
    // L=256 chunk=128 forces 2 chunks both at m=128.
    let l = 256usize;
    let chunk_size = 128usize;
    let prompt: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();

    let snap_a = run_prefill_logits_single_shot(&model, &cluster, &prompt, l)?;
    let snap_b = run_prefill_pp_no_chunking(&model, &cluster, &prompt, l)?;
    let snap_c = run_prefill_pp_chunked(&model, &cluster, &prompt, chunk_size)?;

    // Hypothesis check: does kernel dispatch depend on scratch.max_tokens?
    // Run L=64 with scratch=64 (D) and L=64 with scratch=128 (E). If D != E,
    // then the same call shape produces different KV depending on scratch
    // size — that's a kernel-dispatch-by-max_tokens issue and the root
    // cause of the chunking divergence.
    let prompt_short: Vec<u32> = prompt[..chunk_size].to_vec();
    let snap_d = run_prefill_pp_no_chunking(&model, &cluster, &prompt_short, chunk_size)?;
    let snap_e = run_prefill_pp_no_chunking(&model, &cluster, &prompt_short, l)?;
    // G: manual two-call WITHOUT triggering chunking. scratch = L so each
    // call is in the no-chunking branch. Tells us: is this just "2 calls
    // produce different state than 1 call" regardless of scratch?
    let snap_g = run_prefill_pp_two_manual_calls(&model, &cluster, &prompt, l, chunk_size)?;

    // Isolation table — each comparison answers a specific question.
    eprintln!("--- A vs B (forward_prefill_pp_logits L=128 scratch=128 vs forward_prefill_pp L=128 scratch=128) ---");
    let ab_diff = compare_pp_snapshots(&snap_a, &snap_b);
    eprintln!("--- B vs C (forward_prefill_pp scratch=128 NO chunking vs scratch=64 RECURSIVE chunking) ---");
    let bc_diff = compare_pp_snapshots(&snap_b, &snap_c);
    eprintln!("--- A vs C (single-shot _logits vs chunked _pp) ---");
    let ac_diff = compare_pp_snapshots(&snap_a, &snap_c);
    eprintln!("--- D vs E (L=64 scratch=64 vs L=64 scratch=128 — same call, different scratch size) ---");
    let de_diff = compare_pp_snapshots(&snap_d, &snap_e);
    eprintln!("--- B vs G (1 call L=128 vs 2 manual calls of L=64, both scratch=128 — no chunking branch) ---");
    let bg_diff = compare_pp_snapshots(&snap_b, &snap_g);
    eprintln!("--- C vs G (recursive chunking vs manual 2 calls — both make 2 calls; only difference is scratch size) ---");
    let cg_diff = compare_pp_snapshots(&snap_c, &snap_g);

    model.dispose(&cluster)?;
    cluster.dispose()?;

    // Diagnostic verdict.
    let mut diagnoses = Vec::<String>::new();
    if ab_diff > 0 {
        diagnoses.push(format!("_logits vs _pp differ at same L ({ab_diff} layers) — different code paths"));
    }
    if de_diff > 0 {
        diagnoses.push(format!(
            "scratch.max_tokens affects KV output for the SAME call ({de_diff} layers): kernel dispatch keys on max_tokens, not n_tokens — this is the chunking root cause (chunk size != single-shot's max → different kernels → different K)"
        ));
    }
    if bc_diff > 0 && de_diff == 0 {
        diagnoses.push(format!("recursive chunking diverges ({bc_diff} layers) but scratch-size alone doesn't — bug is in the recursion loop, not kernel dispatch"));
    }
    if bg_diff > 0 {
        diagnoses.push(format!(
            "MULTI-CALL ALONE diverges from single-call ({bg_diff} layer diffs at scratch=L=128) — fundamental bug in cross-call state plumbing, not specific to the chunking branch"
        ));
    }
    if bg_diff == 0 && cg_diff > 0 {
        diagnoses.push(format!(
            "Manual 2-call (scratch=128) matches single-call, but recursive chunking (scratch=64) doesn't — only difference is scratch size at call time → kernel dispatch DOES depend on max_tokens"
        ));
    }
    if !diagnoses.is_empty() {
        panic!("{}", diagnoses.join("\n"));
    }
    eprintln!("OK: all runs produce byte-equal KV across all layers");
    Ok(())
}

fn run_prefill_logits_single_shot(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    prompt: &[u32],
    l: usize,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut scratch = ShardedForwardPrefillScratch::new(model, cluster, l)?;
    let mut sink: Vec<f32> = Vec::new();
    let res = forward_prefill_pp_logits(model, &mut session, cluster, &mut scratch, prompt, 0, &mut sink);
    let snap = if res.is_ok() {
        snapshot_pp_session(&session, cluster)
    } else {
        Err(res.unwrap_err())
    };
    let _ = scratch.dispose(cluster);
    let _ = session.dispose(cluster);
    snap
}

fn run_prefill_pp_no_chunking(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    prompt: &[u32],
    l: usize,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut scratch = ShardedForwardPrefillScratch::new(model, cluster, l)?;
    let res = forward_prefill_pp(model, &mut session, cluster, &mut scratch, prompt, 0);
    let snap = if res.is_ok() {
        snapshot_pp_session(&session, cluster)
    } else {
        Err(res.unwrap_err())
    };
    let _ = scratch.dispose(cluster);
    let _ = session.dispose(cluster);
    snap
}

fn run_prefill_pp_two_manual_calls(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    prompt: &[u32],
    scratch_size: usize,
    chunk: usize,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut scratch = ShardedForwardPrefillScratch::new(model, cluster, scratch_size)?;
    let mut start = 0usize;
    let mut last_err: Option<anyhow::Error> = None;
    while start < prompt.len() {
        let end = (start + chunk).min(prompt.len());
        if let Err(e) = forward_prefill_pp(
            model,
            &mut session,
            cluster,
            &mut scratch,
            &prompt[start..end],
            start,
        ) {
            last_err = Some(e);
            break;
        }
        start = end;
    }
    let snap = if last_err.is_none() {
        snapshot_pp_session(&session, cluster)
    } else {
        Err(last_err.unwrap())
    };
    let _ = scratch.dispose(cluster);
    let _ = session.dispose(cluster);
    snap
}

fn run_prefill_pp_chunked(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    prompt: &[u32],
    chunk_size: usize,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    let mut session = Qwen3MoEShardedSession::new(model, cluster)?;
    let mut scratch = ShardedForwardPrefillScratch::new(model, cluster, chunk_size)?;
    let res = forward_prefill_pp(model, &mut session, cluster, &mut scratch, prompt, 0);
    let snap = if res.is_ok() {
        snapshot_pp_session(&session, cluster)
    } else {
        Err(res.unwrap_err())
    };
    let _ = scratch.dispose(cluster);
    let _ = session.dispose(cluster);
    snap
}

fn compare_pp_snapshots(
    a: &[Vec<LayerCacheSnapshot>],
    b: &[Vec<LayerCacheSnapshot>],
) -> usize {
    let mut diffs = 0;
    for (rank_idx, (sr, cr)) in a.iter().zip(b.iter()).enumerate() {
        for (layer_local, (sa, cb)) in sr.iter().zip(cr.iter()).enumerate() {
            if let Some(msg) = describe_diff(sa, cb) {
                diffs += 1;
                if diffs <= 4 {
                    eprintln!("  rank {rank_idx} local-layer {layer_local}: {msg}");
                }
            }
        }
    }
    if diffs > 4 {
        eprintln!("  ... ({} more diffs not shown)", diffs - 4);
    }
    eprintln!("  total layer diffs: {diffs}");
    diffs
}
