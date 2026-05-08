//! **#242 Phase B2** — chunked-prefill-then-decode token-id parity test (PP).
//! Phase A2/A3/B1 (`chunked_prefill_kv_parity.rs`) verified that chunked
//! prefill leaves the per-layer KV state byte-equal to single-shot
//! prefill. Phase B2 closes the next gap: that the decode loop driven
//! from a chunked-prefill session produces the same token-id sequence
//! as decode from a single-shot prefill session.
//! KV parity is necessary but not sufficient — even if KV bytes
//! match, divergent scratch contents, residual stream snapshots, or
//! position-tracking off-by-ones inside `forward_one_token_pp` could
//! make decode disagree. This test runs prefill + 16 greedy decodes
//! through both paths and asserts the 16-token sequences are
//! identical.
//! Skipped when the GGUF is missing or fewer than 4 HIP devices are
//! available. Runs under `FLAMBEAU_MAX_CTX=2048`.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, forward_prefill_pp_logits,
    ShardedForwardOneTokenScratch, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use std::path::PathBuf;

fn gguf_path() -> Option<PathBuf> {
    std::env::var("FLAMBEAU_QWEN35_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/artefact/models/Qwen3.5-9B-Q4_1.gguf")))
        .filter(|p| p.exists())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

/// Single-shot prefill (scratch sized to L) + greedy decode loop.
/// Returns the `n_decode`-token sequence, the first token coming from
/// the last-prompt-position prefill logits.
fn run_single_shot_then_decode(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    prompt: &[u32],
    n_decode: usize,
) -> Result<Vec<u32>> {
    let l = prompt.len();
    let mut session = Qwen3MoEShardedSession::new(model, cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(model, cluster, l)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(model, cluster)?;

    let mut prefill_logits: Vec<f32> = Vec::new();
    let prefill_res = forward_prefill_pp_logits(
        model,
        &mut session,
        cluster,
        &mut prefill_scratch,
        prompt,
        0,
        &mut prefill_logits,
    );
    let result = match prefill_res {
        Ok(()) => decode_loop(
            model,
            &mut session,
            cluster,
            &mut decode_scratch,
            argmax(&prefill_logits),
            l,
            n_decode,
        ),
        Err(e) => Err(e),
    };
    let _ = prefill_scratch.dispose(cluster);
    let _ = decode_scratch.dispose(cluster);
    let _ = session.dispose(cluster);
    result
}

/// Chunked prefill (scratch sized to chunk_size, mirrors the
/// production `prefill_logits` PP arm: forward_prefill_pp for non-
/// final chunks, forward_prefill_pp_logits for the final chunk) +
/// greedy decode loop.
fn run_chunked_then_decode(
    model: &Qwen3MoEShardedModel,
    cluster: &HipCluster,
    prompt: &[u32],
    chunk_size: usize,
    n_decode: usize,
) -> Result<Vec<u32>> {
    let l = prompt.len();
    let mut session = Qwen3MoEShardedSession::new(model, cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut prefill_scratch = ShardedForwardPrefillScratch::new(model, cluster, chunk_size)?;
    let mut decode_scratch = ShardedForwardOneTokenScratch::new(model, cluster)?;

    let result = (|| -> Result<Vec<u32>> {
        let mut start = 0usize;
        let mut prefill_logits: Vec<f32> = Vec::new();
        while start < l {
            let end = (start + chunk_size).min(l);
            let is_last = end == l;
            if is_last {
                forward_prefill_pp_logits(
                    model,
                    &mut session,
                    cluster,
                    &mut prefill_scratch,
                    &prompt[start..end],
                    start,
                    &mut prefill_logits,
                )?;
            } else {
                forward_prefill_pp(
                    model,
                    &mut session,
                    cluster,
                    &mut prefill_scratch,
                    &prompt[start..end],
                    start,
                )?;
            }
            start = end;
        }
        let first_next = argmax(&prefill_logits);
        decode_loop(
            model,
            &mut session,
            cluster,
            &mut decode_scratch,
            first_next,
            l,
            n_decode,
        )
    })();

    let _ = prefill_scratch.dispose(cluster);
    let _ = decode_scratch.dispose(cluster);
    let _ = session.dispose(cluster);
    result
}

fn decode_loop(
    model: &Qwen3MoEShardedModel,
    session: &mut Qwen3MoEShardedSession,
    cluster: &HipCluster,
    decode_scratch: &mut ShardedForwardOneTokenScratch,
    first_next: u32,
    prompt_len: usize,
    n_decode: usize,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(n_decode);
    tokens.push(first_next);
    let mut tok = first_next;
    let mut pos = prompt_len;
    for _ in 1..n_decode {
        let next = forward_one_token_pp(model, session, cluster, decode_scratch, tok, pos)?;
        tokens.push(next);
        tok = next;
        pos += 1;
    }
    Ok(tokens)
}

#[test]
fn chunked_prefill_decode_token_parity_pp4() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    if n_available < 4 {
        eprintln!("skip — need 4 HIP devices for PP4 test, have {n_available}");
        return Ok(());
    }

    std::env::set_var("FLAMBEAU_MAX_CTX", "2048");

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let cluster = HipCluster::new(&[0, 1, 2, 3])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;

    // L=256, chunk=128 → 2 chunks both at the same MMQ dispatch
    // bucket as the L=256 single-shot (both m>=128 → 4warp_lds).
    // n_decode=16 catches a divergence in the first ~half second of
    // generation; greedy means any drift is bit-deterministic.
    let l = 256usize;
    let chunk = 128usize;
    let n_decode = 16usize;
    let prompt: Vec<u32> = (0..l as u32).map(|i| (1 + i * 37) % 151000).collect();

    let toks_a = run_single_shot_then_decode(&model, &cluster, &prompt, n_decode)?;
    let toks_b = run_chunked_then_decode(&model, &cluster, &prompt, chunk, n_decode)?;

    eprintln!("single-shot decode: {:?}", toks_a);
    eprintln!("chunked    decode: {:?}", toks_b);

    let mut first_diff: Option<usize> = None;
    for (i, (a, b)) in toks_a.iter().zip(toks_b.iter()).enumerate() {
        if a != b {
            first_diff = Some(i);
            break;
        }
    }

    model.dispose(&cluster)?;
    cluster.dispose()?;

    if let Some(idx) = first_diff {
        panic!(
            "chunked vs single-shot decode diverge at token {idx}: \
             single={} chunked={} (full A={:?} B={:?})",
            toks_a[idx], toks_b[idx], toks_a, toks_b
        );
    }
    eprintln!(
        "OK: chunked + decode matches single-shot + decode for {n_decode} tokens"
    );
    Ok(())
}
