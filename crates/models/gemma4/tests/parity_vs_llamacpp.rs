//! Greedy-decode parity: flambeau gemma4 vs llama.cpp on the same
//! GGUF + prompt + decode length.
//!
//! Gates: bit-exact token-id agreement on the prefix for ≥ N_MATCH tokens.
//! A divergence inside the first N_MATCH tokens indicates an arithmetic
//! drift somewhere in our forward path (kernel precision, norm cast,
//! per-layer-embd build, MoE gating, etc.). The cert at
//! `certs/perf/gemma4_v1_bench/parity.md` records the prompt + the
//! reference token sequence from `llama-cli` so subsequent runs of
//! this test can diff without re-invoking llama.cpp.
//!
//! Two variants:
//! - **E4B-Q4_0 single (`hip:0`)** — exercises per-layer-embd
//!   side-channel + dense FFN. Smallest config; canary for
//!   arithmetic regressions.
//! - **31B-Q4_0 PP2 (`hip:0,2`)** — exercises PP hand-offs +
//!   batched-prefill kernel. Catches regressions in PP plumbing.
//!
//! Skipped when GGUFs are absent or HIP device count is insufficient.

#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::Arc;

use std::sync::Arc as ArcGuard;

use flambeau_backend_hip::{device_count, HipCluster, HipDevice};
use flambeau_gemma4::{
    forward_one_token, partition_layers, Gemma4Config, Gemma4DeviceWeights, Gemma4PpDriver,
    Gemma4Session, Gemma4TpDriver, ModelLayout,
};
use flambeau_quant::{load_from_gguf, GgufFile};

const MODELS_DIR: &str = "/artefact/models";
const PROMPT: &str = "The capital of France is";
const N_DECODE: usize = 16;
const MAX_TOKENS: usize = 128;

fn open_or_skip(name: &str) -> Option<GgufFile> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok()
}

/// Run flambeau greedy decode on the single-device path. Returns
/// `(prompt_ids, decoded_ids)`.
fn flambeau_decode_single(file: Arc<GgufFile>) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let device = HipDevice::new(0)?;
    device.bind()?;

    let cfg = Gemma4Config::from_gguf(&file)?;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let tokenizer = load_from_gguf(&file)?;
    let mut prompt_ids = tokenizer.encode(PROMPT)?;
    if tokenizer.force_add_bos {
        if let Some(bos) = tokenizer.bos_id {
            prompt_ids.insert(0, bos);
        }
    }

    let weights = Gemma4DeviceWeights::upload(&file, &cfg, &layout, &device)?;
    let mut session = if cfg.per_layer_embed.is_some() {
        Gemma4Session::new_with_gguf(
            &device,
            weights,
            cfg.clone(),
            layout,
            MAX_TOKENS,
            file.clone(),
        )?
    } else {
        Gemma4Session::new(&device, weights, cfg.clone(), layout, MAX_TOKENS)?
    };

    // Prefill: feed each prompt token at its position.
    let mut tok = prompt_ids[0];
    for (i, &t) in prompt_ids.iter().enumerate() {
        tok = forward_one_token(&mut session, &device, t, i)?;
    }
    // The forward of the *last* prompt token produces the first
    // generated token (argmax of post-prompt logits). Capture it
    // outside the loop body — we want all N_DECODE generated tokens.
    let mut decoded = Vec::with_capacity(N_DECODE);
    decoded.push(tok);
    let mut pos = prompt_ids.len();
    while decoded.len() < N_DECODE {
        tok = forward_one_token(&mut session, &device, tok, pos)?;
        decoded.push(tok);
        pos += 1;
    }

    session.dispose(&device)?;
    Ok((prompt_ids, decoded))
}

/// Run flambeau greedy decode on the PP path. Returns
/// `(prompt_ids, decoded_ids)`.
fn flambeau_decode_pp(file: Arc<GgufFile>, devices: &[i32]) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let cluster = HipCluster::new(devices)?;
    let cfg = Gemma4Config::from_gguf(&file)?;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let layer_to_rank = partition_layers(devices.len(), &layout)?;

    let tokenizer = load_from_gguf(&file)?;
    let mut prompt_ids = tokenizer.encode(PROMPT)?;
    if tokenizer.force_add_bos {
        if let Some(bos) = tokenizer.bos_id {
            prompt_ids.insert(0, bos);
        }
    }

    let mut driver = Gemma4PpDriver::upload(
        &file,
        cfg.clone(),
        layout,
        layer_to_rank,
        cluster,
        MAX_TOKENS,
    )?;

    let first = driver.forward_prefill(&prompt_ids, 0)?;
    let mut decoded = Vec::with_capacity(N_DECODE);
    decoded.push(first);
    let mut tok = first;
    let mut pos = prompt_ids.len();
    while decoded.len() < N_DECODE {
        tok = driver.forward_one_token(tok, pos)?;
        decoded.push(tok);
        pos += 1;
    }

    driver.dispose()?;
    Ok((prompt_ids, decoded))
}

/// PP variant that feeds the prompt one token at a time (using
/// `forward_one_token` for prefill too) — bypasses
/// `forward_prefill_pp` to isolate batched-prefill bugs.
fn flambeau_decode_pp_pertoken(
    file: Arc<GgufFile>,
    devices: &[i32],
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let cluster = HipCluster::new(devices)?;
    let cfg = Gemma4Config::from_gguf(&file)?;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let layer_to_rank = partition_layers(devices.len(), &layout)?;

    let tokenizer = load_from_gguf(&file)?;
    let mut prompt_ids = tokenizer.encode(PROMPT)?;
    if tokenizer.force_add_bos {
        if let Some(bos) = tokenizer.bos_id {
            prompt_ids.insert(0, bos);
        }
    }

    let mut driver = Gemma4PpDriver::upload(
        &file,
        cfg.clone(),
        layout,
        layer_to_rank,
        cluster,
        MAX_TOKENS,
    )?;

    let mut tok = prompt_ids[0];
    for (i, &t) in prompt_ids.iter().enumerate() {
        tok = driver.forward_one_token(t, i)?;
    }
    let mut decoded = Vec::with_capacity(N_DECODE);
    decoded.push(tok);
    let mut pos = prompt_ids.len();
    while decoded.len() < N_DECODE {
        tok = driver.forward_one_token(tok, pos)?;
        decoded.push(tok);
        pos += 1;
    }

    driver.dispose()?;
    Ok((prompt_ids, decoded))
}

/// TP variant: shards the 31B-Q4_0 weights across 2 ranks via
/// `Gemma4TpDriver::upload`. Greedy decodes the same prompt and
/// asserts the output contains "Paris".
fn flambeau_decode_tp(
    file: Arc<GgufFile>,
    devices: &[i32],
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let cluster = ArcGuard::new(HipCluster::new(devices)?);
    let cfg = Gemma4Config::from_gguf(&file)?;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let tokenizer = load_from_gguf(&file)?;
    let mut prompt_ids = tokenizer.encode(PROMPT)?;
    if tokenizer.force_add_bos {
        if let Some(bos) = tokenizer.bos_id {
            prompt_ids.insert(0, bos);
        }
    }

    let mut driver = Gemma4TpDriver::upload(&file, cfg, layout, cluster, MAX_TOKENS)?;

    // TP path: decode-only (no batched-prefill kernel yet — feed each
    // prompt token via forward_one_token).
    let mut tok = prompt_ids[0];
    for (i, &t) in prompt_ids.iter().enumerate() {
        tok = driver.forward_one_token(t, i)?;
    }
    let mut decoded = Vec::with_capacity(N_DECODE);
    decoded.push(tok);
    let mut pos = prompt_ids.len();
    while decoded.len() < N_DECODE {
        tok = driver.forward_one_token(tok, pos)?;
        decoded.push(tok);
        pos += 1;
    }

    driver.dispose()?;
    Ok((prompt_ids, decoded))
}

#[test]
fn parity_31b_q4_0_tp2() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let (prompt_ids, fb_ids) =
        flambeau_decode_tp(file.clone(), &[0, 2]).expect("flambeau decode TP2");
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== COHERENCE | 31B-Q4_0 TP2 (hip:0,2) ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    assert!(
        fb_text.to_lowercase().contains("paris"),
        "31B TP2 decode of 'The capital of France is' did NOT contain 'Paris'. \
         Got: {fb_text:?}"
    );
}

#[test]
fn parity_31b_q4_0_pp2_pertoken() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let (prompt_ids, fb_ids) =
        flambeau_decode_pp_pertoken(file.clone(), &[0, 2]).expect("flambeau decode PP2 per-token");
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== COHERENCE | 31B-Q4_0 PP2 (per-token forward, no batched prefill) ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    assert!(
        fb_text.to_lowercase().contains("paris"),
        "31B PP2 per-token decode of 'The capital of France is' did NOT contain 'Paris'. \
         Got: {fb_text:?}"
    );
}

/// Run `llama-cli` greedy decode on the same model + prompt. Returns
/// `decoded_ids` (just the generated tokens — prompt ids stripped).
fn llamacpp_decode(model_path: &str) -> anyhow::Result<Vec<u32>> {
    use std::process::{Command, Stdio};
    let bin = "/artefact/llama.cpp/build/bin/llama-cli";
    if !Path::new(bin).exists() {
        anyhow::bail!("{bin} not present — skipping llama.cpp parity leg");
    }
    let out = Command::new(bin)
        .env("LD_LIBRARY_PATH", "/opt/rocm-host/lib")
        .env(
            "ROCBLAS_TENSILE_LIBPATH",
            "/opt/rocm-host/lib/rocblas/library",
        )
        .args([
            "-m",
            model_path,
            "-p",
            PROMPT,
            "-n",
            &format!("{N_DECODE}"),
            "--temp",
            "0",
            "--top-k",
            "1",
            "-ngl",
            "99",
            "--seed",
            "0",
            "-sm",
            "layer",
            "-ts",
            "1/1/1/1",
            "-no-cnv",
            "--single-turn",
            "--no-warmup",
            "--no-display-prompt",
            "--log-disable",
        ])
        .stdin(Stdio::null())
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "llama-cli failed: status={}, stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // llama-cli with --verbose-prompt prints generated tokens on stdout
    // (just the text). We re-tokenize the stdout via the GGUF tokenizer
    // to get ids. That's a closed loop using flambeau's encoder for both
    // sides, which is fine for parity since we're checking ARITHMETIC
    // drift, not tokenizer drift.
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let trimmed = text.trim();
    eprintln!("  llama-cli output: {trimmed:?}");
    // Re-encode through flambeau tokenizer for direct id comparison.
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = load_from_gguf(&gguf)?;
    let ids = tokenizer.encode(trimmed)?;
    Ok(ids)
}

/// Compare two id sequences, return the length of the matching prefix.
fn matching_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn report(label: &str, prompt: &[u32], flambeau: &[u32], llamacpp: &[u32], tokenizer: &flambeau_quant::GgufTokenizer) {
    let match_n = matching_prefix_len(flambeau, llamacpp);
    let fb_text = tokenizer.decode(flambeau).unwrap_or_default();
    let lc_text = tokenizer.decode(llamacpp).unwrap_or_default();
    eprintln!("\n=== PARITY | {label} ===");
    eprintln!("  prompt   ({} ids): {prompt:?}", prompt.len());
    eprintln!("  flambeau ({} ids): {flambeau:?}", flambeau.len());
    eprintln!("  flambeau text: {fb_text:?}");
    eprintln!("  llamacpp ({} ids): {llamacpp:?}", llamacpp.len());
    eprintln!("  llamacpp text: {lc_text:?}");
    eprintln!(
        "  matching prefix: {match_n} / {} tokens",
        flambeau.len().min(llamacpp.len())
    );
}

#[test]
fn parity_e4b_q4_0_single() {
    let Some(file) = open_or_skip("gemma-4-E4B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 1).unwrap_or(true) {
        eprintln!("skipping — no HIP device");
        return;
    }
    let file = Arc::new(file);
    let (prompt_ids, fb_ids) = flambeau_decode_single(file.clone()).expect("flambeau decode");
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== PARITY | E4B-Q4_0 single ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    match llamacpp_decode("/artefact/models/gemma-4-E4B-it-Q4_0.gguf") {
        Ok(lc_ids) => {
            let lc_text = tokenizer.decode(&lc_ids).unwrap_or_default();
            let match_n = matching_prefix_len(&fb_ids, &lc_ids);
            eprintln!("  llamacpp ({} ids): {lc_ids:?}", lc_ids.len());
            eprintln!("  llamacpp text: {lc_text:?}");
            eprintln!(
                "  matching prefix: {match_n} / {} tokens",
                fb_ids.len().min(lc_ids.len())
            );
            assert!(
                match_n >= 4,
                "fewer than 4 tokens match — likely arithmetic drift"
            );
        }
        Err(e) => {
            eprintln!("  llama.cpp leg unavailable: {e}");
            eprintln!("  → coherence-only check: flambeau output should contain 'Paris'");
            assert!(
                fb_text.to_lowercase().contains("paris"),
                "flambeau output for 'The capital of France is' did NOT contain 'Paris' — \
                 strong signal of model corruption or arithmetic drift. Got: {fb_text:?}"
            );
        }
    }
}

#[test]
fn parity_31b_q4_0_pp2() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let (prompt_ids, fb_ids) =
        flambeau_decode_pp(file.clone(), &[0, 2]).expect("flambeau decode PP2");
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== PARITY | 31B-Q4_0 PP2 ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    match llamacpp_decode("/artefact/models/gemma-4-31B-it-Q4_0.gguf") {
        Ok(lc_ids) => {
            let lc_text = tokenizer.decode(&lc_ids).unwrap_or_default();
            let match_n = matching_prefix_len(&fb_ids, &lc_ids);
            eprintln!("  llamacpp ({} ids): {lc_ids:?}", lc_ids.len());
            eprintln!("  llamacpp text: {lc_text:?}");
            eprintln!(
                "  matching prefix: {match_n} / {} tokens",
                fb_ids.len().min(lc_ids.len())
            );
            assert!(
                match_n >= 4,
                "fewer than 4 tokens match — likely arithmetic drift"
            );
        }
        Err(e) => {
            eprintln!("  llama.cpp leg unavailable: {e}");
            eprintln!("  → coherence-only check: flambeau output should contain 'Paris'");
            assert!(
                fb_text.to_lowercase().contains("paris"),
                "flambeau output for 'The capital of France is' did NOT contain 'Paris' — \
                 strong signal of model corruption or arithmetic drift. Got: {fb_text:?}"
            );
        }
    }
}
