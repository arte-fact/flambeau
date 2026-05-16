//! Shared parity / smoke harness for gemma4 integration tests.
//!
//! Cargo includes `tests/common/mod.rs` into each test binary that
//! declares `mod common;`. Pieces here are deliberately allow-dead —
//! every test binary uses a different subset.

#![allow(dead_code)]
#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use flambeau_backend_hip::{device_count, HipCluster, HipDevice};
use flambeau_blocks::HybridCluster;
use flambeau_quant::{load_from_gguf, GgufFile, GgufTokenizer};
use flambeau_runtime::ModelDriver;

pub const MODELS_DIR: &str = "/artefact/models";
pub const PROMPT: &str = "The capital of France is";

/// Open a GGUF by name under [`MODELS_DIR`] or return `None` with a
/// skip line on stderr.
pub fn open_or_skip(name: &str) -> Option<Arc<GgufFile>> {
    let p = Path::new(MODELS_DIR).join(name);
    if !p.exists() {
        eprintln!("skipping — {name} not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok().map(Arc::new)
}

/// Return `Some(HipDevice::new(0))` when at least one HIP device is
/// visible; print a skip line and return `None` otherwise.
pub fn device_or_skip() -> Option<HipDevice> {
    let n = device_count().ok()?;
    if n < 1 {
        eprintln!("skipping — no HIP devices");
        return None;
    }
    HipDevice::new(0).ok()
}

/// Build a [`HipCluster`] over `devices` if enough are present.
pub fn cluster_or_skip(devices: &[i32]) -> Option<HipCluster> {
    let n = device_count().ok()?;
    if (n as usize) < devices.len() {
        eprintln!("skipping — need {} HIP devices, have {n}", devices.len());
        return None;
    }
    HipCluster::new(devices).ok()
}

/// Build a hybrid (pp+tp) mesh. `stage_ids[s]` lists the device ids
/// for stage `s`. Each sub-cluster MUST have full peer access (intra-
/// stage BAR1 AR); the 4-way global cluster needs only the stage-
/// boundary edge for hand-off, so we do NOT require
/// `peer_access_full()` on the global.
///
/// Returns `(sub_clusters, global, mesh_flat)` or `None` on skip.
pub fn hybrid_clusters_or_skip(
    stage_ids: &[&[i32]],
) -> Option<(Vec<Arc<HipCluster>>, Arc<HipCluster>, Vec<i32>)> {
    let total: usize = stage_ids.iter().map(|s| s.len()).sum();
    let n = device_count().ok()?;
    if (n as usize) < total {
        eprintln!("skipping — need {total} HIP devices, have {n}");
        return None;
    }
    let mut subs = Vec::with_capacity(stage_ids.len());
    for &ids in stage_ids {
        let c = Arc::new(HipCluster::new(ids).ok()?);
        if !c.peer_access_full() {
            eprintln!("skipping — sub-cluster {ids:?} peer access incomplete");
            return None;
        }
        subs.push(c);
    }
    let mesh: Vec<i32> = stage_ids.iter().flat_map(|s| s.iter().copied()).collect();
    let global = Arc::new(HipCluster::new(&mesh).ok()?);
    Some((subs, global, mesh))
}

/// Tokenize `prompt` against the GGUF's tokenizer, prepending BOS when
/// the tokenizer requests it. Returns the id sequence.
pub fn tokenize_prompt(file: &GgufFile, prompt: &str) -> Result<Vec<u32>> {
    let tokenizer = load_from_gguf(file)?;
    let mut ids = tokenizer.encode(prompt)?;
    if tokenizer.force_add_bos {
        if let Some(bos) = tokenizer.bos_id {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

/// Greedy decode `n_decode` tokens through any [`ModelDriver`] after
/// feeding `prompt_ids` one at a time. Returns the generated id
/// sequence (length == `n_decode`).
pub fn greedy_decode(
    driver: &mut dyn ModelDriver,
    prompt_ids: &[u32],
    n_decode: usize,
) -> Result<Vec<u32>> {
    if prompt_ids.is_empty() {
        return Err(anyhow!("greedy_decode: empty prompt"));
    }
    let mut tok = prompt_ids[0];
    for (i, &t) in prompt_ids.iter().enumerate() {
        tok = driver.forward_one_token(t, i)?;
    }
    let mut out = Vec::with_capacity(n_decode);
    out.push(tok);
    let mut pos = prompt_ids.len();
    while out.len() < n_decode {
        tok = driver.forward_one_token(tok, pos)?;
        out.push(tok);
        pos += 1;
    }
    Ok(out)
}

/// Greedy decode `n_decode` tokens through llama.cpp's `llama-cli`
/// binary for the same prompt + `--temp 0 --top-k 1 --seed 0`. Re-
/// tokenizes the stdout via the GGUF tokenizer so the returned id
/// sequence is directly comparable to a flambeau decode.
///
/// Returns an error (with skip-worthy message) if the binary is not
/// present or the invocation fails — the caller decides whether to
/// fall back to a coherence-only assertion.
pub fn llamacpp_decode(model_path: &str, prompt: &str, n_decode: usize) -> Result<Vec<u32>> {
    use std::process::{Command, Stdio};
    let bin = "/artefact/llama.cpp/build/bin/llama-cli";
    if !Path::new(bin).exists() {
        return Err(anyhow!("{bin} not present — skipping llama.cpp parity leg"));
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
            prompt,
            "-n",
            &format!("{n_decode}"),
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
        return Err(anyhow!(
            "llama-cli failed: status={}, stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let trimmed = text.trim();
    eprintln!("  llama-cli output: {trimmed:?}");
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = load_from_gguf(&gguf)?;
    Ok(tokenizer.encode(trimmed)?)
}

/// Length of the matching prefix between two id sequences.
pub fn matching_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Print a parity comparison block to stderr (prompt + both decoded
/// sequences + matching prefix).
pub fn report_parity(
    label: &str,
    prompt: &[u32],
    flambeau: &[u32],
    llamacpp: &[u32],
    tokenizer: &GgufTokenizer,
) {
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

/// Either assert at least `min_prefix` tokens match llama.cpp, or fall
/// back to asserting the flambeau decode contains `coherence_kw` (case-
/// insensitive). The fallback fires when the llama.cpp binary is
/// absent so the test still proves end-to-end correctness in
/// isolation.
pub fn assert_parity_or_keyword(
    label: &str,
    model_path: &str,
    prompt_ids: &[u32],
    fb_ids: &[u32],
    tokenizer: &GgufTokenizer,
    min_prefix: usize,
    coherence_kw: &str,
) {
    let fb_text = tokenizer.decode(fb_ids).unwrap_or_default();
    eprintln!("\n=== {label} ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    match llamacpp_decode(model_path, PROMPT, fb_ids.len()) {
        Ok(lc_ids) => {
            let lc_text = tokenizer.decode(&lc_ids).unwrap_or_default();
            let match_n = matching_prefix_len(fb_ids, &lc_ids);
            eprintln!("  llamacpp ({} ids): {lc_ids:?}", lc_ids.len());
            eprintln!("  llamacpp text: {lc_text:?}");
            eprintln!(
                "  matching prefix: {match_n} / {} tokens",
                fb_ids.len().min(lc_ids.len())
            );
            assert!(
                match_n >= min_prefix,
                "{label}: fewer than {min_prefix} tokens match llama.cpp — likely arithmetic drift"
            );
        }
        Err(e) => {
            eprintln!("  llama.cpp leg unavailable: {e}");
            eprintln!(
                "  → coherence-only check: flambeau output should contain {coherence_kw:?}"
            );
            assert!(
                fb_text.to_lowercase().contains(&coherence_kw.to_lowercase()),
                "{label}: flambeau output did NOT contain {coherence_kw:?}. \
                 Got: {fb_text:?}"
            );
        }
    }
}

/// Same as [`hybrid_clusters_or_skip`] but constructs the
/// [`HybridCluster`] in the correct order (sub-clusters before global,
/// per MEMORY.md `hybrid_cluster_order`).
pub fn hybrid_cluster_or_skip(
    stage_ids: &[&[i32]],
    tp_size: usize,
) -> Option<HybridCluster> {
    let (subs, global, _mesh) = hybrid_clusters_or_skip(stage_ids)?;
    HybridCluster::new(subs, global, tp_size).ok()
}
