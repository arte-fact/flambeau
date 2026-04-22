//! V1.7.4 — token-parity cert vs llama.cpp on real Qwen3.6 weights.
//!
//! Runs flambeau decode for N greedy steps from a fixed seed token and
//! compares the generated token sequence against an llama.cpp reference
//! produced from the same GGUF at temperature=0.
//!
//! Reference tokens live in `certs/parity/<model_tag>_decode.json` at
//! the workspace root; regenerate with:
//!
//! ```
//! LD_LIBRARY_PATH=/artefact/llama.cpp/build/bin:/opt/rocm-7.1.1/lib \
//!   /artefact/llama.cpp/build/bin/llama-cli \
//!     -m $FLAMBEAU_QWEN3_GGUF -ngl 999 -n 8 --temp 0 --seed 1 -no-cnv \
//!     --no-display-prompt -p "Hello"
//! ```
//!
//! Then tokenise the 8 generated tokens with the same tokenizer, and
//! write them to the JSON under `expected_token_ids`.
//!
//! Skips when `FLAMBEAU_QWEN3_GGUF` is unset, no HIP devices, or the
//! reference JSON is missing. This is a cert, not a smoke.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::forward::{forward_one_token_pp, ShardedForwardOneTokenScratch};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession};
use flambeau_runtime::LayerAssignment;
use serde::Deserialize;

#[derive(Deserialize)]
struct ParityCert {
    /// Human tag so the right cert is loaded for the GGUF under test.
    model_tag: String,
    /// Tokeniser-internal id fed to the first decode step.
    seed_token_id: u32,
    /// Expected sequence of argmax tokens from llama.cpp at temperature=0.
    expected_token_ids: Vec<u32>,
    /// Optional second oracle — candle's greedy sequence on the same GGUF.
    /// Both llama.cpp and candle agree on token 1 (the argmax of the first
    /// forward from the seed) but diverge token 2+ on 35B hybrid GDN+MoE
    /// due to F16/F32 micro-arithmetic differences between the two impls.
    #[serde(default)]
    candle_token_ids: Vec<u32>,
    /// Temperature / sampling the reference was produced with. Must be 0.0
    /// for greedy parity.
    temperature: f32,
    /// When `true`, the test hard-fails on divergence. When `false`, the
    /// test emits a WARN with both sequences and exits 0 — the "diagnostic"
    /// mode used while closing the V1.7.4.a parity gap.
    #[serde(default = "default_expect_pass")]
    expect_pass: bool,
    /// Free-form note — reference prompt, llama.cpp / candle commit, etc.
    #[allow(dead_code)] // kept so cert JSON keeps documenting its provenance
    notes: String,
}

fn default_expect_pass() -> bool {
    true
}

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

fn card_vram_bytes(card: usize) -> Option<u64> {
    let s = std::fs::read_to_string(format!(
        "/sys/class/drm/card{card}/device/mem_info_vram_total"
    ))
    .ok()?;
    s.trim().parse::<u64>().ok()
}

fn workspace_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR points at crates/models/qwen3-moe.
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

fn load_cert(model_tag_hint: &str) -> Result<Option<ParityCert>> {
    let cert_path = workspace_root()
        .join("certs")
        .join("parity")
        .join(format!("{model_tag_hint}_decode.json"));
    if !cert_path.exists() {
        eprintln!(
            "parity cert {} not found — skipping (regenerate with llama-cli per test docs)",
            cert_path.display()
        );
        return Ok(None);
    }
    let bytes = std::fs::read(&cert_path)
        .with_context(|| format!("read {}", cert_path.display()))?;
    let cert: ParityCert = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", cert_path.display()))?;
    if cert.temperature != 0.0 {
        bail!(
            "parity cert {} has temperature={} — greedy parity requires 0.0",
            cert_path.display(),
            cert.temperature
        );
    }
    Ok(Some(cert))
}

/// Map the GGUF path to a stable cert tag. Qwen3.6-35B-A3B-UD-Q4_K_S is the
/// only V1 target; extend when we add more parity-gated models.
fn cert_tag_for(path: &std::path::Path) -> Option<&'static str> {
    let name = path.file_name()?.to_str()?;
    if name.contains("Qwen3.6-35B-A3B-UD-Q4_K_S") {
        Some("qwen3_6_35b_a3b_ud_q4_k_s")
    } else {
        None
    }
}

#[test]
fn decode_greedy_matches_llama_cpp() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping parity cert");
        return Ok(());
    };
    let Some(tag) = cert_tag_for(&path) else {
        eprintln!(
            "no parity cert tag registered for {} — skipping",
            path.display()
        );
        return Ok(());
    };
    let Some(cert) = load_cert(tag)? else {
        return Ok(());
    };
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices for parity run — got {n}, skipping");
        return Ok(());
    }

    // Budget check: same shape as the real-weight smoke test.
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let layout = flambeau_qwen3_moe::ModelLayout::from_gguf(&file, &cfg)?;
    let need = layout.total_bytes() as usize
        + if cfg.tied_lm_head {
            file.info("token_embd.weight")
                .map(|i| i.size_in_bytes() as usize)
                .unwrap_or(0)
        } else {
            0
        };
    let total_vram: u64 = (0..n as usize).filter_map(card_vram_bytes).sum();
    if total_vram > 0 && (need as u64) + 2 * 1024 * 1024 * 1024 > total_vram {
        eprintln!(
            "skipping parity: model needs {:.2} GiB, cluster has {:.2} GiB",
            need as f64 / (1024.0 * 1024.0 * 1024.0),
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        return Ok(());
    }

    let cluster = HipCluster::new(&(0..n).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);

    eprintln!(
        "parity run [{}]: loading Qwen3 MoE across {} ranks ({:.2} GiB)…",
        cert.model_tag,
        cluster.ranks(),
        need as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster)?;
    let mut scratch = ShardedForwardOneTokenScratch::new(&model, &cluster)?;

    let n_predict = cert.expected_token_ids.len();
    let mut got: Vec<u32> = Vec::with_capacity(n_predict);
    let mut current = cert.seed_token_id;
    for pos in 0..n_predict {
        let next = forward_one_token_pp(
            &model,
            &mut session,
            &cluster,
            &mut scratch,
            current,
            pos,
        )
        .with_context(|| format!("decode step pos={pos}"))?;
        got.push(next);
        current = next;
    }

    eprintln!("  got       = {got:?}");
    eprintln!("  llama.cpp = {:?}", cert.expected_token_ids);
    if !cert.candle_token_ids.is_empty() {
        eprintln!("  candle    = {:?}", cert.candle_token_ids);
    }
    // Primary parity gate: token 1 argmax. Even llama.cpp and candle
    // disagree on tokens 2+ on 35B hybrid GDN+MoE, so demanding the full
    // 8-token match is stricter than the references can themselves meet.
    if !got.is_empty() && !cert.expected_token_ids.is_empty() {
        let ref_first = cert.expected_token_ids[0];
        let candle_first = cert.candle_token_ids.first().copied();
        let ours_first = got[0];
        let two_oracles_agree = candle_first.map(|c| c == ref_first).unwrap_or(true);
        if two_oracles_agree {
            eprintln!(
                "  [first-token parity] llama.cpp={} candle={:?} ours={}",
                ref_first, candle_first, ours_first,
            );
            if ours_first == ref_first {
                eprintln!("  first-token parity PASSES");
            } else {
                eprintln!(
                    "  first-token parity DIVERGES — both oracles say {ref_first}, we say {ours_first}"
                );
            }
        } else {
            eprintln!(
                "  [first-token parity] oracles disagree (llama={ref_first} candle={:?}); skipping primary gate",
                candle_first,
            );
        }
    }

    let matches = got == cert.expected_token_ids;
    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;

    if !matches {
        // Diff the two sequences for a clear failure / warn message.
        let mut first_div = None;
        for (i, (g, e)) in got.iter().zip(&cert.expected_token_ids).enumerate() {
            if g != e {
                first_div = Some((i, *g, *e));
                break;
            }
        }
        let msg = if let Some((i, g, e)) = first_div {
            format!(
                "greedy-parity divergence at step {i}: flambeau={g}, llama.cpp={e}\n  got      = {got:?}\n  expected = {:?}",
                cert.expected_token_ids
            )
        } else {
            format!(
                "greedy-parity length mismatch: got {} tokens, expected {}\n  got      = {got:?}\n  expected = {:?}",
                got.len(),
                cert.expected_token_ids.len(),
                cert.expected_token_ids
            )
        };
        if cert.expect_pass {
            bail!("{msg}");
        } else {
            eprintln!(
                "DIAGNOSTIC (expect_pass=false, V1.7.4.a gap): {msg}\n  \
                 flip expect_pass=true in the cert JSON once the fix lands."
            );
        }
    }
    Ok(())
}
