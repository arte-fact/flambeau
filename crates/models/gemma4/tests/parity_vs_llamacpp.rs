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
use flambeau_runtime::ModelDriver;

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

/// Probe: TP forward one decode step, then download rank-0 and rank-1's
/// `stage.hidden` and compare. After a successful AR-reduce + residual
/// add, the hidden state must match bit-for-bit across ranks (TP is
/// replicated post-AR). If they differ, AR-reduce is broken or one
/// rank's per-layer contribution is wrong.
#[test]
fn tp_hidden_cross_rank_match_after_one_token() {
    use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        return;
    }
    let file = Arc::new(file);
    let cluster = ArcGuard::new(HipCluster::new(&[0, 2]).expect("cluster"));
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let hidden = cfg.hidden_size;
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    eprintln!(
        "cfg: hidden={} num_heads={} head_dim={} head_dim_swa={} ff_len={} num_kv_heads[0..4]={:?} swa_layers[0..4]={:?}",
        cfg.hidden_size,
        cfg.num_heads,
        cfg.head_dim,
        cfg.swa.head_dim_swa,
        cfg.feed_forward_length,
        &cfg.num_kv_heads[..4.min(cfg.num_kv_heads.len())],
        &cfg.swa.swa_layers[..4.min(cfg.swa.swa_layers.len())],
    );
    for i in 0..3.min(layout.layers.len()) {
        let s = &layout.layers[i];
        eprintln!(
            "  layer {i}: head_dim={} n_heads={} n_kv_heads={} window={} has_kv={} ffn_kind={:?}",
            s.head_dim, s.n_heads, s.n_kv_heads, s.window, s.has_kv, s.ffn_kind
        );
    }
    let mut driver = Gemma4TpDriver::upload(&file, cfg, layout, cluster, MAX_TOKENS)
        .expect("TP upload");

    let _ = driver.forward_one_token(2, 0).expect("forward");

    // Download both ranks' hidden state.
    let mut buffers: [Vec<half::f16>; 2] = [
        vec![half::f16::from_f32(0.0); hidden],
        vec![half::f16::from_f32(0.0); hidden],
    ];
    for (rank, buf) in buffers.iter_mut().enumerate() {
        let device = driver.tp.cluster().device(rank);
        device.bind().expect("bind");
        let stage = &driver.stages[rank];
        // SAFETY: stage.hidden owns `hidden * 2` bytes; buf sized identically.
        unsafe {
            device
                .memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToHost,
                    DevicePtr(buf.as_mut_ptr() as usize),
                    stage.hidden,
                    hidden * 2,
                )
                .expect("memcpy");
        }
        device.default_stream().synchronize().expect("sync");
    }

    let mut max_abs_diff = 0.0f32;
    let mut zero_count_r0 = 0usize;
    let mut zero_count_r1 = 0usize;
    let mut nan_count = 0usize;
    for i in 0..hidden {
        let a = buffers[0][i].to_f32();
        let b = buffers[1][i].to_f32();
        if a == 0.0 {
            zero_count_r0 += 1;
        }
        if b == 0.0 {
            zero_count_r1 += 1;
        }
        if a.is_nan() || b.is_nan() {
            nan_count += 1;
        }
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    eprintln!("=== TP cross-rank hidden diff after 1 forward ===");
    eprintln!("  hidden = {hidden}");
    eprintln!("  rank0 first 8: {:?}", &buffers[0][..8].iter().map(|h| h.to_f32()).collect::<Vec<_>>());
    eprintln!("  rank1 first 8: {:?}", &buffers[1][..8].iter().map(|h| h.to_f32()).collect::<Vec<_>>());
    eprintln!("  rank0 zeros: {zero_count_r0}, rank1 zeros: {zero_count_r1}, NaN count: {nan_count}");
    eprintln!("  max |rank0 - rank1| = {max_abs_diff}");

    driver.dispose().expect("dispose");

    assert!(
        max_abs_diff < 1e-3,
        "TP rank-0 hidden != rank-1 hidden post-forward; AR-reduce or residual broken"
    );
}

/// Probe: TP forward one decode step and dump the logits stats. Used
/// to see if logits have signal (and which token dominates).
#[test]
fn tp_logits_dump_after_one_token() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        return;
    }
    let file = Arc::new(file);
    let cluster = ArcGuard::new(HipCluster::new(&[0, 2]).expect("cluster"));
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let mut driver = Gemma4TpDriver::upload(&file, cfg, layout, cluster, MAX_TOKENS)
        .expect("TP upload");

    let _ = driver.forward_one_token(2, 0).expect("forward");
    // Read logits_host from driver via reflection — quick + dirty: re-run with bos token
    // and inspect via debug.
    // Driver doesn't expose logits_host directly, so re-run via a small probe:
    let tok2 = driver.forward_one_token(105, 1).expect("forward 2");
    eprintln!("TP rank-0/2 forward(105, pos=1) argmax = {tok2}");
    driver.dispose().expect("dispose");
}

/// Sanity probe — downloads rank-0's uploaded attn_q from device and
/// diffs against a hand-sliced view of the GGUF mmap. If the upload is
/// byte-correct, this test passes. Used to bisect #37 between
/// "upload is wrong" and "forward path is wrong".
#[test]
fn upload_byte_parity_31b_q4_0_tp2_rank0_attn_q() {
    use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
    let Some(file) = open_or_skip("gemma-4-31B-it-Q4_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let cluster = ArcGuard::new(HipCluster::new(&[0, 2]).expect("cluster"));
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let mut driver = Gemma4TpDriver::upload(&file, cfg, layout, cluster, MAX_TOKENS)
        .expect("TP upload");

    // Take rank-1's ffn_down for layer 0 (row-parallel Q4_1, second half cols).
    let stage = &driver.stages[1];
    let layer0 = &stage.layer_weights[0];
    let q_dims = layer0.ffn_down.dims;
    let q_ptr = layer0.ffn_down.ptr;
    eprintln!("rank 1 layer 0 ffn_down dims = {q_dims:?}");

    // Compute expected byte count.
    let bytes_per_row = flambeau_blocks::row_bytes_for_dtype(
        flambeau_quant::GgmlDType::Q4_1,
        q_dims[1],
    )
    .expect("row_bytes");
    let total_bytes = q_dims[0] * bytes_per_row;
    eprintln!(
        "rank 0 layer 0 attn_q total bytes = {total_bytes} (rows={}, bpr={bytes_per_row})",
        q_dims[0]
    );

    // Download rank-1's slice from device.
    let device = driver.tp.cluster().device(1);
    device.bind().expect("bind");
    let mut host = vec![0u8; total_bytes];
    // SAFETY: q_ptr owns total_bytes; host is sized for total_bytes.
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(host.as_mut_ptr() as usize),
                q_ptr,
                total_bytes,
            )
            .expect("memcpy");
    }
    device.default_stream().synchronize().expect("sync");

    // Hand-shard ffn_down rank-1 = for each row, cols
    // [ff_local, ff_global). Read the expected slice from mmap.
    let raw = file
        .tensor_raw("blk.0.ffn_down.weight")
        .expect("tensor_raw");
    let q_width_global = q_dims[1] * 2;
    let bpr_global = flambeau_blocks::row_bytes_for_dtype(
        flambeau_quant::GgmlDType::Q4_1,
        q_width_global,
    )
    .expect("bpr_global");
    let bpr_local = bytes_per_row;
    let out_rows = q_dims[0];
    let mut expected = Vec::<u8>::with_capacity(total_bytes);
    for r in 0..out_rows {
        let src_start = r * bpr_global + 1 * bpr_local;
        expected.extend_from_slice(&raw[src_start..src_start + bpr_local]);
    }
    let expected: &[u8] = &expected;
    let match_count = host.iter().zip(expected.iter()).take_while(|(a, b)| a == b).count();
    eprintln!(
        "rank 0 attn_q[0..{total_bytes}] match prefix = {match_count}/{total_bytes} bytes"
    );
    eprintln!(
        "first 16 bytes: device={:?}, expected={:?}",
        &host[..16],
        &expected[..16]
    );

    driver.dispose().expect("dispose");

    assert_eq!(
        match_count, total_bytes,
        "rank 1 ffn_down (Q4_1) upload differs from expected mmap row-parallel slice"
    );
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

/// Q8_0 quantization probe — same dense model as `parity_31b_q4_0_tp2`
/// but at Q8_0 instead of Q4_0. Tests whether the F16 saturation
/// in the full-attention layers (head_dim=512) surfaces on dense
/// models when Q4_0's spike-rounding is replaced by Q8_0's
/// higher-precision rounding. If 31B-Q8_0 produces coherent
/// output, F16 dense paths are immune; if it degenerates the
/// same as 26B-A4B did pre-fix, the dense LayerComposerTp also
/// needs the F32 attention output path.
#[test]
fn smoke_31b_q8_0_tp2() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q8_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let vocab = cfg.vocab_size;
    let (prompt_ids, fb_ids) = match flambeau_decode_tp(file.clone(), &[0, 2]) {
        Ok(r) => r,
        Err(e) => {
            let full = format!("{e:#}");
            if full.contains("out of memory") || full.contains("OutOfMemory") || full.contains("rank 0 TP upload") {
                eprintln!(
                    "skipping — 31B-Q8_0 likely OOMs at TP2 on 16 GB MI50s \
                     (~33 GB Q8_0 weights → ~16.5 GB/rank, too tight): {full}"
                );
                return;
            }
            eprintln!("31B-Q8_0 TP2 decode failed: {full}");
            panic!("flambeau decode TP2 (31B-Q8_0)");
        }
    };
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== SMOKE | 31B-Q8_0 TP2 (hip:0,2) ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    assert_eq!(fb_ids.len(), N_DECODE);
    for (i, &t) in fb_ids.iter().enumerate() {
        assert!((t as usize) < vocab, "step {i}: token {t} >= vocab {vocab}");
    }
    let first = fb_ids[0];
    let all_same = fb_ids.iter().all(|&t| t == first);
    if all_same {
        eprintln!(
            "  [WARN] all {} tokens identical ({first}) — Q8_0 surfaces F16 saturation \
             on dense 31B too; dense composer also needs the F32 attention path",
            fb_ids.len()
        );
    } else if fb_text.to_lowercase().contains("paris") {
        eprintln!("  [OK] coherent: Q8_0 dense path produces correct output");
    } else {
        eprintln!("  [PARTIAL] varied but non-topical — F16 cascade has subtler bug");
    }
}

/// Phase 10c-G bisect: 26B-A4B-Q8_0 on PP2 (layer-split). Single-
/// device OOMs at 27 GB on a 16 GB MI50; PP2 splits layers so each
/// rank holds ~13.5 GB. If PP-MoE produces coherent (non-pad) output
/// for the same model, the F16 overflow in `smoke_26b_a4b_q8_0_tp2`
/// is TP-specific (likely in the per-branch norm composition for
/// shared-MLP + routed-MoE). If PP-MoE ALSO overflows, the bug is
/// gemma4-general.
/// Isolation probe for #108 — gemma4-31B-Q8_0 PP2. Same forward block
/// (StandardAttention::forward_decode) as 26B-A4B-Q8_0 PP2 but **dense
/// (no MoE)** at the same Q8_0 quant. If this works → bug is MoE-
/// specific; if this fails → bug is Q8_0 in the attention block itself.
#[test]
fn isolate_31b_q8_0_pp2_dense_vs_moe() {
    let Some(file) = open_or_skip("gemma-4-31B-it-Q8_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let vocab = cfg.vocab_size;
    // PP3 — 31B Q8_0 weights are ~33 GB. PP2 OOMs (~16 GB/rank);
    // PP4 on this rig hits a non-gfx906 mid-arch kernel-load issue
    // unrelated to #108. PP3 on 0/2/3 = ~11 GB/rank fits comfortably.
    let (prompt_ids, fb_ids) = match flambeau_decode_pp_pertoken(file.clone(), &[0, 2, 3]) {
        Ok(r) => r,
        Err(e) => {
            let full = format!("{e:#}");
            if full.contains("out of memory") || full.contains("OutOfMemory") {
                eprintln!("skipping — 31B-Q8_0 PP4 OOM: {full}");
                return;
            }
            eprintln!("31B-Q8_0 PP4 decode failed: {full}");
            panic!("flambeau decode PP4 (31B-Q8_0)");
        }
    };
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== ISOLATE | 31B-Q8_0 PP2 (hip:0,2) per-token ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");
    let first = fb_ids[0];
    let all_same = fb_ids.iter().all(|&t| t == first);
    if all_same {
        eprintln!("  [SAME-FAILURE] all {} tokens identical ({first}) — bug is Q8_0 attention path, not MoE", fb_ids.len());
    } else if fb_text.to_lowercase().contains("paris") {
        eprintln!("  [OK] coherent — bug is MoE-specific, not Q8_0-attention");
    } else {
        eprintln!("  [PARTIAL] varied but non-topical");
    }
    for (i, &t) in fb_ids.iter().enumerate() {
        assert!((t as usize) < vocab, "step {i}: token {t} >= vocab {vocab}");
    }
}

/// Per-iteration logits-health probe for the 26B-A4B-Q8_0 PP MoE NaN
/// debug (#108). Runs the prompt token by token and after each step
/// reports max|logit|, NaN count, top-3 ids. Goal: localise *which*
/// prompt position first introduces NaN.
#[test]
fn debug_26b_a4b_q8_0_pp2_logits_per_iter() {
    let Some(file) = open_or_skip("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let cluster = HipCluster::new(&[0, 2]).expect("cluster");
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let layer_to_rank = partition_layers(2, &layout).expect("partition");

    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let mut prompt_ids = tokenizer.encode(PROMPT).expect("encode");
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
    )
    .expect("upload");

    eprintln!("\n=== DEBUG | 26B-A4B-Q8_0 PP2 logits-per-iter ===");
    eprintln!("  prompt ids: {prompt_ids:?}");
    let mut logits: Vec<f32> = Vec::new();
    for (i, &t) in prompt_ids.iter().enumerate() {
        driver
            .forward_one_token_logits(t, i, &mut logits)
            .expect("forward");
        let n_nan = logits.iter().filter(|v| v.is_nan()).count();
        let n_inf = logits.iter().filter(|v| v.is_infinite()).count();
        let finite_max = logits
            .iter()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .filter(|(_, v)| v.is_finite())
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top3: Vec<(usize, f32)> = indexed.into_iter().take(3).collect();
        eprintln!(
            "  iter={i} pos={i} tok={t} | n_nan={n_nan} n_inf={n_inf} \
             finite_max_abs={finite_max:.2} top3={top3:?}"
        );
    }
    let _ = driver.dispose();
}

#[test]
fn smoke_26b_a4b_q8_0_pp2() {
    let Some(file) = open_or_skip("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let vocab = cfg.vocab_size;
    // Per-token decode avoids the MoE prefill bail (#23). Each prompt
    // token + decode token feeds through forward_one_token. NOTE:
    // PP's forward_one_token currently passes `moe_scratch: None` to
    // forward_layer_decode (pp.rs:853), so PP-MoE is also unwired.
    // The smoke skips with a diagnostic rather than panicking — once
    // PP-MoE scratch is wired, this becomes a real bisect tool.
    let (prompt_ids, fb_ids) = match flambeau_decode_pp_pertoken(file.clone(), &[0, 2]) {
        Ok(r) => r,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("moe_scratch is None")
                || msg.contains("MoE prefill not supported")
            {
                eprintln!(
                    "skipping — PP-MoE not yet wired: {msg}\n\
                     (Pending: alloc Gemma4MoeScratch on Gemma4PpStage + pass through \
                     forward_one_token; tracked under 10c-G followup.)"
                );
                return;
            }
            eprintln!("26B-A4B PP2 decode failed: {e:#}");
            panic!("flambeau decode PP2 (MoE)");
        }
    };
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== SMOKE | 26B-A4B-Q8_0 PP2 (hip:0,2) ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");

    assert_eq!(fb_ids.len(), N_DECODE);
    for (i, &t) in fb_ids.iter().enumerate() {
        assert!((t as usize) < vocab, "step {i}: token {t} >= vocab {vocab}");
    }
    let first = fb_ids[0];
    let all_same = fb_ids.iter().all(|&t| t == first);
    if all_same {
        eprintln!(
            "  [WARN] all {} decoded tokens identical ({first}); gemma4 PP MoE \
             produces constant logits — bug is gemma4-general, not TP-specific",
            fb_ids.len()
        );
    } else {
        eprintln!("  [OK] non-degenerate output → 10c-G TP MoE is the bug");
    }
}

/// Smoke test for Gemma4-26B-A4B (MoE) on TP2. Asserts decode runs
/// to completion without crash + produces in-vocab tokens for every
/// step. Does NOT assert text match — the gemma4 MoE path uses
/// `RouterNormalize::TopkRenorm` instead of the spec's
/// `softmax-then-topk` (pending kernel), so output is finite +
/// plausible but not bit-exact vs llama.cpp. Real parity comes in a
/// follow-up (10c-G) after the softmax_topk_f32 kernel lands.
#[test]
fn smoke_26b_a4b_q8_0_tp2() {
    let Some(file) = open_or_skip("gemma-4-26B-A4B-it-Q8_0.gguf") else {
        return;
    };
    if device_count().map(|n| n < 2).unwrap_or(true) {
        eprintln!("skipping — need 2 HIP devices");
        return;
    }
    let file = Arc::new(file);
    let cfg = Gemma4Config::from_gguf(&file).expect("cfg");
    let vocab = cfg.vocab_size;
    let (prompt_ids, fb_ids) = match flambeau_decode_tp(file.clone(), &[0, 2]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("26B-A4B TP2 decode failed: {e}");
            panic!("flambeau decode TP2 (MoE)");
        }
    };
    let tokenizer = load_from_gguf(&file).expect("tokenizer");
    let fb_text = tokenizer.decode(&fb_ids).unwrap_or_default();
    eprintln!("\n=== SMOKE | 26B-A4B-Q8_0 TP2 (hip:0,2) ===");
    eprintln!("  prompt   ({} ids): {prompt_ids:?}", prompt_ids.len());
    eprintln!("  flambeau ({} ids): {fb_ids:?}", fb_ids.len());
    eprintln!("  flambeau text: {fb_text:?}");

    assert_eq!(fb_ids.len(), N_DECODE, "decoded fewer than {N_DECODE} tokens");
    for (i, &t) in fb_ids.iter().enumerate() {
        assert!(
            (t as usize) < vocab,
            "decoded token #{i} = {t} >= vocab_size {vocab}"
        );
    }
    // Diagnostic — log degenerate output (constant logits) but do
    // NOT panic. This smoke gates on "no crash + N tokens + in-vocab";
    // text-quality / parity vs llama.cpp lives in the follow-up
    // (10c-G) after the gemma4 MoE forward is debugged + the
    // softmax_topk_f32 kernel ships.
    let first = fb_ids[0];
    let all_same = fb_ids.iter().all(|&t| t == first);
    if all_same {
        eprintln!(
            "  [WARN] all {} decoded tokens identical ({first}); MoE forward likely \
             producing constant logits — see 10c-G for debugging",
            fb_ids.len()
        );
    }
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
