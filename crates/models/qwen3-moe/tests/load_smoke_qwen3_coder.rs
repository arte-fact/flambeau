//! V2.28.b-i0 — load + config smoke for Qwen3-Coder-30B-A3B-Instruct
//! (arch=qwen3moe, pure MoE, 48 layers, 128 experts, 8 top-k).
//!
//! This cert just proves:
//!   - `Qwen3MoEConfig::from_gguf` parses the qwen3moe arch correctly.
//!   - `Qwen3MoEShardedModel::load` walks every layer, resolves the
//!     Dense attention + routed MoE tensors, uploads to Mesh<N>.
//!   - Reports VRAM footprint per rank.
//!
//! FORWARD IS NOT TESTED HERE. qwen3moe's dense attention path (plain Q
//! projection, no fused gate, no post-attn sigmoid) is different from
//! qwen35moe full-attn (Q|gate fused + sigmoid-gated output). That needs
//! a separate `forward_dense_attn_{decode,prefill}` wiring.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::{AttentionFamily, Qwen3MoEConfig, Qwen3MoEShardedModel};
use flambeau_runtime::LayerAssignment;
use std::time::Instant;

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_CODER_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            Some(std::path::PathBuf::from(
                "/artefact/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
            ))
        })
        .filter(|p| p.exists())
}

#[test]
fn load_smoke_qwen3_coder() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("skip — Qwen3-Coder GGUF not present");
        return Ok(());
    };
    let n_available: i32 = device_count().unwrap_or(0);
    let n_requested: i32 = std::env::var("FLAMBEAU_MESH_RANKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    if n_requested <= 0 || n_requested > n_available {
        eprintln!(
            "FLAMBEAU_MESH_RANKS={n_requested} unavailable (have {n_available}) — skipping"
        );
        return Ok(());
    }

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;

    // Config sanity.
    assert_eq!(cfg.arch, "qwen3moe");
    assert_eq!(cfg.family, AttentionFamily::Dense);
    assert!(cfg.gdn.is_none(), "qwen3moe has no SSM/GDN");
    assert!(
        cfg.full_attention_interval.is_none(),
        "qwen3moe has no full-attn interval (every layer is full-attn)"
    );
    assert!(!cfg.is_dense_ffn(), "qwen3moe is MoE, not dense-FFN");
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.shared_expert_intermediate_size, None);
    assert_eq!(cfg.num_layers, 48);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.num_kv_heads, 4);
    assert_eq!(cfg.head_dim, 128);
    // 0..num_layers all full-attn on dense family.
    for il in 0..cfg.num_layers {
        assert!(!cfg.is_recurrent(il), "layer {il} should be full-attn");
    }

    eprintln!(
        "config: arch={} family={:?} layers={} hidden={} heads={}/{} head_dim={} experts={}/{}",
        cfg.arch,
        cfg.family,
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.num_experts,
        cfg.num_experts_per_tok,
    );

    // Load across the mesh.
    let cluster = HipCluster::new(&(0..n_requested).collect::<Vec<_>>())?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    eprintln!(
        "loading {} across Mesh<{}> (contiguous layer assignment)…",
        path.file_name().unwrap().to_string_lossy(),
        cluster.ranks(),
    );

    let t0 = Instant::now();
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    let load_dt = t0.elapsed();

    let total_gib = model.total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    let per_rank_gib = total_gib / cluster.ranks() as f64;
    eprintln!(
        "load ok: {:.2}s total_bytes={:.2} GiB per_rank={:.2} GiB",
        load_dt.as_secs_f64(),
        total_gib,
        per_rank_gib,
    );

    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
