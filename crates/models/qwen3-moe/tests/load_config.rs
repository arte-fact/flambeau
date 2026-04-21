//! Integration test — parse config + layout from a real Qwen3-family MoE GGUF.
//!
//! Set `FLAMBEAU_QWEN3_GGUF=/path/to/qwen3moe.gguf` to enable the real-file
//! tests; otherwise they skip. Any GGUF whose `general.architecture` is one
//! of `qwen3moe` / `qwen35moe` / `qwen36moe` works.
//!
//! The V1 reference is `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (arch=qwen35moe,
//! hybrid GDN + full-attention + MoE-with-shared-expert).

use anyhow::Result;
use flambeau_qwen3_moe::{
    AttentionFamily, LayerAttnBlock, ModelLayout, Qwen3MoEConfig,
};
use flambeau_quant::GgufFile;

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn config_from_real_qwen3_gguf() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!(
            "FLAMBEAU_QWEN3_GGUF unset — skipping config_from_real_qwen3_gguf"
        );
        return Ok(());
    };
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;

    assert!(
        matches!(cfg.arch.as_str(), "qwen3moe" | "qwen35moe" | "qwen36moe"),
        "arch = {}", cfg.arch
    );
    assert!(cfg.num_layers >= 40, "num_layers = {}", cfg.num_layers);
    assert!(cfg.num_heads >= 16, "num_heads = {}", cfg.num_heads);
    assert!(cfg.num_kv_heads <= cfg.num_heads);
    assert!(cfg.head_dim >= 64);
    assert!(cfg.num_experts >= 64, "num_experts = {}", cfg.num_experts);
    assert!(cfg.num_experts_per_tok >= 4);
    assert!(cfg.num_experts_per_tok <= cfg.num_experts);
    assert!(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps < 1e-3);
    assert!(cfg.rope.freq_base >= 10_000.0);
    assert!(cfg.vocab_size >= 100_000);

    // Hybrid arches must have the full companion of SSM + shared-expert
    // + full_attention_interval metadata. Dense arches must not.
    match cfg.family {
        AttentionFamily::Dense => {
            assert_eq!(cfg.arch, "qwen3moe");
            assert!(cfg.gdn.is_none());
            assert!(cfg.full_attention_interval.is_none());
            assert!(cfg.shared_expert_intermediate_size.is_none());
        }
        AttentionFamily::Hybrid => {
            assert!(cfg.gdn.is_some(), "hybrid arch must carry SSM dims");
            let gdn = cfg.gdn.as_ref().unwrap();
            assert!(gdn.d_inner > 0);
            assert!(gdn.conv_kernel >= 2);
            assert!(gdn.num_v_heads > 0);
            assert!(cfg.full_attention_interval.is_some());
            assert!(cfg.shared_expert_intermediate_size.is_some());
            assert!(cfg.rope.rotated_dims <= cfg.head_dim);
            if cfg.arch == "qwen35moe" {
                // Qwen3.6 reports sections = [11, 11, 10, 0].
                assert!(cfg.rope.sections.is_some());
            }
        }
    }

    eprintln!(
        "Qwen3.x config [{}]: layers={} heads={}/{} head_dim={} experts={}({} top_k) hidden={} vocab={} rope_rotated={}/{} sections={:?} gdn={} full_attn_every={:?} shared_expert={:?}",
        cfg.arch,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.rope.rotated_dims,
        cfg.head_dim,
        cfg.rope.sections,
        cfg.gdn.is_some(),
        cfg.full_attention_interval,
        cfg.shared_expert_intermediate_size,
    );
    Ok(())
}

#[test]
fn layout_enumerates_every_tensor() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!(
            "FLAMBEAU_QWEN3_GGUF unset — skipping layout_enumerates_every_tensor"
        );
        return Ok(());
    };
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let layout = ModelLayout::from_gguf(&file, &cfg)?;

    assert_eq!(layout.layers.len(), cfg.num_layers);

    // Layer classification must agree with config on every index.
    let mut num_full_attn = 0usize;
    let mut num_gdn = 0usize;
    let mut num_dense = 0usize;
    for (i, l) in layout.layers.iter().enumerate() {
        match (&cfg.family, cfg.is_recurrent(i), &l.attn) {
            (AttentionFamily::Dense, false, LayerAttnBlock::Dense(_)) => num_dense += 1,
            (AttentionFamily::Hybrid, true, LayerAttnBlock::Gdn(_)) => num_gdn += 1,
            (AttentionFamily::Hybrid, false, LayerAttnBlock::FullAttn(_)) => {
                num_full_attn += 1
            }
            other => panic!("layer {i} classification mismatch: {other:?}"),
        }

        // Shared-expert presence mirrors config.
        assert_eq!(
            l.ffn.shared.is_some(),
            cfg.shared_expert_intermediate_size.is_some(),
            "layer {i} shared-expert mismatch"
        );

        // Every layer has an attn_norm; hybrids additionally have
        // post_attention_norm, denses additionally have ffn_norm.
        match cfg.family {
            AttentionFamily::Dense => {
                assert!(l.post_attention_norm.is_none());
                assert!(l.ffn_norm.is_some());
            }
            AttentionFamily::Hybrid => {
                assert!(l.post_attention_norm.is_some());
                assert!(l.ffn_norm.is_none());
            }
        }
    }
    if cfg.family == AttentionFamily::Hybrid {
        assert_eq!(num_gdn, cfg.num_recurrent_layers());
        assert_eq!(num_full_attn, cfg.num_full_attn_layers());
    } else {
        assert_eq!(num_dense, cfg.num_layers);
    }

    let total_gb = layout.total_bytes() as f64 / 1e9;
    eprintln!(
        "layout [{}]: {} layers (full-attn={} / gdn={} / dense={}), total weight bytes = {:.2} GB",
        cfg.arch,
        layout.layers.len(),
        num_full_attn,
        num_gdn,
        num_dense,
        total_gb,
    );
    // Q4_K-quantised Qwen3-class models are in the 15–30 GB range.
    assert!(total_gb > 10.0 && total_gb < 60.0);
    Ok(())
}
