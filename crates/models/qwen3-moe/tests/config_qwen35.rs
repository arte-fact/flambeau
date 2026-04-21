//! V2.2 scaffold: `Qwen3MoEConfig::from_gguf` now accepts arch=`qwen35`
//! (dense-hybrid) without requiring MoE metadata. Full loader + forward
//! for dense FFN is V2.2 follow-up.

use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::Qwen3MoEConfig;

fn gguf_path(env_key: &str, default: &str) -> Option<std::path::PathBuf> {
    std::env::var(env_key)
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from(default)))
        .filter(|p| p.exists())
}

#[test]
fn qwen35_config_parses() {
    let Some(path) = gguf_path("FLAMBEAU_QWEN35_GGUF", "/artefact/models/Qwen3.5-9B-Q4_1.gguf")
    else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let cfg = Qwen3MoEConfig::from_gguf(&gguf).expect("parse qwen35 config");

    eprintln!(
        "qwen35 cfg: arch={} layers={} hidden={} heads={}/{} head_dim={} ffn={}",
        cfg.arch,
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.moe_intermediate_size
    );

    assert_eq!(cfg.arch, "qwen35");
    assert_eq!(cfg.num_experts, 0, "qwen35 is dense — num_experts must be 0");
    assert_eq!(cfg.num_layers, 32);
    // ffn dim comes from `qwen35.feed_forward_length`.
    assert_eq!(cfg.moe_intermediate_size, 12288);
    // Hybrid family (GDN + full-attn).
    assert!(cfg.gdn.is_some(), "qwen35 is hybrid, GDN dims required");
    assert_eq!(cfg.full_attention_interval, Some(4));
}
