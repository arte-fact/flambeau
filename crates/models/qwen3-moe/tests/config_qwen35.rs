//! V2.2.a scaffold: `Qwen3MoEConfig::from_gguf` + `ModelLayout::from_gguf`
//! + sharded weight upload accept arch=`qwen35` (dense-hybrid). The forward
//! + parity land in V2.2.b (Q4_1 MMVQ) and V2.2.c (dense FFN forward).

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

#[test]
fn qwen35_layout_resolves_dense_ffn() {
    let Some(path) = gguf_path("FLAMBEAU_QWEN35_GGUF", "/artefact/models/Qwen3.5-9B-Q4_1.gguf")
    else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let cfg = Qwen3MoEConfig::from_gguf(&gguf).expect("parse qwen35 config");
    let layout = flambeau_qwen3_moe::layout::ModelLayout::from_gguf(&gguf, &cfg)
        .expect("resolve qwen35 layout");

    assert_eq!(layout.layers.len(), 32);
    // Every layer must have a dense FFN and no MoE fields.
    for (i, l) in layout.layers.iter().enumerate() {
        assert!(l.ffn.dense.is_some(), "layer {i} missing dense FFN");
        assert!(l.ffn.ffn_gate_exps.is_none(), "layer {i} has unexpected MoE ffn_gate_exps");
        assert!(l.ffn.ffn_up_exps.is_none());
        assert!(l.ffn.ffn_down_exps.is_none());
        assert!(l.ffn.ffn_gate_inp.is_none());
        assert!(l.ffn.shared.is_none());
        let d = l.ffn.dense.as_ref().unwrap();
        // GGUF dims are [inner, outer] on disk — gate/up = [hidden, inter],
        // down = [inter, hidden].
        assert_eq!(d.ffn_gate.dims, vec![cfg.moe_intermediate_size as u64, cfg.hidden_size as u64]);
        assert_eq!(d.ffn_up.dims, vec![cfg.moe_intermediate_size as u64, cfg.hidden_size as u64]);
        assert_eq!(d.ffn_down.dims, vec![cfg.hidden_size as u64, cfg.moe_intermediate_size as u64]);
    }
    eprintln!("qwen35 layout: total_bytes={} MiB", layout.total_bytes() / (1024 * 1024));
}

#[cfg(feature = "hip")]
#[test]
fn qwen35_loads_on_mesh1() -> anyhow::Result<()> {
    use flambeau_backend_hip::{device_count, HipCluster};
    use flambeau_qwen3_moe::Qwen3MoEShardedModel;
    use flambeau_runtime::LayerAssignment;

    let Some(path) = gguf_path("FLAMBEAU_QWEN35_GGUF", "/artefact/models/Qwen3.5-9B-Q4_1.gguf")
    else {
        eprintln!("skip — Qwen3.5 GGUF not present");
        return Ok(());
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("skip — no HIP device");
        return Ok(());
    }
    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    assert_eq!(cfg.arch, "qwen35");

    let cluster = HipCluster::new(&[0])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, 1);
    // Exit criterion for V2.2.a: upload succeeds without error. Forward
    // fails until V2.2.b (Q4_1 MMVQ) + V2.2.c (forward_dense_ffn_decode)
    // land.
    let model = Qwen3MoEShardedModel::load(&file, &cluster, &assignment)?;
    let bytes = model.shards[0].total_bytes();
    eprintln!(
        "qwen35 Mesh<1> loaded: {:.2} GiB on device 0",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    assert!(bytes > 4_000_000_000, "expected ~5 GiB on device");
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
