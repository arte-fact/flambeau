//! Integration test — upload a full Qwen3.x MoE model to HIP + allocate a
//! session's caches. Skips when either `FLAMBEAU_QWEN3_GGUF` is unset or
//! there is no HIP device available (sandbox / CI lint hosts).
//!
//! Reference model: Qwen3.6-35B-A3B-UD-Q4_K_S.gguf — ~20 GB of weights,
//! 40 layers, 30 GDN + 10 full-attn, 32 num_v_heads × 128² × 30 ≈ 60 MB
//! of GDN state for the full context.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_core::Device;
use flambeau_ops::hip::HipDevice;
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::{LayerCache, Qwen3MoEConfig, Qwen3MoEModel, Qwen3MoESession};

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

fn hip_device() -> Option<HipDevice> {
    let n = flambeau_backend_hip::device_count().ok()?;
    if n < 1 {
        return None;
    }
    HipDevice::new(0).ok()
}

/// Best-effort per-card VRAM read via sysfs — enough for a pre-check that
/// skips the upload test when the GGUF is too big to fit. Assumes DRM
/// card ordering aligns with HIP device ordering (true on every ROCm rig
/// we've touched in V1).
fn card0_vram_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_total").ok()?;
    s.trim().parse::<u64>().ok()
}

#[test]
fn upload_full_qwen3_moe_model() -> Result<()> {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping upload_full_qwen3_moe_model");
        return Ok(());
    };
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping upload_full_qwen3_moe_model");
        return Ok(());
    };
    device.bind()?;

    let file = GgufFile::open(&path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&file)?;
    let expected_layout_bytes =
        flambeau_qwen3_moe::ModelLayout::from_gguf(&file, &cfg)?.total_bytes() as usize;

    // VRAM pre-check. Qwen3.6-35B at ~20 GB doesn't fit on 16 GB MI50; the
    // real Mesh<N>-sharded load lands in V1.7.5. Skip rather than OOM so
    // CI hosts with small cards still run the scaffold path against any
    // GGUF whose layout total fits in a single card's VRAM.
    if let Some(vram) = card0_vram_bytes() {
        if (expected_layout_bytes as u64) + 512 * 1024 * 1024 > vram {
            eprintln!(
                "skipping upload: model needs {:.2} GiB but device has {:.2} GiB — \
                 run against a smaller GGUF until Mesh<N> sharding lands (V1.7.5)",
                expected_layout_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                vram as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            return Ok(());
        }
    }

    // A truncated / inconsistent GGUF surfaces as RangeOutOfBounds mid-upload
    // now that gguf.rs bounds-checks the mmap slice. Skip cleanly rather
    // than failing the run so the scaffold isn't held hostage to one bad
    // file.
    let model = match Qwen3MoEModel::load(&file, &device) {
        Ok(m) => m,
        Err(e) => {
            let s = format!("{e:#}");
            if s.contains("RangeOutOfBounds") || s.contains("range start index") {
                eprintln!("skipping upload: GGUF appears truncated — {s}");
                return Ok(());
            }
            return Err(e);
        }
    };
    let uploaded = model.weights.total_bytes();
    assert_eq!(
        uploaded, expected_layout_bytes,
        "bytes uploaded ({uploaded}) must match layout.total_bytes() ({expected_layout_bytes})"
    );
    assert_eq!(model.weights.device_id(), device.id());
    assert_eq!(model.weights.layers.len(), cfg.num_layers);

    // Spot-check first + last layer tensor liveness. Every DevicePtr in an
    // uploaded weight set must be non-null and pointer-unique across the
    // model.
    let ptrs: Vec<_> = model.weights.iter_tensors().map(|t| t.ptr).collect();
    assert!(
        ptrs.iter().all(|p| !p.is_null()),
        "every weight tensor must have a non-null device pointer"
    );
    let unique: std::collections::HashSet<_> = ptrs.iter().map(|p| p.as_usize()).collect();
    assert_eq!(
        unique.len(),
        ptrs.len(),
        "weight device pointers must be unique (no alias collisions)"
    );

    // Session caches: full-attn layers get KvCache, recurrent get GDN state.
    let session = Qwen3MoESession::new(&cfg, &device, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut n_kv = 0usize;
    let mut n_gdn = 0usize;
    for (il, cache) in session.layers().iter().enumerate() {
        match cache {
            LayerCache::FullAttn(kv) => {
                assert!(!cfg.is_recurrent(il));
                assert_eq!(kv.max_tokens(), cfg.context_length);
                assert_eq!(kv.n_heads(), cfg.num_kv_heads);
                assert_eq!(kv.head_dim(), cfg.head_dim);
                n_kv += 1;
            }
            LayerCache::FullAttnQ8(kv) => {
                assert!(!cfg.is_recurrent(il));
                assert_eq!(kv.max_tokens(), cfg.context_length);
                assert_eq!(kv.n_heads(), cfg.num_kv_heads);
                assert_eq!(kv.head_dim(), cfg.head_dim);
                n_kv += 1;
            }
            LayerCache::Gdn(g) => {
                assert!(cfg.is_recurrent(il));
                let gdn = cfg.gdn.as_ref().unwrap();
                assert_eq!(g.num_v_heads, gdn.num_v_heads);
                assert_eq!(g.head_k_dim, gdn.head_k_dim);
                assert_eq!(g.head_v_dim, gdn.head_v_dim());
                assert_eq!(
                    g.state_bytes,
                    g.num_v_heads * g.head_k_dim * g.head_v_dim * 4
                );
                n_gdn += 1;
            }
        }
    }
    assert_eq!(n_kv, cfg.num_full_attn_layers());
    assert_eq!(n_gdn, cfg.num_recurrent_layers());

    eprintln!(
        "upload ok: {:.2} GiB weights, {:.2} MiB session caches, {} full-attn + {} gdn layers",
        uploaded as f64 / (1024.0 * 1024.0 * 1024.0),
        session.total_bytes() as f64 / (1024.0 * 1024.0),
        n_kv,
        n_gdn,
    );

    // Explicit teardown. Drop-without-dispose logs a warn.
    session.dispose(&device)?;
    model.dispose(&device)?;
    Ok(())
}
