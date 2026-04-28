//! MTP-3 — smoke test: load `Qwen3.6-27B-mtp.gguf` to a HIP device and
//! verify the 15 mtp.* tensors land with expected shapes and dtypes.
//!
//! Skipped when:
//!   - no HIP device available, or
//!   - `/artefact/models/Qwen3.6-27B-mtp.gguf` not present (run
//!     `python3 tools/convert_qwen36_mtp.py --shards-dir tools/mtp_cache
//!     --out /artefact/models/Qwen3.6-27B-mtp.gguf` to produce it).
//!
//! Forward-pass parity vs the Python reference (`tools/mtp_reference.py`)
//! is the next step (MTP-3.5) — needs a couple of activation-precision
//! choices to be made (Q8_1 quantize-on-input for the linears, F16↔F32
//! negotiation between ops, gated-attn split).

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_quant::{GgmlDType, GgufFile};
use flambeau_qwen3_moe::mtp::{
    derive_sibling_mtp_path, load_mtp_head, verify_pairing, MTP_TENSOR_NAMES,
};
use std::path::PathBuf;

fn mtp_gguf_path() -> Option<PathBuf> {
    let p = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skip: {} not present (run tools/convert_qwen36_mtp.py)", p.display());
        None
    }
}

#[test]
fn mtp_load_smoke() -> Result<()> {
    let Some(mtp_path) = mtp_gguf_path() else {
        return Ok(());
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("skip: no HIP device");
        return Ok(());
    }

    let device = HipDevice::new(0)?;
    let file = GgufFile::open(&mtp_path)?;

    // Pairing: thin sibling stamps target_arch/hidden/vocab. Verify
    // against Qwen3.6-27B's known config values.
    verify_pairing(&file, "qwen35", 5120, 248320)?;

    let mtp = load_mtp_head(&file, &device)?;

    // Basic shape sanity — what we expect for Qwen3.6-27B.
    let h = 5120usize;
    let inter = 17408usize;
    let n_q = 24usize;
    let n_kv = 4usize;
    let head_dim = 256usize;

    fn elem_count(t: &flambeau_qwen3_moe::DeviceTensor) -> u64 {
        t.dims.iter().product()
    }

    // mtp.fc.weight: shape stored as [in=2*hidden, out=hidden] in GGUF
    // (logical [out, in] in PyTorch); element count = hidden * 2*hidden.
    assert_eq!(elem_count(&mtp.fc) as usize, h * 2 * h);
    assert_eq!(elem_count(&mtp.norm) as usize, h);
    assert_eq!(elem_count(&mtp.pre_fc_norm_hidden) as usize, h);
    assert_eq!(elem_count(&mtp.pre_fc_norm_embedding) as usize, h);

    let b = &mtp.block;
    assert_eq!(elem_count(&b.input_layernorm) as usize, h);
    assert_eq!(elem_count(&b.post_attention_layernorm) as usize, h);
    assert_eq!(elem_count(&b.q_proj) as usize, h * 2 * n_q * head_dim); // gated: Q || gate
    assert_eq!(elem_count(&b.k_proj) as usize, h * n_kv * head_dim);
    assert_eq!(elem_count(&b.v_proj) as usize, h * n_kv * head_dim);
    assert_eq!(elem_count(&b.o_proj) as usize, n_q * head_dim * h);
    assert_eq!(elem_count(&b.q_norm) as usize, head_dim);
    assert_eq!(elem_count(&b.k_norm) as usize, head_dim);
    assert_eq!(elem_count(&b.gate_proj) as usize, h * inter);
    assert_eq!(elem_count(&b.up_proj) as usize, h * inter);
    assert_eq!(elem_count(&b.down_proj) as usize, h * inter);

    // Dtype check: norms F32, linears Q8_0 (per the converter).
    for (name, expect) in [
        (&mtp.norm, GgmlDType::F32),
        (&mtp.pre_fc_norm_hidden, GgmlDType::F32),
        (&mtp.pre_fc_norm_embedding, GgmlDType::F32),
        (&b.input_layernorm, GgmlDType::F32),
        (&b.post_attention_layernorm, GgmlDType::F32),
        (&b.q_norm, GgmlDType::F32),
        (&b.k_norm, GgmlDType::F32),
        (&mtp.fc, GgmlDType::Q8_0),
        (&b.q_proj, GgmlDType::Q8_0),
        (&b.k_proj, GgmlDType::Q8_0),
        (&b.v_proj, GgmlDType::Q8_0),
        (&b.o_proj, GgmlDType::Q8_0),
        (&b.gate_proj, GgmlDType::Q8_0),
        (&b.up_proj, GgmlDType::Q8_0),
        (&b.down_proj, GgmlDType::Q8_0),
    ] {
        assert_eq!(name.dtype, expect, "tensor `{}` dtype", name.name);
    }

    // We loaded all 15.
    let total = mtp.total_bytes();
    eprintln!("MTP loaded: {} bytes ({:.1} MB)", total, total as f64 / 1e6);
    assert!(total > 400 * 1024 * 1024, "expected >400 MB; got {total}");
    assert!(total < 500 * 1024 * 1024, "expected <500 MB; got {total}");

    // MTP_TENSOR_NAMES sanity: 15 entries.
    assert_eq!(MTP_TENSOR_NAMES.len(), 15);
    Ok(())
}

#[test]
fn mtp_pairing_metadata_present() -> Result<()> {
    let Some(mtp_path) = mtp_gguf_path() else {
        return Ok(());
    };
    let file = GgufFile::open(&mtp_path)?;
    assert_eq!(file.metadata_str("general.architecture"), Some("qwen35-mtp"));
    assert_eq!(file.metadata_str("mtp.target_arch"), Some("qwen35"));
    assert_eq!(file.metadata_u32("mtp.target_hidden_size"), Some(5120));
    assert_eq!(file.metadata_u32("mtp.target_vocab_size"), Some(248320));
    assert_eq!(file.metadata_u32("mtp.num_layers"), Some(1));
    Ok(())
}

#[test]
fn mtp_pairing_rejects_mismatch() -> Result<()> {
    let Some(mtp_path) = mtp_gguf_path() else {
        return Ok(());
    };
    let file = GgufFile::open(&mtp_path)?;
    // Wrong base arch
    assert!(verify_pairing(&file, "wrongarch", 5120, 248320).is_err());
    // Wrong base hidden_size
    assert!(verify_pairing(&file, "qwen35", 4096, 248320).is_err());
    // Wrong base vocab_size
    assert!(verify_pairing(&file, "qwen35", 5120, 32000).is_err());
    // Correct
    verify_pairing(&file, "qwen35", 5120, 248320)?;
    Ok(())
}

#[test]
fn sibling_path_derivation() {
    use std::path::Path;
    let cases = [
        ("/m/Qwen3.6-27B-Q4_0.gguf",         "/m/Qwen3.6-27B-mtp.gguf"),
        ("/m/Qwen3.6-27B-Q8_0.gguf",         "/m/Qwen3.6-27B-mtp.gguf"),
        ("/m/Qwen3.6-27B-UD-Q4_K_XL.gguf",   "/m/Qwen3.6-27B-mtp.gguf"),
        ("/m/Qwen3.6-27B-UD-Q6_K_XL.gguf",   "/m/Qwen3.6-27B-mtp.gguf"),
    ];
    for (base, expected) in cases {
        let got = derive_sibling_mtp_path(Path::new(base));
        assert_eq!(got, std::path::PathBuf::from(expected), "for base {base}");
    }
}
