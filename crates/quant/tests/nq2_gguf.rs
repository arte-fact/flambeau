//! Cross-check of the NQ2 GGUF path against the Python converter
//! (scripts/nq_pt_to_gguf.py). Skips unless NQ2_GGUF / NQ2_REF_F32 are set.
//!
//!   NQ2_GGUF=/path/Qwen3-0.6B-NQ2.gguf \
//!   NQ2_REF_F32=/tmp/nq_ref_attn_q.f32 \
//!   NQ2_TENSOR=blk.0.attn_q.weight \
//!   cargo test -p flambeau-quant --test nq2_gguf -- --nocapture

use flambeau_quant::{dequantize_nq2, GgmlDType, GgufFile};

#[test]
fn nq2_gguf_roundtrip_vs_python() {
    let Ok(gguf_path) = std::env::var("NQ2_GGUF") else {
        eprintln!("NQ2_GGUF not set — skipping");
        return;
    };
    let Ok(ref_path) = std::env::var("NQ2_REF_F32") else {
        eprintln!("NQ2_REF_F32 not set — skipping");
        return;
    };
    let tensor = std::env::var("NQ2_TENSOR").unwrap_or_else(|_| "blk.0.attn_q.weight".into());

    let file = GgufFile::open(&gguf_path).expect("open gguf");
    let info = file.info(&tensor).expect("tensor present");
    assert_eq!(info.dtype, GgmlDType::NQ2, "{tensor} dtype");
    let dout = info.dims[0] as usize;
    let din = info.dims[1] as usize;
    let r = flambeau_quant::gguf::nq2_rank(dout as u64, din as u64) as usize;
    let wpr = r.div_ceil(32);
    let expected_bytes = (dout + din) * wpr * 4 + (dout + din) * 2;
    assert_eq!(
        info.size_in_bytes() as usize,
        expected_bytes,
        "{tensor} blob size"
    );

    // raw blob dequant vs python reference
    let raw = file.tensor_raw(&tensor).expect("raw");
    assert_eq!(raw.len(), expected_bytes);
    let mut out = vec![0.0f32; dout * din];
    dequantize_nq2(raw, &mut out, dout, din).expect("dequant");
    let reference = std::fs::read(&ref_path).expect("read ref");
    let ref_f32: &[f32] = bytemuck::cast_slice(&reference);
    assert_eq!(ref_f32.len(), dout * din);
    let max_abs = out
        .iter()
        .zip(ref_f32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = ref_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("{tensor}: dout={dout} din={din} r={r} max|diff|={max_abs:.3e} max|ref|={scale:.3e}");
    assert!(max_abs <= 1e-5 * scale, "max_abs={max_abs} scale={scale}");

    // and through the GgufFile convenience path
    let via_file = file.dequantize_tensor(&tensor).expect("dequantize_tensor");
    assert_eq!(via_file.len(), out.len());
    let max_abs2 = via_file
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_abs2, 0.0);
}
