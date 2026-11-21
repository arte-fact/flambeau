//! Every row of a real GGUF's Q4_K `token_embd.weight`, gathered on device by
//! `embed_gather_q4_k_f16`, must match the host load path bit for bit.
//! Skips unless `FLAMBEAU_TEST_Q4K_GGUF` names the GGUF, e.g.
//!   FLAMBEAU_TEST_Q4K_GGUF=~/models/Qwen3.8-27B-UD-Q4_K_M.gguf \
//!   cargo test -p flambeau-bench --features cuda --test embed_gather_q4k_real_gguf

#![cfg(any(feature = "hip", feature = "cuda"))]

use std::path::PathBuf;

use flambeau_bench::sweep_embed_gather_q4k::check_gguf_token_embd;

fn gguf() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("FLAMBEAU_TEST_Q4K_GGUF")?);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skip: {} absent", p.display());
        None
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_token_embd_bit_identical() {
    let Some(path) = gguf() else { return };
    let (rows, bad) = check_gguf_token_embd::<flambeau_bench::backend::Cuda>(&path).expect("gather check");
    assert_eq!(bad, 0, "{bad} differing F16 elements over {rows} rows");
}

#[cfg(feature = "hip")]
#[test]
fn hip_token_embd_bit_identical() {
    let Some(path) = gguf() else { return };
    let (rows, bad) = check_gguf_token_embd::<flambeau_bench::backend::Hip>(&path).expect("gather check");
    assert_eq!(bad, 0, "{bad} differing F16 elements over {rows} rows");
}
