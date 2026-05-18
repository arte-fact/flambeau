#![cfg(feature = "hip")]

use std::io::Write;
use std::path::PathBuf;

use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::session::KvLayout;
use flambeau_qwen3_moe::Qwen3MoEPpDriver;
use flambeau_runtime::ModelDriver;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

const PROMPT: &[u32] = &[248045, 846, 198, 20206];

#[test]
#[ignore = "parity dump — run via scripts/parity/v2_vs_legacy.sh"]
fn parity_dump_legacy() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let out_path = std::env::var("FLAMBEAU_PARITY_OUT")
        .expect("set FLAMBEAU_PARITY_OUT to the dump path");

    let file = GgufFile::open(&path).expect("open gguf");
    let mut driver = Qwen3MoEPpDriver::load(&file, &[0], PROMPT.len() + 4, KvLayout::F16)
        .expect("legacy PP=1 load");

    let mut f = std::fs::File::create(&out_path).expect("create dump");
    for (i, &t) in PROMPT.iter().enumerate() {
        let mut logits = Vec::new();
        driver
            .forward_one_token_logits(t, i, &mut logits)
            .expect("legacy fwd");
        let bytes = bytemuck::cast_slice::<f32, u8>(&logits);
        let len = logits.len() as u32;
        f.write_all(&len.to_le_bytes()).expect("write len");
        f.write_all(bytes).expect("write logits");
        let (mut bi, mut bv) = (0_u32, f32::NEG_INFINITY);
        for (k, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = k as u32;
            }
        }
        eprintln!("step {i} tok={t} vocab={} argmax={bi}", logits.len());
    }
    driver.dispose().expect("dispose");
    eprintln!("wrote {out_path}");
}
