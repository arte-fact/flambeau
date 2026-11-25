//! End-to-end NQ2 integration through the real `qmatmul()` dispatch path:
//! GGUF → raw upload → `qmatmul(.., QDtype::NQ2)` on HIP, checked against the
//! CPU `dequantize_nq2` reference, then benched over the full 0.6B layer
//! sequence (m = 1 and m = 4) for decode t/s. With `--q4 <gguf>` the same
//! sequence is benched with Q4_K_M weights through the dispatch table.
//!
//! Usage:
//!   cargo run --release -p flambeau-ops --features hip --example nq2_e2e -- \
//!     /path/Qwen3-0.6B-NQ2.gguf [--q4 /path/Qwen3-0.6B-Q4_K_M.gguf] [--tensor blk.0.attn_q.weight]

use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, QDtype, Stream};
use flambeau_ops::qmatmul::qmatmul;
use flambeau_ops::{MatmulShape, OpCtx, OpsRegistry, QmatmulBuffers};
use flambeau_quant::{dequantize_nq2, BlockQ8_1, GgmlDType, GgufFile};
use half::f16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn download(dev: &HipDevice, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            ptr,
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

/// Host Q8_1 quantization matching BlockQ8_1 { d, s, qs[32] }; the mmvq_nq_bin
/// stage-1 kernel reads `yb->d * yb->qs[j & 31]`.
fn quantize_q8_1(x: &[f32]) -> Vec<BlockQ8_1> {
    assert_eq!(x.len() % 32, 0);
    x.chunks(32)
        .map(|c| {
            let amax = c.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            let mut qs = [0i8; 32];
            for (i, &v) in c.iter().enumerate() {
                qs[i] = (v / d).round().clamp(-127.0, 127.0) as i8;
            }
            BlockQ8_1 {
                d: f16::from_f32(d),
                s: f16::from_f32(0.0),
                qs,
            }
        })
        .collect()
}

fn dequant_q8_1(blocks: &[BlockQ8_1]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * 32);
    for b in blocks {
        let d = b.d.to_f32();
        for &q in &b.qs {
            out.push(d * q as f32);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gguf_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| panic!("usage: nq2_e2e <nq2.gguf> [--q4 <q4.gguf>] [--tensor NAME]"));
    let q4_path = args
        .windows(2)
        .find(|w| w[0] == "--q4")
        .map(|w| w[1].clone());
    let tensor_name = args
        .windows(2)
        .find(|w| w[0] == "--tensor")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "blk.0.attn_q.weight".to_string());

    let dev = HipDevice::new(0).expect("HipDevice::new(0)");
    let reg = OpsRegistry::new(&dev).expect("OpsRegistry::new");
    let stream = dev.default_stream();
    let ctx = OpCtx {
        reg: &reg,
        stream,
    };

    let file = GgufFile::open(&gguf_path).expect("open gguf");
    let nq2_names: Vec<String> = file
        .tensor_order
        .iter()
        .filter(|n| file.tensors[*n].dtype == GgmlDType::NQ2)
        .cloned()
        .collect();
    eprintln!("{}: {} NQ2 tensors", gguf_path, nq2_names.len());
    assert!(!nq2_names.is_empty());

    // ---- correctness on one tensor through the real qmatmul() dispatch ----
    let info = file.info(&tensor_name).expect("tensor");
    assert_eq!(info.dtype, GgmlDType::NQ2);
    let n = info.dims[0] as usize; // dout
    let k = info.dims[1] as usize; // din
    let raw = file.tensor_raw(&tensor_name).expect("raw").to_vec();
    let w_dev = upload(&dev, &raw);

    let mut rng = Rng(0x5EED_0001);
    let x: Vec<f32> = (0..k).map(|_| rng.f32()).collect();
    let blocks = quantize_q8_1(&x);
    let act_dev = upload(&dev, &blocks);
    let dst_dev = dev.alloc(n * 4).unwrap();

    qmatmul(
        ctx,
        QmatmulBuffers {
            weights: w_dev,
            act_q8_1: act_dev,
            act_q8_1_mmq: DevicePtr(0),
            dst: dst_dev,
        },
        MatmulShape { m: 1, k, n },
        QDtype::NQ2,
    )
    .expect("qmatmul NQ2");

    let y_raw = download(&dev, dst_dev, n * 4);
    let y_gpu: Vec<f32> = y_raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // CPU reference: y = W_hat @ x_deq, W_hat from dequantize_nq2
    let mut w_hat = vec![0.0f32; n * k];
    dequantize_nq2(&raw, &mut w_hat, n, k).expect("dequant");
    let x_deq = dequant_q8_1(&blocks);
    let mut max_rel = 0.0f32;
    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += w_hat[i * k + j] * x_deq[j];
        }
        let denom = acc.abs().max(1e-3);
        max_rel = max_rel.max((y_gpu[i] - acc).abs() / denom);
    }
    eprintln!("correctness {tensor_name}: n={n} k={k} max_rel_err={max_rel:.3e}");
    assert!(max_rel < 1e-3, "NQ2 qmatmul mismatch: {max_rel:.3e}");

    // ---- bench: one full pass over all NQ2 linears = one decode token ----
    let mut bench = |label: &str, dtype: QDtype, path: &str, m: usize| {
        let f = GgufFile::open(path).unwrap();
        let names: Vec<String> = f
            .tensor_order
            .iter()
            .filter(|n| f.tensors[*n].dtype == (if dtype == QDtype::NQ2 { GgmlDType::NQ2 } else { GgmlDType::Q4K }))
            .cloned()
            .collect();
        let max_k = names
            .iter()
            .map(|n| f.tensors[n].dims[1] as usize)
            .max()
            .unwrap();
        let max_n = names
            .iter()
            .map(|n| f.tensors[n].dims[0] as usize)
            .max()
            .unwrap();
        // per-tensor device weights + per-k activation rows (slot-major)
        let mut w_devs = Vec::new();
        let mut act_by_k: std::collections::HashMap<usize, DevicePtr> = Default::default();
        for name in &names {
            let raw = f.tensor_raw(name).unwrap().to_vec();
            w_devs.push(upload(&dev, &raw));
            let kk = f.tensors[name].dims[1] as usize;
            act_by_k.entry(kk).or_insert_with(|| {
                let x: Vec<f32> = (0..kk).map(|_| rng.f32()).collect();
                let b: Vec<BlockQ8_1> = quantize_q8_1(&x);
                // replicate to m slots
                let mut all = Vec::with_capacity(b.len() * m);
                for _ in 0..m {
                    all.extend_from_slice(&b);
                }
                upload(&dev, &all)
            });
        }
        let dst_dev = dev.alloc(max_n * m * 4).unwrap();
        // warmup
        for (i, name) in names.iter().enumerate() {
            let nn = f.tensors[name].dims[0] as usize;
            let kk = f.tensors[name].dims[1] as usize;
            qmatmul(
                ctx,
                QmatmulBuffers {
                    weights: w_devs[i],
                    act_q8_1: act_by_k[&kk],
                    act_q8_1_mmq: DevicePtr(0),
                    dst: dst_dev,
                },
                MatmulShape { m, k: kk, n: nn },
                dtype,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for (i, name) in names.iter().enumerate() {
                let nn = f.tensors[name].dims[0] as usize;
                let kk = f.tensors[name].dims[1] as usize;
                qmatmul(
                    ctx,
                    QmatmulBuffers {
                        weights: w_devs[i],
                        act_q8_1: act_by_k[&kk],
                        act_q8_1_mmq: DevicePtr(0),
                        dst: dst_dev,
                    },
                    MatmulShape { m, k: kk, n: nn },
                    dtype,
                )
                .unwrap();
            }
            stream.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let ms = best * 1000.0;
        eprintln!(
            "bench {label} m={m}: {ms:.3} ms/token-equivalent -> {:.2} t/s ({} layers)",
            1000.0 / ms,
            names.len()
        );
        (ms, names.len())
    };

    bench("NQ2", QDtype::NQ2, &gguf_path, 1);
    bench("NQ2", QDtype::NQ2, &gguf_path, 4);
    if let Some(q4) = &q4_path {
        bench("Q4_K_M", QDtype::Q4_K, q4, 1);
        bench("Q4_K_M", QDtype::Q4_K, q4, 4);
    }
    eprintln!("OK");
}
