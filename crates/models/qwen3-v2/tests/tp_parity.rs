//! tp_size=2 parity vs SD on Qwen3-Embedding-0.6B. One thread per
//! rank; AR via shared `Mutex<partials>` + `Barrier` (host roundtrip).
//! F32 reduction-tree order differs from SD so bit-equality isn't
//! mathematically guaranteed — the test tolerates ≤ 5e-2 abs diff
//! and asserts argmax match.

#![cfg(feature = "hip")]

use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::tp::{TpForwardCtx, TpHooks};
use flambeau_forward::ForwardCtx;
use flambeau_ops::OpsRegistry;
use flambeau_qwen3_v2::{forward_one_token, load_from_gguf, load_tp_shard_from_gguf};
use flambeau_quant::GgufFile;

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

struct ArCoordinator {
    n_ranks: usize,
    partials: Mutex<Vec<Option<Vec<f32>>>>,
    barrier: Barrier,
}

impl ArCoordinator {
    fn new(n_ranks: usize) -> Self {
        Self {
            n_ranks,
            partials: Mutex::new(vec![None; n_ranks]),
            barrier: Barrier::new(n_ranks),
        }
    }
}

fn ar_sum_via_coordinator(
    coord: Arc<ArCoordinator>,
    rank: usize,
    _n_ranks: usize,
    buf: DevicePtr,
    n_elems: usize,
    device: &flambeau_backend_hip::HipDevice,
    stream: &flambeau_backend_hip::HipStream,
) -> anyhow::Result<()> {
    let bytes = n_elems * 4;
    let mut host = vec![0.0_f32; n_elems];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            buf,
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;

    {
        let mut p = coord.partials.lock().unwrap();
        p[rank] = Some(host);
    }
    coord.barrier.wait();

    // Every rank sums (cheap; result is identical across ranks).
    let summed: Vec<f32> = {
        let p = coord.partials.lock().unwrap();
        let mut s = p[0].as_ref().unwrap().clone();
        for r in 1..coord.n_ranks {
            let other = p[r].as_ref().unwrap();
            for i in 0..n_elems {
                s[i] += other[i];
            }
        }
        s
    };

    // Second barrier: no rank may clear partials before all have read.
    coord.barrier.wait();

    if rank == 0 {
        let mut p = coord.partials.lock().unwrap();
        for r in 0..coord.n_ranks {
            p[r] = None;
        }
    }
    coord.barrier.wait();

    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            buf,
            DevicePtr(summed.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream)?;
    Ok(())
}

#[test]
fn tp_size_2_logits_match_single_device() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");

    // SD baseline.
    let baseline: Vec<f32> = {
        let device = HipDevice::new(0).expect("HIP device 0");
        device.bind().expect("bind");
        let mut model = load_from_gguf(&file, &device).expect("load_from_gguf");
        let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
        let stream = device.default_stream();
        let max_seq_len = 64.min(model.config.context_length);
        let cfg = ScratchConfig {
            hidden: model.config.hidden,
            intermediate: model.config.intermediate,
            q_width: model.config.n_heads * model.config.head_dim,
            kv_width: model.config.n_kv_heads * model.config.head_dim,
            vocab: model.config.vocab_size,
            max_seq_len,
            num_layers: model.config.num_layers,
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("SD pool");
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        forward_one_token(&model, &mut ctx, 1, 0).expect("SD forward");
        let logits = ctx.logits().to_vec();
        drop(ctx);
        pool.dispose(&device).expect("pool dispose");
        model.dispose(&device).expect("model dispose");
        logits
    };

    // tp_size=2.
    let n_ranks = 2;
    let coord = Arc::new(ArCoordinator::new(n_ranks));
    let mut handles = Vec::with_capacity(n_ranks);
    let model_path = path.clone();
    for rank in 0..n_ranks {
        let coord = Arc::clone(&coord);
        let model_path = model_path.clone();
        let handle = thread::spawn(move || -> anyhow::Result<Vec<f32>> {
            let file = GgufFile::open(&model_path)?;
            let device = HipDevice::new(0)?;
            device.bind()?;
            let mut model = load_tp_shard_from_gguf(&file, &device, rank, n_ranks)?;
            let reg = OpsRegistry::new(&device)?;
            let stream = device.default_stream();
            let max_seq_len = 64.min(model.config.context_length);
            let cfg = ScratchConfig {
                hidden: model.config.hidden,
                intermediate: model.config.intermediate / n_ranks,
                q_width: (model.config.n_heads / n_ranks) * model.config.head_dim,
                kv_width: (model.config.n_kv_heads / n_ranks) * model.config.head_dim,
                vocab: model.config.vocab_size,
                max_seq_len,
                num_layers: model.config.num_layers,
            };
            let mut pool = ScratchPool::new(&device, cfg)?;
            let coord_for_hook = Arc::clone(&coord);
            let hooks = TpHooks {
                rank,
                n_ranks,
                ar_callback: Box::new(move |r, nr, buf, n, dev, str_| {
                    ar_sum_via_coordinator(
                        Arc::clone(&coord_for_hook),
                        r,
                        nr,
                        buf,
                        n,
                        dev,
                        str_,
                    )
                }),
            };
            let mut ctx = TpForwardCtx::new(&device, stream, &reg, &mut pool, hooks);
            forward_one_token(&model, &mut ctx, 1, 0)?;
            let logits = ctx.logits().to_vec();
            drop(ctx);
            pool.dispose(&device)?;
            model.dispose(&device)?;
            Ok(logits)
        });
        handles.push(handle);
    }

    let mut per_rank_logits: Vec<Vec<f32>> = handles
        .into_iter()
        .enumerate()
        .map(|(rank, h)| {
            h.join()
                .unwrap_or_else(|e| panic!("rank {rank} thread panicked: {e:?}"))
                .unwrap_or_else(|e| panic!("rank {rank} forward failed: {e:#}"))
        })
        .collect();
    let tp_rank0 = per_rank_logits.remove(0);
    let tp_rank1 = per_rank_logits.remove(0);
    drop(per_rank_logits);

    // output_head is replicated post-AR → every rank must agree.
    let mut inter_rank_diff = 0.0_f32;
    for (a, b) in tp_rank0.iter().zip(tp_rank1.iter()) {
        inter_rank_diff = inter_rank_diff.max((a - b).abs());
    }
    eprintln!("TP rank0 vs rank1 max_abs_diff = {inter_rank_diff:.6}");
    assert!(
        inter_rank_diff == 0.0,
        "TP ranks diverged: {inter_rank_diff:.6} (AR not deterministic?)"
    );

    let mut max_abs_diff = 0.0_f32;
    let mut worst = 0;
    for (i, (&a, &b)) in tp_rank0.iter().zip(baseline.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            worst = i;
        }
    }
    let baseline_argmax = baseline
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let tp_argmax = tp_rank0
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "TP vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst} (SD={:.4} TP={:.4})  argmax SD={baseline_argmax} TP={tp_argmax}",
        baseline[worst],
        tp_rank0[worst]
    );
    assert_eq!(tp_argmax, baseline_argmax, "argmax differs SD vs TP");
    // Bound is generous; F32 noise from differing reduction-tree order.
    assert!(
        max_abs_diff < 5e-2,
        "TP logits diverge from SD by {max_abs_diff:.6} at idx {worst}"
    );
}
