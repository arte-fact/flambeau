//! pp_size=2 × tp_size=2 (4 ranks) parity vs SD on Qwen3-Embedding-0.6B.
//! Two per-stage AR coordinators + one inter-stage peer_buffer + a
//! 4-rank handoff barrier.

#![cfg(feature = "hip")]

use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_forward::hybrid::HybridForwardCtx;
use flambeau_forward::single_device::{ScratchConfig, ScratchPool, SingleDeviceForwardCtx};
use flambeau_forward::ForwardCtx;
use flambeau_ops::OpsRegistry;
use flambeau_qwen3_v2::{forward_one_token, load_from_gguf, load_tp_shard_from_gguf};
use flambeau_quant::GgufFile;
use half::f16;

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
fn pp2_tp2_logits_match_single_device() {
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

    // Shape probe (cheap re-parse to learn num_layers).
    let probe_cfg = flambeau_qwen3_v2::Qwen3V2Config::from_gguf(&file)
        .expect("parse config for shape probe");
    let num_layers = probe_cfg.num_layers;
    let hidden = probe_cfg.hidden;

    // pp_size=2, tp_size=2.
    let n_stages = 2;
    let tp_size = 2;
    let total_ranks = n_stages * tp_size;
    let split = num_layers / n_stages;

    let stage_ar = vec![
        Arc::new(ArCoordinator::new(tp_size)),
        Arc::new(ArCoordinator::new(tp_size)),
    ];
    let peer_buffer: Arc<Mutex<Vec<f16>>> = Arc::new(Mutex::new(vec![f16::ZERO; hidden]));
    let handoff = Arc::new(Barrier::new(total_ranks));
    let model_path = path.clone();

    let mut handles = Vec::with_capacity(total_ranks);
    for stage_idx in 0..n_stages {
        for rank_in_stage in 0..tp_size {
            let stage_ar = Arc::clone(&stage_ar[stage_idx]);
            let peer_buffer = Arc::clone(&peer_buffer);
            let handoff = Arc::clone(&handoff);
            let model_path = model_path.clone();
            let handle = thread::spawn(move || -> anyhow::Result<(usize, usize, Vec<f32>)> {
                let file = GgufFile::open(&model_path)?;
                let device = HipDevice::new(0)?;
                device.bind()?;
                let mut model =
                    load_tp_shard_from_gguf(&file, &device, rank_in_stage, tp_size)?;
                let reg = OpsRegistry::new(&device)?;
                let stream = device.default_stream();
                let max_seq_len = 64.min(model.config.context_length);

                let layer_start = stage_idx * split;
                let layer_end = if stage_idx + 1 == n_stages {
                    model.config.num_layers
                } else {
                    layer_start + split
                };

                let cfg = ScratchConfig {
                    hidden: model.config.hidden,
                    intermediate: model.config.intermediate / tp_size,
                    q_width: (model.config.n_heads / tp_size) * model.config.head_dim,
                    kv_width: (model.config.n_kv_heads / tp_size) * model.config.head_dim,
                    vocab: model.config.vocab_size,
                    max_seq_len,
                    num_layers: layer_end - layer_start,
                };
                let mut pool = ScratchPool::new(&device, cfg)?;

                let ar = Arc::clone(&stage_ar);
                let ar_callback: Box<
                    dyn FnMut(
                            usize,
                            usize,
                            DevicePtr,
                            usize,
                            &flambeau_backend_hip::HipDevice,
                            &flambeau_backend_hip::HipStream,
                        ) -> anyhow::Result<()>
                        + Send,
                > = Box::new(move |r, n, buf, n_elems, dev, str_| {
                    ar_sum_via_coordinator(Arc::clone(&ar), r, n, buf, n_elems, dev, str_)
                });

                let mut ctx = HybridForwardCtx::new(
                    &device,
                    stream,
                    &reg,
                    &mut pool,
                    stage_idx,
                    n_stages,
                    rank_in_stage,
                    tp_size,
                    layer_start,
                    layer_end,
                    ar_callback,
                    Arc::clone(&peer_buffer),
                    Arc::clone(&handoff),
                );
                forward_one_token(&model, &mut ctx, 1, 0)?;
                let logits = ctx.logits().to_vec();
                drop(ctx);
                pool.dispose(&device)?;
                model.dispose(&device)?;
                Ok((stage_idx, rank_in_stage, logits))
            });
            handles.push(handle);
        }
    }

    let mut per_rank: Vec<(usize, usize, Vec<f32>)> = handles
        .into_iter()
        .map(|h| {
            h.join()
                .unwrap_or_else(|e| panic!("thread panicked: {e:?}"))
                .unwrap_or_else(|e| panic!("rank forward failed: {e:#}"))
        })
        .collect();

    // Last-stage ranks own the logits; other ranks return an empty buffer.
    let last_stage_logits: Vec<Vec<f32>> = per_rank
        .iter()
        .filter(|(s, _, _)| *s == n_stages - 1)
        .map(|(_, _, l)| l.clone())
        .collect();
    assert_eq!(last_stage_logits.len(), tp_size);
    let head_logits = &last_stage_logits[0];

    // Last-stage TP ranks must agree (output_head is replicated post-AR).
    let mut inter_rank_diff = 0.0_f32;
    for other in &last_stage_logits[1..] {
        for (a, b) in head_logits.iter().zip(other.iter()) {
            inter_rank_diff = inter_rank_diff.max((a - b).abs());
        }
    }
    eprintln!("hybrid last-stage rank0 vs rank1 max_abs_diff = {inter_rank_diff:.6}");
    assert_eq!(inter_rank_diff, 0.0, "last-stage ranks diverged");

    // Non-last stages return empty logits.
    for (s, r, l) in &per_rank {
        if *s != n_stages - 1 {
            assert!(l.is_empty(), "stage {s} rank {r} has logits but shouldn't");
        }
    }

    let mut max_abs_diff = 0.0_f32;
    let mut worst = 0;
    for (i, (&a, &b)) in head_logits.iter().zip(baseline.iter()).enumerate() {
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
    let hyb_argmax = head_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "hybrid pp2tp2 vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst} (SD={:.4} Hyb={:.4})  argmax SD={baseline_argmax} Hyb={hyb_argmax}",
        baseline[worst],
        head_logits[worst]
    );
    assert_eq!(hyb_argmax, baseline_argmax);
    assert!(
        max_abs_diff < 5e-2,
        "hybrid logits diverge from SD by {max_abs_diff:.6} at idx {worst}"
    );

    per_rank.clear();
}
