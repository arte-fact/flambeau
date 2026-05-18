//! Per-topology worker launch + dispatch.
//!
//! `launch` spawns one worker per rank and arranges the inter-thread
//! state they share (`ArCoordinator`, `peer_buffer`, handoff barrier).
//!
//! `run_forward` drives one decode step across every rank. Dispatch
//! shape per topology:
//! * SD — single Forward, wait.
//! * TP — Forward to every rank in parallel, wait on all (every rank
//!   produces post-AR logits; we keep the first).
//! * PP — Forward to every rank in parallel (workers serialise
//!   themselves on the peer_buffer mutex), keep the last stage's
//!   logits.
//! * Hybrid — Forward to every rank in parallel; per-stage TP
//!   coordinates through `ArCoordinator`, stages chain through the
//!   shared `peer_buffer` + handoff barrier.

use std::sync::{Arc, Barrier};

use anyhow::{anyhow, Context, Result};
use flambeau_quant::GgufFile;

use super::ar::{new_peer_buffer, ArCoordinator};
use super::workers::{WorkerHandle, WorkerRole};
use super::{Arch, Topology};

pub fn launch<A: Arch>(
    file: GgufFile,
    topology: &Topology,
    ctx_cap: Option<usize>,
    prefill_ubatch: usize,
    max_slots: usize,
) -> Result<Vec<WorkerHandle<A>>> {
    let file = Arc::new(file);
    match topology {
        Topology::SingleDevice { device } => {
            let h = WorkerHandle::<A>::spawn(
                *device,
                WorkerRole::Sd,
                Arc::clone(&file),
                ctx_cap,
                prefill_ubatch,
                max_slots,
            )?;
            Ok(vec![h])
        }
        Topology::Tp { devices } => {
            launch_tp::<A>(devices, file, ctx_cap, prefill_ubatch, max_slots)
        }
        Topology::Pp {
            devices,
            layer_split,
        } => launch_pp::<A>(
            devices,
            layer_split.as_deref(),
            file,
            ctx_cap,
            prefill_ubatch,
            max_slots,
        ),
        Topology::Hybrid {
            stages,
            layer_split,
        } => launch_hybrid::<A>(
            stages,
            layer_split.as_deref(),
            file,
            ctx_cap,
            prefill_ubatch,
            max_slots,
        ),
    }
}

fn launch_tp<A: Arch>(
    devices: &[i32],
    file: Arc<GgufFile>,
    ctx_cap: Option<usize>,
    prefill_ubatch: usize,
    max_slots: usize,
) -> Result<Vec<WorkerHandle<A>>> {
    let n = devices.len();
    let ar = Arc::new(ArCoordinator::new(n));
    let mut handles = Vec::with_capacity(n);
    for (rank, &dev) in devices.iter().enumerate() {
        let role = WorkerRole::Tp {
            rank,
            n_ranks: n,
            ar: Arc::clone(&ar),
        };
        handles.push(
            WorkerHandle::<A>::spawn(dev, role, Arc::clone(&file), ctx_cap, prefill_ubatch, max_slots)
                .with_context(|| format!("TP rank {rank} on hip:{dev}"))?,
        );
    }
    Ok(handles)
}

fn launch_pp<A: Arch>(
    devices: &[i32],
    layer_split: Option<&[usize]>,
    file: Arc<GgufFile>,
    ctx_cap: Option<usize>,
    prefill_ubatch: usize,
    max_slots: usize,
) -> Result<Vec<WorkerHandle<A>>> {
    let n = devices.len();
    let split = match layer_split {
        Some(s) => {
            if s.len() != n {
                return Err(anyhow!(
                    "layer_split.len() {} != devices.len() {n}",
                    s.len()
                ));
            }
            s.to_vec()
        }
        None => Vec::new(),
    };
    // peer_buffer hidden size is filled in lazily by the workers on
    // first DtoH deposit; we pre-allocate an empty Vec<f16> and let
    // the producer resize it.
    let peer = new_peer_buffer(0);
    let mut handles = Vec::with_capacity(n);
    let mut layer_cursor = 0usize;
    for (rank, &dev) in devices.iter().enumerate() {
        let (layer_start, layer_end) = if !split.is_empty() {
            let count = split[rank];
            let s = layer_cursor;
            layer_cursor += count;
            (s, s + count)
        } else {
            // Even split filled in by the caller via `layer_split` is
            // recommended — without it we punt with start=end=0 and
            // the model's Arch::forward must derive its own slice.
            (0, 0)
        };
        let role = WorkerRole::Pp {
            rank,
            n_ranks: n,
            layer_start,
            layer_end,
            peer_buffer: Arc::clone(&peer),
        };
        handles.push(
            WorkerHandle::<A>::spawn(dev, role, Arc::clone(&file), ctx_cap, prefill_ubatch, max_slots)
                .with_context(|| format!("PP rank {rank} on hip:{dev}"))?,
        );
    }
    Ok(handles)
}

fn launch_hybrid<A: Arch>(
    stages: &[Vec<i32>],
    layer_split: Option<&[usize]>,
    file: Arc<GgufFile>,
    ctx_cap: Option<usize>,
    prefill_ubatch: usize,
    max_slots: usize,
) -> Result<Vec<WorkerHandle<A>>> {
    let n_stages = stages.len();
    let total_ranks: usize = stages.iter().map(|s| s.len()).sum();
    let split: Vec<usize> = match layer_split {
        Some(s) => {
            if s.len() != n_stages {
                return Err(anyhow!(
                    "Hybrid layer_split.len() {} != stages.len() {n_stages}",
                    s.len()
                ));
            }
            s.to_vec()
        }
        None => Vec::new(),
    };
    let handoff = Arc::new(Barrier::new(total_ranks));
    let peer = new_peer_buffer(0);

    let mut handles = Vec::with_capacity(total_ranks);
    let mut layer_cursor = 0usize;
    for (stage_idx, ranks) in stages.iter().enumerate() {
        let tp_size = ranks.len();
        let ar = Arc::new(ArCoordinator::new(tp_size));
        let (layer_start, layer_end) = if !split.is_empty() {
            let s = layer_cursor;
            let count = split[stage_idx];
            layer_cursor += count;
            (s, s + count)
        } else {
            (0, 0)
        };
        for (rank_in_stage, &dev) in ranks.iter().enumerate() {
            let role = WorkerRole::Hybrid {
                stage_idx,
                n_stages,
                rank_in_stage,
                tp_size,
                layer_start,
                layer_end,
                ar: Arc::clone(&ar),
                peer_buffer: Arc::clone(&peer),
                handoff: Arc::clone(&handoff),
            };
            handles.push(
                WorkerHandle::<A>::spawn(
                    dev,
                    role,
                    Arc::clone(&file),
                    ctx_cap,
                    prefill_ubatch,
                    max_slots,
                )
                .with_context(|| {
                    format!("Hybrid stage {stage_idx} rank {rank_in_stage} on hip:{dev}")
                })?,
            );
        }
    }
    Ok(handles)
}

pub fn run_forward<A: Arch>(
    topology: &Topology,
    handles: &mut [WorkerHandle<A>],
    tokens: Vec<u32>,
    start_position: usize,
) -> Result<Vec<f32>> {
    match topology {
        Topology::SingleDevice { .. } => {
            let rx = handles[0].send_forward(tokens, start_position)?;
            rx.recv()
                .map_err(|e| anyhow!("SD reply channel closed: {e}"))?
        }
        Topology::Pp { .. } => {
            // PP has no barrier — `embed` on rank > 0 strictly requires
            // peer_buffer populated by rank N-1. Drive ranks in order.
            let mut last_logits = Vec::new();
            for h in handles.iter_mut() {
                let rx = h.send_forward(tokens.clone(), start_position)?;
                last_logits = rx
                    .recv()
                    .map_err(|e| anyhow!("PP reply channel closed: {e}"))??;
            }
            Ok(last_logits)
        }
        Topology::Tp { .. } | Topology::Hybrid { .. } => {
            // Parallel dispatch — workers coordinate through
            // ArCoordinator (TP) or ArCoordinator + handoff barrier
            // (Hybrid). All ranks must be live concurrently or the
            // barriers deadlock.
            let mut rxs = Vec::with_capacity(handles.len());
            for h in handles.iter_mut() {
                rxs.push(h.send_forward(tokens.clone(), start_position)?);
            }
            let mut last_nonempty: Option<Vec<f32>> = None;
            for rx in rxs {
                let logits = rx
                    .recv()
                    .map_err(|e| anyhow!("worker reply channel closed: {e}"))??;
                if !logits.is_empty() {
                    last_nonempty = Some(logits);
                }
            }
            last_nonempty.ok_or_else(|| anyhow!("no rank produced logits"))
        }
    }
}
