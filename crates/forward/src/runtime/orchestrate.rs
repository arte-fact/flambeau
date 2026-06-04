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

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster};
use flambeau_quant::GgufFile;

use super::ar::{new_peer_edge, new_peer_edge_prealloc, ArCoordinator, BarArCoordinator};
use super::workers::{WorkerHandle, WorkerRole};
use super::{Arch, Topology};

/// Build a BAR1 P2P AR coordinator for a TP cluster, if peer-access
/// permits. Returns `None` on partial peer-access matrices, missing
/// hsaco, or BAR1-incompatible topologies — callers fall back to the
/// host-bounce coordinator.
fn try_build_bar_ar(devices: &[i32]) -> Option<Arc<BarArCoordinator>> {
    if devices.len() < 2 {
        return None;
    }
    let cluster = Arc::new(HipCluster::new(devices).ok()?);
    if !cluster.peer_access_full() {
        return None;
    }
    let bar = Arc::new(BarP2pAllReduce::new(Arc::clone(&cluster)).ok()?);
    Some(Arc::new(BarArCoordinator::new(bar).ok()?))
}

pub fn launch<A: Arch>(
    file: GgufFile,
    topology: &Topology,
    params: LaunchParams,
) -> Result<Vec<WorkerHandle<A>>> {
    let file = Arc::new(file);
    match topology {
        Topology::SingleDevice { device } => {
            let h = WorkerHandle::<A>::spawn(*device, WorkerRole::Sd, Arc::clone(&file), params)?;
            Ok(vec![h])
        }
        Topology::Tp { devices } => launch_tp::<A>(devices, file, params),
        Topology::Pp {
            devices,
            layer_split,
        } => launch_pp::<A>(
            devices,
            layer_split.as_deref(),
            LaunchConfig { file, params },
        ),
        Topology::Hybrid {
            stages,
            layer_split,
        } => launch_hybrid::<A>(
            stages,
            layer_split.as_deref(),
            LaunchConfig { file, params },
        ),
    }
}

fn launch_tp<A: Arch>(
    devices: &[i32],
    file: Arc<GgufFile>,
    params: LaunchParams,
) -> Result<Vec<WorkerHandle<A>>> {
    let n = devices.len();
    let ar = Arc::new(ArCoordinator::new(n));
    let bar = if params.deterministic_ar {
        None
    } else {
        try_build_bar_ar(devices)
    };
    let mut handles = Vec::with_capacity(n);
    for (rank, &dev) in devices.iter().enumerate() {
        let role = WorkerRole::Tp {
            rank,
            n_ranks: n,
            ar: Arc::clone(&ar),
            bar: bar.as_ref().map(Arc::clone),
        };
        handles.push(
            WorkerHandle::<A>::spawn(dev, role, Arc::clone(&file), params)
                .with_context(|| format!("TP rank {rank} on hip:{dev}"))?,
        );
    }
    Ok(handles)
}

/// Runtime knobs shared by every worker (orchestrate-level launchers,
/// `WorkerHandle::spawn`, `init_rank`). Separate from the GGUF `Arc`
/// because the worker variants take it by ref or by clone independently.
#[derive(Copy, Clone, Debug)]
pub struct LaunchParams {
    pub ctx_cap: Option<usize>,
    pub prefill_ubatch: usize,
    pub max_slots: usize,
    pub paged_kv_pages: Option<usize>,
    pub kv_layout: crate::core::KvLayout,
    /// Route TP/Hybrid AllReduce through the host-bounce coordinator
    /// (DtoH → CPU sum → HtoD) instead of BAR1 P2P. The BAR1 aperture
    /// read is non-coherent on gfx906 (see
    /// `doc/DETERMINISM_INVESTIGATION.md`), so BAR1 AR is not bit-
    /// reproducible at temp=0; host-bounce is. Costs the DtoH/HtoD
    /// bytes BAR1 avoids — opt-in via the server `--deterministic` flag.
    pub deterministic_ar: bool,
}

/// Shared runtime configuration for the topology launchers.
pub struct LaunchConfig {
    pub file: Arc<GgufFile>,
    pub params: LaunchParams,
}

fn launch_pp<A: Arch>(
    devices: &[i32],
    layer_split: Option<&[usize]>,
    cfg: LaunchConfig,
) -> Result<Vec<WorkerHandle<A>>> {
    let LaunchConfig { file, params } = cfg;
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
    // One PeerEdge per PP boundary (n-1 total). Each carries a device
    // buffer pre-bound to the consumer's device (allocated lazily on
    // first peer_send) + an event for cross-stream ordering. Replaces
    // the single shared host-bounce vec that used to serialise all
    // PP hand-offs through one mutex.
    let mut edges: Vec<super::ar::PeerBuffer> = Vec::with_capacity(n.saturating_sub(1));
    for edge_idx in 0..n.saturating_sub(1) {
        let consumer_dev = devices[edge_idx + 1];
        edges.push(
            new_peer_edge(consumer_dev)
                .with_context(|| format!("PP edge {edge_idx}→{} (consumer hip:{consumer_dev})", edge_idx + 1))?,
        );
    }
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
        let send_edge = if rank + 1 < n {
            Some(Arc::clone(&edges[rank]))
        } else {
            None
        };
        let recv_edge = if rank > 0 {
            Some(Arc::clone(&edges[rank - 1]))
        } else {
            None
        };
        let role = WorkerRole::Pp {
            rank,
            n_ranks: n,
            layer_start,
            layer_end,
            send_edge,
            recv_edge,
        };
        handles.push(
            WorkerHandle::<A>::spawn(
                dev,
                role,
                Arc::clone(&file),
                params,
            )
            .with_context(|| format!("PP rank {rank} on hip:{dev}"))?,
        );
    }
    Ok(handles)
}

fn launch_hybrid<A: Arch>(
    stages: &[Vec<i32>],
    layer_split: Option<&[usize]>,
    cfg: LaunchConfig,
) -> Result<Vec<WorkerHandle<A>>> {
    let LaunchConfig { file, params } = cfg;
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
    // S8b: per-transition per-rank edges, one DtoD peer slot per
    // producer-consumer pair (rank_in_stage k of stage i → rank k of
    // stage i+1). Replaces the single shared host-bounce slot +
    // handoff barrier with PpStage's event-based async DtoD pattern.
    // Requires equal `tp_size` across adjacent stages; pp2tp2's
    // canonical config satisfies this.
    let mut transition_edges: Vec<Vec<super::ar::PeerBuffer>> =
        Vec::with_capacity(n_stages.saturating_sub(1));
    for ti in 0..n_stages.saturating_sub(1) {
        let src_ranks = &stages[ti];
        let dst_ranks = &stages[ti + 1];
        if src_ranks.len() != dst_ranks.len() {
            return Err(anyhow!(
                "Hybrid transition {ti}→{}: src tp_size {} != dst tp_size {} \
                 (S8b async DtoD path assumes equal TP across stages)",
                ti + 1,
                src_ranks.len(),
                dst_ranks.len()
            ));
        }
        let mut edges_at_transition = Vec::with_capacity(src_ranks.len());
        // Pre-allocate dst to a safe upper bound on prefill_ubatch *
        // hidden * 2 bytes (F16). 8192 is the largest hidden across
        // v2-supported archs (gemma4-31B = 5376, qwen3.6-27B = 5120,
        // qwen3.5-9B = 4096). Eliminates the race where the
        // concurrently-started receiver hits peer_recv before the
        // producer's lazy peer_send alloc runs.
        const MAX_HIDDEN: usize = 8192;
        let max_bytes = params.prefill_ubatch * MAX_HIDDEN * 2;
        for (k, &consumer_dev) in dst_ranks.iter().enumerate().take(src_ranks.len()) {
            edges_at_transition.push(
                new_peer_edge_prealloc(consumer_dev, max_bytes).with_context(|| {
                    format!(
                        "Hybrid edge stage {ti}→{} rank_in_stage {k} (consumer hip:{consumer_dev}, prealloc {max_bytes} B)",
                        ti + 1
                    )
                })?,
            );
        }
        transition_edges.push(edges_at_transition);
    }

    let mut handles = Vec::with_capacity(total_ranks);
    let mut layer_cursor = 0usize;
    for (stage_idx, ranks) in stages.iter().enumerate() {
        let tp_size = ranks.len();
        let ar = Arc::new(ArCoordinator::new(tp_size));
        let bar = if params.deterministic_ar {
            None
        } else {
            try_build_bar_ar(ranks)
        };
        let (layer_start, layer_end) = if !split.is_empty() {
            let s = layer_cursor;
            let count = split[stage_idx];
            layer_cursor += count;
            (s, s + count)
        } else {
            (0, 0)
        };
        for (rank_in_stage, &dev) in ranks.iter().enumerate() {
            let send_edge = if stage_idx + 1 < n_stages {
                Some(Arc::clone(&transition_edges[stage_idx][rank_in_stage]))
            } else {
                None
            };
            let recv_edge = if stage_idx > 0 {
                Some(Arc::clone(&transition_edges[stage_idx - 1][rank_in_stage]))
            } else {
                None
            };
            let role = WorkerRole::Hybrid {
                stage_idx,
                n_stages,
                rank_in_stage,
                tp_size,
                layer_start,
                layer_end,
                ar: Arc::clone(&ar),
                bar: bar.as_ref().map(Arc::clone),
                send_edge,
                recv_edge,
            };
            handles.push(
                WorkerHandle::<A>::spawn(dev, role, Arc::clone(&file), params)
                    .with_context(|| {
                        format!("Hybrid stage {stage_idx} rank {rank_in_stage} on hip:{dev}")
                    })?,
            );
        }
    }
    Ok(handles)
}

pub fn run_forward_mixed<A: Arch>(
    topology: &Topology,
    handles: &mut [WorkerHandle<A>],
    tokens: Vec<u32>,
    positions: Vec<usize>,
    slot_ids: Vec<usize>,
    prefill_rows: usize,
) -> Result<Vec<f32>> {
    match topology {
        Topology::SingleDevice { .. } => {
            let rx = handles[0].send_forward_mixed(tokens, positions, slot_ids, prefill_rows)?;
            rx.recv()
                .map_err(|e| anyhow!("SD mixed reply channel closed: {e}"))?
        }
        Topology::Pp { .. } => {
            let mut last_logits = Vec::new();
            for h in handles.iter_mut() {
                let rx = h.send_forward_mixed(
                    tokens.clone(),
                    positions.clone(),
                    slot_ids.clone(),
                    prefill_rows,
                )?;
                last_logits = rx
                    .recv()
                    .map_err(|e| anyhow!("PP mixed reply channel closed: {e}"))??;
            }
            Ok(last_logits)
        }
        Topology::Tp { .. } | Topology::Hybrid { .. } => {
            let mut rxs = Vec::with_capacity(handles.len());
            for h in handles.iter_mut() {
                rxs.push(h.send_forward_mixed(
                    tokens.clone(),
                    positions.clone(),
                    slot_ids.clone(),
                    prefill_rows,
                )?);
            }
            let mut last_nonempty: Option<Vec<f32>> = None;
            for rx in rxs {
                let logits = rx
                    .recv()
                    .map_err(|e| anyhow!("mixed worker reply channel closed: {e}"))??;
                if !logits.is_empty() {
                    last_nonempty = Some(logits);
                }
            }
            last_nonempty.ok_or_else(|| anyhow!("no rank produced mixed logits"))
        }
    }
}

pub fn run_forward<A: Arch>(
    topology: &Topology,
    handles: &mut [WorkerHandle<A>],
    tokens: Vec<u32>,
    positions: Vec<usize>,
    slot_ids: Vec<usize>,
) -> Result<Vec<f32>> {
    match topology {
        Topology::SingleDevice { .. } => {
            let rx = handles[0].send_forward(tokens, positions, slot_ids)?;
            rx.recv()
                .map_err(|e| anyhow!("SD reply channel closed: {e}"))?
        }
        Topology::Pp { .. } => {
            let mut last_logits = Vec::new();
            for h in handles.iter_mut() {
                let rx = h.send_forward(tokens.clone(), positions.clone(), slot_ids.clone())?;
                last_logits = rx
                    .recv()
                    .map_err(|e| anyhow!("PP reply channel closed: {e}"))??;
            }
            Ok(last_logits)
        }
        Topology::Tp { .. } | Topology::Hybrid { .. } => {
            let mut rxs = Vec::with_capacity(handles.len());
            for h in handles.iter_mut() {
                rxs.push(h.send_forward(tokens.clone(), positions.clone(), slot_ids.clone())?);
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
