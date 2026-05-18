//! Persistent worker thread per rank. Holds the per-rank model +
//! scratch pool; receives `Forward { token, position }` commands over
//! mpsc and replies with the logits (empty Vec on ranks that don't
//! produce them, e.g. PP non-last). Lifetime constraints on
//! `ForwardCtx` force per-step ctx construction inside the worker.

use std::marker::PhantomData;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgufFile;

use crate::core::ScratchPool;
use crate::engine::{HybridForwardCtx, PpForwardCtx, SingleDeviceForwardCtx, TpForwardCtx, TpHooks};
use crate::loader::ShardMode;
use crate::ForwardCtx;

use super::ar::{make_ar_callback, ArCoordinator, PeerBuffer};
use super::Arch;

/// Per-rank role + the inter-thread state needed to construct the
/// right `ForwardCtx` each step.
pub enum WorkerRole {
    Sd,
    Tp {
        rank: usize,
        n_ranks: usize,
        ar: Arc<ArCoordinator>,
    },
    Pp {
        rank: usize,
        n_ranks: usize,
        layer_start: usize,
        layer_end: usize,
        peer_buffer: PeerBuffer,
    },
    Hybrid {
        stage_idx: usize,
        n_stages: usize,
        rank_in_stage: usize,
        tp_size: usize,
        layer_start: usize,
        layer_end: usize,
        ar: Arc<ArCoordinator>,
        peer_buffer: PeerBuffer,
        handoff: Arc<Barrier>,
    },
}

impl WorkerRole {
    fn shard(&self) -> ShardMode {
        match self {
            WorkerRole::Sd | WorkerRole::Pp { .. } => ShardMode::Replicated,
            WorkerRole::Tp { rank, n_ranks, .. } => ShardMode::Tp {
                rank: *rank,
                n_ranks: *n_ranks,
            },
            WorkerRole::Hybrid {
                rank_in_stage,
                tp_size,
                ..
            } => ShardMode::Tp {
                rank: *rank_in_stage,
                n_ranks: *tp_size,
            },
        }
    }

    fn layer_slice(&self) -> Option<(usize, usize)> {
        match self {
            WorkerRole::Pp {
                layer_start,
                layer_end,
                ..
            }
            | WorkerRole::Hybrid {
                layer_start,
                layer_end,
                ..
            } => Some((*layer_start, *layer_end)),
            _ => None,
        }
    }
}

enum Command {
    Forward {
        tokens: Vec<u32>,
        positions: Vec<usize>,
        slot_ids: Vec<usize>,
        reply: SyncSender<Result<Vec<f32>>>,
    },
    ResetKv {
        reply: SyncSender<Result<()>>,
    },
    Shutdown,
}

pub struct WorkerHandle<A: Arch> {
    cmd_tx: mpsc::Sender<Command>,
    thread: Option<JoinHandle<()>>,
    _phantom: PhantomData<A>,
}

impl<A: Arch> WorkerHandle<A> {
    /// Spawn a worker on `device_id`. Returns once `Arch::load` succeeds
    /// (the worker blocks the spawning thread until loading is done so
    /// errors surface synchronously).
    pub fn spawn(
        device_id: i32,
        role: WorkerRole,
        file: Arc<GgufFile>,
        ctx_cap: Option<usize>,
        prefill_ubatch: usize,
        max_slots: usize,
    ) -> Result<Self> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<()>>(1);

        let thread = std::thread::spawn(move || {
            let (mut state, init_err) =
                match init_rank::<A>(device_id, &role, &file, ctx_cap, prefill_ubatch, max_slots) {
                Ok(s) => (Some(s), None),
                Err(e) => (None, Some(e)),
            };
            let send_result = match &init_err {
                None => ready_tx.send(Ok(())),
                Some(e) => ready_tx.send(Err(anyhow::anyhow!("{e:#}"))),
            };
            if send_result.is_err() || init_err.is_some() {
                return;
            }
            let mut state = state.take().unwrap();

            // Stage 2: command loop.
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    Command::Forward {
                        tokens,
                        positions,
                        slot_ids,
                        reply,
                    } => {
                        let res = run_forward_once::<A>(
                            &mut state,
                            &role,
                            &tokens,
                            &positions,
                            &slot_ids,
                        );
                        let _ = reply.send(res);
                    }
                    Command::ResetKv { reply } => {
                        let res = state.pool.reset_gdn_state(&state.device);
                        let _ = reply.send(res);
                    }
                    Command::Shutdown => break,
                }
            }

            // Stage 3: cleanup.
            let _ = state.pool.dispose(&state.device);
            let _ = A::dispose(&mut state.model, &state.device);
        });

        // Wait for the worker to report load success/failure.
        match ready_rx
            .recv()
            .context("worker init channel closed before load completed")?
        {
            Ok(()) => Ok(Self {
                cmd_tx,
                thread: Some(thread),
                _phantom: PhantomData,
            }),
            Err(e) => {
                let _ = thread.join();
                Err(e)
            }
        }
    }

    /// Queue a Forward command; returns the reply receiver.
    pub fn send_forward(
        &self,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        slot_ids: Vec<usize>,
    ) -> Result<Receiver<Result<Vec<f32>>>> {
        let (reply_tx, reply_rx) = mpsc::sync_channel::<Result<Vec<f32>>>(1);
        self.cmd_tx
            .send(Command::Forward {
                tokens,
                positions,
                slot_ids,
                reply: reply_tx,
            })
            .map_err(|e| anyhow::anyhow!("worker channel closed: {e}"))?;
        Ok(reply_rx)
    }

    /// Queue a ResetKv command. Each worker zeroes its rank's GDN
    /// state slabs; reply fires once that rank's stream is drained.
    pub fn send_reset_kv(&self) -> Result<Receiver<Result<()>>> {
        let (reply_tx, reply_rx) = mpsc::sync_channel::<Result<()>>(1);
        self.cmd_tx
            .send(Command::ResetKv { reply: reply_tx })
            .map_err(|e| anyhow::anyhow!("worker channel closed: {e}"))?;
        Ok(reply_rx)
    }

    /// Tell the worker to exit and join.
    pub fn shutdown(mut self) -> Result<()> {
        let _ = self.cmd_tx.send(Command::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        Ok(())
    }
}

/// Per-rank state owned by the worker thread. Fields drop in
/// declaration order — model + pool + reg release HIP resources
/// (hipFree, hipModuleUnload) that need the device's context alive,
/// so `device` is declared LAST.
struct RankState<A: Arch> {
    model: A::Model,
    pool: ScratchPool,
    reg: OpsRegistry,
    device: HipDevice,
}

fn init_rank<A: Arch>(
    device_id: i32,
    role: &WorkerRole,
    file: &GgufFile,
    ctx_cap: Option<usize>,
    prefill_ubatch: usize,
    max_slots: usize,
) -> Result<RankState<A>> {
    let device = HipDevice::new(device_id).context("HipDevice::new")?;
    device.bind().context("device.bind")?;
    let shard = role.shard();
    let model = A::load(file, &device, shard, role.layer_slice(), ctx_cap)
        .context("Arch::load")?;
    let mut cfg = A::scratch_config(&model, shard, prefill_ubatch, max_slots);
    if let Some((ls, le)) = role.layer_slice() {
        cfg.num_layers = le - ls;
        // KV cache slots are per-owned-layer; slice the per-layer
        // widths to match. Pool's `num_layers` must equal the slice's
        // length under PP/Hybrid.
        if let Some(per) = cfg.per_layer_kv_widths.as_ref() {
            if le > per.len() {
                anyhow::bail!(
                    "layer range [{ls}..{le}) out of per_layer_kv_widths.len() {}",
                    per.len()
                );
            }
            cfg.per_layer_kv_widths = Some(per[ls..le].to_vec());
        }
    }
    let pool = ScratchPool::new(&device, cfg).context("ScratchPool::new")?;
    let reg = OpsRegistry::new(&device).context("OpsRegistry::new")?;
    Ok(RankState {
        device,
        pool,
        reg,
        model,
    })
}

fn run_forward_once<A: Arch>(
    state: &mut RankState<A>,
    role: &WorkerRole,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<Vec<f32>> {
    let stream = state.device.default_stream();
    match role {
        WorkerRole::Sd => {
            let mut ctx =
                SingleDeviceForwardCtx::new(&state.device, stream, &state.reg, &mut state.pool);
            A::forward(&state.model, &mut ctx, tokens, positions, slot_ids)?;
            Ok(ctx.logits().to_vec())
        }
        WorkerRole::Tp { rank, n_ranks, ar } => {
            let hooks = TpHooks {
                rank: *rank,
                n_ranks: *n_ranks,
                ar_callback: make_ar_callback(Arc::clone(ar), *rank),
            };
            let mut ctx = TpForwardCtx::new(
                &state.device,
                stream,
                &state.reg,
                &mut state.pool,
                hooks,
            );
            A::forward(&state.model, &mut ctx, tokens, positions, slot_ids)?;
            Ok(ctx.logits().to_vec())
        }
        WorkerRole::Pp {
            rank,
            n_ranks,
            layer_start,
            layer_end,
            peer_buffer,
        } => {
            let mut buf = peer_buffer
                .lock()
                .map_err(|e| anyhow::anyhow!("peer_buffer poisoned: {e}"))?;
            let mut ctx = PpForwardCtx::new(
                &state.device,
                stream,
                &state.reg,
                &mut state.pool,
                *rank,
                *n_ranks,
                *layer_start,
                *layer_end,
                &mut *buf,
            );
            A::forward(&state.model, &mut ctx, tokens, positions, slot_ids)?;
            Ok(ctx.logits().to_vec())
        }
        WorkerRole::Hybrid {
            stage_idx,
            n_stages,
            rank_in_stage,
            tp_size,
            layer_start,
            layer_end,
            ar,
            peer_buffer,
            handoff,
        } => {
            let mut ctx = HybridForwardCtx::new(
                &state.device,
                stream,
                &state.reg,
                &mut state.pool,
                *stage_idx,
                *n_stages,
                *rank_in_stage,
                *tp_size,
                *layer_start,
                *layer_end,
                make_ar_callback(Arc::clone(ar), *rank_in_stage),
                Arc::clone(peer_buffer),
                Arc::clone(handoff),
            );
            A::forward(&state.model, &mut ctx, tokens, positions, slot_ids)?;
            Ok(ctx.logits().to_vec())
        }
    }
}
