//! Topology-abstracted execution. `Session<A: Arch>` hides the
//! per-topology orchestration (worker threads, AR coordinator,
//! peer-copy, stage handoff) behind a uniform forward-one-token API.

pub mod ar;
pub mod orchestrate;
pub mod workers;

use anyhow::Result;
use flambeau_backend_hip::HipDevice;
use flambeau_quant::GgufFile;

use crate::core::ScratchConfig;
use crate::ctx::{ForwardCtx, LayerKind};
use crate::loader::ShardMode;

/// Operator-facing topology selection.
#[derive(Clone, Debug)]
pub enum Topology {
    SingleDevice {
        device: i32,
    },
    /// Pipeline-parallel across `devices` (one rank per device). If
    /// `layer_split` is None, layers are split evenly.
    Pp {
        devices: Vec<i32>,
        layer_split: Option<Vec<usize>>,
    },
    /// Tensor-parallel across `devices`. Every rank holds a shard of
    /// every layer; AR collapses partials after each row-parallel matmul.
    Tp { devices: Vec<i32> },
    /// PP-of-TP. `stages` is `[stage_idx][rank_in_stage]` — each stage
    /// is a TP cluster, stages chain via host peer_buffer. `layer_split`
    /// gives the per-stage layer count (length = `stages.len()`); when
    /// None the caller must size the model's layer-range iterator
    /// itself.
    Hybrid {
        stages: Vec<Vec<i32>>,
        layer_split: Option<Vec<usize>>,
    },
}

impl Topology {
    pub fn total_ranks(&self) -> usize {
        match self {
            Topology::SingleDevice { .. } => 1,
            Topology::Pp { devices, .. } | Topology::Tp { devices } => devices.len(),
            Topology::Hybrid { stages, .. } => stages.iter().map(|s| s.len()).sum(),
        }
    }
}

/// What an arch crate has to provide. The trait lives in the runtime
/// module because the Session takes `A: Arch` generic; arch crates
/// `impl Arch for ()` on a marker type (e.g. `pub struct Qwen3V2;`).
pub trait Arch: Send + Sync + 'static {
    /// On-device model handle. Must be `Send` because workers own it
    /// across the channel boundary.
    type Model: Send + 'static;

    /// `general.architecture` value the loader expects.
    fn arch_tag() -> &'static str;

    /// Load this rank's slice of weights. `ShardMode::Replicated` for
    /// SD/PP; `ShardMode::Tp { rank, n_ranks }` for TP/Hybrid (TP
    /// being the inner ring of Hybrid stages).
    fn load(
        file: &GgufFile,
        device: &HipDevice,
        shard: ShardMode,
    ) -> Result<Self::Model>;

    /// Run one decode step. Logits land in `ctx.logits()`.
    fn forward<C: ForwardCtx>(
        model: &Self::Model,
        ctx: &mut C,
        token: u32,
        position: usize,
    ) -> Result<()>;

    /// Build the ScratchPool config for this rank given the
    /// effective `ShardMode` (TP divides per-rank widths).
    fn scratch_config(model: &Self::Model, shard: ShardMode) -> ScratchConfig;

    /// Per-layer attention kind (FullAttn / Gdn) if the arch is
    /// hybrid; `None` if every layer is uniform (qwen3 dense, gemma4
    /// dense are FullAttn everywhere).
    fn layer_kind(_model: &Self::Model, _li: usize) -> Option<LayerKind> {
        None
    }

    /// Release device memory. Called from the worker thread.
    fn dispose(model: &mut Self::Model, device: &HipDevice) -> Result<()>;
}

/// Topology-abstracted forward session. Holds persistent worker threads
/// (one per rank). Forward steps go over an mpsc command channel.
pub struct Session<A: Arch> {
    topology: Topology,
    handles: Vec<workers::WorkerHandle<A>>,
    /// Last forward step's logits, captured from the rank that owns
    /// them (last PP stage / any TP rank / SD).
    last_logits: Vec<f32>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arch> Session<A> {
    pub fn new(file: GgufFile, topology: Topology) -> Result<Self> {
        let handles = orchestrate::launch::<A>(file, &topology)?;
        Ok(Self {
            topology,
            handles,
            last_logits: Vec::new(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Drive one decode step across every rank. Returns when the
    /// rank that owns the LM head (SD: rank 0; PP/Hybrid: last
    /// stage; TP: any rank) finishes.
    pub fn forward_one_token(&mut self, token: u32, position: usize) -> Result<()> {
        self.last_logits = orchestrate::run_forward(&self.topology, &mut self.handles, token, position)?;
        Ok(())
    }

    /// The logits emitted by the most recent `forward_one_token`.
    pub fn logits(&self) -> &[f32] {
        &self.last_logits
    }

    /// Shut workers down + free device memory.
    pub fn dispose(mut self) -> Result<()> {
        for h in self.handles.drain(..) {
            h.shutdown()?;
        }
        Ok(())
    }
}

impl<A: Arch> Drop for Session<A> {
    fn drop(&mut self) {
        for h in self.handles.drain(..) {
            let _ = h.shutdown();
        }
    }
}
