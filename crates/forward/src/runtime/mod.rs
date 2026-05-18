//! Topology-abstracted execution. `Session<A: Arch>` hides the
//! per-topology orchestration (worker threads, AR coordinator,
//! peer-copy, stage handoff) behind a uniform forward-one-token API.

pub mod ar;
pub mod driver;
pub mod orchestrate;
pub mod workers;

use anyhow::{anyhow, Result};
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

/// What an arch crate has to provide. Arch crates declare a marker
/// type (e.g. `pub struct Qwen35V2;`) and implement this trait on it;
/// the Session takes `A: Arch` generic.
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

    /// Decode wrapper that returns last-token logits via `out` (caller-
    /// owned). Matches the `ModelDriver::forward_one_token_logits`
    /// shape so server adapters call one method per request step.
    pub fn forward_one_token_logits(
        &mut self,
        token: u32,
        position: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_one_token(token, position)?;
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    /// Multi-token prefill returning the LAST token's logits via `out`.
    /// v1 implementation loops `forward_one_token` for each prompt
    /// token; tracked as [P9-OUT real batched-prefill kernel] —
    /// switching to a single prefill pass per layer is the lever.
    pub fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward_prefill_logits: empty tokens");
        }
        for (i, &t) in tokens.iter().enumerate() {
            self.forward_one_token(t, start_position + i)?;
        }
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    /// Zero per-rank GDN state + conv history. KV-cache positions are
    /// caller-supplied; this is the only stateful slot that survives a
    /// request boundary, so it's the entire reset surface.
    pub fn reset_kv(&mut self) -> Result<()> {
        let rxs: Vec<_> = self
            .handles
            .iter()
            .map(|h| h.send_reset_kv())
            .collect::<Result<_>>()?;
        for rx in rxs {
            rx.recv()
                .map_err(|e| anyhow!("reset_kv reply channel closed: {e}"))??;
        }
        Ok(())
    }

    /// The logits emitted by the most recent `forward_one_token`.
    pub fn logits(&self) -> &[f32] {
        &self.last_logits
    }

    /// Vocab size — `self.last_logits.len()` after the first forward,
    /// `0` before. Callers needing a pre-forward value must look at the
    /// model handle directly (`A::Model` exposes it).
    pub fn vocab_size(&self) -> usize {
        self.last_logits.len()
    }

    /// Shut workers down + free device memory.
    pub fn dispose(mut self) -> Result<()> {
        Self::dispose_in_place(&mut self)?;
        Ok(())
    }

    /// `&mut`-only dispose so callers behind a trait method (e.g.
    /// `ModelDriver::dispose`, which cannot consume `self`) can release
    /// the workers without owning the `Session`. Drops in `Drop` as a
    /// safety net catch what this misses.
    pub fn dispose_in_place(&mut self) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for h in self.handles.drain(..) {
            if let Err(e) = h.shutdown() {
                first_err.get_or_insert(e);
            }
        }
        if let Some(e) = first_err {
            Err(e)
        } else {
            Ok(())
        }
    }
}

impl<A: Arch> Drop for Session<A> {
    fn drop(&mut self) {
        for h in self.handles.drain(..) {
            let _ = h.shutdown();
        }
    }
}
