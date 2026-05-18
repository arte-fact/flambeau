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
    /// being the inner ring of Hybrid stages). `layer_range` is
    /// `Some((start, end))` for PP / Hybrid ranks (load only layers
    /// `[start, end)` to keep per-rank VRAM bounded); `None` for SD /
    /// TP (every layer loads). `ctx_cap` clamps the model's GGUF
    /// `context_length` to the operator's `--ctx-cap` so per-layer KV
    /// slabs don't allocate for the full 128–256k context.
    fn load(
        file: &GgufFile,
        device: &HipDevice,
        shard: ShardMode,
        layer_range: Option<(usize, usize)>,
        ctx_cap: Option<usize>,
    ) -> Result<Self::Model>;

    /// Run forward over `tokens` (length 1 for decode; longer for
    /// prefill). `start_position` is the KV write tail for `tokens[0]`.
    /// Logits for the LAST token land in `ctx.logits()`.
    fn forward<C: ForwardCtx>(
        model: &Self::Model,
        ctx: &mut C,
        tokens: &[u32],
        start_position: usize,
    ) -> Result<()>;

    /// Build the ScratchPool config for this rank given the
    /// effective `ShardMode` (TP divides per-rank widths), the
    /// operator's prefill chunk size (sizes the per-token scratch
    /// slots), and the number of inflight slots (sizes per-layer KV
    /// + GDN state).
    fn scratch_config(
        model: &Self::Model,
        shard: ShardMode,
        prefill_ubatch: usize,
        max_slots: usize,
    ) -> ScratchConfig;

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
    last_logits: Vec<f32>,
    prefill_ubatch: usize,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Arch> Session<A> {
    pub fn new(
        file: GgufFile,
        topology: Topology,
        ctx_cap: Option<usize>,
        prefill_ubatch: usize,
        max_slots: usize,
    ) -> Result<Self> {
        if prefill_ubatch == 0 {
            anyhow::bail!("Session::new: prefill_ubatch must be > 0");
        }
        if max_slots == 0 {
            anyhow::bail!("Session::new: max_slots must be > 0");
        }
        let handles =
            orchestrate::launch::<A>(file, &topology, ctx_cap, prefill_ubatch, max_slots)?;
        Ok(Self {
            topology,
            handles,
            last_logits: Vec::new(),
            prefill_ubatch,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn prefill_ubatch(&self) -> usize {
        self.prefill_ubatch
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Drive a forward pass across every rank. `tokens.len() == 1` is
    /// decode; longer is prefill. Returns when the rank that owns the
    /// LM head (SD: rank 0; PP/Hybrid: last stage; TP: any rank) finishes.
    /// Logits for the LAST token land in `self.last_logits`.
    pub fn forward(&mut self, tokens: &[u32], start_position: usize) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward: empty tokens");
        }
        self.last_logits = orchestrate::run_forward(
            &self.topology,
            &mut self.handles,
            tokens.to_vec(),
            start_position,
        )?;
        Ok(())
    }

    pub fn forward_one_token(&mut self, token: u32, position: usize) -> Result<()> {
        self.forward(&[token], position)
    }

    pub fn forward_one_token_logits(
        &mut self,
        token: u32,
        position: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward(&[token], position)?;
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    pub fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward_prefill_logits: empty tokens");
        }
        let chunk_size = self.prefill_ubatch;
        let mut pos = start_position;
        let mut i = 0;
        while i < tokens.len() {
            let end = (i + chunk_size).min(tokens.len());
            self.forward(&tokens[i..end], pos)?;
            pos += end - i;
            i = end;
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
