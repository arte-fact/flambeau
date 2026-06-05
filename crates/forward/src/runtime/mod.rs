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

use crate::core::{KvLayout, ScratchConfig};
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
    Tp {
        devices: Vec<i32>,
    },
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

    /// Run forward over `tokens` (length 1 = single decode; >1 =
    /// prefill chunk or batched-decode N slots). `positions[i]` is
    /// the KV write row for `tokens[i]`; `slot_ids[i]` selects the
    /// inflight slot whose KV/GDN slab receives that write. Logits
    /// for the LAST token land in `ctx.logits()`.
    fn forward<C: ForwardCtx>(
        model: &Self::Model,
        ctx: &mut C,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
    ) -> Result<()>;

    /// Sarathi-Serve mixed-batch forward (Phase K4). Rows
    /// `[0..prefill_rows)` are a prefill chunk for `slot_ids[0]`;
    /// rows `[prefill_rows..n)` are batched decodes across N distinct
    /// slots. One forward call covering both phases; logit emission
    /// is `(N + 1)`-wide. Default impl bails — only archs that override
    /// (qwen35-v2, qwen35moe-v2) support mixed-batch today.
    fn forward_mixed<C: ForwardCtx>(
        _model: &Self::Model,
        _ctx: &mut C,
        _tokens: &[u32],
        _positions: &[usize],
        _slot_ids: &[usize],
        _prefill_rows: usize,
    ) -> Result<()> {
        anyhow::bail!("Arch::forward_mixed: not implemented for this arch")
    }

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
        paged_kv_pages: Option<usize>,
        kv_layout: KvLayout,
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
        params: orchestrate::LaunchParams,
    ) -> Result<Self> {
        let orchestrate::LaunchParams {
            ctx_cap: _,
            prefill_ubatch,
            max_slots,
            paged_kv_pages: _,
            kv_layout: _,
        } = params;
        if prefill_ubatch == 0 {
            anyhow::bail!("Session::new: prefill_ubatch must be > 0");
        }
        if max_slots == 0 {
            anyhow::bail!("Session::new: max_slots must be > 0");
        }
        let handles = orchestrate::launch::<A>(file, &topology, params)?;
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

    /// Drive a forward pass across every rank. `positions[i]` is the
    /// KV write row for `tokens[i]`; `slot_ids[i]` is the inflight
    /// slot. Logits for the LAST token land in `self.last_logits`.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward: empty tokens");
        }
        if positions.len() != tokens.len() || slot_ids.len() != tokens.len() {
            anyhow::bail!(
                "Session::forward: positions.len {} / slot_ids.len {} != tokens.len {}",
                positions.len(),
                slot_ids.len(),
                tokens.len()
            );
        }
        self.last_logits = orchestrate::run_forward(
            &self.topology,
            &mut self.handles,
            tokens.to_vec(),
            positions.to_vec(),
            slot_ids.to_vec(),
        )?;
        Ok(())
    }

    pub fn forward_one_token(&mut self, token: u32, position: usize) -> Result<()> {
        self.forward(&[token], &[position], &[0])
    }

    /// Sarathi-Serve mixed-batch dispatch (Phase K4). Rows
    /// `[0..prefill_rows)` are a prefill chunk for `slot_ids[0]`;
    /// rows `[prefill_rows..n)` are batched decodes across N distinct
    /// slots. Logits emitted are `(N + 1)` rows: the last prefill
    /// chunk token + N decode rows, in that order. Arch must override
    /// `Arch::forward_mixed`; default impl bails.
    pub fn forward_mixed(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        slot_ids: &[usize],
        prefill_rows: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward_mixed: empty tokens");
        }
        if positions.len() != tokens.len() || slot_ids.len() != tokens.len() {
            anyhow::bail!(
                "Session::forward_mixed: positions.len {} / slot_ids.len {} != tokens.len {}",
                positions.len(),
                slot_ids.len(),
                tokens.len()
            );
        }
        if prefill_rows == 0 || prefill_rows >= tokens.len() {
            anyhow::bail!(
                "Session::forward_mixed: prefill_rows must satisfy 0 < K < n (got K={prefill_rows}, n={})",
                tokens.len()
            );
        }
        self.last_logits = orchestrate::run_forward_mixed(
            &self.topology,
            &mut self.handles,
            tokens.to_vec(),
            positions.to_vec(),
            slot_ids.to_vec(),
            prefill_rows,
        )?;
        Ok(())
    }

    /// Read the i-th mixed-output row from the last `forward_mixed` call.
    /// Row 0 = prefill slot's next-token logit; rows 1..=N = decode slots.
    pub fn mixed_logits_row(&self, i: usize, vocab: usize) -> &[f32] {
        &self.last_logits[i * vocab..(i + 1) * vocab]
    }

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
            let chunk_len = end - i;
            let chunk_positions: Vec<usize> = (0..chunk_len).map(|k| pos + k).collect();
            let chunk_slots = vec![0usize; chunk_len];
            self.forward(&tokens[i..end], &chunk_positions, &chunk_slots)?;
            pos += chunk_len;
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

    /// Zero GDN state + conv history for one inflight slot only. Used
    /// by the server's shared-Session pool when a single conversation
    /// resets without disturbing peers.
    pub fn reset_kv_slot(&mut self, slot_id: usize) -> Result<()> {
        let rxs: Vec<_> = self
            .handles
            .iter()
            .map(|h| h.send_reset_kv_slot(slot_id))
            .collect::<Result<_>>()?;
        for rx in rxs {
            rx.recv()
                .map_err(|e| anyhow!("reset_kv_slot reply channel closed: {e}"))??;
        }
        Ok(())
    }

    /// Recycle every page held by `slot_id` across every layer's
    /// `PagePool` back into the free list. No-op on non-paged ranks.
    /// Called by the server when a slot is released.
    pub fn release_paged_slot(&mut self, slot_id: usize) -> Result<()> {
        let rxs: Vec<_> = self
            .handles
            .iter()
            .map(|h| h.send_release_paged_slot(slot_id))
            .collect::<Result<_>>()?;
        for rx in rxs {
            rx.recv()
                .map_err(|e| anyhow!("release_paged_slot reply channel closed: {e}"))??;
        }
        Ok(())
    }

    /// Single-token decode into a specific KV/GDN slot. Mirrors
    /// `forward_one_token_logits` but routes the write to `slot_id`.
    pub fn forward_one_token_logits_slot(
        &mut self,
        token: u32,
        position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward(&[token], &[position], &[slot_id])?;
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    /// Chunked prefill into a specific slot. Each chunk is forwarded
    /// with `slot_ids = [slot_id; chunk_len]`; the slot's KV slab
    /// receives the full prompt's K/V rows, and GDN state advances
    /// only for that slot.
    pub fn forward_prefill_logits_slot(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        slot_id: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward_prefill_logits_slot: empty tokens");
        }
        let chunk_size = self.prefill_ubatch;
        let mut pos = start_position;
        let mut i = 0;
        while i < tokens.len() {
            let end = (i + chunk_size).min(tokens.len());
            let chunk_len = end - i;
            let chunk_positions: Vec<usize> = (0..chunk_len).map(|k| pos + k).collect();
            let chunk_slots = vec![slot_id; chunk_len];
            self.forward(&tokens[i..end], &chunk_positions, &chunk_slots)?;
            pos += chunk_len;
            i = end;
        }
        out.clear();
        out.extend_from_slice(&self.last_logits);
        Ok(())
    }

    /// Logits emitted by the most recent forward call. For prefill /
    /// single decode this is a single `vocab` row. For batched-decode
    /// (distinct slot_ids) it is `N * vocab` row-major.
    pub fn logits(&self) -> &[f32] {
        &self.last_logits
    }

    /// Slice the `i`-th token's logits row from the most recent forward.
    /// `vocab` must match the model's emit width (use `Session::logits().len() / n_rows`).
    pub fn logits_row(&self, i: usize, vocab: usize) -> &[f32] {
        &self.last_logits[i * vocab..(i + 1) * vocab]
    }

    pub fn vocab_size(&self) -> usize {
        self.last_logits.len()
    }

    /// Batched-decode entry: forwards N pairs of (token, position) each
    /// targeting its slot's KV history; emits N logits rows in
    /// `self.last_logits` row-major `[N, vocab]`.
    pub fn forward_decode_batched(&mut self, slots: &[(u32, usize, usize)]) -> Result<()> {
        if slots.is_empty() {
            anyhow::bail!("Session::forward_decode_batched: empty slots");
        }
        let tokens: Vec<u32> = slots.iter().map(|s| s.0).collect();
        let positions: Vec<usize> = slots.iter().map(|s| s.1).collect();
        let slot_ids: Vec<usize> = slots.iter().map(|s| s.2).collect();
        // Guard: distinct slot_ids so output_head emits all N logits rows.
        if tokens.len() > 1 && slot_ids.iter().all(|&s| s == slot_ids[0]) {
            anyhow::bail!(
                "Session::forward_decode_batched: slot_ids all equal — \
                 use forward_prefill_logits for single-slot multi-token"
            );
        }
        self.forward(&tokens, &positions, &slot_ids)
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
