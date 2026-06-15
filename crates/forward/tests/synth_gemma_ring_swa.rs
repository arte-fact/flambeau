//! gemma4 ring-buffered SWA parity. A window-depth KV slab (ring_depth =
//! window + prefill_ubatch) must produce the same attention output as a
//! full-context slab — including when a chunked prefill STRADDLES the ring
//! boundary and a subsequent decode WRAPS past it. Run identical token
//! sequences through a ring pool and a full-depth pool and compare.
//!
//! Covers both gemma4 write paths: F16 (kv_append_v_unit_norm_f16, per-token
//! wrap — otherwise only exercised here) and Q8 (kv_append_f16_to_q8, the
//! 2-segment straddle split). Single-device ctx, single slot.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::core::{KvLayout, MixedBatch, ScratchConfig};
use flambeau_forward::ctx::{AttnWeights, ForwardCtx, RopeVariant};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;

const HIDDEN: usize = 128;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 64;
const RMS_EPS: f32 = 1e-5;
const WINDOW: i32 = 2;
const UBATCH: usize = 4;
// ring_depth = window + prefill_ubatch. Chunk 2 (positions UBATCH..2*UBATCH)
// writes rows {UBATCH..} mod RING_DEPTH and straddles the wrap; the decode at
// position 2*UBATCH wraps again.
const RING_DEPTH: usize = WINDOW as usize + UBATCH;

fn build_attn(allocs: &mut DeviceAllocs) -> AttnWeights {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;
    AttnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, 301), q_width, HIDDEN),
        attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 302), kv_width, HIDDEN),
        attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 303), kv_width, HIDDEN)),
        attn_v_unit_norm_w: Some(allocs.upload_f16(&vec![1.0_f32; HEAD_DIM])),
        attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, 304), HIDDEN, q_width),
        attn_q_norm: None,
        attn_k_norm: None,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        rotated_dims: HEAD_DIM,
        rope_theta: 10000.0,
        rope_variant: RopeVariant::NeoxSplit,
        window_size: WINDOW,
        rms_eps: RMS_EPS,
        softmax_scale: None,
        attn_q_gated: false,
        kv_share_src: None,
        post_attn_norm: Some(allocs.upload_f16(&vec![1.0_f32; HIDDEN])),
    }
}

fn build_scratch_cfg(ring: bool, layout: KvLayout, max_slots: usize) -> ScratchConfig {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;
    ScratchConfig {
        hidden: HIDDEN,
        intermediate: 256,
        q_width,
        kv_width,
        vocab: 64,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: 1,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: UBATCH,
        max_slots,
        per_layer_embd: 0,
        paged_kv: None,
        kv_layout: layout,
        per_layer_kv_layouts: None,
        // Ring: window-depth slab. Full: None → max_seq_len (no ring).
        per_layer_kv_depths: if ring { Some(vec![RING_DEPTH]) } else { None },
    }
}

fn upload_residual(device: &HipDevice, host: &[f32]) -> DevicePtr {
    let host_f16: Vec<f16> = host.iter().map(|&v| f16::from_f32(v * 0.05)).collect();
    let bytes = host_f16.len() * 2;
    let ptr = device.alloc(bytes).expect("alloc resid");
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::HostToDevice,
                ptr,
                DevicePtr(host_f16.as_ptr() as usize),
                bytes,
            )
            .expect("HtoD resid");
    }
    device.default_stream().synchronize().expect("sync");
    ptr
}

fn read_f16(device: &HipDevice, src: DevicePtr, n_elems: usize) -> Vec<f32> {
    let mut host = vec![f16::from_f32(0.0); n_elems];
    let bytes = n_elems * 2;
    unsafe {
        device
            .memcpy_async(
                device.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(host.as_mut_ptr() as usize),
                src,
                bytes,
            )
            .expect("DtoH");
    }
    device.default_stream().synchronize().expect("sync");
    host.into_iter().map(|v| v.to_f32()).collect()
}

/// Chunk 1 (pos 0..U), chunk 2 (pos U..2U, straddles the ring wrap), then one
/// decode (pos 2U, wraps). Returns (chunk2 prefill output, decode output).
fn run(
    device: &HipDevice,
    reg: &OpsRegistry,
    attn: &AttnWeights,
    resid_dev: DevicePtr,
    ring: bool,
    layout: KvLayout,
) -> (Vec<f32>, Vec<f32>) {
    let stream = device.default_stream();
    let mut pool = ScratchPool::new(device, build_scratch_cfg(ring, layout, 1)).expect("pool");
    let mut ctx = SingleDeviceForwardCtx::new(device, stream, reg, &mut pool);

    // Chunk 1: prefill positions [0, U) into slot 0.
    let c1 = unsafe { Tensor::<F16>::from_raw(resid_dev, UBATCH * HIDDEN) };
    let pos1: Vec<usize> = (0..UBATCH).collect();
    let slot1 = vec![0usize; UBATCH];
    ctx.standard_attn(&c1, attn, 0, &pos1, &slot1, None)
        .expect("chunk1")
        .expect("chunk1 residual");

    // Chunk 2: prefill positions [U, 2U) into slot 0 — straddles the wrap.
    let c2_ptr = resid_dev.offset_bytes(UBATCH * HIDDEN * 2);
    let c2 = unsafe { Tensor::<F16>::from_raw(c2_ptr, UBATCH * HIDDEN) };
    let pos2: Vec<usize> = (UBATCH..2 * UBATCH).collect();
    let slot2 = vec![0usize; UBATCH];
    let d2 = ctx
        .standard_attn(&c2, attn, 0, &pos2, &slot2, None)
        .expect("chunk2")
        .expect("chunk2 residual");
    let chunk2_out = read_f16(device, d2.ptr, UBATCH * HIDDEN);

    // Decode: position 2U into slot 0 — n_tokens_kv > ring_depth, ring active.
    let dec_ptr = resid_dev.offset_bytes(2 * UBATCH * HIDDEN * 2);
    let dec = unsafe { Tensor::<F16>::from_raw(dec_ptr, HIDDEN) };
    let dd = ctx
        .standard_attn(&dec, attn, 0, &[2 * UBATCH], &[0usize], None)
        .expect("decode")
        .expect("decode residual");
    let decode_out = read_f16(device, dd.ptr, HIDDEN);

    pool.dispose(device).expect("pool dispose");
    (chunk2_out, decode_out)
}

fn assert_close(label: &str, full: &[f32], ring: &[f32]) {
    assert_eq!(full.len(), ring.len(), "{label}: len mismatch");
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (&x, &y)) in full.iter().zip(ring).enumerate() {
        let abs = (x - y).abs();
        let rel = abs / x.abs().max(y.abs()).max(1e-3);
        if rel > worst_rel {
            worst_rel = rel;
            worst_abs = abs;
            worst_idx = i;
        }
    }
    let abs_tol = 0.5_f32;
    let rel_tol = 1e-2_f32;
    eprintln!(
        "[{label}] worst rel={worst_rel:.3e} abs={worst_abs:.3e} at idx={worst_idx} \
         (full={}, ring={})",
        full[worst_idx], ring[worst_idx]
    );
    assert!(
        worst_rel <= rel_tol || worst_abs <= abs_tol,
        "{label}: rel={worst_rel:.3e} > {rel_tol:.3e} AND abs={worst_abs:.3e} > {abs_tol}"
    );
}

fn ring_vs_full(layout: KvLayout, label: &str) {
    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));
    let attn = build_attn(&mut allocs);

    let n_total = 2 * UBATCH + 1;
    let resid_host = det_signal(n_total * HIDDEN, 17);
    // Independent device copies so the two runs can't share state.
    let resid_full = upload_residual(&device, &resid_host);
    let resid_ring = upload_residual(&device, &resid_host);

    let (full_c2, full_d) = run(&device, &reg, &attn, resid_full, false, layout);
    let (ring_c2, ring_d) = run(&device, &reg, &attn, resid_ring, true, layout);

    assert_close(&format!("{label} chunk2 straddle prefill"), &full_c2, &ring_c2);
    assert_close(&format!("{label} decode wrap"), &full_d, &ring_d);
}

/// Mixed-batch (`standard_attn_mixed`) on a ring layer: populate a decode
/// slot past the wrap, then run a mixed batch (K prefill rows for a fresh
/// slot + 1 decode row for the wrapped slot). A ring slab must match a
/// full-context slab. Returns the (K+1)-row delta.
fn run_mixed(
    device: &HipDevice,
    reg: &OpsRegistry,
    attn: &AttnWeights,
    resid_pop: DevicePtr,
    resid_mix: DevicePtr,
    ring: bool,
    layout: KvLayout,
) -> Vec<f32> {
    let stream = device.default_stream();
    let mut pool = ScratchPool::new(device, build_scratch_cfg(ring, layout, 2)).expect("pool mix");
    let mut ctx = SingleDeviceForwardCtx::new(device, stream, reg, &mut pool);

    // Populate slot 1 to position 2U-1 via two prefill chunks (the second
    // straddles the wrap for a ring slab).
    let pa = unsafe { Tensor::<F16>::from_raw(resid_pop, UBATCH * HIDDEN) };
    ctx.standard_attn(&pa, attn, 0, &(0..UBATCH).collect::<Vec<_>>(), &[1usize; UBATCH], None)
        .expect("populate A")
        .expect("populate A resid");
    let pb_ptr = resid_pop.offset_bytes(UBATCH * HIDDEN * 2);
    let pb = unsafe { Tensor::<F16>::from_raw(pb_ptr, UBATCH * HIDDEN) };
    ctx.standard_attn(&pb, attn, 0, &(UBATCH..2 * UBATCH).collect::<Vec<_>>(), &[1usize; UBATCH], None)
        .expect("populate B")
        .expect("populate B resid");

    // Mixed: K=UBATCH prefill rows for fresh slot 0 (pos 0..U) + 1 decode row
    // for slot 1 at position 2U (n_kv > ring_depth → wraps).
    // K+N must fit max_prefill_tokens (= UBATCH): UBATCH-1 prefill rows + 1
    // decode. The decode row (slot 1 @ pos 2U, wrapped) is the ring target.
    let kp = UBATCH - 1;
    let mix = unsafe { Tensor::<F16>::from_raw(resid_mix, (kp + 1) * HIDDEN) };
    let mut positions: Vec<usize> = (0..kp).collect();
    positions.push(2 * UBATCH);
    let mut slot_ids = vec![0usize; kp];
    slot_ids.push(1);
    let delta = ctx
        .standard_attn_mixed(
            &mix,
            attn,
            0,
            MixedBatch { positions: &positions, slot_ids: &slot_ids, prefill_rows: kp },
            None,
        )
        .expect("mixed")
        .expect("mixed resid");
    let out = read_f16(device, delta.ptr, (kp + 1) * HIDDEN);
    pool.dispose(device).expect("dispose");
    out
}

fn ring_vs_full_mixed(layout: KvLayout, label: &str) {
    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));
    let attn = build_attn(&mut allocs);

    let pop_host = det_signal(2 * UBATCH * HIDDEN, 23);
    let mix_host = det_signal(UBATCH * HIDDEN, 29);
    let pop_full = upload_residual(&device, &pop_host);
    let mix_full = upload_residual(&device, &mix_host);
    let pop_ring = upload_residual(&device, &pop_host);
    let mix_ring = upload_residual(&device, &mix_host);

    let full = run_mixed(&device, &reg, &attn, pop_full, mix_full, false, layout);
    let ring = run_mixed(&device, &reg, &attn, pop_ring, mix_ring, true, layout);
    assert_close(&format!("{label} mixed wrap"), &full, &ring);
}

#[test]
fn gemma_ring_swa_mixed_matches_full_depth() {
    // standard_attn_mixed on a ring layer.
    ring_vs_full_mixed(KvLayout::F16Contig, "ring-swa mixed");
}

fn splitk_ring_vs_full(layout: KvLayout, label: &str) {
    // Decode with a > 256-key SWA window on a ring slab triggers windowed
    // split-K (chunk_base): the chunk kernel chunks [n_kv - window, n_kv) over
    // the ring, addressing rows mod ring_depth. Must match the full-context
    // (slide → split-K) path, which chunks the same window off the slid pointer.
    const W: i32 = 288; // > 256 so the window splits into >1 chunk (split-K)
    const UB: usize = 128; // max_prefill_tokens
    const D: usize = W as usize + UB; // ring slab depth = 416
    const MSL: usize = 512; // > D (ring active) and > TGT (full slab fits)
    const CHUNK: usize = 128; // prefill chunk (<= UB and <= D - W)
    const TGT: usize = 450; // decode pos → n_kv 451 > D (ring); window 288 > 256 (split-K)

    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));
    let mut attn = build_attn(&mut allocs);
    attn.window_size = W;
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;

    let run = |ring: bool, resid: DevicePtr| -> Vec<f32> {
        let stream = device.default_stream();
        let cfg = ScratchConfig {
            hidden: HIDDEN,
            intermediate: 256,
            q_width,
            kv_width,
            vocab: 64,
            max_seq_len: MSL,
            num_layers: 1,
            max_experts: 0,
            max_experts_per_tok: 0,
            gdn: None,
            per_layer_kv_widths: None,
            attn_q_gated: false,
            shared_intermediate: 0,
            max_prefill_tokens: UB,
            max_slots: 1,
            per_layer_embd: 0,
            paged_kv: None,
            kv_layout: layout,
            per_layer_kv_layouts: None,
            per_layer_kv_depths: if ring { Some(vec![D]) } else { None },
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("pool");
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
        // Chunked prefill [0, TGT) into slot 0 (chunks straddle the ring wrap).
        let mut pos = 0;
        while pos < TGT {
            let n = CHUNK.min(TGT - pos);
            let t = unsafe { Tensor::<F16>::from_raw(resid.offset_bytes(pos * HIDDEN * 2), n * HIDDEN) };
            let positions: Vec<usize> = (pos..pos + n).collect();
            ctx.standard_attn(&t, &attn, 0, &positions, &vec![0usize; n], None)
                .expect("prefill")
                .expect("prefill resid");
            pos += n;
        }
        // Decode at TGT (n_kv > 256 → split-K; > D → ring active).
        let dt = unsafe { Tensor::<F16>::from_raw(resid.offset_bytes(TGT * HIDDEN * 2), HIDDEN) };
        let delta = ctx
            .standard_attn(&dt, &attn, 0, &[TGT], &[0usize], None)
            .expect("decode")
            .expect("decode resid");
        let out = read_f16(&device, delta.ptr, HIDDEN);
        pool.dispose(&device).expect("dispose");
        out
    };

    let resid_host = det_signal((TGT + 1) * HIDDEN, 41);
    let full = run(false, upload_residual(&device, &resid_host));
    let ring = run(true, upload_residual(&device, &resid_host));
    assert_close(label, &full, &ring);
}

#[test]
fn gemma_ring_swa_splitk_f16_matches_full_depth() {
    splitk_ring_vs_full(KvLayout::F16Contig, "ring-swa splitk f16");
}

#[test]
fn gemma_ring_swa_splitk_q8_matches_full_depth() {
    // Q8 split-K over the ring — caught the chunk_base kernel bug the F16-only
    // test missed (Q8 chunk kernel's t_start wasn't using chunk_base).
    splitk_ring_vs_full(KvLayout::Q8Contig, "ring-swa splitk q8");
}

#[test]
fn gemma_ring_swa_f16_matches_full_depth() {
    // F16 KV → kv_append_v_unit_norm_f16 per-token wrap write path.
    ring_vs_full(KvLayout::F16Contig, "ring-swa f16");
}

#[test]
fn gemma_ring_swa_q8_matches_full_depth() {
    // Q8 KV → kv_append_f16_to_q8 2-segment straddle-split write path.
    ring_vs_full(KvLayout::Q8Contig, "ring-swa q8");
}
