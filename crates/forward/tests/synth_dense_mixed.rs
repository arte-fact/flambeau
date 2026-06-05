//! Phase K1a parity test: `standard_attn_mixed` vs separate
//! `standard_attn` calls. Runs K prefill rows for slot 0 + N decode rows
//! for slots 1..=N through the mixed path once, and compares the F16
//! attention output to the reference path that runs the prefill and
//! the batched decode as two independent `standard_attn` calls.
//!
//! Single-device ctx (no TP AR, no PP handoff) so the residual returned
//! by `standard_attn` is the unmodified attention output projection
//! (F16). The reference and mixed paths must produce bit-equal or
//! F16-tolerance-equal F16 deltas — modulo F32 reduction-order noise
//! on the output proj's per-row accumulators.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::core::{MixedBatch, ScratchConfig};
use flambeau_forward::ctx::{AttnWeights, ForwardCtx, RopeVariant};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;

const HIDDEN: usize = 128;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 16;
const RMS_EPS: f32 = 1e-5;
const K_PREFILL: usize = 4;
const N_DECODE: usize = 3;

fn build_attn_weights(allocs: &mut DeviceAllocs) -> AttnWeights {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;
    AttnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, 101), q_width, HIDDEN),
        attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 102), kv_width, HIDDEN),
        attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, 103), kv_width, HIDDEN)),
        attn_v_unit_norm_w: None,
        attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, 104), HIDDEN, q_width),
        attn_q_norm: None,
        attn_k_norm: None,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        rotated_dims: HEAD_DIM,
        rope_theta: 10000.0,
        rope_variant: RopeVariant::NeoxSplit,
        window_size: 0,
        rms_eps: RMS_EPS,
        softmax_scale: None,
        attn_q_gated: false,
        kv_share_src: None,
        post_attn_norm: None,
    }
}

fn build_scratch_cfg() -> ScratchConfig {
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
        max_prefill_tokens: K_PREFILL + N_DECODE,
        max_slots: N_DECODE + 1,
        per_layer_embd: 0,
        paged_kv: None,
        kv_layout: flambeau_forward::core::KvLayout::F16Contig,
        per_layer_kv_layouts: None,
        per_layer_kv_depths: None,    }
}

fn det_signal_f16(n: usize, seed: u32) -> Vec<f32> {
    det_signal(n, seed)
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

#[test]
fn synth_mixed_matches_separate_calls() {
    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

    let attn = build_attn_weights(&mut allocs);
    let n_total = K_PREFILL + N_DECODE;

    // Deterministic residual inputs. We reuse the SAME residual for both
    // the reference and mixed runs so the only difference under test is
    // the path through the attention dispatch.
    let resid_host = det_signal_f16(n_total * HIDDEN, 7);
    let resid_dev_ref = upload_residual(&device, &resid_host);
    let resid_dev_mix = upload_residual(&device, &resid_host);

    // -------- Reference: two separate standard_attn calls --------
    let ref_pref_out: Vec<f32>;
    let ref_dec_out: Vec<f32>;
    {
        let mut pool = ScratchPool::new(&device, build_scratch_cfg()).expect("pool ref");
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

        // Pass 1: K prefill rows for slot 0 at positions [0..K).
        let resid_pref = unsafe { Tensor::<F16>::from_raw(resid_dev_ref, K_PREFILL * HIDDEN) };
        let positions_p: Vec<usize> = (0..K_PREFILL).collect();
        let slot_ids_p = vec![0usize; K_PREFILL];
        let delta_p = ctx
            .standard_attn(&resid_pref, &attn, 0, &positions_p, &slot_ids_p, None)
            .expect("standard_attn prefill")
            .expect("prefill delta");
        ref_pref_out = read_f16(&device, delta_p.ptr, K_PREFILL * HIDDEN);

        // Pass 2: N decode rows for slots 1..=N at position 0.
        // Resid input view starts after the K rows.
        let resid_dec_ptr = resid_dev_ref.offset_bytes(K_PREFILL * HIDDEN * 2);
        let resid_dec = unsafe { Tensor::<F16>::from_raw(resid_dec_ptr, N_DECODE * HIDDEN) };
        let positions_d = vec![0usize; N_DECODE];
        let slot_ids_d: Vec<usize> = (1..=N_DECODE).collect();
        let delta_d = ctx
            .standard_attn(&resid_dec, &attn, 0, &positions_d, &slot_ids_d, None)
            .expect("standard_attn decode")
            .expect("decode delta");
        ref_dec_out = read_f16(&device, delta_d.ptr, N_DECODE * HIDDEN);

        pool.dispose(&device).expect("pool dispose ref");
    }

    // -------- Mixed: one standard_attn_mixed call --------
    let mix_pref_out: Vec<f32>;
    let mix_dec_out: Vec<f32>;
    {
        let mut pool = ScratchPool::new(&device, build_scratch_cfg()).expect("pool mix");
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

        let resid_mix = unsafe { Tensor::<F16>::from_raw(resid_dev_mix, n_total * HIDDEN) };
        let mut positions = (0..K_PREFILL).collect::<Vec<_>>();
        positions.extend(vec![0usize; N_DECODE]);
        let mut slot_ids = vec![0usize; K_PREFILL];
        slot_ids.extend(1..=N_DECODE);

        let delta = ctx
            .standard_attn_mixed(
                &resid_mix,
                &attn,
                0,
                MixedBatch {
                    positions: &positions,
                    slot_ids: &slot_ids,
                    prefill_rows: K_PREFILL,
                },
                None,
            )
            .expect("standard_attn_mixed")
            .expect("mixed delta");
        let delta_all = read_f16(&device, delta.ptr, n_total * HIDDEN);
        mix_pref_out = delta_all[..K_PREFILL * HIDDEN].to_vec();
        mix_dec_out = delta_all[K_PREFILL * HIDDEN..].to_vec();

        pool.dispose(&device).expect("pool dispose mix");
    }

    // -------- Compare. Hybrid abs+rel tolerance for F16 reduction noise. --------
    fn assert_close(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{label}: len mismatch");
        let mut worst_rel = 0.0f32;
        let mut worst_abs = 0.0f32;
        let mut worst_idx = 0usize;
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
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
             (ref={}, mix={}), abs_tol={abs_tol} rel_tol={rel_tol}",
            a[worst_idx], b[worst_idx]
        );
        assert!(
            worst_rel <= rel_tol || worst_abs <= abs_tol,
            "{label}: rel={worst_rel:.3e} > {rel_tol:.3e} AND abs={worst_abs:.3e} > {abs_tol}"
        );
    }

    assert_close("prefill K rows", &ref_pref_out, &mix_pref_out);
    assert_close("decode N rows", &ref_dec_out, &mix_dec_out);
}
