//! Phase K2 parity test: `gdn_layer_mixed` vs separate `gdn_layer`
//! calls. Runs K prefill rows for slot 0 + N decode rows for slots
//! 1..=N through the mixed path once, and compares the F16 GDN delta
//! outputs to a reference path that runs the prefill and the
//! batched-decode as two independent `gdn_layer` calls.
//!
//! Both paths start from zero-initialised GDN state + conv history,
//! so the per-slot updates are deterministic.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::core::{GdnMixedBatch, ScratchConfig};
use flambeau_forward::ctx::{ForwardCtx, GdnDims, GdnWeights};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;
use half::f16;

fn upload_f32_tensor(allocs: &mut DeviceAllocs, host: &[f32]) -> Tensor<flambeau_model_ops::F32> {
    let (ptr, _) = allocs.upload(host);
    unsafe { Tensor::<flambeau_model_ops::F32>::from_raw(ptr, host.len()) }
}

const HIDDEN: usize = 256;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const NUM_V_HEADS: usize = 2;
const NUM_K_HEADS: usize = 2;
const CONV_KERNEL: usize = 4;
const RMS_EPS: f32 = 1e-5;
const K_PREFILL: usize = 4;
const N_DECODE: usize = 3;

fn build_gdn_weights(allocs: &mut DeviceAllocs, dims: GdnDims) -> GdnWeights {
    let d_inner = dims.d_inner;
    let conv_channels = dims.conv_channels;
    let seed = 200;
    GdnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_qkv: allocs.upload_q8_0(
            &det_signal(conv_channels * HIDDEN, seed + 1),
            conv_channels,
            HIDDEN,
        ),
        attn_gate: allocs.upload_q8_0(&det_signal(d_inner * HIDDEN, seed + 2), d_inner, HIDDEN),
        ssm_alpha: allocs.upload_q8_0(
            &det_signal(NUM_V_HEADS * HIDDEN, seed + 3),
            NUM_V_HEADS,
            HIDDEN,
        ),
        ssm_beta: allocs.upload_q8_0(
            &det_signal(NUM_V_HEADS * HIDDEN, seed + 4),
            NUM_V_HEADS,
            HIDDEN,
        ),
        ssm_out: allocs.upload_q8_0(&det_signal(HIDDEN * d_inner, seed + 5), HIDDEN, d_inner),
        ssm_dt_bias: upload_f32_tensor(allocs, &det_signal(NUM_V_HEADS, seed + 6)),
        ssm_a: upload_f32_tensor(allocs, &det_signal(NUM_V_HEADS, seed + 7)),
        ssm_conv1d: upload_f32_tensor(allocs, &det_signal(CONV_KERNEL * conv_channels, seed + 8)),
        ssm_norm_w: upload_f32_tensor(allocs, &vec![1.0_f32; HEAD_V_DIM]),
        dims,
        rms_eps: RMS_EPS,
        rep_inner_layout: false,
    }
}

fn build_scratch_cfg(dims: GdnDims) -> ScratchConfig {
    ScratchConfig {
        hidden: HIDDEN,
        intermediate: 0,
        q_width: 0,
        kv_width: 0,
        vocab: 64,
        max_seq_len: 1,
        num_layers: 1,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: Some(dims),
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

fn zero_gdn_state(device: &HipDevice, pool: &ScratchPool, dims: GdnDims) {
    let state_n = NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;
    let conv_n = (CONV_KERNEL - 1) * dims.conv_channels;
    let n_slots = pool.config.max_slots;
    let state_zero = vec![0.0_f32; state_n * n_slots];
    let conv_zero = vec![0.0_f32; conv_n * n_slots];
    let stream = device.default_stream();
    for ls in &pool.gdn_state {
        unsafe {
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    ls.state,
                    DevicePtr(state_zero.as_ptr() as usize),
                    state_zero.len() * 4,
                )
                .expect("zero state");
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    ls.conv_history,
                    DevicePtr(conv_zero.as_ptr() as usize),
                    conv_zero.len() * 4,
                )
                .expect("zero conv");
        }
    }
    stream.synchronize().expect("sync zero-init");
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
fn synth_gdn_mixed_matches_separate_calls() {
    let d_inner = NUM_V_HEADS * HEAD_V_DIM;
    let conv_channels = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM;
    let dims = GdnDims {
        d_inner,
        num_v_heads: NUM_V_HEADS,
        num_k_heads: NUM_K_HEADS,
        head_k_dim: HEAD_K_DIM,
        head_v_dim: HEAD_V_DIM,
        conv_channels,
        conv_kernel: CONV_KERNEL,
    };

    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

    let weights = build_gdn_weights(&mut allocs, dims);
    let n_total = K_PREFILL + N_DECODE;
    let resid_host = det_signal(n_total * HIDDEN, 7);
    let resid_dev_ref = upload_residual(&device, &resid_host);
    let resid_dev_mix = upload_residual(&device, &resid_host);

    // -------- Reference: separate prefill + decode-batched --------
    let ref_pref: Vec<f32>;
    let ref_dec: Vec<f32>;
    {
        let mut pool = ScratchPool::new(&device, build_scratch_cfg(dims)).expect("pool ref");
        zero_gdn_state(&device, &pool, dims);
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

        // Prefill: K rows for slot 0.
        let resid_pref = unsafe { Tensor::<F16>::from_raw(resid_dev_ref, K_PREFILL * HIDDEN) };
        let slot_ids_p = vec![0usize; K_PREFILL];
        let delta_p = ctx
            .gdn_layer(&resid_pref, &weights, 0, &slot_ids_p, None)
            .expect("gdn_layer prefill")
            .expect("prefill delta");
        ref_pref = read_f16(&device, delta_p.ptr, K_PREFILL * HIDDEN);

        // Decode-batched: N rows for slots 1..=N.
        let resid_dec_ptr = resid_dev_ref.offset_bytes(K_PREFILL * HIDDEN * 2);
        let resid_dec = unsafe { Tensor::<F16>::from_raw(resid_dec_ptr, N_DECODE * HIDDEN) };
        let slot_ids_d: Vec<usize> = (1..=N_DECODE).collect();
        let delta_d = ctx
            .gdn_layer(&resid_dec, &weights, 0, &slot_ids_d, None)
            .expect("gdn_layer decode")
            .expect("decode delta");
        ref_dec = read_f16(&device, delta_d.ptr, N_DECODE * HIDDEN);

        pool.dispose(&device).expect("pool dispose ref");
    }

    // -------- Mixed: one gdn_layer_mixed call --------
    let mix_pref: Vec<f32>;
    let mix_dec: Vec<f32>;
    {
        let mut pool = ScratchPool::new(&device, build_scratch_cfg(dims)).expect("pool mix");
        zero_gdn_state(&device, &pool, dims);
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

        let resid_mix = unsafe { Tensor::<F16>::from_raw(resid_dev_mix, n_total * HIDDEN) };
        let mut slot_ids = vec![0usize; K_PREFILL];
        slot_ids.extend(1..=N_DECODE);
        let delta = ctx
            .gdn_layer_mixed(
                &resid_mix,
                &weights,
                0,
                GdnMixedBatch {
                    slot_ids: &slot_ids,
                    prefill_rows: K_PREFILL,
                },
                None,
            )
            .expect("gdn_layer_mixed")
            .expect("mixed delta");
        let all = read_f16(&device, delta.ptr, n_total * HIDDEN);
        mix_pref = all[..K_PREFILL * HIDDEN].to_vec();
        mix_dec = all[K_PREFILL * HIDDEN..].to_vec();

        pool.dispose(&device).expect("pool dispose mix");
    }

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

    assert_close("prefill K rows", &ref_pref, &mix_pref);
    assert_close("decode N rows", &ref_dec, &mix_dec);
}
