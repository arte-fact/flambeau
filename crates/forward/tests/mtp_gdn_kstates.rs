//! MTP verify rollback of the GDN state by per-token states + device index
//! (`mtp_spec_rows` + `gdn_select_rows`) against the previous snapshot +
//! replay path, bit for bit. Five chained verify cycles of 5 rows keep 1, 2,
//! 3, 4 and 5 rows (0 to 4 drafts accepted); every cycle after the first
//! reads its input state through the index. The reference is what the old
//! path computed: the certified `gdn_state_step_alphabeta_f32_s128` over the
//! first `n_keep` stored rows from the cycle's input state (the replay), and
//! the conv history taken from the stored conv rows.

#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend::Backend;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_forward::core::scratch::MtpScratch;
use flambeau_forward::core::ScratchConfig;
use flambeau_forward::ctx::{ForwardCtx, GdnDims, GdnWeights};
use flambeau_forward::{ForwardEngine, ScratchPool};
use flambeau_model_ops::{Tensor, F32};
use flambeau_ops::Ops;

const HIDDEN: usize = 256;
const NUM_V_HEADS: usize = 4;
const NUM_K_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
const VERIFY_ROWS: usize = 5;

fn read_f32<D: Device>(dev: &D, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    // SAFETY: `src` holds `n` f32; synced below.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .expect("DtoH");
    }
    dev.default_stream().synchronize().expect("sync");
    out
}

fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn run<B: Backend>(dev: &B::Device, reg: &B::Registry, allocs: &mut DeviceAllocs<B::Device>) {
    let d_inner = NUM_V_HEADS * HEAD_DIM;
    let conv_channels = 2 * NUM_K_HEADS * HEAD_DIM + d_inner;
    let dims = GdnDims {
        d_inner,
        num_v_heads: NUM_V_HEADS,
        num_k_heads: NUM_K_HEADS,
        head_k_dim: HEAD_DIM,
        head_v_dim: HEAD_DIM,
        conv_channels,
        conv_kernel: CONV_KERNEL,
    };
    let f32_tensor = |allocs: &mut DeviceAllocs<B::Device>, host: &[f32]| -> Tensor<F32> {
        let (ptr, _) = allocs.upload(host);
        unsafe { Tensor::<F32>::from_raw(ptr, host.len()) }
    };
    let ssm_a: Vec<f32> = (0..NUM_V_HEADS).map(|i| -(0.5 + 0.1 * i as f32)).collect();
    let w = GdnWeights {
        attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        attn_qkv: allocs.upload_q8_0(
            &det_signal(conv_channels * HIDDEN, 301),
            conv_channels,
            HIDDEN,
        ),
        attn_gate: allocs.upload_q8_0(&det_signal(d_inner * HIDDEN, 302), d_inner, HIDDEN),
        ssm_alpha: allocs.upload_q8_0(&det_signal(NUM_V_HEADS * HIDDEN, 303), NUM_V_HEADS, HIDDEN),
        ssm_beta: allocs.upload_q8_0(&det_signal(NUM_V_HEADS * HIDDEN, 304), NUM_V_HEADS, HIDDEN),
        ssm_out: allocs.upload_q8_0(&det_signal(HIDDEN * d_inner, 305), HIDDEN, d_inner),
        ssm_dt_bias: f32_tensor(allocs, &det_signal(NUM_V_HEADS, 306)),
        ssm_a: f32_tensor(allocs, &ssm_a),
        ssm_conv1d: f32_tensor(allocs, &det_signal(CONV_KERNEL * conv_channels, 307)),
        ssm_norm_w: f32_tensor(allocs, &vec![1.0_f32; HEAD_DIM]),
        dims,
        rms_eps: 1e-5,
        rep_inner_layout: false,
    };

    let cfg = ScratchConfig {
        mtp: Some(MtpScratch {
            kv_width: 0,
            max_draft: VERIFY_ROWS - 1,
        }),
        hidden: HIDDEN,
        intermediate: 0,
        q_width: 0,
        kv_width: 0,
        vocab: 64,
        max_seq_len: 64,
        num_layers: 1,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: Some(dims),
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: VERIFY_ROWS,
        max_slots: 1,
        per_layer_embd: 0,
        paged_kv: None,
        kv_layout: flambeau_forward::core::KvLayout::F16Contig,
        per_layer_kv_layouts: None,
        per_layer_kv_depths: None,
    };
    let mut pool = ScratchPool::new(dev, cfg).expect("ScratchPool::new");
    pool.reset_gdn_state(dev).expect("reset gdn");
    let stream = dev.default_stream();
    let state_n = NUM_V_HEADS * HEAD_DIM * HEAD_DIM;
    let hist_n = (CONV_KERNEL - 1) * conv_channels;

    // A short prefill so the cycles start from a non-trivial state.
    let warm = allocs.upload_f16(&det_signal(3 * HIDDEN, 11));
    {
        let mut ctx = ForwardEngine::<B>::new_solo(dev, stream, reg, &mut pool);
        ctx.gdn_layer(&warm, &w, 0, &[0; 3], None)
            .expect("warm gdn");
    }
    let mut ref_state = read_f32(dev, pool.gdn_state[0].state, state_n);

    let d_ref_in = allocs.upload(&vec![0.0f32; state_n]).0;
    let d_ref_out = allocs.upload(&vec![0.0f32; state_n]).0;
    let d_ref_ao = allocs.upload(&vec![0.0f32; VERIFY_ROWS * d_inner]).0;
    for (cycle, n_keep) in [1usize, 2, 3, 4, 5].into_iter().enumerate() {
        let input = allocs.upload_f16(&det_signal(VERIFY_ROWS * HIDDEN, 40 + cycle as u32));
        {
            let mut ctx = ForwardEngine::<B>::new_solo(dev, stream, reg, &mut pool);
            ctx.mtp_spec_rows(true);
            ctx.gdn_layer(&input, &w, 0, &[0; VERIFY_ROWS], None)
                .expect("verify gdn");
            ctx.mtp_spec_rows(false);
            ctx.gdn_select_rows(0, n_keep, VERIFY_ROWS)
                .expect("select rows");
        }
        assert_eq!(pool.gdn_rs_pending, n_keep);

        let rows = pool.gdn_spec_rows(0, &dims).expect("spec rows");
        // SAFETY: d_ref_in holds state_n f32, ref_state is state_n f32.
        unsafe {
            dev.memcpy_async(
                stream,
                CopyDirection::HostToDevice,
                d_ref_in,
                DevicePtr(ref_state.as_ptr() as usize),
                state_n * 4,
            )
            .expect("HtoD ref state");
        }
        B::ops(reg, stream)
            .gdn_state_step_alphabeta_f32_s128(
                flambeau_ops::GdnStepAlphaBetaBuffers {
                    q: rows.q,
                    k: rows.k,
                    v: rows.v,
                    alpha_in: rows.alpha,
                    beta_in: rows.beta,
                    ssm_dt_bias: w.ssm_dt_bias.ptr,
                    ssm_a: w.ssm_a.ptr,
                    state_in: d_ref_in,
                    state_out: d_ref_out,
                    attn_out: d_ref_ao,
                },
                flambeau_ops::GdnStepShape {
                    b: 1,
                    h_v: NUM_V_HEADS,
                    l: n_keep,
                    n_rep: NUM_V_HEADS / NUM_K_HEADS,
                    rep_inner_layout: false,
                },
            )
            .expect("reference replay");
        let want_state = read_f32(dev, d_ref_out, state_n);
        let want_hist = read_f32(
            dev,
            rows.conv_input.offset_bytes(n_keep * conv_channels * 4),
            hist_n,
        );

        let got_state = read_f32(
            dev,
            pool.mtp_gdn_kstates
                .offset_bytes((n_keep - 1) * state_n * 4),
            state_n,
        );
        let got_hist = read_f32(dev, pool.gdn_state[0].conv_history, hist_n);
        assert_eq!(
            bit_diff(&got_state, &want_state),
            0,
            "cycle {cycle}: state after keeping {n_keep} rows"
        );
        assert_eq!(
            bit_diff(&got_hist, &want_hist),
            0,
            "cycle {cycle}: conv history after keeping {n_keep} rows"
        );
        assert!(got_state.iter().any(|&x| x != 0.0));
        ref_state = got_state;
    }

    // Any other GDN reader first copies the selected state back to the live
    // slab: the host snapshot sees the last cycle's state and the index drops.
    let snap = {
        let mut ctx = ForwardEngine::<B>::new_solo(dev, stream, reg, &mut pool);
        ctx.gdn_snapshot(0).expect("gdn snapshot")
    };
    assert_eq!(pool.gdn_rs_pending, 0);
    let snap_state: Vec<f32> = snap[..state_n * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(bit_diff(&snap_state, &ref_state), 0, "materialized state");
    pool.dispose(dev).expect("pool dispose");
}

#[cfg(feature = "hip")]
#[test]
fn mtp_gdn_kstates_match_snapshot_replay_hip() {
    use flambeau_backend::HipBackend;
    use flambeau_backend_hip::HipDevice;
    let dev = HipDevice::new(0).expect("HipDevice 0");
    dev.bind().expect("hip bind");
    let reg = flambeau_ops::OpsRegistry::new(&dev).expect("OpsRegistry");
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("hip alias"));
    run::<HipBackend>(&dev, &reg, &mut allocs);
}

#[cfg(feature = "cuda")]
#[test]
fn mtp_gdn_kstates_match_snapshot_replay_cuda() {
    use flambeau_backend::CudaBackend;
    use flambeau_backend_cuda::CudaDevice;
    let dev = CudaDevice::new(0).expect("CudaDevice 0");
    dev.bind().expect("cuda bind");
    let reg = flambeau_ops::CudaOpsRegistry::new(&dev).expect("CudaOpsRegistry");
    let mut allocs = DeviceAllocs::new(CudaDevice::new(0).expect("cuda alias"));
    run::<CudaBackend>(&dev, &reg, &mut allocs);
}
