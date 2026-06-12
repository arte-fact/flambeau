//! Slot snapshot/restore parity: prefill a synthetic prompt, snapshot the
//! slot, decode a reference continuation, restore the snapshot, decode
//! again — every step's logits must be bit-identical. Covers the dense
//! KV path (F16Contig + Q8Contig rows) and the GDN recurrent state.

#![cfg(feature = "hip")]

mod common;

use common::{det_signal, DeviceAllocs};
use flambeau_backend_hip::HipDevice;
use flambeau_core::Device;
use flambeau_forward::core::{KvLayout, ScratchConfig};
use flambeau_forward::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, GdnDims, GdnWeights,
    LmHeadWeights,
};
use flambeau_forward::{ScratchPool, SingleDeviceForwardCtx};
use flambeau_model_ops::Tensor;
use flambeau_ops::OpsRegistry;

const RMS_EPS: f32 = 1e-5;
const PROMPT: [u32; 4] = [7, 11, 13, 5];
const DECODE_STEPS: usize = 8;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

fn assert_logits_identical(reference: &[Vec<f32>], replay: &[Vec<f32>], tag: &str) {
    assert_eq!(reference.len(), replay.len());
    for (step, (a, b)) in reference.iter().zip(replay.iter()).enumerate() {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x == y,
                "{tag}: step {step} logits[{i}] diverged: reference {x} vs replay {y}"
            );
        }
    }
}

fn kv_parity(layout: KvLayout, swa: bool) {
    const VOCAB: usize = 64;
    const HIDDEN: usize = 128;
    const INTERMEDIATE: usize = 256;
    const NUM_LAYERS: usize = 2;
    const N_HEADS: usize = 4;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const MAX_SEQ_LEN: usize = 64;
    const WINDOW: usize = 8;
    const RING_DEPTH: usize = WINDOW + 1; // window + max_prefill_tokens (1)

    // SWA: a prompt longer than the ring so layer 0 WRAPS — the case the
    // old snapshot bailed on; the resident window is captured whole.
    // Dense: the short shared PROMPT, no wrap. Both stay < MAX_SEQ_LEN so
    // the global layer (layer 1) keeps full context.
    let swa_prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let prompt: &[u32] = if swa { &swa_prompt } else { &PROMPT };

    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;

    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

    let embd = EmbeddingWeights {
        token_embd: allocs.upload_f16(&det_signal(VOCAB * HIDDEN, 1)),
        vocab_size: VOCAB,
        hidden: HIDDEN,
        post_scale: None,
    };
    let lm_head = LmHeadWeights {
        output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
        final_logit_softcap: None,
        vocab_size: VOCAB,
        hidden: HIDDEN,
        rms_eps: RMS_EPS,
    };

    let mut attn_weights: Vec<AttnWeights> = Vec::with_capacity(NUM_LAYERS);
    let mut ffn_weights: Vec<FfnWeights> = Vec::with_capacity(NUM_LAYERS);
    for li in 0..NUM_LAYERS {
        let seed = 100 + (li as u32) * 10;
        attn_weights.push(AttnWeights {
            attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, seed + 1), q_width, HIDDEN),
            attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 2), kv_width, HIDDEN),
            attn_v: Some(allocs.upload_q8_0(
                &det_signal(kv_width * HIDDEN, seed + 3),
                kv_width,
                HIDDEN,
            )),
            attn_v_unit_norm_w: None,
            attn_output: allocs.upload_q8_0(
                &det_signal(HIDDEN * q_width, seed + 4),
                HIDDEN,
                q_width,
            ),
            attn_q_norm: None,
            attn_k_norm: None,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            rotated_dims: HEAD_DIM,
            rope_theta: 10000.0,
            rope_variant: flambeau_forward::ctx::RopeVariant::NeoxSplit,
            window_size: if swa && li == 0 { WINDOW as i32 } else { 0 },
            rms_eps: RMS_EPS,
            softmax_scale: None,
            attn_q_gated: false,
            kv_share_src: None,
            post_attn_norm: None,
        });
        ffn_weights.push(FfnWeights {
            ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            ffn_gate: allocs.upload_q8_0(
                &det_signal(INTERMEDIATE * HIDDEN, seed + 5),
                INTERMEDIATE,
                HIDDEN,
            ),
            ffn_up: allocs.upload_q8_0(
                &det_signal(INTERMEDIATE * HIDDEN, seed + 6),
                INTERMEDIATE,
                HIDDEN,
            ),
            ffn_down: allocs.upload_q8_0(
                &det_signal(HIDDEN * INTERMEDIATE, seed + 7),
                HIDDEN,
                INTERMEDIATE,
            ),
            activation: Activation::SwiGLU,
            rms_eps: RMS_EPS,
            post_ffn_norm: None,
        });
    }

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        q_width,
        kv_width,
        vocab: VOCAB,
        max_seq_len: MAX_SEQ_LEN,
        num_layers: NUM_LAYERS,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: None,
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: 1,
        max_slots: 1,
        per_layer_embd: 0,
        paged_kv: None,
        kv_layout: layout,
        per_layer_kv_layouts: None,
        // SWA: layer 0 is a window-sized ring (depth = window + ubatch),
        // layer 1 stays full-context. Dense: both full-context.
        per_layer_kv_depths: if swa {
            Some(vec![RING_DEPTH, MAX_SEQ_LEN])
        } else {
            None
        },
    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");

    let fwd = |pool: &mut ScratchPool, token: u32, position: usize| -> Vec<f32> {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, pool);
        let mut resid = ctx.embed(&embd, &[token]).expect("embed");
        for li in 0..NUM_LAYERS {
            let normed = ctx
                .rmsnorm(&resid, &attn_weights[li].attn_norm, RMS_EPS, 1)
                .expect("attn rmsnorm");
            let delta = ctx
                .standard_attn(&normed, &attn_weights[li], li, &[position], &[0], None)
                .expect("standard_attn")
                .expect("delta");
            resid = ctx.residual_add(resid, delta, 1).expect("attn add");
            let normed = ctx
                .rmsnorm(&resid, &ffn_weights[li].ffn_norm, RMS_EPS, 1)
                .expect("ffn rmsnorm");
            let delta = ctx
                .dense_ffn(&normed, &ffn_weights[li], 1, None)
                .expect("dense_ffn")
                .expect("delta");
            resid = ctx.residual_add(resid, delta, 1).expect("ffn add");
        }
        ctx.output_head(&resid, &lm_head, &[0])
            .expect("output_head");
        ctx.logits().to_vec()
    };

    let mut last_logits = Vec::new();
    for (p, &t) in prompt.iter().enumerate() {
        last_logits = fwd(&mut pool, t, p);
    }
    let snap = pool
        .snapshot_slot_bytes(0, prompt.len(), &device)
        .expect("snapshot");

    let decode = |pool: &mut ScratchPool, seed_logits: &[f32]| -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(DECODE_STEPS);
        let mut token = argmax(seed_logits);
        for step in 0..DECODE_STEPS {
            let logits = fwd(pool, token, prompt.len() + step);
            token = argmax(&logits);
            out.push(logits);
        }
        out
    };

    let reference = decode(&mut pool, &last_logits);

    // Wrong-token-count restores must be rejected, not silently applied.
    assert!(pool
        .restore_slot_bytes(0, prompt.len() + 1, &snap, &device)
        .is_err());

    pool.restore_slot_bytes(0, prompt.len(), &snap, &device)
        .expect("restore");
    let replay = decode(&mut pool, &last_logits);

    let tag = if swa { "swa" } else { "dense" };
    assert_logits_identical(&reference, &replay, &format!("{tag} {layout:?}"));
    pool.dispose(&device).expect("pool dispose");
}

#[test]
fn dense_kv_snapshot_restore_decode_parity_f16() {
    kv_parity(KvLayout::F16Contig, false);
}

#[test]
fn swa_kv_snapshot_restore_decode_parity_f16() {
    kv_parity(KvLayout::F16Contig, true);
}

#[test]
fn swa_kv_snapshot_restore_decode_parity_q8() {
    kv_parity(KvLayout::Q8Contig, true);
}

#[test]
fn dense_kv_snapshot_restore_decode_parity_q8() {
    kv_parity(KvLayout::Q8Contig, false);
}

#[test]
fn gdn_snapshot_restore_decode_parity() {
    const HIDDEN: usize = 256;
    const VOCAB: usize = 64;
    const NUM_LAYERS: usize = 2;
    const HEAD_K_DIM: usize = 128;
    const HEAD_V_DIM: usize = 128;
    const NUM_V_HEADS: usize = 2;
    const NUM_K_HEADS: usize = 2;
    const CONV_KERNEL: usize = 4;

    let d_inner = NUM_V_HEADS * HEAD_V_DIM;
    let conv_channels = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM;

    let device = HipDevice::new(0).expect("HipDevice 0");
    device.bind().expect("bind");
    let reg = OpsRegistry::new(&device).expect("OpsRegistry::new");
    let stream = device.default_stream();
    let mut allocs = DeviceAllocs::new(HipDevice::new(0).expect("HipDevice 0 alias"));

    let upload_f32 = |allocs: &mut DeviceAllocs, host: &[f32]| {
        let (ptr, _) = allocs.upload(host);
        unsafe { Tensor::<flambeau_model_ops::F32>::from_raw(ptr, host.len()) }
    };

    let embd = EmbeddingWeights {
        token_embd: allocs.upload_f16(&det_signal(VOCAB * HIDDEN, 1)),
        vocab_size: VOCAB,
        hidden: HIDDEN,
        post_scale: None,
    };
    let lm_head = LmHeadWeights {
        output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
        lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
        final_logit_softcap: None,
        vocab_size: VOCAB,
        hidden: HIDDEN,
        rms_eps: RMS_EPS,
    };

    let dims = GdnDims {
        d_inner,
        num_v_heads: NUM_V_HEADS,
        num_k_heads: NUM_K_HEADS,
        head_k_dim: HEAD_K_DIM,
        head_v_dim: HEAD_V_DIM,
        conv_channels,
        conv_kernel: CONV_KERNEL,
    };

    let mut gdn_weights: Vec<GdnWeights> = Vec::with_capacity(NUM_LAYERS);
    for li in 0..NUM_LAYERS {
        let seed = 200 + (li as u32) * 30;
        gdn_weights.push(GdnWeights {
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
            ssm_dt_bias: upload_f32(&mut allocs, &det_signal(NUM_V_HEADS, seed + 6)),
            ssm_a: upload_f32(&mut allocs, &det_signal(NUM_V_HEADS, seed + 7)),
            ssm_conv1d: upload_f32(
                &mut allocs,
                &det_signal(CONV_KERNEL * conv_channels, seed + 8),
            ),
            ssm_norm_w: upload_f32(&mut allocs, &vec![1.0_f32; HEAD_V_DIM]),
            dims,
            rms_eps: RMS_EPS,
            rep_inner_layout: false,
        });
    }

    let cfg = ScratchConfig {
        hidden: HIDDEN,
        intermediate: 0,
        q_width: 0,
        kv_width: 0,
        vocab: VOCAB,
        max_seq_len: 16,
        num_layers: NUM_LAYERS,
        max_experts: 0,
        max_experts_per_tok: 0,
        gdn: Some(dims),
        per_layer_kv_widths: None,
        attn_q_gated: false,
        shared_intermediate: 0,
        max_prefill_tokens: 1,
        max_slots: 1,
        per_layer_embd: 0,
        paged_kv: None,
        kv_layout: KvLayout::F16Contig,
        per_layer_kv_layouts: None,
        per_layer_kv_depths: None,
    };
    let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");

    let fwd = |pool: &mut ScratchPool, token: u32| -> Vec<f32> {
        let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, pool);
        let mut resid = ctx.embed(&embd, &[token]).expect("embed");
        for li in 0..NUM_LAYERS {
            let normed = ctx
                .rmsnorm(&resid, &gdn_weights[li].attn_norm, RMS_EPS, 1)
                .expect("attn_norm");
            let delta = ctx
                .gdn_layer(&normed, &gdn_weights[li], li, &[0], None)
                .expect("gdn_layer")
                .expect("delta");
            resid = ctx.residual_add(resid, delta, 1).expect("residual_add");
        }
        ctx.output_head(&resid, &lm_head, &[0]).expect("output_head");
        ctx.logits().to_vec()
    };

    let mut last_logits = Vec::new();
    for &t in PROMPT.iter() {
        last_logits = fwd(&mut pool, t);
    }
    let snap = pool
        .snapshot_slot_bytes(0, PROMPT.len(), &device)
        .expect("snapshot");

    let decode = |pool: &mut ScratchPool, seed_logits: &[f32]| -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(DECODE_STEPS);
        let mut token = argmax(seed_logits);
        for _ in 0..DECODE_STEPS {
            let logits = fwd(pool, token);
            token = argmax(&logits);
            out.push(logits);
        }
        out
    };

    let reference = decode(&mut pool, &last_logits);

    // The reference decode advanced the recurrent state past the snapshot
    // point; zero it so a passing replay can only come from the restore.
    pool.reset_gdn_state_slot(0, &device).expect("reset gdn");
    pool.restore_slot_bytes(0, PROMPT.len(), &snap, &device)
        .expect("restore");
    let replay = decode(&mut pool, &last_logits);

    assert_logits_identical(&reference, &replay, "gdn");
    pool.dispose(&device).expect("pool dispose");
}
