//! Single-GPU forward context: one device, one stream, `NoopHooks`.

use anyhow::Result;
use flambeau_backend_hip::{HipDevice, HipStream};
use flambeau_model_ops::{Tensor, F16};
use flambeau_ops::OpsRegistry;

use crate::core::{composites, CoreState, NoopHooks};
use crate::ctx::{
    AttnWeights, EmbeddingWeights, FfnWeights, ForwardCtx, LmHeadWeights, ModelLayout,
    MoeWeights,
};

pub use crate::core::{ScratchConfig, ScratchPool};

pub struct SingleDeviceForwardCtx<'a> {
    core: CoreState<'a>,
    hooks: NoopHooks,
}

impl<'a> SingleDeviceForwardCtx<'a> {
    pub fn new(
        device: &'a HipDevice,
        stream: &'a HipStream,
        reg: &'a OpsRegistry,
        pool: &'a mut ScratchPool,
    ) -> Self {
        Self {
            core: CoreState::new(device, stream, reg, pool),
            hooks: NoopHooks,
        }
    }

    /// Reset slot selection between forward passes. KV caches stay
    /// populated; caller's invariant is that `position` matches.
    pub fn reset(&mut self) {
        self.core.pool.current_residual_is_a = true;
    }
}

impl ForwardCtx for SingleDeviceForwardCtx<'_> {
    fn embed(&mut self, weights: &EmbeddingWeights, token_id: u32) -> Result<Tensor<F16>> {
        composites::embed_local(&mut self.core, &mut self.hooks, weights, token_id)
    }

    fn rmsnorm(
        &mut self,
        input: &Tensor<F16>,
        weight: &Tensor<F16>,
        eps: f32,
    ) -> Result<Tensor<F16>> {
        composites::rmsnorm_local(&mut self.core, &mut self.hooks, input, weight, eps)
    }

    fn residual_add(&mut self, a: Tensor<F16>, b: Tensor<F16>) -> Result<Tensor<F16>> {
        composites::residual_add_local(&mut self.core, &mut self.hooks, a, b)
    }

    fn standard_attn(
        &mut self,
        input: &Tensor<F16>,
        weights: &AttnWeights,
        layer_idx: usize,
        position: usize,
    ) -> Result<Tensor<F16>> {
        composites::standard_attn_local(
            &mut self.core,
            &mut self.hooks,
            input,
            weights,
            layer_idx,
            position,
        )
    }

    fn gdn_layer(
        &mut self,
        input: &Tensor<F16>,
        weights: &crate::ctx::GdnWeights,
        layer_idx: usize,
    ) -> Result<Tensor<F16>> {
        composites::gdn_layer_local(&mut self.core, &mut self.hooks, input, weights, layer_idx)
    }

    fn dense_ffn(&mut self, input: &Tensor<F16>, weights: &FfnWeights) -> Result<Tensor<F16>> {
        composites::dense_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn moe_ffn(&mut self, input: &Tensor<F16>, weights: &MoeWeights) -> Result<Tensor<F16>> {
        composites::moe_ffn_local(&mut self.core, &mut self.hooks, input, weights)
    }

    fn output_head(&mut self, input: &Tensor<F16>, lm_head: &LmHeadWeights) -> Result<()> {
        composites::output_head_local(&mut self.core, &mut self.hooks, input, lm_head)
    }

    fn layer_range<'b>(
        &'b mut self,
        layout: &'b ModelLayout,
    ) -> Box<dyn Iterator<Item = usize> + 'b> {
        Box::new(0..layout.num_layers)
    }

    fn logits(&self) -> &[f32] {
        &self.core.logits_host
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctx::{Activation, QuantWeight};
    use bytemuck::Pod;
    use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
    use flambeau_model_ops::Q8_0;
    use flambeau_quant::quantize_k::quantize_row_q8_0;
    use half::f16;

    struct DeviceAllocs {
        device: HipDevice,
        allocs: Vec<(DevicePtr, usize)>,
    }

    impl DeviceAllocs {
        fn new(device: HipDevice) -> Self {
            Self {
                device,
                allocs: Vec::new(),
            }
        }

        fn upload<P: Pod>(&mut self, host: &[P]) -> (DevicePtr, usize) {
            let bytes = std::mem::size_of_val(host);
            let ptr = self.device.alloc(bytes).expect("alloc");
            let stream = self.device.default_stream();
            unsafe {
                self.device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ptr,
                        DevicePtr(host.as_ptr() as usize),
                        bytes,
                    )
                    .expect("memcpy HtoD");
            }
            stream.synchronize().expect("sync");
            self.allocs.push((ptr, bytes));
            (ptr, bytes)
        }

        fn upload_f16(&mut self, host_f32: &[f32]) -> Tensor<F16> {
            let host_f16: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let (ptr, _) = self.upload(&host_f16);
            unsafe { Tensor::<F16>::from_raw(ptr, host_f16.len()) }
        }

        fn upload_q8_0(
            &mut self,
            host_f32: &[f32],
            rows: usize,
            cols: usize,
        ) -> QuantWeight {
            assert_eq!(host_f32.len(), rows * cols);
            assert!(cols % 32 == 0, "Q8_0 needs cols % 32 == 0");
            let mut bytes: Vec<u8> = Vec::with_capacity(rows * cols / 32 * 34);
            for r in 0..rows {
                quantize_row_q8_0(&host_f32[r * cols..(r + 1) * cols], &mut bytes);
            }
            let (ptr, _) = self.upload(&bytes);
            let tensor = unsafe { Tensor::<Q8_0>::from_raw(ptr, rows * cols) };
            QuantWeight::Q8_0(tensor)
        }
    }

    impl Drop for DeviceAllocs {
        fn drop(&mut self) {
            for (ptr, bytes) in self.allocs.drain(..) {
                unsafe {
                    let _ = self.device.dealloc(ptr, bytes);
                }
            }
        }
    }

    fn det_signal(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let s = (seed as f32) * 0.013 + (i as f32) * 0.027;
                (s.sin() + (s * 1.7).cos()) * 0.1
            })
            .collect()
    }

    #[test]
    fn single_device_synthetic_qwen_dense_one_token_forward() {
        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        const INTERMEDIATE: usize = 256;
        const NUM_LAYERS: usize = 2;
        const N_HEADS: usize = 4;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const MAX_SEQ_LEN: usize = 16;
        const RMS_EPS: f32 = 1e-5;

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
        let lm_head_weights = LmHeadWeights {
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
                attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 3), kv_width, HIDDEN)),
                attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, seed + 4), HIDDEN, q_width),
                attn_q_norm: None,
                attn_k_norm: None,
                n_heads: N_HEADS,
                n_kv_heads: N_KV_HEADS,
                head_dim: HEAD_DIM,
                rotated_dims: HEAD_DIM,
                rope_theta: 10000.0,
                window_size: 0,
                rms_eps: RMS_EPS,
                softmax_scale: None,
            });
            ffn_weights.push(FfnWeights {
                ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
                ffn_gate: allocs.upload_q8_0(&det_signal(INTERMEDIATE * HIDDEN, seed + 5), INTERMEDIATE, HIDDEN),
                ffn_up: allocs.upload_q8_0(&det_signal(INTERMEDIATE * HIDDEN, seed + 6), INTERMEDIATE, HIDDEN),
                ffn_down: allocs.upload_q8_0(&det_signal(HIDDEN * INTERMEDIATE, seed + 7), HIDDEN, INTERMEDIATE),
                activation: Activation::SwiGLU,
                rms_eps: RMS_EPS,
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
            gdn: None,
            per_layer_kv_widths: None,
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
        let layout = ModelLayout {
            num_layers: NUM_LAYERS,
            hidden: HIDDEN,
            kv_max_seq_len: MAX_SEQ_LEN,
        };

        {
            let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);

            let token_id: u32 = 7;
            let position: usize = 0;

            let mut resid = ctx.embed(&embd, token_id).expect("embed");
            let layers: Vec<usize> = ctx.layer_range(&layout).collect();
            assert_eq!(layers, vec![0, 1]);

            for li in layers {
                let normed = ctx
                    .rmsnorm(&resid, &attn_weights[li].attn_norm, RMS_EPS)
                    .expect("attn rmsnorm");
                let delta = ctx
                    .standard_attn(&normed, &attn_weights[li], li, position)
                    .expect("standard_attn");
                resid = ctx.residual_add(resid, delta).expect("attn residual_add");

                let normed = ctx
                    .rmsnorm(&resid, &ffn_weights[li].ffn_norm, RMS_EPS)
                    .expect("ffn rmsnorm");
                let delta = ctx.dense_ffn(&normed, &ffn_weights[li]).expect("dense_ffn");
                resid = ctx.residual_add(resid, delta).expect("ffn residual_add");
            }

            ctx.output_head(&resid, &lm_head_weights).expect("output_head");
            let logits = ctx.logits();
            assert_eq!(logits.len(), VOCAB);
            for (i, &l) in logits.iter().enumerate() {
                assert!(l.is_finite(), "logits[{i}] = {l} not finite");
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(max - min > 1e-3);
        }

        pool.dispose(&device).expect("pool dispose");
    }

    use crate::ctx::{GdnDims, GdnWeights};
    use flambeau_model_ops::F32;

    impl DeviceAllocs {
        fn upload_f32(&mut self, host: &[f32]) -> Tensor<F32> {
            let (ptr, _) = self.upload(host);
            unsafe { Tensor::<F32>::from_raw(ptr, host.len()) }
        }
    }

    #[test]
    fn single_device_synthetic_gdn_one_token_forward() {
        // DeltaNetLayer is hardwired to head_k_dim = head_v_dim = 128.
        const HIDDEN: usize = 256;
        const VOCAB: usize = 64;
        const NUM_LAYERS: usize = 2;
        const HEAD_K_DIM: usize = 128;
        const HEAD_V_DIM: usize = 128;
        const NUM_V_HEADS: usize = 2;
        const NUM_K_HEADS: usize = 2;
        const CONV_KERNEL: usize = 4;
        const RMS_EPS: f32 = 1e-5;

        let d_inner = NUM_V_HEADS * HEAD_V_DIM;
        let conv_channels = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM;

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
                attn_gate: allocs.upload_q8_0(
                    &det_signal(d_inner * HIDDEN, seed + 2),
                    d_inner,
                    HIDDEN,
                ),
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
                ssm_out: allocs.upload_q8_0(
                    &det_signal(HIDDEN * d_inner, seed + 5),
                    HIDDEN,
                    d_inner,
                ),
                ssm_dt_bias: allocs.upload_f32(&det_signal(NUM_V_HEADS, seed + 6)),
                ssm_a: allocs.upload_f32(&det_signal(NUM_V_HEADS, seed + 7)),
                ssm_conv1d: allocs.upload_f32(&det_signal(CONV_KERNEL * conv_channels, seed + 8)),
                ssm_norm_w: allocs.upload_f16(&vec![1.0_f32; HEAD_V_DIM]),
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
            max_seq_len: 1,
            num_layers: NUM_LAYERS,
            max_experts: 0,
            gdn: Some(dims),
            per_layer_kv_widths: None,
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");

        // Zero per-layer GDN state + conv history (residual stream
        // expects the recurrent state to start at 0).
        for ls in &pool.gdn_state {
            let state_n = NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;
            let conv_n = (CONV_KERNEL - 1) * conv_channels;
            let state_zero = vec![0.0_f32; state_n];
            let conv_zero = vec![0.0_f32; conv_n];
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ls.state,
                        DevicePtr(state_zero.as_ptr() as usize),
                        state_n * 4,
                    )
                    .expect("zero state");
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ls.conv_history,
                        DevicePtr(conv_zero.as_ptr() as usize),
                        conv_n * 4,
                    )
                    .expect("zero conv_history");
            }
        }
        stream.synchronize().expect("sync zero-init");

        let layout = ModelLayout {
            num_layers: NUM_LAYERS,
            hidden: HIDDEN,
            kv_max_seq_len: 1,
        };

        {
            let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
            let mut resid = ctx.embed(&embd, 7).expect("embed");
            let layers: Vec<usize> = ctx.layer_range(&layout).collect();
            for li in layers {
                let normed = ctx
                    .rmsnorm(&resid, &gdn_weights[li].attn_norm, RMS_EPS)
                    .expect("attn_norm");
                let delta = ctx
                    .gdn_layer(&normed, &gdn_weights[li], li)
                    .expect("gdn_layer");
                resid = ctx.residual_add(resid, delta).expect("residual_add");
            }
            ctx.output_head(&resid, &lm_head).expect("output_head");
            let logits = ctx.logits();
            assert_eq!(logits.len(), VOCAB);
            for (i, &l) in logits.iter().enumerate() {
                assert!(l.is_finite(), "logits[{i}] = {l} not finite");
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(max - min > 1e-3, "GDN logits collapsed");
        }

        pool.dispose(&device).expect("pool dispose");
    }

    #[test]
    fn single_device_synthetic_moe_one_token_forward() {
        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        const INTERMEDIATE: usize = 256;
        const NUM_LAYERS: usize = 2;
        const N_HEADS: usize = 4;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const MAX_SEQ_LEN: usize = 16;
        const N_EXPERTS: usize = 4;
        const EXPERTS_PER_TOK: usize = 2;
        const RMS_EPS: f32 = 1e-5;

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
        let lm_head_weights = LmHeadWeights {
            output_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
            lm_head: allocs.upload_q8_0(&det_signal(VOCAB * HIDDEN, 9), VOCAB, HIDDEN),
            final_logit_softcap: None,
            vocab_size: VOCAB,
            hidden: HIDDEN,
            rms_eps: RMS_EPS,
        };

        let mut attn_weights: Vec<AttnWeights> = Vec::with_capacity(NUM_LAYERS);
        let mut moe_weights: Vec<MoeWeights> = Vec::with_capacity(NUM_LAYERS);
        for li in 0..NUM_LAYERS {
            let seed = 100 + (li as u32) * 100;
            attn_weights.push(AttnWeights {
                attn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
                attn_q: allocs.upload_q8_0(&det_signal(q_width * HIDDEN, seed + 1), q_width, HIDDEN),
                attn_k: allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 2), kv_width, HIDDEN),
                attn_v: Some(allocs.upload_q8_0(&det_signal(kv_width * HIDDEN, seed + 3), kv_width, HIDDEN)),
                attn_output: allocs.upload_q8_0(&det_signal(HIDDEN * q_width, seed + 4), HIDDEN, q_width),
                attn_q_norm: None,
                attn_k_norm: None,
                n_heads: N_HEADS,
                n_kv_heads: N_KV_HEADS,
                head_dim: HEAD_DIM,
                rotated_dims: HEAD_DIM,
                rope_theta: 10000.0,
                window_size: 0,
                rms_eps: RMS_EPS,
                softmax_scale: None,
            });

            let mut experts_gate = Vec::with_capacity(N_EXPERTS);
            let mut experts_up = Vec::with_capacity(N_EXPERTS);
            let mut experts_down = Vec::with_capacity(N_EXPERTS);
            for e in 0..N_EXPERTS as u32 {
                let s = seed + 100 + e * 7;
                experts_gate.push(allocs.upload_q8_0(
                    &det_signal(INTERMEDIATE * HIDDEN, s + 1),
                    INTERMEDIATE,
                    HIDDEN,
                ));
                experts_up.push(allocs.upload_q8_0(
                    &det_signal(INTERMEDIATE * HIDDEN, s + 2),
                    INTERMEDIATE,
                    HIDDEN,
                ));
                experts_down.push(allocs.upload_q8_0(
                    &det_signal(HIDDEN * INTERMEDIATE, s + 3),
                    HIDDEN,
                    INTERMEDIATE,
                ));
            }
            moe_weights.push(MoeWeights {
                ffn_norm: allocs.upload_f16(&vec![1.0_f32; HIDDEN]),
                router: allocs.upload_q8_0(&det_signal(N_EXPERTS * HIDDEN, seed + 50), N_EXPERTS, HIDDEN),
                experts_gate,
                experts_up,
                experts_down,
                n_experts: N_EXPERTS,
                experts_per_tok: EXPERTS_PER_TOK,
                activation: Activation::SwiGLU,
                rms_eps: RMS_EPS,
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
            max_experts: N_EXPERTS,
            gdn: None,
            per_layer_kv_widths: None,
        };
        let mut pool = ScratchPool::new(&device, cfg).expect("ScratchPool::new");
        let layout = ModelLayout {
            num_layers: NUM_LAYERS,
            hidden: HIDDEN,
            kv_max_seq_len: MAX_SEQ_LEN,
        };

        {
            let mut ctx = SingleDeviceForwardCtx::new(&device, stream, &reg, &mut pool);
            let token_id: u32 = 7;
            let position: usize = 0;

            let mut resid = ctx.embed(&embd, token_id).expect("embed");
            let layers: Vec<usize> = ctx.layer_range(&layout).collect();
            for li in layers {
                let normed = ctx
                    .rmsnorm(&resid, &attn_weights[li].attn_norm, RMS_EPS)
                    .expect("attn rmsnorm");
                let delta = ctx
                    .standard_attn(&normed, &attn_weights[li], li, position)
                    .expect("standard_attn");
                resid = ctx.residual_add(resid, delta).expect("attn residual_add");

                let normed = ctx
                    .rmsnorm(&resid, &moe_weights[li].ffn_norm, RMS_EPS)
                    .expect("moe rmsnorm");
                let delta = ctx.moe_ffn(&normed, &moe_weights[li]).expect("moe_ffn");
                resid = ctx.residual_add(resid, delta).expect("moe residual_add");
            }
            ctx.output_head(&resid, &lm_head_weights).expect("output_head");
            let logits = ctx.logits();
            assert_eq!(logits.len(), VOCAB);
            for (i, &l) in logits.iter().enumerate() {
                assert!(l.is_finite(), "logits[{i}] = {l} not finite");
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(max - min > 1e-3, "MoE logits collapsed to a constant");
        }

        pool.dispose(&device).expect("pool dispose");
    }
}
