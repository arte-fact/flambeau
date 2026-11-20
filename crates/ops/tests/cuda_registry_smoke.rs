#![cfg(feature = "cuda")]
//! Keystone for the CUDA runtime path: `Registry<CudaKernels>::new` constructs
//! on a real sm_86 device by loading the stems this backend's `.cubin`
//! catalogue actually contains, skipping the unported legacy / out-of-scope
//! ones (instead of failing fast on the first absent stem). Asserts the
//! decode-path stems qwen35moe-v2 Q4_K dispatches are present.

use flambeau_ops::cuda::CudaDevice;
use flambeau_ops::CudaOpsRegistry;

#[test]
fn cuda_registry_loads_present_skips_unported() {
    let dev = CudaDevice::new(0).expect("CUDA device 0 (RTX 3090)");
    let reg = CudaOpsRegistry::new(&dev).expect("registry constructs under load-present");

    eprintln!(
        "CUDA registry skipped {} of {} stems: {:?}",
        reg.skipped().len(),
        flambeau_ops::KERNEL_STEMS.len(),
        reg.skipped(),
    );

    // Attention / norm / activation decode spine.
    for stem in ["attention_decode_f16", "rmsnorm_f16", "swiglu_f32_to_f16"] {
        assert!(
            reg.module(stem).is_some(),
            "decode-path stem `{stem}` not loaded (expected in the sm_86 catalogue)"
        );
    }

    // Every in-scope quant dtype's committed sm_86 decode dispatch impl_id must
    // resolve through the launch recipe to a stem present in the catalogue —
    // otherwise a real model of that quant dies at the first qmatmul with
    // "no launch recipe registered" (Q8_0) or "stem not loaded" (Q4_0/Q5_x).
    let decode_impls = [
        "qmatmul_q2_k_mmvq_dp4a",
        "qmatmul_q3_k_mmvq_dp4a_t128_vdr2",
        "qmatmul_q4_k_mmvq_dp4a_t128_vdr2",
        "qmatmul_q5_k_mmvq_dp4a_t128_vdr2",
        "qmatmul_q6_k_mmvq_dp4a",
        "qmatmul_q8_k_mmvq_single_row",
        "qmatmul_q8_0_mmvq_dp4a",
        "qmatmul_q4_0_mmvq_dp4a",
        "qmatmul_q4_1_mmvq_dp4a",
        "qmatmul_q5_0_mmvq_dp4a",
        "qmatmul_q5_1_mmvq_dp4a",
        // IQ i-quants (decode dispatch → the dp4a variant, which is built).
        "qmatmul_iq1_s_mmvq_dp4a",
        "qmatmul_iq1_m_mmvq_dp4a",
        "qmatmul_iq2_xxs_mmvq_dp4a",
        "qmatmul_iq2_xs_mmvq_dp4a",
        "qmatmul_iq2_s_mmvq_dp4a",
        "qmatmul_iq3_xxs_mmvq_dp4a",
        "qmatmul_iq3_s_mmvq_dp4a",
        "qmatmul_iq4_nl_mmvq_dp4a_t128_vdr2",
        "qmatmul_iq4_xs_mmvq_dp4a_t128_vdr4",
    ];
    for impl_id in decode_impls {
        let stem = flambeau_ops::qmatmul::recipe_stem_for(impl_id)
            .unwrap_or_else(|e| panic!("impl_id `{impl_id}` has no launch recipe: {e}"));
        assert!(
            reg.module(stem).is_some(),
            "dispatch impl_id `{impl_id}` resolves to stem `{stem}`, not loaded on sm_86"
        );
    }
}
