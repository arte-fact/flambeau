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

    // Dense Q4_K decode dispatches the dp4a MMVQ variant (committed in
    // dispatch/cuda/sm_86.toml as `qmatmul_q4_k_mmvq_dp4a`); attention / norm /
    // activation are the rest of the decode spine. All must be loaded.
    for stem in [
        "mmvq_q4_k_r2_dp4a",
        "attention_decode_f16",
        "rmsnorm_f16",
        "swiglu_f32_to_f16",
    ] {
        assert!(
            reg.module(stem).is_some(),
            "decode-path stem `{stem}` not loaded (expected in the sm_86 catalogue)"
        );
    }
}
