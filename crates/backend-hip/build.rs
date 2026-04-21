use std::env;

fn main() {
    // Prefer ROCm 7.1.1 (candle-validated on gfx906: matched hipcc +
    // libamdhip64 + librccl + rocprofv3). `.env` sets ROCM_PATH=/opt/rocm-7.1.1.
    // Fall back to /opt/rocm (7.2.1 in this sandbox) if 7.1.1 isn't installed.
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    // On 7.1.1 the real libamdhip64 + librccl + libroctx + rocprofv3 plugins
    // all live under `$ROCM_PATH/core-7.13/lib`; the top-level `$ROCM_PATH/lib`
    // holds stubs that are missing RCCL symbols (ncclRecv, ncclGetErrorString,
    // ...). We put core-7.13/lib FIRST in the search path so the real `.so`
    // wins both at link time and at runtime via rpath.
    let core_lib = format!("{rocm_path}/core-7.13/lib");
    if std::path::Path::new(&core_lib).exists() {
        println!("cargo:rustc-link-search=native={core_lib}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{core_lib}");
    }
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm_path}/lib");

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // RCCL linkage is opt-in via the `rccl` feature so builds on non-ROCm
    // hosts (CI lint, developer laptop without ROCm lib files) succeed.
    //
    // On 7.1.1 the real librccl is at $ROCM_PATH/core-7.13/lib, not
    // $ROCM_PATH/lib (the top-level .so is a stub missing symbols).
    // Link order matters: the core-7.13 search path must come FIRST.
    if env::var("CARGO_FEATURE_RCCL").is_ok() {
        println!("cargo:rustc-link-lib=dylib=rccl");
    }
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}
