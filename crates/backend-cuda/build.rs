use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Link the CUDA driver library `libcuda.so.1` (ships with the NVIDIA
    // driver, not the toolkit). The stubs dir lets a build host link with no
    // driver present; the real `.so.1` resolves at runtime.
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
    for dir in [
        format!("{cuda_path}/lib64"),
        format!("{cuda_path}/lib64/stubs"),
        "/usr/lib/x86_64-linux-gnu".to_string(),
    ] {
        if Path::new(&dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}
