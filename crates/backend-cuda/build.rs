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

    // NCCL (multi-GPU Collectives seam) — only under the `nccl` feature. The
    // wheel ships `libnccl.so.2` with no `.so` dev symlink, so `-lnccl` can't
    // find it directly: symlink `libnccl.so` into OUT_DIR and point both the
    // link search and the runtime rpath at the real lib dir.
    if env::var("CARGO_FEATURE_NCCL").is_ok() {
        link_nccl();
    }
}

fn link_nccl() {
    use std::path::PathBuf;
    println!("cargo:rerun-if-env-changed=NCCL_LIB_DIR");
    let lib_dir = env::var("NCCL_LIB_DIR")
        .ok()
        .or_else(find_nccl_lib_dir)
        .expect("NCCL requested (feature `nccl`) but libnccl.so.2 not found; set NCCL_LIB_DIR");
    let real = Path::new(&lib_dir).join("libnccl.so.2");
    assert!(real.exists(), "NCCL_LIB_DIR={lib_dir} has no libnccl.so.2");

    let out_dir = env::var("OUT_DIR").unwrap();
    let link = PathBuf::from(&out_dir).join("libnccl.so");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&real, &link).expect("symlink libnccl.so");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=dylib=nccl");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
}

fn find_nccl_lib_dir() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.local/lib/python3.14/site-packages/nvidia/nccl/lib"),
        format!("{home}/.local/lib/python3.12/site-packages/nvidia/nccl/lib"),
        "/usr/lib/x86_64-linux-gnu".to_string(),
    ];
    candidates
        .into_iter()
        .find(|d| Path::new(d).join("libnccl.so.2").exists())
}
