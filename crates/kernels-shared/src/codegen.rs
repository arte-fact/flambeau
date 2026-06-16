//! Build-time codegen for the kernel instantiation manifest.
//!
//! Shared by `kernels-hip/build.rs` and `kernels-cuda/build.rs` (as a
//! build-dependency) so wrapper generation has one source of truth. Lives next
//! to the `instantiations.toml` it parses.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    kernel: Vec<KernelCell>,
}

#[derive(Deserialize)]
struct KernelCell {
    stem: String,
    includes: Vec<String>,
    entry: Vec<EntrySpec>,
}

#[derive(Deserialize)]
struct EntrySpec {
    name: String,
    params: String,
    body: String,
}

/// Generate one wrapper `.cu` per manifest cell into `gen_dir`.
///
/// Each cell emits one or more `extern "C" __global__` entries sharing the
/// cell's includes and `.cu` (so a single backend module — e.g. `mmvq_q4_0` —
/// can carry both its F32 and F16-dst entry points).
///
/// `arch_header` is the calling backend's arch-primitives header
/// (`gfx906.cuh` for HIP, `sm_80.cuh` for CUDA); it is injected ahead of each
/// cell's shared includes so the templated body finds the neutral primitives.
/// Returns the generated paths for the build script's compile set, or an empty
/// vec if the manifest is absent. A wrapper is rewritten only when its content
/// changes, so unchanged cells don't churn the build cache.
///
/// # Panics
///
/// Panics (failing the build) if the manifest exists but is malformed, or if a
/// generated file cannot be written.
pub fn generate_instantiations(manifest: &Path, gen_dir: &Path, arch_header: &str) -> Vec<PathBuf> {
    let Ok(text) = fs::read_to_string(manifest) else {
        return Vec::new();
    };
    let parsed: Manifest = toml::from_str(&text).expect("parse instantiations.toml");
    fs::create_dir_all(gen_dir).expect("create generated kernel dir");

    let mut out = Vec::with_capacity(parsed.kernel.len());
    for cell in &parsed.kernel {
        let mut src = String::new();
        src.push_str("// Auto-generated from kernels-shared/instantiations.toml.\n");
        src.push_str("// Do not edit; change the manifest cell instead.\n");
        writeln!(src, "#include \"{arch_header}\"").expect("write to String is infallible");
        for inc in &cell.includes {
            writeln!(src, "#include \"{inc}\"").expect("write to String is infallible");
        }
        for ent in &cell.entry {
            writeln!(src, "\nextern \"C\" __global__ void {}(", ent.name)
                .expect("write to String is infallible");
            writeln!(src, "{}\n) {{", ent.params).expect("write to String is infallible");
            writeln!(src, "    {}", ent.body).expect("write to String is infallible");
            src.push_str("}\n");
        }

        let path = gen_dir.join(format!("{}.cu", cell.stem));
        let changed = fs::read_to_string(&path).map(|existing| existing != src).unwrap_or(true);
        if changed {
            fs::write(&path, &src).expect("write generated .cu");
        }
        out.push(path);
    }
    out
}
