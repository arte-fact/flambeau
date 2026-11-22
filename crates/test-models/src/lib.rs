//! Locates the GGUF files integration tests load.
//!
//! Models live in `$FLAMBEAU_MODELS_DIR` when it is set, else in `~/models`
//! when that directory exists, else in `/artefact/models`. A test whose
//! model is missing prints `SKIP <test> : <path> absent` and returns; the
//! line is written to stderr directly, past libtest's output capture, so a
//! skipped test never passes silently.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Directory the test GGUFs are looked up in.
pub fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FLAMBEAU_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let dir = Path::new(&home).join("models");
        if dir.is_dir() {
            return dir;
        }
    }
    PathBuf::from("/artefact/models")
}

/// `file` in [`models_dir`], or `None` after reporting the skip.
pub fn model(test: &str, file: &str) -> Option<PathBuf> {
    present(test, models_dir().join(file))
}

/// The first of `files` present in [`models_dir`], or `None` after
/// reporting the skip.
pub fn first_model(test: &str, files: &[&str]) -> Option<PathBuf> {
    let dir = models_dir();
    let found = files.iter().map(|f| dir.join(f)).find(|p| p.exists());
    if found.is_none() {
        skip(test, &dir.join(format!("{{{}}}", files.join(","))));
    }
    found
}

/// `$var` when set, else `file` in [`models_dir`]; `None` after reporting
/// the skip when the resolved path is missing.
pub fn model_from_env(test: &str, var: &str, file: &str) -> Option<PathBuf> {
    match std::env::var_os(var) {
        Some(p) => present(test, PathBuf::from(p)),
        None => model(test, file),
    }
}

fn present(test: &str, path: PathBuf) -> Option<PathBuf> {
    if path.exists() {
        Some(path)
    } else {
        skip(test, &path);
        None
    }
}

fn skip(test: &str, path: &Path) {
    let _ = writeln!(std::io::stderr(), "SKIP {test} : {} absent", path.display());
}
