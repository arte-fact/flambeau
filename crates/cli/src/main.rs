//! `flambeau` CLI — subcommand dispatch.
//!
//! V1.0 stub: subcommands parse but print a "not yet implemented" message until
//! their target step lands. See `doc/ROADMAP-V1-QWEN36-GFX906.md` for what each
//! subcommand requires.

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "flambeau", version, about = "Max-perf inference server for modern LLMs on HIP + CUDA.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump GGUF tensor list, dtype audit, metadata (V1.1).
    InspectGguf {
        path: String,
    },
    /// Dump HIP `.hsaco` kernel symbols + VGPR budgets (V1.3+).
    InspectHsaco {
        path: String,
    },
    /// Single-prompt inference; prints completion to stdout (V1.7).
    Infer {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Comma-separated device list, e.g. `hip:0,1,2,3`.
        #[arg(long, default_value = "hip:0")]
        devices: String,
    },
    /// OpenAI-compatible HTTP server (V1.8).
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "hip:0")]
        devices: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// T-track warmup-tuner (V1.x side-track).
    Tune {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "hip:0")]
        devices: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// M-track MCP server (V1.x side-track). Dev-only.
    Mcp {
        #[arg(long, default_value_t = 9090)]
        port: u16,
    },
    /// Correctness-sweep harness; emits certs (V1.3+).
    Sweep {
        #[arg(long)]
        arch: String,
        #[arg(long)]
        op: Option<String>,
    },
    /// Perf-regression matrix (V1.4+).
    Matrix {
        #[arg(long)]
        models: Vec<String>,
        #[arg(long, default_value_t = 512)]
        prompt_len: usize,
        #[arg(long, default_value_t = 64)]
        tg_len: usize,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::InspectGguf { path } => todo!("V1.1: implement inspect-gguf for {path}"),
        Cmd::InspectHsaco { path } => todo!("V1.3+: implement inspect-hsaco for {path}"),
        Cmd::Infer { model, .. } => todo!("V1.7: implement infer for {model}"),
        Cmd::Serve { model, .. } => todo!("V1.8: implement serve for {model}"),
        Cmd::Tune { model, .. } => todo!("T-track: implement tune for {model}"),
        Cmd::Mcp { port } => todo!("M-track: implement mcp on :{port}"),
        Cmd::Sweep { arch, .. } => todo!("V1.3+: implement sweep for {arch}"),
        Cmd::Matrix { .. } => todo!("V1.4+: implement matrix"),
    }
}
