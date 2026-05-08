//! flambeau-mcp-server — M-track, dev-only.
//! Exposes flambeau's internal dev-surface (sweep, cert-check, pmc-probe,
//! inspect-gguf, dispatch reads, cert-diff) as MCP tools over the
//! official Rust SDK (`rmcp`). Every tool's return type is a JSON
//! artefact that is itself committable — cert rows, PMC snapshots,
//! dispatch-row TOML fragments.
//! **Never load-bearing for production.** `flambeau serve` in prod
//! must not depend on this crate — enforced by the workspace
//! dependency graph (server crate has no path = "../mcp-server" link).
//! Transports today:
//! - stdio (T:1.1 / M1.1) — Claude Code / MCP CLI default.
//! - HTTP (M1.5 follow-up).
//! See `doc/ROADMAP-V2-TOOL-CALLING-AND-MCP.md` §M1 for the tool
//! catalogue and CLAUDE.md §M-track for the "every finding
//! round-trips into a committable artefact" contract.

use std::future::Future;

use anyhow::{Context, Result};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt};

/// Shorthand: every tool returns this. We construct `CallToolResult`
/// by hand so the response carries BOTH:
/// - `structured_content`: the parsed JSON object — what an MCP-spec-
/// compliant client (Claude Code, MCP Inspector, agents) renders.
/// - `content[0]`: a *pretty-printed* JSON dump as text fallback —
/// what older clients see. `CallToolResult::structured()` uses
/// `to_string()` (compact) for the fallback which renders as a
/// wall of escaped JSON; pretty-printing makes it readable too.
/// - `is_error`: false for success, true for tool-level errors.
type ToolReturn = CallToolResult;

/// The M-track MCP server. Each `#[tool]` method corresponds to one
/// flambeau dev-surface operation. M1.1 ships the scaffold + a single
/// `flambeau_ping` smoke tool so the stdio transport can be exercised
/// end-to-end without depending on any of the real sweep/inspect
/// machinery yet. M1.2 onwards swap the real tools in.
#[derive(Debug, Clone)]
pub struct FlambeauMcp {
    // Used by `#[tool_router]`'s generated impl (proc-macro expansion
    // isn't traced by rustc's dead-code analysis).
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl FlambeauMcp {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

impl Default for FlambeauMcp {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Tool parameter schemas -----------------------------------------------

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PingRequest {
    #[schemars(description = "Echoed back in the response, for round-trip testing.")]
    pub note: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct InspectGgufRequest {
    #[schemars(description = "Absolute path to a GGUF file.")]
    pub path: String,
    #[schemars(description = "Max metadata entries to include in the return. Default 64.")]
    pub max_metadata: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CertCheckRequest {
    #[schemars(description = "Backend name, typically \"hip\".")]
    pub backend: Option<String>,
    #[schemars(description = "Architecture tag, e.g. \"gfx906\".")]
    pub arch: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DispatchReadRequest {
    #[schemars(description = "Backend name, typically \"hip\".")]
    pub backend: Option<String>,
    #[schemars(description = "Architecture tag, e.g. \"gfx906\".")]
    pub arch: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct TuneDryRequest {
    #[schemars(description = "Model id or GGUF path.")]
    pub model: String,
    #[schemars(description = "Device list, e.g. \"hip:0,1\".")]
    pub devices: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MatrixRequest {
    #[schemars(description = "Model ids to run. Empty = all configured models.")]
    pub models: Option<Vec<String>>,
    #[schemars(description = "Prompt length for prefill. Default 512.")]
    pub prompt_len: Option<u32>,
    #[schemars(description = "Token-generation length. Default 64.")]
    pub tg_len: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DispatchAbRequest {
    #[schemars(description = "Impl ID for the baseline.")]
    pub impl_a: String,
    #[schemars(description = "Impl ID for the candidate.")]
    pub impl_b: String,
    #[schemars(description = "Model id or GGUF path to exercise.")]
    pub model: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CertDiffRequest {
    #[schemars(description = "Repo-relative cert path, e.g. `certs/hip/gfx906/mmvq_q8_0_gfx906.json`.")]
    pub cert_path: String,
    #[schemars(description = "Git ref for the LHS of the diff. Default: `HEAD~1`.")]
    pub commit_a: Option<String>,
    #[schemars(description = "Git ref for the RHS. Default: `HEAD`.")]
    pub commit_b: Option<String>,
}

#[tool_router]
impl FlambeauMcp {
    /// M1.1 smoke: round-trip test tool. Not one of the M1.2 wrappers;
    /// exists solely so `mcp-cli ls-tools` shows something and
    /// `mcp-cli call flambeau_ping` returns a predictable artefact
    /// without any GPU / sweep / kernel machinery needing to be live.
    #[tool(description = "Round-trip smoke test. Echoes `note` and returns build info.")]
    fn flambeau_ping(
        &self,
        Parameters(PingRequest { note }): Parameters<PingRequest>,
    ) -> ToolReturn {
        let artefact = serde_json::json!({
            "flambeau_version": env!("CARGO_PKG_VERSION"),
            "note": note.unwrap_or_default(),
            "message": "flambeau-mcp-server is up",
        });
        // Ping doesn't produce a committable artefact, but we still
        // ship the M3.1 envelope for schema consistency across every
        // real tool. canonical_path is the crate's Cargo.toml as a
        // neutral marker; commit_msg is diagnostic.
        wrap_artefact(
            artefact,
            concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
            "flambeau_ping smoke",
        )
    }

    /// M1.2: dump a GGUF's metadata + tensor list. Wraps the same path
    /// as `flambeau inspect-gguf` but returns a structured JSON artefact
    /// instead of human-readable output.
    #[tool(description = "Inspect a GGUF file — metadata, tensor list, dtype audit.")]
    fn flambeau_inspect_gguf(
        &self,
        Parameters(InspectGgufRequest { path, max_metadata }):
            Parameters<InspectGgufRequest>,
    ) -> ToolReturn {
        let max = max_metadata.unwrap_or(64);
        tool_inspect_gguf(&path, max)
            .unwrap_or_else(|e| error_payload("flambeau_inspect_gguf", e))
    }

    /// M1.2: validate every dispatch row has a matching green cert.
    /// Wraps `flambeau cert-check`. Returns `{rows_checked, failures[]}`
    /// as JSON — suitable for committing directly into a PR body.
    #[tool(description = "Validate that every dispatch row has a matching green cert.")]
    fn flambeau_cert_check(
        &self,
        Parameters(CertCheckRequest { backend, arch }):
            Parameters<CertCheckRequest>,
    ) -> ToolReturn {
        let backend = backend.as_deref().unwrap_or("hip");
        let arch = arch.as_deref().unwrap_or("gfx906");
        tool_cert_check(backend, arch)
            .unwrap_or_else(|e| error_payload("flambeau_cert_check", e))
    }

    /// M1.2: read the raw dispatch TOML and return its parsed contents.
    /// Stable shape: `{backend, arch, path, rows: [{...toml row...}]}`.
    /// Lets an agent introspect live dispatch without shelling to a
    /// text editor.
    #[tool(description = "Read the dispatch TOML and return the parsed rows.")]
    fn flambeau_dispatch_read(
        &self,
        Parameters(DispatchReadRequest { backend, arch }):
            Parameters<DispatchReadRequest>,
    ) -> ToolReturn {
        let backend = backend.as_deref().unwrap_or("hip");
        let arch = arch.as_deref().unwrap_or("gfx906");
        tool_dispatch_read(backend, arch)
            .unwrap_or_else(|e| error_payload("flambeau_dispatch_read", e))
    }

    /// M1.4: intentional stub — T-track warmup-tuner isn't shipped
    /// yet. Surface the tool so downstream agents can code against the
    /// schema, and return a structured `{error: "not_shipped", ...}`
    /// payload today. Flipping this to a real impl is the T-track
    /// landing PR (separate from this roadmap).
    #[tool(description = "T-track warmup-tuner dry-run. NOT SHIPPED — returns a not_shipped error today.")]
    fn flambeau_tune_dry(
        &self,
        Parameters(TuneDryRequest { model, devices }): Parameters<TuneDryRequest>,
    ) -> ToolReturn {
        err_result(serde_json::json!({
            "error": "not_shipped",
            "tool": "flambeau_tune_dry",
            "tracking": "ROADMAP-V2 T-track / CLAUDE.md §Side-tracks T1–T5",
            "reason": "T-track autotuner crate is still a stub (crates/autotune is 8 lines). \
                       Implementation landing is out of scope for the V2 MCP track.",
            "received": { "model": model, "devices": devices },
        }))
    }

    /// M1.4 stub: perf-regression matrix. Not shipped — the `flambeau
    /// matrix` CLI itself is a `todo!()` today. Same rationale as
    /// `flambeau_tune_dry`.
    #[tool(description = "Perf-regression matrix (pp/tg sweep across models). NOT SHIPPED — returns a not_shipped error today.")]
    fn flambeau_matrix(
        &self,
        Parameters(MatrixRequest { models, prompt_len, tg_len }):
            Parameters<MatrixRequest>,
    ) -> ToolReturn {
        err_result(serde_json::json!({
            "error": "not_shipped",
            "tool": "flambeau_matrix",
            "tracking": "ROADMAP-V2 §M1.4 → CLI matrix subcommand at crates/cli/src/main.rs todo!()",
            "reason": "`flambeau matrix` CLI subcommand is a todo!() placeholder. \
                       Landing a real bench runner is out of scope for V2.",
            "received": {
                "models": models,
                "prompt_len": prompt_len,
                "tg_len": tg_len,
            },
        }))
    }

    /// M1.4 stub: dispatch A/B compare. Not shipped.
    #[tool(description = "A/B-compare two dispatch impls on a model. NOT SHIPPED — returns a not_shipped error today.")]
    fn flambeau_dispatch_ab(
        &self,
        Parameters(DispatchAbRequest { impl_a, impl_b, model }):
            Parameters<DispatchAbRequest>,
    ) -> ToolReturn {
        err_result(serde_json::json!({
            "error": "not_shipped",
            "tool": "flambeau_dispatch_ab",
            "tracking": "ROADMAP-V2 §M1.b (scoped as follow-up to M1)",
            "reason": "`bench ab` mode isn't implemented in flambeau-bench yet. \
                       Landing is deferred behind M1.5 HTTP transport + M2 client.",
            "received": { "impl_a": impl_a, "impl_b": impl_b, "model": model },
        }))
    }

    /// M1.3: diff a cert JSON at two git commits. Loads each blob via
    /// `git show <commit>:<path>`, parses both, and returns a
    /// structured summary: `{pass_flip, results_added, results_removed,
    /// results_changed, pmc_delta, notes}`.
    /// Intended flow: a driver agent sweeps a new kernel, commits the
    /// cert, then calls this tool with `commit_a=HEAD~1, commit_b=HEAD`
    /// to auto-generate PR-body content — the JSON artefact
    /// round-trips into a commit message / PR description.
    #[tool(description = "Diff a cert JSON between two git commits. Used to build PR-body content from a sweep.")]
    fn flambeau_cert_diff(
        &self,
        Parameters(CertDiffRequest { cert_path, commit_a, commit_b }):
            Parameters<CertDiffRequest>,
    ) -> ToolReturn {
        let a = commit_a.as_deref().unwrap_or("HEAD~1");
        let b = commit_b.as_deref().unwrap_or("HEAD");
        tool_cert_diff(&cert_path, a, b)
            .unwrap_or_else(|e| error_payload("flambeau_cert_diff", e))
    }
}

// ---- Tool implementations --------------------------------------------------

/// Walks the GGUF at `path` and packages metadata + tensors into a
/// JSON artefact. Truncates string metadata beyond 4096 bytes to keep
/// the response size bounded (Qwen3.6's chat template alone is ~8 KB
/// and would otherwise dominate the payload).
fn tool_inspect_gguf(path: &str, max_metadata: usize) -> Result<ToolReturn> {
    use flambeau_quant::gguf::{GgufFile, Value};
    let file = GgufFile::open(path).context("open GGUF")?;
    let mut entries: Vec<(&String, &Value)> = file.metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let truncated = entries.len().saturating_sub(max_metadata);
    let metadata_json: serde_json::Value = serde_json::Value::Object(
        entries
            .into_iter()
            .take(max_metadata)
            .map(|(k, v)| (k.clone(), value_to_json(v)))
            .collect(),
    );
    let tensors_json: Vec<serde_json::Value> = file
        .tensor_order
        .iter()
        .map(|name| {
            let info = &file.tensors[name];
            serde_json::json!({
                "name": name,
                "dtype": info.dtype.name(),
                "dims": info.dims,
                "size_bytes": info.size_in_bytes(),
            })
        })
        .collect();
    let artefact = serde_json::json!({
        "path": file.path.display().to_string(),
        "gguf_version": format!("{:?}", file.version),
        "metadata": metadata_json,
        "metadata_truncated": truncated,
        "tensor_count": file.tensors.len(),
        "tensors": tensors_json,
    });
    // M3.1: inspect-gguf is a read tool — there's nothing to write.
    // The canonical_path points back at the inspected file so an
    // agent can quote it in a report; the commit message is
    // diagnostic-only ("inspected" rather than "updated").
    Ok(wrap_artefact(
        artefact,
        file.path.display().to_string(),
        format!("inspect-gguf: {}", file.path.display()),
    ))
}

/// Convert GGUF metadata values into JSON. Mirrors the `inspect-gguf`
/// CLI's display logic but preserves structure (arrays stay arrays,
/// numeric types stay numeric) instead of pretty-printing.
fn value_to_json(v: &flambeau_quant::gguf::Value) -> serde_json::Value {
    use flambeau_quant::gguf::Value;
    match v {
        Value::U8(x) => serde_json::json!(*x),
        Value::I8(x) => serde_json::json!(*x),
        Value::U16(x) => serde_json::json!(*x),
        Value::I16(x) => serde_json::json!(*x),
        Value::U32(x) => serde_json::json!(*x),
        Value::I32(x) => serde_json::json!(*x),
        Value::U64(x) => serde_json::json!(*x),
        Value::I64(x) => serde_json::json!(*x),
        Value::F32(x) => serde_json::json!(*x),
        Value::F64(x) => serde_json::json!(*x),
        Value::Bool(x) => serde_json::json!(*x),
        Value::String(s) if s.len() <= 4096 => serde_json::json!(s),
        Value::String(s) => serde_json::json!({
            "truncated_string": &s[..4096.min(s.len())],
            "bytes": s.len(),
        }),
        Value::Array(a) if a.len() <= 64 => {
            serde_json::Value::Array(a.iter().map(value_to_json).collect())
        }
        Value::Array(a) => serde_json::json!({
            "truncated_array_head": a.iter().take(16).map(value_to_json).collect::<Vec<_>>(),
            "total_elems": a.len(),
        }),
    }
}

/// Wraps `flambeau_bench::dispatch::cert_check`.
fn tool_cert_check(backend: &str, arch: &str) -> Result<ToolReturn> {
    let repo_root = std::env::current_dir().context("cwd for cert-check")?;
    let dispatch_path = repo_root.join(format!("dispatch/{backend}/{arch}.toml"));
    let report = flambeau_bench::dispatch::cert_check(&repo_root, &dispatch_path)
        .context("cert_check")?;
    let failures: Vec<serde_json::Value> = report
        .failures
        .iter()
        .map(|(impl_id, err)| {
            serde_json::json!({
                "impl_id": impl_id,
                "error": err.to_string(),
            })
        })
        .collect();
    let artefact = serde_json::json!({
        "backend": backend,
        "arch": arch,
        "dispatch_path": dispatch_path.display().to_string(),
        "rows_checked": report.rows_checked,
        "failures": failures,
        "ok": report.ok(),
    });
    // M3.1: cert-check is also a read tool, but its result is PR-body
    // material when `ok = false` (names the failing rows). Commit
    // message spells out the pass/fail state so an agent summarising
    // into a PR has a one-liner to paste.
    let msg = if report.ok() {
        format!("cert-check {backend}/{arch}: {} rows, all green", report.rows_checked)
    } else {
        format!(
            "cert-check {backend}/{arch}: {} rows, {} failure(s)",
            report.rows_checked,
            report.failures.len()
        )
    };
    Ok(wrap_artefact(
        artefact,
        dispatch_path.display().to_string(),
        msg,
    ))
}

/// Raw read + TOML parse of the dispatch file.
fn tool_dispatch_read(backend: &str, arch: &str) -> Result<ToolReturn> {
    let repo_root = std::env::current_dir().context("cwd for dispatch-read")?;
    let dispatch_path = repo_root.join(format!("dispatch/{backend}/{arch}.toml"));
    let toml_src = std::fs::read_to_string(&dispatch_path)
        .with_context(|| format!("read {}", dispatch_path.display()))?;
    let parsed: toml::Value = toml::from_str(&toml_src).context("parse dispatch TOML")?;
    // Re-serialise via serde_json so the MCP response is pure JSON.
    let artefact = serde_json::json!({
        "backend": backend,
        "arch": arch,
        "path": dispatch_path.display().to_string(),
        "raw": serde_json::to_value(&parsed).context("toml → json")?,
    });
    Ok(wrap_artefact(
        artefact,
        dispatch_path.display().to_string(),
        format!("dispatch-read {backend}/{arch}"),
    ))
}

/// M1.3: fetch a blob from git at a specific ref and parse as JSON.
fn git_show_json(commit: &str, path: &str) -> Result<serde_json::Value> {
    let out = std::process::Command::new("git")
        .arg("show")
        .arg(format!("{commit}:{path}"))
        .output()
        .with_context(|| format!("git show {commit}:{path}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "git show {commit}:{path} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let blob = std::str::from_utf8(&out.stdout)
        .with_context(|| format!("{commit}:{path} not utf-8"))?;
    serde_json::from_str(blob).with_context(|| format!("parse {commit}:{path} as JSON"))
}

/// Diff two cert JSONs into an agent-friendly summary. Focuses on what
/// a reviewer would want in a PR body: whether `pass` flipped at all,
/// which shapes were added / removed / changed, and any PMC deltas.
/// Cert schema (see `certs/hip/gfx906/*.json`): `{impl_id, backend,
/// arch, op, dtype_weight, dtype_activation, results: [{m, k, n, seed,
/// max_rel_err, tolerance, pass, pmc?}], ...}`. We diff `results[]` by
/// `(m, k, n)` key.
fn tool_cert_diff(cert_path: &str, commit_a: &str, commit_b: &str) -> Result<ToolReturn> {
    let a_json = git_show_json(commit_a, cert_path)?;
    let b_json = git_show_json(commit_b, cert_path)?;

    let a_results = a_json
        .get("results")
        .and_then(|v| v.as_array())
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let b_results = b_json
        .get("results")
        .and_then(|v| v.as_array())
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    let shape_key = |r: &serde_json::Value| -> String {
        format!(
            "m={:?},k={:?},n={:?}",
            r.get("m").unwrap_or(&serde_json::Value::Null),
            r.get("k").unwrap_or(&serde_json::Value::Null),
            r.get("n").unwrap_or(&serde_json::Value::Null),
        )
    };
    let a_map: std::collections::BTreeMap<String, &serde_json::Value> =
        a_results.iter().map(|r| (shape_key(r), r)).collect();
    let b_map: std::collections::BTreeMap<String, &serde_json::Value> =
        b_results.iter().map(|r| (shape_key(r), r)).collect();

    let mut added: Vec<String> = Vec::new();
    let mut removed: Vec<String> = Vec::new();
    let mut changed: Vec<serde_json::Value> = Vec::new();

    for (k, b_row) in &b_map {
        match a_map.get(k) {
            None => added.push(k.clone()),
            Some(a_row) if a_row != b_row => {
                let a_pass = a_row.get("pass").and_then(|v| v.as_bool()).unwrap_or(false);
                let b_pass = b_row.get("pass").and_then(|v| v.as_bool()).unwrap_or(false);
                let a_err = a_row
                    .get("max_rel_err")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NAN);
                let b_err = b_row
                    .get("max_rel_err")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NAN);
                changed.push(serde_json::json!({
                    "shape": k,
                    "pass": [a_pass, b_pass],
                    "pass_flip": a_pass != b_pass,
                    "max_rel_err": [a_err, b_err],
                }));
            }
            _ => {}
        }
    }
    for k in a_map.keys() {
        if !b_map.contains_key(k) {
            removed.push(k.clone());
        }
    }

    // Top-level pass_flip: any row that flipped green→red (regression)
    // or red→green (recovery).
    let pass_flips: Vec<&serde_json::Value> = changed
        .iter()
        .filter(|c| c.get("pass_flip").and_then(|v| v.as_bool()) == Some(true))
        .collect();

    let any_regression = pass_flips.iter().any(|c| {
        let p = c.get("pass").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        p.first().and_then(|v| v.as_bool()) == Some(true)
            && p.get(1).and_then(|v| v.as_bool()) == Some(false)
    });
    let pass_flip_count = pass_flips.len();
    let added_count = added.len();
    let removed_count = removed.len();
    let changed_count = changed.len();
    let artefact = serde_json::json!({
        "cert_path": cert_path,
        "commit_a": commit_a,
        "commit_b": commit_b,
        "impl_id_a": a_json.get("impl_id"),
        "impl_id_b": b_json.get("impl_id"),
        "results_added": added,
        "results_removed": removed,
        "results_changed": changed,
        "pass_flip_count": pass_flip_count,
        "any_regression": any_regression,
    });
    // M3.1: cert-diff's commit message is the PR-body hook. Summarise
    // the shape diff in one line so the agent can quote it verbatim.
    let msg = if any_regression {
        format!(
            "cert-diff {commit_a}..{commit_b} on {cert_path}: REGRESSION ({pass_flip_count} pass flip(s))"
        )
    } else if changed_count + added_count + removed_count == 0 {
        format!("cert-diff {commit_a}..{commit_b} on {cert_path}: no changes")
    } else {
        format!(
            "cert-diff {commit_a}..{commit_b} on {cert_path}: +{added_count} -{removed_count} ~{changed_count}"
        )
    };
    Ok(wrap_artefact(artefact, cert_path.to_owned(), msg))
}

/// Build a tool-success result from any JSON value, with both a
/// pretty-printed text fallback and the parsed `structured_content`.
/// Centralised here so every tool body uses the same envelope shape.
fn ok_result(value: serde_json::Value) -> ToolReturn {
    let mut r = CallToolResult::structured(value.clone());
    let pretty =
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
    r.content = vec![Content::text(pretty)];
    r
}

/// Same shape but flagged `is_error=true` so the client surfaces a
/// failure indicator. Tool *invocation* errors (transport, schema)
/// stay as McpError; this is for tool-*business* errors (e.g.
/// "not_shipped", "remote tool returned error").
fn err_result(value: serde_json::Value) -> ToolReturn {
    let mut r = CallToolResult::structured_error(value.clone());
    let pretty =
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
    r.content = vec![Content::text(pretty)];
    r
}

/// M3.1: wrap every real tool response in the committable-artefact
/// envelope. An agent driving flambeau-A → flambeau-B can take this
/// response, write `artefact` to `canonical_path` in its working tree,
/// and open a PR with `suggested_commit_message`. The shape is stable
/// across tools so the agent only has to learn it once.
fn wrap_artefact(
    artefact: serde_json::Value,
    canonical_path: impl Into<String>,
    suggested_commit_message: impl Into<String>,
) -> ToolReturn {
    ok_result(serde_json::json!({
        "artefact": artefact,
        "canonical_path": canonical_path.into(),
        "suggested_commit_message": suggested_commit_message.into(),
    }))
}

/// Shared error payload. Tools never panic — on failure they return a
/// JSON `{error, tool}` blob that the MCP client can surface.
fn error_payload(tool: &str, err: anyhow::Error) -> ToolReturn {
    err_result(serde_json::json!({
        "error": format!("{err:#}"),
        "tool": tool,
    }))
}

// `#[tool_handler]` wires the inherent `tool_router` into
// `ServerHandler`'s `list_tools` + `call_tool`. Without it both methods
// fall back to rmcp's defaults (empty list, method-not-found).
#[tool_handler]
impl ServerHandler for FlambeauMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "flambeau-mcp-server — dev-surface for the flambeau inference framework. \
                 Tools wrap sweep / cert-check / pmc-probe / inspect / dispatch operations \
                 and return committable JSON artefacts. Never load-bearing for production.",
            )
    }
}

/// Boot the MCP server over stdio. Blocks the current task until the
/// peer closes the connection or an error is raised.
/// Invoked by `flambeau mcp --stdio` (the default CLI mode).
pub fn run_stdio() -> impl Future<Output = Result<()>> {
    async move {
        tracing::info!(
            target: "flambeau.mcp",
            transport = "stdio",
            "booting MCP server"
        );
        let server = FlambeauMcp::new();
        let (input, output) = stdio();
        let running = server
            .serve((input, output))
            .await
            .context("serve stdio")?;
        // Run until the MCP peer disconnects. `waiting()` returns once
        // the transport closes.
        running
            .waiting()
            .await
            .context("mcp service waiting")?;
        Ok(())
    }
}

/// Boot the MCP server over streamable HTTP on `127.0.0.1:<port>`.
/// Mounts the service at `/mcp`.
/// Binds 127.0.0.1 by default — this is a DEV surface and exposing
/// it on a routable address is an explicit opt-in. Crosses the line
/// into real network exposure only when the operator passes a
/// different bind address; until M2, the only caller is another
/// flambeau process on the same host.
/// Invoked by `flambeau mcp --port N`.
pub fn run_http(port: u16) -> impl Future<Output = Result<()>> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    async move {
        tracing::info!(
            target: "flambeau.mcp",
            transport = "http",
            bind = %format!("127.0.0.1:{port}"),
            "booting MCP server"
        );

        // A fresh `FlambeauMcp` per connection via the service factory.
        // Cheap — `FlambeauMcp::new()` is just `ToolRouter::new()`.
        let ct = CancellationToken::new();
        let http_service: StreamableHttpService<FlambeauMcp, LocalSessionManager> =
            StreamableHttpService::new(
                || Ok(FlambeauMcp::new()),
                Arc::new(LocalSessionManager::default()),
                StreamableHttpServerConfig::default()
                    .with_cancellation_token(ct.child_token()),
            );

        let router = axum::Router::new().nest_service("/mcp", http_service);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .with_context(|| format!("bind 127.0.0.1:{port}"))?;
        let addr = listener.local_addr().context("local_addr")?;
        tracing::info!(
            target: "flambeau.mcp",
            listen = %addr,
            "MCP HTTP server listening"
        );
        // Signal handler: Ctrl-C cancels the server task.
        tokio::spawn({
            let ct = ct.clone();
            async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    tracing::info!(target: "flambeau.mcp", "Ctrl-C — shutting down");
                    ct.cancel();
                }
            }
        });
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { ct.cancelled().await })
            .await
            .context("axum serve")?;
        Ok(())
    }
}
