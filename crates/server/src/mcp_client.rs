//! MCP client that `flambeau serve` uses to enumerate / call tools on
//! remote MCP servers passed via `--mcp <url>` (ROADMAP-V2 §M2.1).
//!
//! Non-goals for M2.1: the full agent loop (tool-call bridging + re-
//! prompt) is M2.2; telemetry and context-budgeting are M2.3. This
//! module ships the read-only "enumerate tools at startup" primitive
//! plus the data structures the chat handler renders into the model's
//! `tools[]` list.
//!
//! Disambiguation: tool names are prefixed with a short alias derived
//! from the URL host+port (`alias.tool_name`), so two servers exposing
//! `flambeau_sweep` don't collide in the model's view. If the client
//! also supplied `tools[]` in the request, those keep their unprefixed
//! names — only the `--mcp`-registered servers get a prefix.

use anyhow::{Context, Result};
use rmcp::model::{CallToolRequestParams, ClientInfo};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::ServiceExt;

/// One tool advertised by a remote MCP server.
///
/// `name` is the **prefixed** name the model sees (`alias.tool`). The
/// original, unprefixed name is kept as `remote_name` so the agent
/// loop (M2.2) can forward `tools/call` using the name the server
/// knows.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RemoteTool {
    /// Source URL the tool was discovered on.
    pub server_url: String,
    /// Short alias we derive for disambiguation (hostname:port).
    pub server_alias: String,
    /// Prefixed name the model sees, e.g. `"flambeau_b.flambeau_sweep"`.
    pub name: String,
    /// Name as advertised by the remote server (no prefix).
    pub remote_name: String,
    /// Human-readable description for the `tools[]` entry.
    pub description: String,
    /// JSON-Schema for the tool's parameters. Rendered as-is into the
    /// model's `tools[].function.parameters`.
    pub input_schema: serde_json::Value,
}

/// Connect to every `url` in parallel, call `list_all_tools` on each,
/// and return a flat `Vec<RemoteTool>`. Errors on one URL do not tank
/// the others — per-server failures are surfaced as warning-level logs
/// and skipped, so a transiently-down MCP server doesn't kill the
/// whole `flambeau serve` startup.
pub async fn enumerate(urls: &[String]) -> Result<Vec<RemoteTool>> {
    if urls.is_empty() {
        return Ok(Vec::new());
    }
    let mut joinset: tokio::task::JoinSet<(String, Result<Vec<RemoteTool>>)> =
        tokio::task::JoinSet::new();
    for url in urls {
        let url = url.clone();
        joinset.spawn(async move {
            let res = enumerate_one(&url).await;
            (url, res)
        });
    }
    let mut out: Vec<RemoteTool> = Vec::new();
    while let Some(joined) = joinset.join_next().await {
        let (url, res) = joined.context("enumerate_one join")?;
        match res {
            Ok(mut tools) => {
                tracing::info!(
                    target: "flambeau.mcp_client",
                    url = %url,
                    tool_count = tools.len(),
                    "remote MCP server enumerated"
                );
                out.append(&mut tools);
            }
            Err(e) => {
                // Don't fail the whole boot — downstream requests that
                // rely on this server will just have an empty tools
                // list. The operator is a dev and can see the warning.
                tracing::warn!(
                    target: "flambeau.mcp_client",
                    url = %url,
                    error = %e,
                    "failed to enumerate remote MCP server — continuing without its tools"
                );
            }
        }
    }
    Ok(out)
}

async fn enumerate_one(url: &str) -> Result<Vec<RemoteTool>> {
    let alias = alias_for(url);
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url.to_owned()),
    );
    let client = ClientInfo::default()
        .serve(transport)
        .await
        .with_context(|| format!("initialize mcp client for {url}"))?;
    let tools = client
        .list_all_tools()
        .await
        .with_context(|| format!("list_all_tools on {url}"))?;
    // Politely close the session so the server sees a clean
    // disconnect. If it fails we don't care — we already have the
    // tools list.
    let _ = client.cancel().await;

    Ok(tools
        .into_iter()
        .map(|t| {
            let input_schema = serde_json::to_value(&t.input_schema)
                .unwrap_or(serde_json::Value::Null);
            let description = t.description.as_deref().unwrap_or("").to_owned();
            let remote_name = t.name.to_string();
            let prefixed = format!("{alias}.{remote_name}");
            RemoteTool {
                server_url: url.to_owned(),
                server_alias: alias.clone(),
                name: prefixed,
                remote_name,
                description,
                input_schema,
            }
        })
        .collect())
}

/// Short alias derived from `url`. Hostname by default, with port
/// appended when the URL specifies one (so `http://localhost:9090` and
/// `http://localhost:9091` don't collide).
///
/// Falls back to `"mcp"` when parsing fails — better than a crash, and
/// the prefix still disambiguates against other sources.
fn alias_for(url: &str) -> String {
    // Minimal URL-parse: find `://`, then `/`, extract authority.
    let authority = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(url)
        .split(|c: char| c == '/' || c == '?' || c == '#')
        .next()
        .unwrap_or("mcp");
    // Scrub to a safe identifier: allow `[A-Za-z0-9_]`, replace others
    // with `_`. Tool-name grammars in some frontends are conservative.
    let mut alias = String::with_capacity(authority.len());
    for ch in authority.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            alias.push(ch);
        } else {
            alias.push('_');
        }
    }
    if alias.is_empty() {
        "mcp".into()
    } else {
        alias
    }
}

/// Look up a [`RemoteTool`] by the prefixed (`alias.tool`) name the
/// model emits. Returns `None` when the name doesn't match any
/// registered upstream tool.
pub fn find_by_prefixed_name<'a>(
    tools: &'a [RemoteTool],
    prefixed_name: &str,
) -> Option<&'a RemoteTool> {
    tools.iter().find(|t| t.name == prefixed_name)
}

/// M2.2: invoke a remote MCP tool. Opens a fresh connection for the
/// call (pooling is a perf lever deferred to a follow-up), sends
/// `tools/call`, and flattens the result's content list into a single
/// string that the model can consume as a `role="tool"` payload.
///
/// `arguments_json` is the JSON-*string* the model emitted (our wire
/// contract, #20198). We parse it here into a `Map` for the MCP call.
pub async fn call_remote(tool: &RemoteTool, arguments_json: &str) -> Result<String> {
    // Parse the arguments string. If the model emitted something that
    // isn't a JSON object, log + pass an empty map so the remote
    // server's own schema validation can decide how to respond.
    let arguments_obj: serde_json::Map<String, serde_json::Value> =
        if arguments_json.trim().is_empty() {
            serde_json::Map::new()
        } else {
            match serde_json::from_str(arguments_json) {
                Ok(serde_json::Value::Object(m)) => m,
                Ok(other) => {
                    tracing::warn!(
                        target: "flambeau.mcp_client",
                        tool = %tool.name,
                        got_kind = %other,
                        "tool_call.arguments was not a JSON object; sending empty args"
                    );
                    serde_json::Map::new()
                }
                Err(e) => {
                    tracing::warn!(
                        target: "flambeau.mcp_client",
                        tool = %tool.name,
                        error = %e,
                        "tool_call.arguments isn't valid JSON; sending empty args"
                    );
                    serde_json::Map::new()
                }
            }
        };

    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(tool.server_url.clone()),
    );
    let client = ClientInfo::default()
        .serve(transport)
        .await
        .with_context(|| format!("mcp connect for call: {}", tool.server_url))?;
    let result = client
        .call_tool(
            CallToolRequestParams::new(tool.remote_name.clone())
                .with_arguments(arguments_obj),
        )
        .await
        .with_context(|| format!("call_tool {}@{}", tool.remote_name, tool.server_url))?;
    let _ = client.cancel().await;

    if result.is_error == Some(true) {
        // MCP spec allows a tool to return content + is_error=true.
        // We still surface the content as the role=tool payload —
        // better than a bare error string for the model to reason
        // about.
        tracing::warn!(
            target: "flambeau.mcp_client",
            tool = %tool.name,
            "remote tool returned is_error=true"
        );
    }
    Ok(content_to_text(&result.content))
}

/// Collapse a rmcp `CallToolResult.content` (which can be text +
/// images + resource links) into a plain string. Non-text content is
/// replaced with a marker for now — images etc. need multi-modal
/// response plumbing which is V3 scope.
fn content_to_text(content: &[rmcp::model::Content]) -> String {
    let mut out = String::new();
    for c in content {
        let raw: &rmcp::model::RawContent = &c.raw;
        match raw {
            rmcp::model::RawContent::Text(txt) => {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&txt.text);
            }
            _ => {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str("[non-text mcp content omitted; multi-modal is V3]");
            }
        }
    }
    out
}

/// M2.3: estimate how many tokens the tools[] array takes up in the
/// rendered prompt. Renders an empty-messages prompt with the full
/// tools list, tokenizes it, subtracts the render output of an
/// empty-tools prompt, and returns the delta. This is an upper bound
/// on "tokens consumed just for tool-definition overhead".
///
/// Returns `None` when rendering or tokenising fails — the caller
/// (startup log) then skips the budget warning rather than crashing.
pub fn estimate_tools_token_cost(
    chat_template: &flambeau_quant::ChatTemplate,
    tokenizer: &flambeau_quant::GgufTokenizer,
    remote_tools: &[RemoteTool],
    additional_client_tools: &[serde_json::Value],
) -> Option<usize> {
    let mut all_tools: Vec<serde_json::Value> = Vec::with_capacity(
        remote_tools.len() + additional_client_tools.len(),
    );
    all_tools.extend(remote_tools.iter().map(to_tool_json));
    all_tools.extend(additional_client_tools.iter().cloned());
    let empty_msgs: Vec<serde_json::Value> = vec![serde_json::json!({
        "role": "user",
        "content": "probe",
    })];
    let with_tools = chat_template
        .render_with_tools(
            &empty_msgs,
            Some(&all_tools),
            /*add_generation_prompt=*/ true,
            /*enable_thinking=*/ Some(false),
        )
        .ok()?;
    let without = chat_template
        .render_with_tools::<_, serde_json::Value>(
            &empty_msgs,
            None,
            /*add_generation_prompt=*/ true,
            /*enable_thinking=*/ Some(false),
        )
        .ok()?;
    let with_ids = tokenizer.encode(&with_tools).ok()?;
    let without_ids = tokenizer.encode(&without).ok()?;
    Some(with_ids.len().saturating_sub(without_ids.len()))
}

/// Render a [`RemoteTool`] into the JSON shape the Jinja chat template
/// expects for `tools[]` entries (OpenAI-compat function tool).
pub fn to_tool_json(t: &RemoteTool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_for_basic_hostport() {
        assert_eq!(alias_for("http://localhost:9090/mcp"), "localhost_9090");
        assert_eq!(alias_for("http://1.2.3.4:8080"), "1_2_3_4_8080");
        assert_eq!(alias_for("http://example.com"), "example_com");
    }

    #[test]
    fn alias_for_fallback() {
        assert_eq!(alias_for(""), "mcp");
        assert_eq!(alias_for("garbage"), "garbage");
    }

    #[test]
    fn to_tool_json_shape_matches_openai() {
        let t = RemoteTool {
            server_url: "http://x/mcp".into(),
            server_alias: "x".into(),
            name: "x.noop".into(),
            remote_name: "noop".into(),
            description: "does nothing".into(),
            input_schema: serde_json::json!({"type":"object","properties":{}}),
        };
        let j = to_tool_json(&t);
        assert_eq!(j["type"], "function");
        assert_eq!(j["function"]["name"], "x.noop");
        assert_eq!(j["function"]["description"], "does nothing");
        assert_eq!(j["function"]["parameters"]["type"], "object");
    }
}
