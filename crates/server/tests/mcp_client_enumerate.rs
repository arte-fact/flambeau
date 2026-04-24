//! T:M2.1 integration test — spin up the real flambeau-mcp-server
//! over HTTP in-process, then use the server crate's `mcp_client`
//! module to connect and enumerate tools. Proves the client side of
//! the `--mcp <url>` flow end-to-end without depending on any external
//! process.

use std::sync::Arc;
use std::time::Duration;

use flambeau_server::mcp_client;

#[tokio::test]
async fn enumerate_live_flambeau_mcp_server_over_http() -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use tokio_util::sync::CancellationToken;

    // Boot the real flambeau_mcp_server via HTTP on an ephemeral port.
    // Same plumbing as `flambeau mcp --port N` — this test exercises
    // the identical code path.
    let ct = CancellationToken::new();
    let service: StreamableHttpService<
        flambeau_mcp_server::FlambeauMcp,
        LocalSessionManager,
    > = StreamableHttpService::new(
        || Ok(flambeau_mcp_server::FlambeauMcp::new()),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default()
            .with_cancellation_token(ct.child_token()),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server_handle = tokio::spawn({
        let ct = ct.clone();
        async move {
            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async move { ct.cancelled_owned().await })
                .await;
        }
    });

    // Give the listener a moment to accept connections.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let url = format!("http://{addr}/mcp");
    let tools = mcp_client::enumerate(&[url.clone()]).await?;

    // Should see at least the eight tools registered by FlambeauMcp
    // (ping + 4 real + 3 stubs = 8). Names are prefix-aliased with the
    // server's host:port.
    let names: std::collections::BTreeSet<_> =
        tools.iter().map(|t| t.remote_name.clone()).collect();
    for expected in [
        "flambeau_ping",
        "flambeau_inspect_gguf",
        "flambeau_cert_check",
        "flambeau_cert_diff",
        "flambeau_dispatch_read",
        "flambeau_tune_dry",
        "flambeau_matrix",
        "flambeau_dispatch_ab",
    ] {
        assert!(
            names.contains(expected),
            "missing remote tool {expected}: got {names:?}"
        );
    }

    // All prefixed names are `<alias>.<remote_name>` and the alias is
    // derived from host:port — so two servers on different ports stay
    // disjoint in the model's view.
    for t in &tools {
        assert_eq!(
            t.name,
            format!("{}.{}", t.server_alias, t.remote_name),
            "prefix format broke"
        );
        assert!(
            t.server_alias.contains("127_0_0_1_"),
            "alias should include ip+port, got {:?}",
            t.server_alias
        );
    }

    // And the to_tool_json shape is OpenAI-compat with the prefixed
    // name (that's what the model will see in the rendered prompt).
    let j = mcp_client::to_tool_json(
        tools.iter().find(|t| t.remote_name == "flambeau_ping").unwrap(),
    );
    assert_eq!(j["type"], "function");
    assert!(
        j["function"]["name"]
            .as_str()
            .unwrap()
            .ends_with(".flambeau_ping"),
        "prefixed name not rendered into tool_json"
    );

    ct.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), server_handle).await;
    Ok(())
}

#[tokio::test]
async fn enumerate_empty_urls_returns_empty() -> anyhow::Result<()> {
    let tools = mcp_client::enumerate(&[]).await?;
    assert!(tools.is_empty());
    Ok(())
}

/// M2.2 integration — enumerate a live flambeau-mcp-server, then
/// actually call the `flambeau_ping` tool on it via
/// `mcp_client::call_remote`, and verify the round-trip payload. This
/// proves the end-to-end client→server tool-call path that the agent
/// loop relies on.
#[tokio::test]
async fn call_remote_round_trips_through_live_server() -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use tokio_util::sync::CancellationToken;

    let ct = CancellationToken::new();
    let service: StreamableHttpService<
        flambeau_mcp_server::FlambeauMcp,
        LocalSessionManager,
    > = StreamableHttpService::new(
        || Ok(flambeau_mcp_server::FlambeauMcp::new()),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default()
            .with_cancellation_token(ct.child_token()),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server_handle = tokio::spawn({
        let ct = ct.clone();
        async move {
            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async move { ct.cancelled_owned().await })
                .await;
        }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let url = format!("http://{addr}/mcp");
    let tools = mcp_client::enumerate(&[url]).await?;
    let ping = tools
        .iter()
        .find(|t| t.remote_name == "flambeau_ping")
        .expect("flambeau_ping not enumerated");

    // Call with a valid JSON object argument string.
    let output = mcp_client::call_remote(
        ping,
        r#"{"note":"hello from call_remote"}"#,
    )
    .await?;
    // The server echoes `note` and includes `flambeau-mcp-server is up`.
    assert!(
        output.contains("hello from call_remote"),
        "round-trip payload missing note: {output:?}"
    );
    assert!(
        output.contains("flambeau-mcp-server is up"),
        "payload missing canonical banner: {output:?}"
    );

    // Also verify that empty / malformed arguments don't crash — the
    // agent loop's most common failure class when a model hallucinates
    // a tool-call argument blob.
    let empty_ok = mcp_client::call_remote(ping, "").await?;
    assert!(
        empty_ok.contains("flambeau-mcp-server is up"),
        "empty-args call should still succeed: {empty_ok:?}"
    );
    let bad_json_ok = mcp_client::call_remote(ping, "{not valid json").await?;
    assert!(
        bad_json_ok.contains("flambeau-mcp-server is up"),
        "malformed-args call should surface a response, not a panic: {bad_json_ok:?}"
    );

    ct.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), server_handle).await;
    Ok(())
}

#[tokio::test]
async fn enumerate_bad_url_logs_and_skips() -> anyhow::Result<()> {
    // Non-existent server: enumerate should not fail — the warning is
    // logged and zero tools contributed.
    let tools = mcp_client::enumerate(&[
        "http://127.0.0.1:1/definitely-not-listening/mcp".into(),
    ])
    .await?;
    assert!(tools.is_empty(), "expected zero tools, got {tools:?}");
    Ok(())
}
