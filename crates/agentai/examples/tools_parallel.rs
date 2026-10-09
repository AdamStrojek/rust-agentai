use agentai::tool::mcp::McpToolBox;
use agentai::tool::ToolBox;
use anyhow::{ensure, Result};
use serde_json::json;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> Result<()> {
    // AgentAI uses stdio; mcp-remote bridges it to Streamable HTTP.
    // Pin the bridge version so this example uses a reproducible dependency.
    let tools = McpToolBox::new(
        "npx",
        [
            "--yes",
            "mcp-remote@0.14.3",
            "https://search.parallel.ai/mcp",
            "--transport",
            "http-only",
            "--enable-proxy",
            "--header",
            concat!(
                "User-Agent: agentai/",
                env!("CARGO_PKG_VERSION"),
                " (tools_parallel example)"
            ),
        ],
        None,
    )
    .await?;

    let definitions = tools.tools_definitions()?;
    for name in ["web_search", "web_fetch"] {
        ensure!(
            definitions.iter().any(|tool| tool.name == name),
            "Server did not advertise {name}"
        );
    }

    // Reuse one session identifier for related calls. No model is involved.
    let session_id = format!(
        "agentai-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    let search = tokio::time::timeout(
        Duration::from_secs(60),
        tools.call_tool(
            "web_search".to_string(),
            json!({
                "objective": "Find the official Rust learning resources",
                "search_queries": ["Rust official learning resources"],
                "session_id": session_id,
            }),
        ),
    )
    .await??;
    println!("Search results:\n{search}");

    let fetch = tokio::time::timeout(
        Duration::from_secs(60),
        tools.call_tool(
            "web_fetch".to_string(),
            json!({
                "urls": ["https://www.rust-lang.org/learn"],
                "objective": "Extract the available Rust learning resources",
                "session_id": session_id,
            }),
        ),
    )
    .await??;
    println!("Fetched page:\n{fetch}");
    Ok(())
}
