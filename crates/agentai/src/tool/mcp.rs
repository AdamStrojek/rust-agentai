//! # Model Context Protocol Tools
//!
//! This module provides [`McpToolBox`], a [`ToolBox`] that exposes all tools of an
//! MCP server to an agent. It is built on the official MCP Rust SDK, [`rmcp`](https://crates.io/crates/rmcp).
//!
//! Supported connection types:
//! - `stdio`: the server is started as a child process ([`McpToolBox::new`]).
//! - Streamable HTTP: the server is already running and reachable by URL ([`McpToolBox::new_http`],
//!   or [`McpToolBox::new_http_with_bearer`] for servers that need a bearer token).
//!
//! The server's tool list is read once, when the toolbox is created.
//!
//! Tool results are returned to the model as text. Text blocks are joined; when there is no
//! text, the structured content is returned as JSON. Images, audio and embedded resources are
//! not forwarded: each one is replaced by a short placeholder such as `[image omitted]`.

use crate::tool::{Tool, ToolBox, ToolError};
use anyhow::{Result as AnyhowResult, anyhow};
use async_trait::async_trait;
use log::trace;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde_json::Value;
use std::collections::HashMap;
use tokio::process::Command;

/// A [`ToolBox`] backed by an MCP server.
pub struct McpToolBox {
    client: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

impl McpToolBox {
    /// Starts an MCP server as a child process and connects to it over stdio.
    ///
    /// # Arguments
    ///
    /// * `cmd` - Command that starts the server, for example `uvx`.
    /// * `args` - Arguments for the command, for example `["mcp-server-time"]`.
    /// * `envs` - Extra environment variables for the server process.
    ///
    /// # Errors
    ///
    /// Returns an error when the process cannot be started, the MCP handshake fails,
    /// or the tool list cannot be read.
    pub async fn new(
        cmd: &str,
        args: impl IntoIterator<Item = impl AsRef<str>>,
        envs: Option<HashMap<String, String>>,
    ) -> AnyhowResult<Self> {
        trace!("McpToolBox::new for cmd: {cmd}");
        let args: Vec<String> = args.into_iter().map(|a| a.as_ref().to_string()).collect();
        let command = Command::new(cmd).configure(|command| {
            command.args(&args);
            if let Some(envs) = envs {
                command.envs(envs);
            }
        });
        let transport = TokioChildProcess::new(command)?;
        let client = ().serve(transport).await?;
        trace!("McpToolBox::new client initialized");

        Self::from_client(client).await
    }

    /// Connects to a running MCP server over the Streamable HTTP transport.
    ///
    /// # Arguments
    ///
    /// * `url` - Endpoint of the MCP server, for example `http://localhost:8000/mcp`.
    ///
    /// # Errors
    ///
    /// Returns an error when the MCP handshake fails or the tool list cannot be read.
    pub async fn new_http(url: &str) -> AnyhowResult<Self> {
        trace!("McpToolBox::new_http for url: {url}");
        let transport = StreamableHttpClientTransport::from_uri(url.to_string());
        let client = ().serve(transport).await?;
        trace!("McpToolBox::new_http client initialized");

        Self::from_client(client).await
    }

    /// Connects to a running MCP server over the Streamable HTTP transport and sends
    /// `Authorization: Bearer <token>` with every request.
    ///
    /// # Arguments
    ///
    /// * `url` - Endpoint of the MCP server, for example `https://example.com/mcp`.
    /// * `token` - Bearer token, without the `Bearer ` prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when the MCP handshake fails or the tool list cannot be read.
    pub async fn new_http_with_bearer(url: &str, token: &str) -> AnyhowResult<Self> {
        trace!("McpToolBox::new_http_with_bearer for url: {url}");
        let config =
            StreamableHttpClientTransportConfig::with_uri(url.to_string()).auth_header(token);
        let transport = StreamableHttpClientTransport::from_config(config);
        let client = ().serve(transport).await?;
        trace!("McpToolBox::new_http_with_bearer client initialized");

        Self::from_client(client).await
    }

    async fn from_client(client: RunningService<RoleClient, ()>) -> AnyhowResult<Self> {
        let tools = client
            .list_all_tools()
            .await?
            .into_iter()
            .map(|mcp_tool| {
                let mut tool = Tool::new(mcp_tool.name.to_string())
                    .with_schema(Value::Object(mcp_tool.input_schema.as_ref().clone()));
                tool.description = mcp_tool.description.map(|d| d.to_string());
                tool
            })
            .collect();

        Ok(Self { client, tools })
    }
}

/// Converts an MCP tool result to text for the model. Text blocks are joined; other blocks
/// become placeholders so the model knows something was returned. Falls back to the
/// structured content when the server returned no text.
fn result_to_text(result: &CallToolResult) -> String {
    let has_text = result.content.iter().any(|block| block.as_text().is_some());
    if !has_text && let Some(structured) = &result.structured_content {
        return structured.to_string();
    }

    result
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => text.text.clone(),
            ContentBlock::Image(_) => "[image omitted]".to_string(),
            ContentBlock::Audio(_) => "[audio omitted]".to_string(),
            ContentBlock::Resource(_) => "[embedded resource omitted]".to_string(),
            ContentBlock::ResourceLink(link) => format!("[resource link: {}]", link.uri),
            _ => "[unsupported content omitted]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl ToolBox for McpToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(self.tools.clone())
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> Result<String, ToolError> {
        // Lets `ToolBoxSet` try the next toolbox for tools this server does not have
        if !self.tools.iter().any(|t| t.name.as_str() == tool_name) {
            return Err(ToolError::NoToolFound(tool_name));
        }

        let arguments = match arguments {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Err(ToolError::LLMError(format!(
                    "Arguments for tool '{tool_name}' must be a JSON object, got: {other}"
                )));
            }
        };

        let mut params = CallToolRequestParams::new(tool_name.clone());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }

        let result = self
            .client
            .call_tool(params)
            .await
            .map_err(|err| ToolError::Other(anyhow!(err)))?;

        let text = result_to_text(&result);
        if result.is_error == Some(true) {
            // The server reports a tool-level failure; pass its message to the model
            return Err(ToolError::LLMError(text));
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result as AnyhowResult;
    use serde_json::json;

    // Helper function to create a McpToolBox for testing
    async fn create_test_toolbox() -> AnyhowResult<McpToolBox> {
        McpToolBox::new("uvx", ["mcp-server-time", "--local-timezone", "UTC"], None).await
    }

    #[tokio::test]
    async fn test_new_and_tools_definitions() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        let tool_defs = mcp_tools.tools_definitions()?;

        // Assert that we get at least two tool definitions
        assert!(tool_defs.len() >= 2);

        // Assert that the "get_current_time" tool exists
        let get_time_tool = tool_defs
            .iter()
            .find(|t| t.name.as_str() == "get_current_time");
        assert!(
            get_time_tool.is_some(),
            "Expected tool 'get_current_time' not found"
        );
        assert_eq!(get_time_tool.unwrap().name.as_str(), "get_current_time");
        assert!(get_time_tool.unwrap().description.is_some());
        assert!(get_time_tool.unwrap().schema.is_some());

        // Assert that the "convert_time" tool exists
        let convert_time_tool = tool_defs.iter().find(|t| t.name.as_str() == "convert_time");
        assert!(
            convert_time_tool.is_some(),
            "Expected tool 'convert_time' not found"
        );
        assert_eq!(convert_time_tool.unwrap().name.as_str(), "convert_time");
        assert!(convert_time_tool.unwrap().description.is_some());
        assert!(convert_time_tool.unwrap().schema.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_call_tool_convert_time() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        // Call the 'convert_time' tool with required arguments
        let arguments = json!({
            "source_timezone": "Europe/Warsaw",
            "target_timezone": "America/New_York",
            "time": "12:00"
        });
        let result = mcp_tools
            .call_tool("convert_time".to_string(), arguments)
            .await?;

        // Assert that the result is a non-empty string (the converted time)
        assert!(!result.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_call_tool_invalid_tool() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        // Call a non-existent tool
        let arguments = json!({});
        let result = mcp_tools
            .call_tool("non_existent_tool".to_string(), arguments)
            .await;

        // Unknown tools return NoToolFound, so ToolBoxSet can try the next toolbox
        assert!(matches!(result, Err(ToolError::NoToolFound(_))));

        Ok(())
    }

    #[tokio::test]
    async fn test_call_tool_server_error_reaches_model() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        // An unknown timezone makes the server report a tool error
        let arguments = json!({ "timezone": "Not/AZone" });
        let result = mcp_tools
            .call_tool("get_current_time".to_string(), arguments)
            .await;

        match result {
            Err(ToolError::LLMError(message)) => assert!(!message.is_empty()),
            other => panic!("Expected ToolError::LLMError, got {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_result_to_text() -> AnyhowResult<()> {
        let mixed: CallToolResult = serde_json::from_value(json!({
            "content": [
                { "type": "text", "text": "first" },
                { "type": "image", "data": "AAAA", "mimeType": "image/png" },
                { "type": "text", "text": "second" }
            ]
        }))?;
        assert_eq!(result_to_text(&mixed), "first\n[image omitted]\nsecond");

        let structured: CallToolResult = serde_json::from_value(json!({
            "content": [],
            "structuredContent": { "sum": 3 }
        }))?;
        assert_eq!(result_to_text(&structured), r#"{"sum":3}"#);

        Ok(())
    }

    #[tokio::test]
    async fn test_call_tool_rejects_non_object_arguments() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        let result = mcp_tools
            .call_tool("get_current_time".to_string(), json!("Europe/Warsaw"))
            .await;

        assert!(matches!(result, Err(ToolError::LLMError(_))));

        Ok(())
    }
}
