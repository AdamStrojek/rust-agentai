//! # Model Context Protocol Tools
//!
//! This module provides [`McpToolBox`], a [`ToolBox`] that exposes all tools of an
//! MCP server to an agent. It is built on the official MCP Rust SDK, [`rmcp`](https://crates.io/crates/rmcp).
//!
//! Supported connection types:
//! - `stdio`: the server is started as a child process ([`McpToolBox::new`]).
//! - Streamable HTTP: the server is already running and reachable by URL ([`McpToolBox::new_http`]).
//!
//! The server's tool list is read once, when the toolbox is created.

use crate::tool::{Tool, ToolBox, ToolError};
use anyhow::{Result as AnyhowResult, anyhow};
use async_trait::async_trait;
use log::trace;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
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

/// Joins the text blocks of an MCP tool result. Falls back to the structured content
/// when the server returned no text.
fn result_to_text(result: &CallToolResult) -> String {
    let text = result
        .content
        .iter()
        .filter_map(|block| block.as_text().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n");

    match (&result.structured_content, text.is_empty()) {
        (Some(structured), true) => structured.to_string(),
        _ => text,
    }
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
