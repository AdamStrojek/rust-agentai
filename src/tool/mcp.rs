//! # Model Context Protocol Tools
//!
//! This module external tools that can connect with MCP Servers.
//!
//! Supported connection types:
//! - `stdio`
//!
//!

use std::collections::HashMap;
use crate::tool::{ToolBox, Tool, ToolError};
use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use mcp_client_rs::{client::{Client, ClientBuilder}, MessageContent};
use serde_json::Value;
use std::sync::Arc;
use log::trace;

pub struct McpToolBox {
    client: Arc<Client>,
    tools: Vec<Tool>,
}

impl McpToolBox {
    pub async fn new(cmd: &str, args: impl IntoIterator<Item = impl AsRef<str>>, envs: Option<HashMap<String, String>>) -> AnyhowResult<Self> {
        trace!("McpToolBox::new for cmd: {}", cmd);
        let mut builder = ClientBuilder::new(cmd)
            .args(args);

        if let Some(envs) = envs {
            for (k, v) in envs {
                builder = builder.env(&k, &v);
            }
        }

        let client = builder.spawn_and_initialize().await?;
        trace!("McpToolBox::new for client initialized");

        let mut tools = vec![];

        for tool_desc in client.list_tools().await?.tools {
            tools.push(Tool {
                    name: tool_desc.name,
                    description: Some(tool_desc.description),
                    schema: Some(tool_desc.input_schema),
                }
            );
        }

        Ok(Self {
            client: Arc::new(client),
            tools,
        })
    }
}

#[async_trait]
impl ToolBox for McpToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(self.tools.clone())
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> Result<String, ToolError> {
        let call_result = self.client.call_tool(&tool_name, arguments).await.map_err(|e| anyhow::Error::new(e))?;

        // TODO: Right now we supports only text response from tool
        let msg = call_result
            .content
            .iter()
            .filter_map(|msg| match msg {
                MessageContent::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(msg)
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
        let get_time_tool = tool_defs.iter().find(|t| t.name == "get_current_time");
        assert!(get_time_tool.is_some(), "Expected tool 'get_current_time' not found");
        assert_eq!(get_time_tool.unwrap().name, "get_current_time");
        assert!(get_time_tool.unwrap().description.is_some());
        assert!(get_time_tool.unwrap().schema.is_some());

        // Assert that the "convert_time" tool exists
        let convert_time_tool = tool_defs.iter().find(|t| t.name == "convert_time");
        assert!(convert_time_tool.is_some(), "Expected tool 'convert_time' not found");
        assert_eq!(convert_time_tool.unwrap().name, "convert_time");
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
        let result = mcp_tools.call_tool("convert_time".to_string(), arguments).await?;

        // Assert that the result is a non-empty string (the converted time)
        assert!(!result.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_call_tool_invalid_tool() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        // Call a non-existent tool
        let arguments = json!({});
        let result = mcp_tools.call_tool("non_existent_tool".to_string(), arguments).await;

        // Assert that calling a non-existent tool returns an error
        assert!(result.is_err());

        Ok(())
    }
}
