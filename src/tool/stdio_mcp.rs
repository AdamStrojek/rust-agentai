//! # Model Context Protocol Tools
//!
//! This module external tools that can connect with STDIOMCP Servers.
//!
//!
//!

use crate::tool::{Tool, ToolBox, ToolError};
use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use log::debug;
use rmcp::{
    model::CallToolRequestParam,
    service::RunningService,
    transport::{ConfigureCommandExt, TokioChildProcess},
    RoleClient, ServiceExt,
};
use serde_json::Value;
use tokio::process::Command;

pub struct StdIoMcp {
    pub tools: Vec<Tool>,
    pub mcp_client: RunningService<RoleClient, ()>,
}

impl StdIoMcp {
    pub async fn try_new(command: String, args: Vec<String>) -> AnyhowResult<Self> {
        let mcp_client = ()
            .serve(TokioChildProcess::new(Command::new(command).configure(
                |cmd| {
                    cmd.args(args);
                },
            ))?)
            .await?;

        // Get server info and list tools
        let server_info = mcp_client.peer_info();
        debug!("Connected to child process server: {server_info:#?}");

        // List tools for this server
        let tools = mcp_client
            .list_tools(Default::default())
            .await?
            .tools
            .into_iter()
            .map(|tool| Tool {
                name: tool.name.to_string(),
                description: tool.description.map(|d| d.to_string()),
                schema: Some(
                    serde_json::to_value(tool.input_schema)
                        .expect("Failed to convert input schema to JSON"),
                ),
            })
            .collect();

        Ok(Self { tools, mcp_client })
    }
}

#[async_trait]
impl ToolBox for StdIoMcp {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(self.tools.clone())
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> Result<String, ToolError> {
        let call_result = self
            .mcp_client
            .call_tool(CallToolRequestParam {
                name: tool_name.clone().into(),
                arguments: Some(arguments.as_object().unwrap().clone()),
            })
            .await
            .map_err(anyhow::Error::new)?;
        if call_result.is_error.unwrap_or(false) {
            let error_message = call_result
                .content
                .iter()
                .find_map(|c| c.raw.as_text().map(|t| t.text.as_str()))
                .unwrap_or("Unknown error");
            if error_message.contains("Unknown tool") {
                return Err(ToolError::NoToolFound(tool_name.clone()));
            }
            return Err(ToolError::Other(anyhow::anyhow!(
                "Tool error: {}",
                error_message
            )));
        }
        let response_json =
            serde_json::to_string(&call_result.content).map_err(|e| ToolError::Other(e.into()))?;
        Ok(response_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result as AnyhowResult;
    use serde_json::json;

    // Helper function to create a McpToolBox for testing
    async fn create_test_toolbox() -> AnyhowResult<StdIoMcp> {
        StdIoMcp::try_new(
            "uvx".to_string(),
            vec![
                "mcp-server-time".to_string(),
                "--local-timezone".to_string(),
                "UTC".to_string(),
            ],
        )
        .await
    }

    #[tokio::test]
    async fn test_new_and_tools_definitions() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        let tool_defs = mcp_tools.tools_definitions()?;

        // Assert that we get at least two tool definitions
        assert!(tool_defs.len() >= 2);

        // Assert that the "get_current_time" tool exists
        let get_time_tool = tool_defs.iter().find(|t| t.name == "get_current_time");
        assert!(
            get_time_tool.is_some(),
            "Expected tool 'get_current_time' not found"
        );
        assert_eq!(get_time_tool.unwrap().name, "get_current_time");
        assert!(get_time_tool.unwrap().description.is_some());
        assert!(get_time_tool.unwrap().schema.is_some());

        // Assert that the "convert_time" tool exists
        let convert_time_tool = tool_defs.iter().find(|t| t.name == "convert_time");
        assert!(
            convert_time_tool.is_some(),
            "Expected tool 'convert_time' not found"
        );
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

        // Assert that calling a non-existent tool returns an error
        assert!(result.is_err());
        Ok(())
    }
}
