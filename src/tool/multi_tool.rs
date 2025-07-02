use async_trait::async_trait;
use genai::chat::Tool;
use serde_json::Value;

use crate::tool::{ToolBox, ToolError};

pub struct MergeTool {
    pub tools: Vec<Box<dyn ToolBox>>,
}

impl MergeTool {
    pub fn new(tools: Vec<Box<dyn ToolBox>>) -> Self {
        Self { tools }
    }
}

#[async_trait]
impl ToolBox for MergeTool {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        let tools = self
            .tools
            .iter()
            .enumerate()
            .map(|(index, tool)| {
                tool.tools_definitions().map(|tool_defs| {
                    tool_defs
                        .into_iter()
                        .map(move |mut tool| {
                            tool.name = format!("{}-{}", index, tool.name);
                            tool
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(tools)
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> Result<String, ToolError> {
        // only split once
        match tool_name.split_once("-") {
            Some((tool_index, original_tool_name)) => {
                let tool_index = tool_index
                    .parse::<usize>()
                    .map_err(|_| ToolError::NoToolFound(tool_name.clone()))?;

                if tool_index >= self.tools.len() {
                    return Err(ToolError::NoToolFound(tool_name));
                }

                let tool = &self.tools[tool_index];
                tool.call_tool(original_tool_name.to_string(), arguments)
                    .await
            }
            None => {
                return Err(ToolError::NoToolFound(tool_name));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use crate::tool::stdio_mcp::StdIoMcp;

    use super::*;

    // Helper function to create a McpToolBox for testing
    async fn create_test_toolbox() -> Result<MergeTool> {
        Ok(MergeTool::new(vec![
            Box::new(
                StdIoMcp::try_new(
                    "uvx".to_string(),
                    vec![
                        "mcp-server-time".to_string(),
                        "--local-timezone".to_string(),
                        "UTC".to_string(),
                    ],
                )
                .await
                .unwrap(),
            ),
            Box::new(
                StdIoMcp::try_new(
                    "uvx".to_string(),
                    vec![
                        "mcp-server-time".to_string(),
                        "--local-timezone".to_string(),
                        "Europe/Paris".to_string(),
                    ],
                )
                .await
                .unwrap(),
            ),
        ]))
    }
    #[tokio::test]
    async fn test_merge_tool() {
        let merge_tool = create_test_toolbox().await.unwrap();
        let tool_defs = merge_tool.tools_definitions().unwrap();
        assert_eq!(tool_defs.len(), 4);
        assert_eq!(tool_defs[0].name, "0-get_current_time");
        assert_eq!(tool_defs[1].name, "0-convert_time");
        assert_eq!(tool_defs[2].name, "1-get_current_time");
        assert_eq!(tool_defs[3].name, "1-convert_time");
    }

    #[tokio::test]
    async fn test_call_tool_convert_time() -> Result<()> {
        let mcp_tools = create_test_toolbox().await?;

        // Call the 'convert_time' tool with required arguments
        let arguments = json!({
            "source_timezone": "Europe/Warsaw",
            "target_timezone": "America/New_York",
            "time": "12:00"
        });
        let result = mcp_tools
            .call_tool("0-convert_time".to_string(), arguments.clone())
            .await?;

        // Assert that the result is a non-empty string (the converted time)
        assert!(!result.is_empty());

        // Call second tool
        let result = mcp_tools
            .call_tool("1-convert_time".to_string(), arguments)
            .await?;
        assert!(!result.is_empty());

        Ok(())
    }
}
