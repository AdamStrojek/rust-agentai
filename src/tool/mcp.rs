//! # Model Context Protocol Tools
//!
//! This module external tools that can connect with MCP Servers.
//!
//! Supported connection types:
//! - `stdio`
//! - `http`
//!
//!

use crate::tool::{Tool, ToolBox, ToolError};
use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use log::{debug, info};
use rmcp::{
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    service::RunningService,
    transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess},
    RoleClient, ServiceExt,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::process::Command;

// Type aliases for the different client types we'll store
type ChildProcessClient = RunningService<RoleClient, ()>;
type HttpClient = RunningService<RoleClient, rmcp::model::InitializeRequestParam>;

pub struct McpToolBox {
    stdio_clients: Vec<Arc<ChildProcessClient>>,
    http_clients: Vec<Arc<HttpClient>>,
    tools: Vec<Tool>,
}

pub enum McpServer {
    StdIo(StdIoMcp),
    StreamableHttp(StreamableHttpMcp),
}

impl McpServer {
    pub fn new_std_io(command: String, args: Vec<String>) -> Self {
        Self::StdIo(StdIoMcp { command, args })
    }

    pub fn new_streamable_http(url: String) -> Self {
        Self::StreamableHttp(StreamableHttpMcp { url })
    }
}

pub struct StdIoMcp {
    pub command: String,
    pub args: Vec<String>,
}

pub struct StreamableHttpMcp {
    pub url: String,
}

impl McpToolBox {
    pub async fn new(servers: Vec<McpServer>) -> AnyhowResult<Self> {
        let mut stdio_clients = Vec::new();
        let mut http_clients = Vec::new();
        let mut tools = Vec::new();

        for server in servers.into_iter() {
            match server {
                McpServer::StdIo(std_io) => {
                    let client = ()
                        .serve(TokioChildProcess::new(
                            Command::new(std_io.command).configure(|cmd| {
                                cmd.args(std_io.args);
                            }),
                        )?)
                        .await?;
                    let tool_index = stdio_clients.len();

                    // Get server info and list tools
                    let server_info = client.peer_info();
                    info!(
                        "Connected to child process server: {server_info:#?}, tool_index: {tool_index}"
                    );

                    // List tools for this server
                    let tools_response = client.list_tools(Default::default()).await?;
                    for tool in tools_response.tools {
                        let name = format!("stdio_{}_{}", tool_index, tool.name);
                        debug!("added stdio tool {name}");
                        tools.push(Tool {
                            name,
                            description: tool.description.map(|d| d.to_string()),
                            schema: Some(serde_json::to_value(tool.input_schema)?),
                        });
                    }

                    stdio_clients.push(Arc::new(client));
                }
                McpServer::StreamableHttp(streamable_http) => {
                    let transport = StreamableHttpClientTransport::from_uri(streamable_http.url);
                    let client_info = ClientInfo {
                        protocol_version: Default::default(),
                        capabilities: ClientCapabilities::default(),
                        client_info: Implementation {
                            name: "sse-client".to_string(),
                            version: "0.0.1".to_string(),
                        },
                    };
                    let client = client_info.serve(transport).await?;
                    let tool_index = http_clients.len();

                    // Get server info and list tools
                    let server_info = client.peer_info();
                    info!("Connected to HTTP server: {server_info:#?} index: {tool_index}");

                    // List tools for this server
                    let tools_response = client.list_tools(Default::default()).await?;
                    for tool in tools_response.tools {
                        let name = format!("http_{}_{}", tool_index, tool.name);
                        debug!("added http tool {name}");
                        tools.push(Tool {
                            name,
                            description: tool.description.map(|d| d.to_string()),
                            schema: Some(serde_json::to_value(tool.input_schema)?),
                        });
                    }

                    http_clients.push(Arc::new(client));
                }
            };
        }

        Ok(Self {
            stdio_clients,
            http_clients,
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
        // Extract server name and actual tool name from the prefixed tool name
        let parts: Vec<String> = tool_name.splitn(3, '_').map(|s| s.to_string()).collect();
        if parts.len() != 3 {
            return Err(ToolError::NoToolFound(tool_name));
        }

        let server_type = &parts[0];
        let server_idx = &parts[1];
        let actual_tool_name = &parts[2];

        match server_type.as_str() {
            "stdio" => {
                if let Some(client) = self.stdio_clients.get(server_idx.parse::<usize>().unwrap()) {
                    let call_result = client
                        .call_tool(CallToolRequestParam {
                            name: actual_tool_name.clone().into(),
                            arguments: Some(arguments.as_object().unwrap().clone()),
                        })
                        .await
                        .map_err(anyhow::Error::new)?;

                    let response_json = serde_json::to_string(&call_result.content)
                        .unwrap_or_else(|_| "Unable to serialize response".to_string());
                    return Ok(response_json);
                } else {
                    return Err(ToolError::NoToolFound(tool_name));
                }
            }
            "http" => {
                if let Some(client) = self.http_clients.get(server_idx.parse::<usize>().unwrap()) {
                    let call_result = client
                        .call_tool(CallToolRequestParam {
                            name: actual_tool_name.clone().into(),
                            arguments: Some(arguments.as_object().unwrap().clone()),
                        })
                        .await
                        .map_err(anyhow::Error::new)?;
                    let response_json = serde_json::to_string(&call_result.content)
                        .unwrap_or_else(|_| "Unable to serialize response".to_string());
                    return Ok(response_json);
                } else {
                    return Err(ToolError::NoToolFound(tool_name));
                }
            }
            _ => return Err(ToolError::NoToolFound(tool_name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result as AnyhowResult;
    use serde_json::json;

    // Helper function to create a McpToolBox for testing
    async fn create_test_toolbox() -> AnyhowResult<McpToolBox> {
        McpToolBox::new(vec![McpServer::new_std_io(
            "uvx".to_string(),
            vec![
                "mcp-server-time".to_string(),
                "--local-timezone".to_string(),
                "UTC".to_string(),
            ],
        )])
        .await
    }

    #[tokio::test]
    async fn test_new_and_tools_definitions() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        let tool_defs = mcp_tools.tools_definitions()?;

        // Assert that we get at least one tool definition
        assert!(tool_defs.len() >= 1);

        // Assert that tools have the server prefix (now using server0_ instead of server_0_)
        let tools_with_prefix: Vec<_> = tool_defs
            .iter()
            .filter(|t| t.name.starts_with("stdio_0_"))
            .collect();
        assert!(!tools_with_prefix.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_call_tool_convert_time() -> AnyhowResult<()> {
        let mcp_tools = create_test_toolbox().await?;

        // First, get the available tools to see what's actually available
        let tool_defs = mcp_tools.tools_definitions()?;
        let convert_time_tool = tool_defs
            .iter()
            .find(|t| t.name.contains("convert_time"))
            .expect("convert_time tool should be available");

        // Call the 'convert_time' tool with required arguments (using the actual tool name)
        let arguments = json!({
            "source_timezone": "Europe/Warsaw",
            "target_timezone": "America/New_York",
            "time": "12:00"
        });
        let result = mcp_tools
            .call_tool(convert_time_tool.name.clone(), arguments)
            .await?;

        // Assert that the result is a non-empty string (the converted time)
        assert!(!result.is_empty());
        println!("Convert time result: {}", result);

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
