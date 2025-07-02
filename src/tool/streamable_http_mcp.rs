//! # Model Context Protocol Tools
//!
//! This module external tools that can connect with Streamable HTTP MCP Servers.
//!
//!

use crate::tool::{Tool, ToolBox, ToolError};
use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use log::debug;
use rmcp::{
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    service::RunningService,
    transport::StreamableHttpClientTransport,
    RoleClient, ServiceExt,
};
use serde_json::Value;

pub struct StreamableHttpMcp {
    pub tools: Vec<Tool>,
    pub mcp_client: RunningService<RoleClient, rmcp::model::InitializeRequestParam>,
}

impl StreamableHttpMcp {
    pub async fn try_new(url: String) -> AnyhowResult<Self> {
        let transport = StreamableHttpClientTransport::from_uri(url);
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: "sse-client".to_string(),
                version: "0.0.1".to_string(),
            },
        };
        let mcp_client = client_info.serve(transport).await?;

        // Get server info and list tools
        let server_info = mcp_client.peer_info();
        debug!("Connected to HTTP server: {server_info:#?}");

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
impl ToolBox for StreamableHttpMcp {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(self.tools.clone())
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> Result<String, ToolError> {
        let call_result = self
            .mcp_client
            .call_tool(CallToolRequestParam {
                name: tool_name.into(),
                arguments: Some(arguments.as_object().unwrap().clone()),
            })
            .await
            .map_err(anyhow::Error::new)?;
        let response_json =
            serde_json::to_string(&call_result.content).map_err(|e| ToolError::Other(e.into()))?;
        Ok(response_json)
    }
}
