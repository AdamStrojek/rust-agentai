//! # Model Context Protocol Tools
//!
//! This module external tools that can connect with MCP Servers.
//!
//! Supported connection types:
//! - `stdio`
//!
//!

use crate::AgentTool;
use anyhow::Result;
use async_trait::async_trait;
use mcp_client_rs::client::{Client, ClientBuilder};
use mcp_client_rs::types::MessageContent;
use serde_json::Value;
use std::sync::Arc;
use log::trace;

pub struct McpClient {
    client: Arc<Client>,
}

impl McpClient {
    pub async fn new(cmd: &str, args: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Self> {
        trace!("McpClient::new for cmd: {}", cmd);
        let client = ClientBuilder::new(cmd)
            .args(args)
            .spawn_and_initialize()
            .await?;
        trace!("McpClient::new for client initialized");
        Ok(Self {
            client: Arc::new(client),
        })
    }

    pub async fn tools<CTX>(&self) -> Result<Vec<Arc<dyn AgentTool<CTX>>>> {
        let mut tools: Vec<Arc<dyn AgentTool<CTX>>> = vec![];

        for tool_desc in self.client.list_tools().await?.tools {
            tools.push(Arc::new(McpTool {
                client: self.client.clone(),
                name: tool_desc.name,
                description: tool_desc.description,
                schema: tool_desc.input_schema,
            }));
        }

        Ok(tools)
    }
}

pub struct McpTool {
    client: Arc<Client>,
    name: String,
    description: String,
    schema: Value,
}

#[async_trait]
impl<CTX> AgentTool<CTX> for McpTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, _: &CTX, params: Value) -> Result<String> {
        let call_result = self.client.call_tool(&self.name, params).await?;

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
