pub mod websearch;

#[cfg(feature = "mcp-client")]
pub mod mcp;

use thiserror::{Error};
use serde_json::Value;

pub use genai::chat::Tool;

/// A container or manager for a collection of `Tool` instances.
/// This provides a way to group and access multiple tools.
#[async_trait::async_trait]
pub trait ToolBox {
    /// Returns a list of all `Tool` instances contained within this ToolBox.
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError>;

    /// This function will call tool
    async fn call_tool(&self, tool_name: String, parameters: Value) -> Result<String, ToolError>;
}

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tools definition not ready")]
    ToolsDefinitionNotReady,
    #[error("Tool named {0} not found")]
    NoToolFound(String),
    #[error("This error will be raised if call fails")]
	ExecutionError,
    // TODO: add opaque type for errors from tool execution
}
