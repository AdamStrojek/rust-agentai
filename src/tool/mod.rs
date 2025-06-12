pub mod websearch;

#[cfg(feature = "mcp-client")]
pub mod mcp;

use thiserror::{Error};
use serde_json::Value;

// Re-export Tool structure, it is being used by ToolBoxes
pub use genai::chat::Tool;

// Re-export tool and toolbox macros, they are used to generate auto implementation of
pub use agentai_macros::{toolbox, tool};

/// A container or manager for a collection of `Tool` instances.
/// This provides a way to group and access multiple tools.
#[async_trait::async_trait]
pub trait ToolBox {
    /// Returns a list of all `Tool` instances contained within this ToolBox.
    /// These definitions include the tool's name, description, and parameters,
    /// which are used by the language model to decide which tool to call.
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError>;

    /// Calls a specific tool by its name with the given parameters.
    ///
    /// This method is the entry point for executing a tool's functionality.
    ///
    /// # Arguments
    /// * `tool_name` - The name of the tool to call.
    /// * `parameters` - A JSON `Value` containing the parameters for the tool call.
    ///
    /// # Returns
    /// A `Result` containing the tool's output as a `String` on success,
    /// or a `ToolError` if the tool call fails or the tool is not found.
    async fn call_tool(&self, tool_name: String, parameters: Value) -> Result<String, ToolError>;
}

#[derive(Error, Debug)]
pub enum ToolError {
    /// Indicates that the tools definitions are not yet ready to be retrieved.
    #[error("Tools definition not ready")]
    ToolsDefinitionNotReady,
    /// Indicates that a requested tool could not be found within the ToolBox.
    #[error("Tool named {0} not found")]
    NoToolFound(String),
    /// Indicates a failure occurred during the execution of a tool.
    #[error("This error will be raised if call fails")]
	ExecutionError,
    /// Represents other underlying errors wrapped from `anyhow::Error`.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
