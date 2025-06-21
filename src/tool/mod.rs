//! # Tools and Tool Boxes
//!
//! This module provides the core infrastructure for defining, organizing, and executing tools within the `agentai` crate.
//! It introduces the concept of a `ToolBox`, which is a collection of callable `Tool` instances.
//!
//! Agents interact with the external world by calling these `Tool`s, which encapsulate specific functionalities
//! like searching the web, interacting with external APIs, or performing calculations.
//!
//! To implement your own `ToolBox`, you have two primary options:
//!
//! 1.  **Using the `#[toolbox]` macro:** This is the recommended approach for most cases. The macro simplifies the process by
//!     automatically generating the necessary boilerplate for a `ToolBox` trait implementation based on methods defined in a struct.
//!     See [`#[toolbox]`](crate::tool::toolbox) for more details.
//!
//! 2.  **Manual implementation:** If you require finer control over the `ToolBox` behavior, you can provide your own implementation
//!     for the [`ToolBox` trait](crate::tool::ToolBox).
//!
//! Ready-to-use `ToolBox` implementations are available:
//! - [crate::tool::buildin]: Provides a set of useful built-in tools.
//! - [crate::tool::mcp]: A `ToolBox` for interacting with the MCP Client. (Requires the `mcp-client` feature).
//!
//! For examples demonstrating how to use tools and toolboxes, look into the `examples` folder.
//! Examples related to tools typically start with the `tools_*` prefix, e.g., [crate::examples::tools_mcp].
//!
//! For example demonstrating how to implement `ToolBox` trait using `#[toolbox]` macro, look into [crate::examples::tools_custom] example.

pub mod websearch;

#[cfg(feature = "mcp-client")]
pub mod mcp;

use thiserror::{Error};
use serde_json::Value;

// Re-export Tool structure, it is being used by ToolBoxes
/// Represents a tool definition that can be exposed to an agent.
///
/// This structure is used by `ToolBox` implementations in their `tools_definitions`
/// function to describe the available tools to the language model.
///
/// **Note:** While this struct defines a tool, actual tool invocation is handled
/// by a `ToolBox` implementing the [`ToolBox`] trait. You must use a `ToolBox`
/// to call any defined tool.
///
/// The `name` field is required, while `description` and `schema` are optional
/// but highly recommended for effective tool use by the agent.
pub use genai::chat::Tool;

// Re-export tool and toolbox macros, they are used to generate auto implementation of
pub use agentai_macros::toolbox;

/// Manages a collection of callable `Tool` instances.
///
/// Implementors of `ToolBox` provide a way to group related tools and expose them to the
/// agent for invocation. The `ToolBox` is responsible for defining the available tools
/// and executing them when requested.
///
/// **Important:** This trait requires the use of the [`#[async_trait::async_trait]`](https://docs.rs/async-trait) attribute macro
/// for proper asynchronous behavior and `dyn ToolBox` compatibility.
///
/// For most use cases, implementing this trait can be significantly simplified by using
/// the [`#[toolbox]`](crate::tool::toolbox) attribute macro. This macro automatically
/// generates the necessary `ToolBox` implementation for a struct based on its methods.
#[async_trait::async_trait]
pub trait ToolBox {
    /// Returns a list of all `Tool` instances contained within this ToolBox.
    /// These definitions include the tool's name, description, and parameters,
    /// which are used by the language model to decide which tool to call.
    ///
    /// The `schema` field of the `Tool` can be conveniently generated from Rust structs using the [`schemars`](https://crates.io/crates/schemars) crate.
    ///
    /// This method is typically invoked internally by the [`Agent`](crate::agent::Agent) structure to discover the available tools and their parameters.
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError>;

    /// Calls a specific tool by its name with the given parameters.
    ///
    /// This method is the entry point for executing a tool's functionality. It is typically invoked internally by the [`Agent`](crate::agent::Agent) structure
    /// when the language model determines that a tool needs to be called.
    /// The `arguments` are provided as a `serde_json::Value`. You can easily deserialize this `Value`
    /// into a Rust struct (e.g., the same struct used to generate the JSON schema
    /// in `tools_definitions`) using the [`serde`](https://crates.io/crates/serde) crate.
    /// The arguments provided by the agent will conform to the JSON schema defined for the tool.
    ///
    /// For example, to deserialize the arguments:
    /// ```rust
    /// let args: ToolArguments = serde_json::from_value(arguments)?;
    /// ```
    /// Replace `ToolArguments` with the actual struct type corresponding to your tool's schema.
    ///
    /// # Arguments
    /// * `tool_name` - The name of the tool to call.
    /// * `arguments` - A JSON `Value` containing the arguments for the tool call.
    ///
    /// # Returns
    /// A `Result` containing the tool's output as a `String` on success,
    /// or a `ToolError` if the tool call fails or the tool is not found.
    async fn call_tool(&self, tool_name: String, arguments: Value) -> Result<String, ToolError>;
}

#[derive(Error, Debug)]
/// Represents potential errors that can occur when working with `ToolBox`es and tools.
///
/// These errors cover scenarios like failing to retrieve tool definitions, attempting to call
/// a non-existent tool, or encountering an issue during tool execution.
pub enum ToolError {
    /// Indicates that the `ToolBox`'s tool definitions are not yet available or ready to be retrieved.
    /// This could happen if the definitions are generated on the fly and required information is missing,
    /// or if the toolbox was not properly initialized or created.
    #[error("Tool definitions are not ready")]
    ToolsDefinitionNotReady,
    /// Indicates that a requested tool could not be found within the `ToolBox`.
    /// This occurs when the `tool_name` provided to `call_tool` does not match any
    /// registered tool in the box.
    #[error("Tool named '{0}' not found")]
    NoToolFound(String),
    /// Indicates a failure occurred during the execution of a specific tool.
    /// This is a general error variant that can encapsulate various runtime issues
    /// encountered while the tool's logic is running.
    #[error("Tool execution failed")]
	ExecutionError,
    /// Represents any other underlying error that occurred, wrapped from the `anyhow::Error` type.
    /// This allows for propagating errors from dependencies or other parts of the system.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
