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

#[cfg(feature = "tools-buildin")]
pub mod buildin;

#[cfg(feature = "mcp-client")]
pub mod mcp;

#[cfg(feature = "tools-web")]
pub mod web;

use serde_json::Value;
use thiserror::Error;

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

pub type ToolResult = Result<String, ToolError>;

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
    async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult;
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
    /// Indicates that returned error should be handled by LLM, for example it may be
    /// missing parameter or malformed data. Tool is responsible to provide
    /// human readable error message, this message will be passed to the LLM
    #[error("{0}")]
    LLMError(String),
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

/// A collection of `ToolBox` instances.
///
/// It allows for managing multiple toolboxes as a single unit, aggregating
/// their tool definitions and dispatching tool calls to the appropriate `ToolBox`.
///
/// When a tool is called, the `ToolBoxSet` will search through its contained
/// toolboxes in the order they were added. The first `ToolBox` that contains
/// a tool with a matching name will be used to execute the call.
#[derive(Default)]
pub struct ToolBoxSet {
    toolboxes: Vec<Box<dyn ToolBox + Send + Sync>>,
}

impl ToolBoxSet {
    /// Creates a new, empty `ToolBoxSet`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a `ToolBox` to the set.
    ///
    /// The order in which toolboxes are added is significant. When a tool call
    /// is made, the `ToolBoxSet` will search for the tool in the order the
    /// toolboxes were added.
    pub fn add_tool(&mut self, toolbox: impl ToolBox + Send + Sync + 'static) {
        self.toolboxes.push(Box::new(toolbox));
    }
}

#[async_trait::async_trait]
impl ToolBox for ToolBoxSet {
    /// Returns a list of all `Tool` instances contained within this ToolBoxSet.
    ///
    /// It aggregates the tool definitions from all the contained toolboxes.
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        let mut all_definitions = Vec::new();
        for toolbox in &self.toolboxes {
            all_definitions.extend(toolbox.tools_definitions()?);
        }
        Ok(all_definitions)
    }

    /// Calls a specific tool by its name with the given parameters.
    ///
    /// It finds the correct `ToolBox` that contains the tool and delegates the call.
    /// If multiple toolboxes contain a tool with the same name, the one that was
    /// added first will be used.
    async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult {
        for toolbox in &self.toolboxes {
            match toolbox
                .call_tool(tool_name.clone(), arguments.clone())
                .await
            {
                Err(ToolError::NoToolFound(_)) => {
                    // No tool in this toolbox, we can check others
                    continue;
                }
                result => {
                    return result;
                }
            };
        }
        Err(ToolError::NoToolFound(tool_name))
    }
}
