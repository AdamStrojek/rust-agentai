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
//! - [crate::tool::buildin]: Provides a set of useful built-in tools. (Requires the `tools-buildin` feature).
//! - [crate::tool::mcp]: Provides a `ToolBox` for interacting with the MCP Client. (Requires the `mcp-client` feature).
//! - [crate::tool::web]: Provides toolboxes for interacting with the web, such as searching and fetching content. (Requires the `tools-web` feature).
//! - [`agentai-image`](https://docs.rs/agentai-image) (a separate crate, `cargo add agentai-image`): image tools for
//!   OpenAI and compatible APIs.
//!   - `GenerateImageToolBox`: `generate_image` and `edit_image`, returning the image or only its metadata as JSON.
//!   - `ReadMediaToolBox`: `read_media` gives the model a stored file (from any `MediaRead`) by its ID.
//!
//! A tool returns text, or a [`ToolOutput`] with text and images for the model. Tools have no
//! structured output: a tool that produces data (a record, a list, a measurement) converts it to
//! text itself, for example as JSON with `serde_json::to_string`, in the format it wants the
//! model to read.
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

pub use crate::message::ToolOutput;
use serde_json::Value;
use thiserror::Error;

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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Tool {
    /// Name of the tool. It must be unique within the toolbox.
    pub name: String,
    /// What the tool does and when to use it. The model reads it.
    pub description: Option<String>,
    /// JSON schema of the arguments.
    pub schema: Option<serde_json::Value>,
    /// Asks providers that support it to make the arguments match the schema exactly.
    pub strict: Option<bool>,
}

impl Tool {
    /// Creates a tool definition with a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            schema: None,
            strict: None,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the JSON schema of the arguments.
    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Marks the tool as strict, see [`Tool::strict`].
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

impl Tool {
    pub(crate) fn into_genai(self) -> genai::chat::Tool {
        let tool = self;
        let mut out = genai::chat::Tool::new(tool.name);
        out.description = tool.description;
        // Anthropic requires a schema, so a tool without arguments gets an empty object schema
        out.schema = Some(
            tool.schema
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
        );
        out.strict = tool.strict;
        out
    }
}

/// Result of a tool. Tool methods in a [`#[toolbox]`](crate::tool::toolbox) usually return
/// `ToolResult` (text) or `ToolResult<ToolOutput>` (text and images);
/// [`ToolBox::call_tool`] returns `ToolResult<ToolOutput>`.
pub type ToolResult<T = String> = Result<T, ToolError>;

// Re-export tool and toolbox macros, they are used to generate auto implementation of
#[cfg(feature = "macros")]
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
pub trait ToolBox: Send + Sync + 'static {
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
    /// ```ignore
    /// # This is only informational example, it represents what in theory is being generated
    /// let args: ToolArguments = serde_json::from_value(arguments)?;
    /// ```
    /// Replace `ToolArguments` with the actual struct type corresponding to your tool's schema.
    ///
    /// # Arguments
    /// * `tool_name` - The name of the tool to call.
    /// * `arguments` - A JSON `Value` containing the arguments for the tool call.
    ///
    /// # Returns
    /// A `Result` containing the tool's [`ToolOutput`] on success,
    /// or a `ToolError` if the tool call fails or the tool is not found.
    async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult<ToolOutput>;
}

/// Errors that can occur when working with `ToolBox`es and tools.
///
/// Variants fall into two groups, and [`ToolError::is_recoverable`] tells them apart:
///
/// - **Recoverable** errors ([`NotFound`](ToolError::NotFound),
///   [`InvalidArguments`](ToolError::InvalidArguments), [`Failed`](ToolError::Failed)) are
///   something the model can react to, for example by calling the tool again with fixed
///   arguments. [`Agent::run`](crate::agent::Agent::run) sends their message to the model as the
///   tool result and the loop continues.
/// - **Fatal** errors ([`Configuration`](ToolError::Configuration), [`Fatal`](ToolError::Fatal))
///   cannot be fixed by calling again, for example a missing API key or a dead MCP server.
///   `Agent::run` stops and returns them as [`AgentError::Tool`](crate::agent::AgentError::Tool).
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ToolError {
    // -- Recoverable: sent to the model as the tool result; the loop continues
    /// The requested tool does not exist in this `ToolBox`. This is the signal for
    /// [`ToolBoxSet`] to try the next toolbox.
    ///
    /// Reserved for unknown tool names: a tool that fails for its own reasons should return
    /// [`InvalidArguments`](ToolError::InvalidArguments) or [`Failed`](ToolError::Failed),
    /// otherwise an outer `ToolBoxSet` silently tries the next toolbox instead.
    #[error("tool '{0}' not found")]
    NotFound(String),
    /// Arguments are missing, malformed or invalid (wrong timezone, bad date format, ...).
    /// The message is passed to the model, so it should explain what to fix.
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    /// The tool ran but failed in a way the model can react to (HTTP 404, an MCP `isError`
    /// result, ...). The message is passed to the model.
    #[error("{0}")]
    Failed(String),

    // -- Fatal: `Agent::run` stops and returns `AgentError::Tool`
    /// Configuration problem that retrying cannot fix: a missing API key, a server that is
    /// not running, tool definitions that are not ready.
    #[error("tool is not configured correctly: {0}")]
    Configuration(String),
    /// Any other unexpected error.
    #[error(transparent)]
    Fatal(Box<dyn std::error::Error + Send + Sync>),
}

impl ToolError {
    /// Creates a [`ToolError::Fatal`] from any error.
    pub fn fatal(err: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Fatal(err.into())
    }

    /// Returns `true` when the model can react to the error, so
    /// [`Agent::run`](crate::agent::Agent::run) sends it back as the tool result and continues.
    /// Returns `false` for errors that end the run.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::NotFound(_) | Self::InvalidArguments(_) | Self::Failed(_)
        )
    }
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
    pub fn add_tool(&mut self, toolbox: impl ToolBox) {
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
    async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult<ToolOutput> {
        for toolbox in &self.toolboxes {
            match toolbox
                .call_tool(tool_name.clone(), arguments.clone())
                .await
            {
                Err(ToolError::NotFound(_)) => {
                    // No tool in this toolbox, we can check others
                    continue;
                }
                result => {
                    return result;
                }
            };
        }
        Err(ToolError::NotFound(tool_name))
    }
}
