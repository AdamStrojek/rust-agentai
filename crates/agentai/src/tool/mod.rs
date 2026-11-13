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
//! - [`agentai-audio`](https://docs.rs/agentai-audio) (a separate crate, `cargo add agentai-audio`): speech tools for
//!   OpenAI and compatible APIs, ElevenLabs and oMLX.
//!   - `SpeechToolBox`: `generate_speech` and `list_voices`, returning the metadata of the audio as JSON or the audio itself.
//!
//! A tool returns text, or a [`ToolOutput`] with text and images for the model, for example an
//! image read from disk (`Media::from_bytes`). The model sees the images in a user message right
//! after the tool results (a later release puts them in the tool result itself for the providers
//! that accept it). Tools have no
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

#[cfg(feature = "tokio")]
pub mod deferred;

#[cfg(feature = "mcp-client")]
pub mod mcp;

#[cfg(feature = "tools-web")]
pub mod web;

#[cfg(feature = "tokio")]
pub use deferred::TokioDeferredTaskToolBox;

use crate::message::{Media, ToolCall};
pub use crate::message::ToolOutput;
use futures_util::future::BoxFuture;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
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
/// `ToolResult` (text), `ToolResult<ToolOutput>` (text and images) or `ToolResult<ToolReturn>`
/// (for deferred tools); [`ToolBox::call_tool`] returns `ToolResult<ToolReturn>`.
pub type ToolResult<T = String> = Result<T, ToolError>;

/// What a tool returns from [`ToolBox::call_tool`]: either an immediate [`ToolOutput`]
/// ([`ToolReturn::Done`]) or an immediate response message paired with a background task
/// ([`ToolReturn::Deferred`]).
///
/// [`ToolOutput`], `String`, `&str` and [`Media`] convert into [`ToolReturn::Done`], so tools
/// in a [`#[toolbox]`](crate::tool::toolbox) that return `ToolResult` or `ToolResult<ToolOutput>`
/// work without changes.
#[non_exhaustive]
pub enum ToolReturn {
    /// The tool finished immediately with this output.
    Done(ToolOutput),
    /// The tool started a background task. `message` is sent to the model right away as the
    /// [`ToolResponse`](crate::ToolResponse) for this [`ToolCall`], and `task` is yielded by the
    /// agent as a [`DeferredTask`] so the caller (for example `TokioDeferredTaskToolBox`) can
    /// spawn it.
    Deferred {
        /// Immediate response text sent to the model for this tool call.
        message: String,
        /// Unstarted background future that produces the final [`ToolOutput`].
        task: BoxFuture<'static, ToolResult<ToolOutput>>,
    },
}

impl ToolReturn {
    /// Creates an immediate [`ToolReturn::Done`] from anything that converts into [`ToolOutput`].
    pub fn done(output: impl Into<ToolOutput>) -> Self {
        Self::Done(output.into())
    }

    /// Creates a [`ToolReturn::Deferred`] with an immediate message for the model and an
    /// unstarted background future.
    pub fn deferred(
        message: impl Into<String>,
        task: impl std::future::Future<Output = ToolResult<ToolOutput>> + Send + 'static,
    ) -> Self {
        Self::Deferred {
            message: message.into(),
            task: Box::pin(task),
        }
    }
}

impl fmt::Debug for ToolReturn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done(output) => f.debug_tuple("Done").field(output).finish(),
            Self::Deferred { message, .. } => f
                .debug_struct("Deferred")
                .field("message", message)
                .finish_non_exhaustive(),
        }
    }
}

impl From<ToolOutput> for ToolReturn {
    fn from(output: ToolOutput) -> Self {
        Self::Done(output)
    }
}

impl From<String> for ToolReturn {
    fn from(text: String) -> Self {
        Self::Done(ToolOutput::from(text))
    }
}

impl From<&str> for ToolReturn {
    fn from(text: &str) -> Self {
        Self::Done(ToolOutput::from(text))
    }
}

impl From<Media> for ToolReturn {
    fn from(media: Media) -> Self {
        Self::Done(ToolOutput::from(media))
    }
}

#[derive(Debug)]
enum TaskState {
    Running,
    Finished(ToolResult<ToolOutput>),
    Taken,
    Cancelled,
}

#[derive(Debug)]
struct SharedTaskState(Mutex<TaskState>);

impl SharedTaskState {
    fn lock(&self) -> MutexGuard<'_, TaskState> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
struct CancelOnDrop(Arc<SharedTaskState>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        if matches!(*state, TaskState::Running) {
            *state = TaskState::Cancelled;
        }
    }
}

/// An unstarted background task returned by a tool via [`ToolReturn::Deferred`].
///
/// The agent yields each `DeferredTask` as [`AgentEvent::DeferredTask`](crate::agent::AgentEvent::DeferredTask)
/// during [`Agent::run_stream`](crate::agent::Agent::run_stream) (or collects it in
/// `AgentResult::unstarted_deferred_tasks` when not streaming), while keeping the matching
/// [`DeferredTaskHandle`] to track completion without depending on an async runtime.
pub struct DeferredTask {
    /// Id of the [`ToolCall`] that created this task.
    pub id: String,
    /// Name of the tool that created this task.
    pub tool: String,
    task: BoxFuture<'static, ToolResult<ToolOutput>>,
    shared: Arc<SharedTaskState>,
    _guard: CancelOnDrop,
}

impl DeferredTask {
    /// Creates a linked `(DeferredTask, DeferredTaskHandle)` pair for `call`.
    pub fn new(
        call: &ToolCall,
        task: BoxFuture<'static, ToolResult<ToolOutput>>,
    ) -> (Self, DeferredTaskHandle) {
        let shared = Arc::new(SharedTaskState(Mutex::new(TaskState::Running)));
        let handle = DeferredTaskHandle {
            id: call.id.clone(),
            tool: call.name.clone(),
            shared: shared.clone(),
        };
        let deferred = Self {
            id: call.id.clone(),
            tool: call.name.clone(),
            task,
            _guard: CancelOnDrop(shared.clone()),
            shared,
        };
        (deferred, handle)
    }

    /// Returns a [`DeferredTaskHandle`] sharing this task's state.
    pub fn handle(&self) -> DeferredTaskHandle {
        DeferredTaskHandle {
            id: self.id.clone(),
            tool: self.tool.clone(),
            shared: self.shared.clone(),
        }
    }

    /// Runs the deferred task to completion and stores its result in the shared state.
    pub async fn run(self) {
        let result = self.task.await;
        *self.shared.lock() = TaskState::Finished(result);
    }
}

impl fmt::Debug for DeferredTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeferredTask")
            .field("id", &self.id)
            .field("tool", &self.tool)
            .finish_non_exhaustive()
    }
}

/// Runtime-neutral handle that tracks the state and result of a [`DeferredTask`].
///
/// Returned in `AgentResult::deferred_tasks` for tasks that are still running when a run ends,
/// and passed to a later run so the agent can notify the model once they complete.
#[derive(Debug, Clone)]
pub struct DeferredTaskHandle {
    /// Id of the [`ToolCall`] that created this task.
    pub id: String,
    /// Name of the tool that created this task.
    pub tool: String,
    shared: Arc<SharedTaskState>,
}

impl DeferredTaskHandle {
    /// Returns `true` once the matching [`DeferredTask`] is no longer running (completed, consumed, or cancelled).
    pub fn is_finished(&self) -> bool {
        !matches!(*self.shared.lock(), TaskState::Running)
    }

    /// Returns `true` when the [`DeferredTask`] was dropped or panicked before completing.
    pub fn is_cancelled(&self) -> bool {
        matches!(*self.shared.lock(), TaskState::Cancelled)
    }

    /// Returns `true` when the completed result has already been taken.
    pub fn is_consumed(&self) -> bool {
        matches!(*self.shared.lock(), TaskState::Taken)
    }

    /// Takes the completed result out of the handle, marking it as consumed.
    ///
    /// Returns `None` if the task is still running, was cancelled, or its result was already taken.
    pub fn take_result(&self) -> Option<ToolResult<ToolOutput>> {
        let mut state = self.shared.lock();
        if matches!(*state, TaskState::Finished(_)) {
            let TaskState::Finished(result) = std::mem::replace(&mut *state, TaskState::Taken)
            else {
                unreachable!()
            };
            Some(result)
        } else {
            None
        }
    }

    pub(crate) fn status_str(&self) -> &'static str {
        match &*self.shared.lock() {
            TaskState::Running => "running",
            TaskState::Finished(Ok(_)) => "completed",
            TaskState::Finished(Err(_)) => "failed",
            TaskState::Taken => "consumed",
            TaskState::Cancelled => "cancelled",
        }
    }
}

/// Replaces arguments that the model sent as a string with JSON inside, where the schema wants
/// an array or an object, with the parsed value.
///
/// Models, local ones especially, often send `"tags": "[\"a\", \"b\"]"` instead of
/// `"tags": ["a", "b"]`. For every top-level argument that is a string, whose schema in
/// `schema` (the tool's JSON schema, `properties` with `$ref`, `anyOf`, `oneOf` and `allOf`
/// followed) allows an array or an object but not a string, and that parses as JSON of an
/// allowed type, the string is replaced. Every other argument is left as it is, so a wrong one
/// still fails to deserialize.
///
/// Code generated by [`#[toolbox]`](crate::tool::toolbox) calls it before deserializing the
/// arguments, unless `parse_json_strings = false` is set. Call it in a hand-written
/// [`ToolBox`] for the same behavior.
pub fn parse_json_string_arguments(arguments: &mut serde_json::Value, schema: &serde_json::Value) {
    let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };
    let Some(arguments) = arguments.as_object_mut() else {
        return;
    };
    for (name, value) in arguments.iter_mut() {
        let serde_json::Value::String(text) = value else {
            continue;
        };
        let Some(property) = properties.get(name) else {
            continue;
        };
        let mut types = SchemaTypes::default();
        types.collect(property, schema, 0);
        if types.string || !(types.array || types.object) {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(parsed @ serde_json::Value::Array(_)) if types.array => *value = parsed,
            Ok(parsed @ serde_json::Value::Object(_)) if types.object => *value = parsed,
            _ => {}
        }
    }
}

/// JSON types a schema allows, as far as [`parse_json_string_arguments`] needs to know.
#[derive(Default)]
struct SchemaTypes {
    string: bool,
    array: bool,
    object: bool,
}

impl SchemaTypes {
    fn collect(&mut self, schema: &serde_json::Value, root: &serde_json::Value, depth: usize) {
        // Recursive `$ref`s end here
        if depth > 16 {
            return;
        }
        match schema.get("type") {
            Some(serde_json::Value::String(name)) => self.add(name),
            Some(serde_json::Value::Array(names)) => names
                .iter()
                .filter_map(|n| n.as_str())
                .for_each(|n| self.add(n)),
            _ => {}
        }
        // A string enum without `type`
        if schema.get("enum").is_some_and(|values| {
            values
                .as_array()
                .is_some_and(|v| v.iter().any(serde_json::Value::is_string))
        }) || schema
            .get("const")
            .is_some_and(serde_json::Value::is_string)
        {
            self.string = true;
        }
        if let Some(target) = schema
            .get("$ref")
            .and_then(serde_json::Value::as_str)
            .and_then(|r| r.strip_prefix('#'))
            .and_then(|pointer| root.pointer(pointer))
        {
            self.collect(target, root, depth + 1);
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(variants) = schema.get(key).and_then(serde_json::Value::as_array) {
                for variant in variants {
                    self.collect(variant, root, depth + 1);
                }
            }
        }
    }

    fn add(&mut self, name: &str) {
        match name {
            "string" => self.string = true,
            "array" => self.array = true,
            "object" => self.object = true,
            _ => {}
        }
    }
}

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

    /// Runs the tool call the model made.
    ///
    /// This method is the entry point for executing a tool's functionality. It is typically
    /// invoked internally by the [`Agent`](crate::agent::Agent) structure when the language
    /// model determines that a tool needs to be called.
    ///
    /// The [`ToolCall`] holds the tool name, the arguments as a `serde_json::Value` (they follow
    /// the JSON schema from `tools_definitions`, so you can deserialize them into the same
    /// struct with [`serde`](https://crates.io/crates/serde)) and the call `id`. The id is the
    /// one the [`ToolResponse`](crate::ToolResponse) and the agent events
    /// ([`AgentEvent::ToolCall`](crate::agent::AgentEvent::ToolCall),
    /// [`AgentEvent::ToolResult`](crate::agent::AgentEvent::ToolResult)) carry, so a tool can
    /// use it to tell calls apart, for example two parallel calls with the same arguments.
    ///
    /// ```ignore
    /// // What the generated code does, in short
    /// let args: ToolArguments = serde_json::from_value(call.arguments.clone())?;
    /// ```
    ///
    /// # Returns
    /// The tool's [`ToolReturn`] on success, or a `ToolError` if the tool call fails or the
    /// tool is not found ([`ToolError::NotFound`]).
    async fn call_tool(&self, call: &ToolCall) -> ToolResult<ToolReturn>;
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
    async fn call_tool(&self, call: &ToolCall) -> ToolResult<ToolReturn> {
        for toolbox in &self.toolboxes {
            match toolbox.call_tool(call).await {
                Err(ToolError::NotFound(_)) => {
                    // No tool in this toolbox, we can check others
                    continue;
                }
                result => {
                    return result;
                }
            };
        }
        Err(ToolError::NotFound(call.name.clone()))
    }
}
