//! # AgentAI
//!
//! AgentAI is a Rust library designed to simplify the creation of AI agents. It leverages
//! the [GenAI](https://crates.io/crates/genai) library to interface with a wide range of popular
//! Large Language Models (LLMs), making it versatile and powerful. Written in Rust, AgentAI
//! benefits from strong static typing and robust error handling, ensuring reliable
//! and maintainable code. Whether you're developing simple or complex AI agents, AgentAI provides
//! a streamlined and efficient development process.
//!
//! ## Warning
//! This library is under heavy development. The interface may change at any time without notice.
//!
//! ## Features
//!
//! - **Connect to any major LLM provider**: Support for OpenAI, Anthropic, Gemini, Ollama, and other OpenAI-compatible APIs.
//! - **Choose the right model for the job**: Flexibly select the best-suited model for each step in your agent's workflow.
//! - **Build custom tools with ease**: A simple interface for creating and managing your own tools using the [`ToolBox`](crate::tool::ToolBox).
//! - **MCP Server Support, local and remote**: Use tools from any Model Context Protocol server, whether you start it locally over **stdio** or connect to a running or hosted server over **Streamable HTTP** (with optional bearer token). Built on the official MCP Rust SDK.
//! - **Structured Output**: No need to parse raw text from model, just provide structure, and AI agent will provide response in defined format.
//! - **Streaming**: Show the answer as the model writes it, together with the tool calls the agent makes. The stream is `Send + 'static`, ready for spawned tasks and server-sent events.
//! - **You own the conversation**: An agent is shareable configuration. You keep the history, decide what to store, trim or persist, and get token usage and the stop reason from every run.
//!
//! ## What's New
//!
//! #### 0.3.0 - Streaming, conversation history in your hands, run details
//!
//! - **Streaming**: [`Agent::run_stream`](crate::Agent::run_stream) takes the same configuration as
//!   [`Agent::run`](crate::Agent::run), and awaiting it returns a stream of [`AgentEvent`](crate::AgentEvent)s:
//!   text and reasoning as the model writes them, each tool call and its result, and finally the result of
//!   the run. Runs with tool calls and structured output stream too. The stream is `Send + 'static`, so you
//!   can spawn it or return it from a web handler.
//! - **You manage the history**: the agent no longer keeps the conversation. Pass your history with
//!   `.with_history(&history)` and add the run's [`new_messages`](crate::AgentResult::new_messages) to it when you
//!   want to continue: `history.extend(result.new_messages)`. Store, trim, persist or reset it however you
//!   like: [`Message`](crate::Message) has its own stable JSON format. A failed or cancelled run never changes
//!   your history, and the system prompt is not part of it: the agent sends it with every request.
//! - **One agent, many conversations**: the model and toolbox now belong to the agent
//!   ([`Agent::with_model`](crate::Agent::with_model), [`Agent::with_toolbox`](crate::Agent::with_toolbox)), and
//!   running takes `&self`, so one agent can be cloned cheaply and shared between tasks. Change the model,
//!   toolbox, iteration limit or options for a single run with the `with_*` methods of
//!   [`AgentRequest`](crate::AgentRequest) or a reusable [`RunOptions`](crate::RunOptions).
//! - **Run details**: awaiting a run returns a [`AgentResult`](crate::AgentResult) with the answer, token usage summed
//!   over all model calls, the stop reason, the model's reasoning and the number of model calls. An answer cut
//!   off by the token limit is reported as [`AgentError::MaxTokens`](crate::AgentError::MaxTokens) instead of a
//!   confusing parse error.
//!
//! **Breaking changes**: `agent.run(model, prompt, toolbox)` is replaced by
//! `agent.run(prompt).with_history(&history).await`, which returns an `AgentResult` instead of the bare answer;
//! the model and toolbox are set with `Agent::with_model` and `Agent::with_toolbox` (or per run), and the agent
//! no longer remembers earlier runs. `Agent::new_with_provider` takes a `Provider`, and `Agent::new_with_client`
//! is removed. Read the [CHANGELOG](https://github.com/AdamStrojek/rust-agentai/blob/master/CHANGELOG.md) before
//! upgrading: it lists the steps to migrate from 0.2.
//!
//! #### 0.2.0 - Agent providers, remote MCP and configuration
//!
//! - **Creating agents**: [`Agent::new_with_provider`](crate::Agent::new_with_provider) binds an agent to any
//!   genai provider with an explicit API key, for example the native OpenRouter adapter, which needs no base URL:
//!   `Agent::new_with_provider(AdapterKind::OpenRouter, None, &api_key, system)?`. `Agent::new` and
//!   `Agent::new_with_url` now return `Result`, and the agent reports typed errors
//!   ([`AgentError`](crate::AgentError)). Tool errors are either recoverable (sent back to the model, which can
//!   try again) or fatal (they stop the run).
//! - **Remote MCP servers, not only stdio**: [`McpToolBox`](crate::tool::mcp::McpToolBox) now connects to MCP
//!   servers over **Streamable HTTP**, so agents can use running and hosted servers, including ones that need a
//!   bearer token (`McpToolBox::new_http`, `McpToolBox::new_http_with_bearer`). Starting a local server over
//!   **stdio** (`McpToolBox::new`) works as before. MCP support is now built on `rmcp`, the official MCP Rust SDK.
//! - **Agent configuration**: [`Agent::with_max_iterations`](crate::Agent::with_max_iterations) and
//!   [`Agent::with_chat_options`](crate::Agent::with_chat_options) set the loop limit and the model options
//!   (temperature, max tokens, tool choice, reasoning effort, ...).
//! - **Strict tools**: `#[tool(strict)]` marks a tool as strict, so providers that support it make the model's
//!   arguments match the parameter schema exactly.
//! - **Dependencies**: genai 0.7 (re-exported as `agentai::genai`; tool calls are no longer lost when a model
//!   also returns text), schemars 1, reqwest 0.13, syn 3, Rust edition 2024 and MSRV 1.88.
//!
//! **Breaking changes**: `Agent::new`, `Agent::new_with_url` and `Agent::run` return `AgentError` instead of
//! `anyhow` errors, `ToolError` variants are renamed and split, `Tool` is built with `Tool::new`, and answer
//! and tool parameter types need `schemars` 1.x. Read the
//! [CHANGELOG](https://github.com/AdamStrojek/rust-agentai/blob/master/CHANGELOG.md) before upgrading: it lists
//! every breaking change and how to migrate.
//!
//! #### 0.1.5 - `ToolBox`
//!
//! This release introduces the [`ToolBox`](crate::tool::ToolBox), a new feature providing an easy-to-use interface for supplying tools to AI agents.
//!
//! ## Future Plans
//!
//! We are continuously working to improve AgentAI. These are the next features we plan; the linked issues have the details:
//!
//! - **Agent Memory**: control what the agent remembers between runs, such as persistence, record limits and context management.
//! - **User input during a run**: let an agent ask the user for more information or confirmation before it continues.
//!
//! ## Installation
//!
//! To add the AgentAI crate to your project, run the following command in your project's root directory:
//!
//! ```bash
//! cargo add agentai
//! ```
//!
//! This command adds the crate and its dependencies to your project.
//!
//! ## Feature Flags
//! <!-- FEATURE FLAGS -->
#![doc = document_features::document_features!()]
//!
//! ## Usage
//!
//! Here is a basic example of how to create an AI agent using AgentAI:
//!
//! ```no_run
//! use agentai::{Agent, Message, AgentResult};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let agent = Agent::new("You are a useful assistant")?.with_model("gpt-4o");
//!
//!     // The conversation history is yours
//!     let mut history: Vec<Message> = Vec::new();
//!
//!     let result: AgentResult<String> = agent.run("Why is the sky blue?").with_history(&history).await?;
//!     println!("Answer: {}", result.answer);
//!     // Keep the question and the answer to continue the conversation
//!     history.extend(result.new_messages);
//!     Ok(())
//! }
//! ```
//!
//! To show the answer while it is written, use `run_stream` with the same configuration:
//!
//! ```no_run
//! use agentai::{Agent, AgentEvent};
//! use futures_util::StreamExt;
//!
//! # async fn run(agent: Agent) -> Result<(), agentai::AgentError> {
//! let mut stream = agent.run_stream::<String>("Why is the sky blue?").await?;
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         AgentEvent::TextDelta(text) => print!("{text}"),
//!         AgentEvent::Finished(result) => println!("\n{} tokens", result.usage.total_tokens),
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Error handling
//!
//! A run returns an [`AgentError`] that you can match on. Provider failures that may
//! succeed on a second try report it with [`AgentError::is_transient`]. Your history is not
//! changed by a failed run, so you can simply try again:
//!
//! ```no_run
//! use agentai::{Agent, AgentError};
//!
//! # async fn run(agent: &Agent) {
//! match agent.run::<String>("Why is the sky blue?").await {
//!     Ok(result) => println!("Answer: {}", result.answer),
//!     Err(err) if err.is_transient() => println!("Try again later: {err}"),
//!     Err(AgentError::InvalidAnswer { raw, .. }) => println!("Unexpected answer: {raw}"),
//!     Err(err) => println!("Failed: {err}"),
//! }
//! # }
//! ```
//!
//! Tools report failures with [`ToolError`](crate::tool::ToolError). Recoverable errors
//! (`NotFound`, `InvalidArguments`, `Failed`) are sent to the model as the tool result and the
//! run continues. Fatal errors (`Configuration`, `Fatal`) stop the run with
//! [`AgentError::Tool`]. [`ToolError::is_recoverable`](crate::tool::ToolError::is_recoverable)
//! tells the two groups apart.
//!
//! ## Examples
//!
#![allow(rustdoc::redundant_explicit_links)]
//! For more examples, check out the [examples](crate::examples) directory. To run an example, use the following command, replacing `<example_name>` with the name of the example file (without the `.rs` extension):
//!
//! ```bash
//! cargo run --example <example_name>
//! ```
//!
//! For instance, to run the `simple` example:
//!
//! ```bash
//! cargo run --example simple
//! ```

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/AdamStrojek/rust-agentai/master/assets/logo-docs.png",
    html_favicon_url = "https://raw.githubusercontent.com/AdamStrojek/rust-agentai/master/assets/favicon.png"
)]

pub mod agent;
pub mod message;
pub mod tool;

// These modules will be enabled only when generating documentation.
#[cfg(doc)]
pub mod examples;

#[cfg(doc)]
pub mod structured_output;

#[allow(unused_imports)]
pub use agent::*;
pub use message::{Media, MediaSource, Message, Part, ProviderData, Role, ToolCall, ToolResponse};

/// Compile-fail checks for the `#[tool(...)]` options. Only built for doc tests.
///
/// An unknown option is rejected:
///
/// ```compile_fail
/// use agentai::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
/// struct T;
/// #[toolbox]
/// impl T {
///     /// Tool
///     #[tool(unknown)]
///     fn a(&self) -> ToolResult { Ok(String::new()) }
/// }
/// ```
///
/// `strict` takes only a boolean:
///
/// ```compile_fail
/// use agentai::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
/// struct T;
/// #[toolbox]
/// impl T {
///     /// Tool
///     #[tool(strict = "yes")]
///     fn a(&self) -> ToolResult { Ok(String::new()) }
/// }
/// ```
///
/// `strict` can be given only once:
///
/// ```compile_fail
/// use agentai::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
/// struct T;
/// #[toolbox]
/// impl T {
///     /// Tool
///     #[tool(strict, strict = false)]
///     fn a(&self) -> ToolResult { Ok(String::new()) }
/// }
/// ```
///
/// Malformed options are reported instead of being ignored (missing comma here):
///
/// ```compile_fail
/// use agentai::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
/// struct T;
/// #[toolbox]
/// impl T {
///     /// Tool
///     #[tool(name = "a" strict)]
///     fn a(&self) -> ToolResult { Ok(String::new()) }
/// }
/// ```
///
/// The same toolbox with valid options compiles:
///
/// ```
/// use agentai::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
/// struct T;
/// #[toolbox]
/// impl T {
///     /// Tool
///     #[tool(name = "a", strict)]
///     fn a(&self) -> ToolResult { Ok(String::new()) }
/// }
/// ```
#[cfg(all(doctest, feature = "macros"))]
pub struct ToolAttributeCompileChecks;
