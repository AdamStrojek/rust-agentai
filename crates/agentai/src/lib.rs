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
//! - **MCP Server Support**: Leverage existing solutions based on the Model-Context-Protocol, eliminating the need to build agent tools from scratch.
//! - **Structured Output**: No need to parse raw text from model, just provide structure, and AI agent will provide response in defined format.
//!
//! ## What's New
//!
//! #### genai 0.7 (version 0.2.0)
//!
//! AgentAI now uses [genai](https://crates.io/crates/genai) 0.7, schemars 1.x and Rust edition 2024 (MSRV 1.88).
//! Models that return text together with tool calls now work: the tool calls are no longer dropped.
//! `Agent::new` and `Agent::new_with_url` now return `Result`, and a tool call without a toolbox
//! is answered with an error message to the model instead of a panic.
//! [`McpToolBox`](crate::tool::mcp::McpToolBox) now uses the official MCP SDK (`rmcp`) and can also
//! connect to running servers over Streamable HTTP.
//! [`Agent::new_with_provider`](crate::Agent::new_with_provider) binds an agent to any genai provider,
//! for example the native OpenRouter adapter (`AdapterKind::OpenRouter`), with an explicit API key.
//! [`Agent::with_max_iterations`](crate::Agent::with_max_iterations) and
//! [`Agent::with_chat_options`](crate::Agent::with_chat_options) configure the loop limit and the model
//! options (temperature, max tokens, tool choice, reasoning effort, ...). The `genai` crate is
//! re-exported as `agentai::genai`.
//! `#[tool(strict)]` marks a tool definition as strict, so supporting providers make the model's
//! arguments match the parameter schema exactly.
//!
//! #### `ToolBox` (version 0.1.5)
//!
//! This release introduces the [`ToolBox`](crate::tool::ToolBox), a new feature providing an easy-to-use interface for supplying tools to AI agents.
//!
//! ## Future Plans
//!
//! We are continuously working to improve AgentAI. Here are some of the features planned for the near future:
//!
//! - **Agent Memory**: Enhance the user experience by adding new functionality for AI agent memory. This will give users control over memory behavior, such as persistence, record limits, and context management for new requests.
//! - **User Input and Streaming Output**: Not every AI agent works silently in the background. Some require additional interaction from the user. This feature will introduce an interface to handle user interactions and provide responses in a streaming format for a better user experience.
//! - **Configurable Behavior**: Introduce a comprehensive way to manage every aspect of an agent's configuration, from model parameters to error-handling behavior for tools.
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
//! use agentai::Agent;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut agent = Agent::new("You are a useful assistant")?;
//!     let answer: String = agent.run("gpt-4o", "Why is the sky blue?", None).await?;
//!     println!("Answer: {}", answer);
//!     Ok(())
//! }
//! ```
//!
//! ## Error handling
//!
//! [`Agent::run`] returns an [`AgentError`] that you can match on. Provider failures that may
//! succeed on a second try report it with [`AgentError::is_transient`]:
//!
//! ```no_run
//! use agentai::{Agent, AgentError};
//!
//! # async fn run(agent: &mut Agent) {
//! match agent.run::<String>("gpt-4o", "Why is the sky blue?", None).await {
//!     Ok(answer) => println!("Answer: {answer}"),
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
/// The [genai](https://crates.io/crates/genai) crate `agentai` is built on, re-exported so you can use
/// its types (`ChatOptions`, `ToolChoice`, `ReasoningEffort`, custom clients, ...) without
/// adding a matching `genai` dependency.
pub use genai;
pub mod tool;

// These modules will be enabled only when generating documentation.
#[cfg(doc)]
pub mod examples;

#[cfg(doc)]
pub mod structured_output;

#[allow(unused_imports)]
pub use agent::*;

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
/// The same toolbox with a valid option compiles:
///
/// ```
/// use agentai::tool::{toolbox, Tool, ToolBox, ToolError, ToolResult};
/// struct T;
/// #[toolbox]
/// impl T {
///     /// Tool
///     #[tool(strict)]
///     fn a(&self) -> ToolResult { Ok(String::new()) }
/// }
/// ```
#[cfg(all(doctest, feature = "macros"))]
pub struct ToolAttributeCompileChecks;
