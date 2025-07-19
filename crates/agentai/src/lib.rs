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
//!     let mut agent = Agent::new("You are a useful assistant");
//!     let answer: String = agent.run("gpt-4o", "Why is the sky blue?", None).await?;
//!     println!("Answer: {}", answer);
//!     Ok(())
//! }
//! ```
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

pub mod agent;
pub mod tool;

// These modules will be enabled only when generating documentation.
#[cfg(doc)]
pub mod examples;

#[cfg(doc)]
pub mod structured_output;

#[allow(unused_imports)]
pub use agent::*;
