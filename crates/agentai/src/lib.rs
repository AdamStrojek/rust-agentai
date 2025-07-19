//! # AgentAI
//!
//! AgentAI is a Rust library designed to simplify the creation of AI agents. It leverages
//! the [GenAI](https://crates.io/crates/genai) library to interface with a wide range of popular
//! Large Language Models (LLMs), making it versatile and powerful. Written in Rust, AgentAI
//! benefits from strong static typing and robust error handling, ensuring more reliable
//! and maintainable code. Whether you're developing simple or complex AI agents, AgentAI provides
//! a streamlined and efficient development process.
//!
//! ## Warning
//!
//! This library is under heavy development. The interface can change at any moment without any notice.
//!
//! ## Features
//!
//! - Use any major LLM API provider -- we support OpenAI, Anthropic, Gemini, Ollama and other OpenAI API Compatible.
//! - You decide what model to use -- depending on step in agentic flow you can choose model that suits best!
//! - Create your own tools with ease using [`ToolBox`es](crate::tool).
//! - Support for MCP Server -- no need to write your own Agent Tools, you can leverage, already existing
//!   solutions, based on Model Context Protocol.
//!
//! ## What's New
//!
//! #### `ToolBox` (version 0.1.5)
//!
//! This release introduces the [`ToolBox`](crate::tool::ToolBox), a new feature that provides a easy-to-use interface for providing tools to AI agents.
//!
//! ## Future Plans
//!
//! We are continuously working on improving AgentAI. Here are some of the features we are planning to introduce in the near future:
//!
//! - **Agent Memory** -- improve experiance and add new functionality around AI agent memory. Currently. user can't manage how memory should
//!   behave, should it be stored, maybe limit amount of records, maybe each request should be started with clean
//! - **User Input and Streaming Output** -- not every AI agent works silently in the background. Some requires additional interaction from user,
//!   also returning response in streaming format improves comfort. This feature will introduce interface to enable that.
//! - **Configurable Behaviour** -- introduce way of managing every aspect of agent configuration, starting from providing model parameters
//!   to how behave on encountering errors from tools.
//!
//! ## Installation
//! To start using AgentAI crate just enter in root directory for your project this command:
//!
//! ```bash
//! cargo add agentai
//! ```
//!
//! This will install this crate with all required dependencies.
//!
//! ## Feature flags
//! <!-- FEATURE FLAGS -->
#![doc = document_features::document_features!()]
//!
//! ## Usage
//! Here is a basic example of how to create an AI agent using AgentAI:
//! ```no_run
//! use agentai::Agent;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut agent = Agent::new("You are a useful assistant");
//!     let answer: String = agent.run("gpt-4o", "Why sky is blue?", None).await?;
//!     println!("Answer: {}", answer);
//!     Ok(())
//! }
//! ```
//!
//!## Examples
//!
#![allow(rustdoc::redundant_explicit_links)]
//! For more examples, check out the [examples](crate::examples) directory. You can build and run them using Cargo with the following command:
//!
//! ```bash
//! cargo run --example <example_name>
//! ```
//!
//! The <example_name> should match the filename of the example you want to run (without the file extension).
//! For example, to run the example that includes the essential parts required to implement an AI agent, use:
//!
//! ```bash
//! cargo run --example simple
//! ```

pub mod agent;
pub mod tool;

// This modules will be enabled only when generating documentation
#[cfg(doc)]
pub mod examples;

#[cfg(doc)]
pub mod structured_output;

#[allow(unused_imports)]
pub use agent::*;
