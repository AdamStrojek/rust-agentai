//! # AgentAI
//!
//! AgentAI is a Rust library designed to simplify the creation of AI agents. It leverages
//! the [GenAI](https://crates.io/crates/genai) library to interface with a wide range of popular
//! Large Language Models (LLMs), making it versatile and powerful. Written in Rust, AgentAI
//! benefits from strong static typing and robust error handling, ensuring more reliable
//! and maintainable code. Whether you're developing simple or complex AI agents, AgentAI provides
//! a streamlined and efficient development process.
//!
//! ## Features
//!
//! - Multi LLM -- we support most of LLM API (OpenAI, Anthropic, Gemini, Ollama and all OpenAI API Compatible)
//! - Choose correct LLM for the task -- use many smaller specialized LLMs to save costs and choose best of all the worlds
//! - Support for MCP Server -- no need to write your own Agent Tools, you can leverage, already existing
//!   solutions, based on Model Context Protocol
//!
//! > **Warning**
//! > This library is under heavy development. The interface can change at any moment without any notice.
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
#![doc = document_features::document_features!()]
//!
//! ## Usage
//! Here is a basic example of how to create an AI agent using AgentAI:
//! ```rust
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
