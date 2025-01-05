//! # AgentAI
//!
//! AgentAI is a Rust library designed to simplify the creation of AI agents. It leverages
//! the [GenAI](https://crates.io/crates/genai) library to interface with a wide range of popular
//! Large Language Models (LLMs), making it versatile and powerful. Written in Rust, AgentAI
//! benefits from strong static typing and robust error handling, ensuring more reliable
//! and maintainable code. Whether you're developing simple or complex AI agents, AgentAI provides
//! a streamlined and efficient development process.
//!
//! > **Warning**
//! > This library is under heavy development. The interface can change at any moment without any notice.
//!
//! ## Installation
//! In your project add the following to your Cargo.toml file:
//!
//! ```toml
//! [dependencies]
//! genai = "0.1.15"
//! agentai = "0.1.1"
//!
//! # Suggested additional dependencies:
//! anyhow = "1"
//! tokio = { version = "1.42.0", features = ["full"] }
//! ```
//! _TODO: Change installation method after release_
//!
//! ## Usage
//! Here is a basic example of how to create an AI agent using AgentAI:
//! ```rust
//! use agentai::Agent;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = genai::Client::default();
//!     let mut agent = Agent::new(&client, "You are a useful assistant", &());
//!     let answer: String = agent.run("gpt-4o", "Why sky is blue?").await?;
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

// This module will be enabled only when generating documentation
#[cfg(doc)]
pub mod examples;

#[allow(unused_imports)]
pub use agent::*;

#[allow(unused_imports)]
pub use tool::*;
