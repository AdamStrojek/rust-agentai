//! # Simple Example
//!
//! Here is a very simple example of how to use AgentAI crate. Here we only initialize
//! LLM client, and later use it to answer a simple question and a follow-up question.
//! The agent does not remember the conversation: the example keeps the history and adds
//! [`RunResult::new_messages`](crate::RunResult::new_messages) to it after every run.
//!
//! Remember to provide return type for answer, it is very important, so a library
//! will knows do you expect structured output or just simple String.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! cargo run --example simple
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/simple.rs")]
//! ```
