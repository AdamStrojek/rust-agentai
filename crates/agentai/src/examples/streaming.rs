//! # Streaming Example
//!
//! This example streams a run with [`AgentRun::stream`](crate::AgentRun::stream) and prints
//! the answer as the model writes it. The stream also reports the tool calls the agent makes
//! and their results, and ends with [`AgentEvent::Finished`](crate::AgentEvent::Finished),
//! which holds the same [`RunResult`](crate::RunResult) as an awaited run. The new messages
//! are then added to the history.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! cargo run --example streaming
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/streaming.rs")]
//! ```
