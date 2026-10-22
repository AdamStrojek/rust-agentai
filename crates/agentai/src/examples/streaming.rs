//! # Streaming Example
//!
//! This example streams a run with [`Agent::run_stream`](crate::Agent::run_stream) and prints
//! the answer as the model writes it. The stream also reports the tool calls the agent makes
//! and their results, and ends with [`AgentEvent::Finished`](crate::AgentEvent::Finished),
//! which holds the same [`AgentResult`](crate::AgentResult) as an awaited run. The new messages
//! are then added to the history.
//!
//! To write the stream to the console without handling the events yourself, use
//! [`AgentStream::print_to`](crate::AgentStream::print_to): it writes the text as it arrives and
//! each tool call on its own line to stdout, stderr, a file or any other `std::io::Write`, and
//! returns the [`AgentResult`](crate::AgentResult).
//! [`AgentStream::print_to_with`](crate::AgentStream::print_to_with) also chooses what is written
//! ([`PrintOptions`](crate::PrintOptions)).
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
