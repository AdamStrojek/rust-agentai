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
//! returns the [`AgentResult`](crate::AgentResult). For async writers there are
//! [`AgentStream::print_to_async`](crate::AgentStream::print_to_async) (`futures::io::AsyncWrite`)
//! and [`AgentStream::print_to_tokio`](crate::AgentStream::print_to_tokio) (`tokio::io::AsyncWrite`).
//! To change what is written, pass a configured [`Printer`](crate::listener::Printer) to
//! [`AgentStream::listen`](crate::AgentStream::listen); to show the run somewhere else, for example
//! in a GUI, implement [`EventListener`](crate::listener::EventListener).
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
