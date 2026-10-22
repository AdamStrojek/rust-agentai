//! # Event Listener Example
//!
//! This example follows streamed runs with event listeners
//! ([`agentai::listener`](crate::listener)). The first run uses a configured
//! [`Printer`](crate::listener::Printer), the ready-made listener, to write the answer, the tool
//! calls and the tool results to stderr. The second run uses a custom
//! [`EventListener`](crate::listener::EventListener) that keeps its own state, as a GUI would: it
//! collects the answer, logs each tool call with the elapsed time and reports the token usage when
//! the run finishes. Both are passed to [`AgentStream::listen`](crate::AgentStream::listen).
//!
//! To run this example from the terminal, enter:
//! ```bash
//! cargo run --example event_listener
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/event_listener.rs")]
//! ```
