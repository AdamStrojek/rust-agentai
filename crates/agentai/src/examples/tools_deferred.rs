//! # Deferred Tool Execution
//!
//! This example shows how to write a background tool with [`ToolReturn::deferred`](crate::tool::ToolReturn::deferred)
//! and run it with [`TokioDeferredTaskToolBox`](crate::tool::TokioDeferredTaskToolBox).
//!
//! To run this example from the terminal, enter:
//! ```bash
//! GEMINI_API_KEY="your-key" cargo run --example tools_deferred
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/tools_deferred.rs")]
//! ```
