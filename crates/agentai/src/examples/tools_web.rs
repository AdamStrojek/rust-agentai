//! # Agent Tools and Web Search
//!
//! This example demonstrates how to update scope of LLM context using
//! tools that have access to the web. To be able to provide two
//! tools we are using `ToolBoxSet` that can combine multiple tools.
//!
//! This example expects `BRAVE_API_KEY` env variable to be exported.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! cargo run --example tools_search
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/tools_web.rs")]
//! ```
