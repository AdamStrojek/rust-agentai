//! # Structured Output Example
//!
//! This example introduces the concept of structured output supported by most of LLMs.
//! Structured outputs may not work on all LLMs!
//!
//! To achieve it, you need to create a new deserializable structure which will be holding
//! answer.
//! Such a structure needs to derive from `serde::Deserialize` and `schemars::JsonSchema`.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! cargo run --example struct_output
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/struct_output.rs")]
//! ```
