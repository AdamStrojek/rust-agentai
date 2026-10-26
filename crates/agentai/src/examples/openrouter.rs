//! # OpenRouter Example
//!
//! This example uses OpenRouter with [`Agent::new`](crate::Agent::new): the `open_router::`
//! namespace in the model name (for example `open_router::nvidia/nemotron-3.5-super-vl:free`)
//! sends the model to OpenRouter, and the API key is read from the `OPEN_ROUTER_API_KEY`
//! environment variable. The agent also gets the built-in date and time tools.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! OPEN_ROUTER_API_KEY=<your key> cargo run --example openrouter
//! ```
//!
//! The example needs an OpenRouter key, so it is not run with the other examples by
//! `cargo test`, only compiled.
//!
//! ## Source Code
//!
//! ```rust,no_run
#![doc = include_str!("../../examples/openrouter.rs")]
//! ```
