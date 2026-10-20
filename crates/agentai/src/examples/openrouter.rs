//! # OpenRouter Example
//!
//! This example uses genai's native OpenRouter adapter with
//! [`Agent::new_with_provider`](crate::Agent::new_with_provider). The adapter already knows
//! OpenRouter's endpoint, so only an API key is needed, and every model name is sent to
//! OpenRouter as is. The agent also gets the built-in date and time tools.
//!
//! You can also use OpenRouter without binding the agent to it: with [`Agent::new`](crate::Agent::new),
//! prefix the model name with the `open_router::` namespace (for example
//! `open_router::nvidia/nemotron-3.5-super-vl:free`) and set the `OPEN_ROUTER_API_KEY`
//! environment variable, which genai reads for this adapter.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! OPENROUTER_API_KEY=<your key> cargo run --example openrouter
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/openrouter.rs")]
//! ```
