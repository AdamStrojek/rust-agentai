//! # AgentAI Core
//!
//! Shared building blocks of the [AgentAI](https://crates.io/crates/agentai) crates.
//!
//! The [`HttpClient`] trait hides the HTTP stack, so crates such as `agentai-image` send
//! their requests without depending on a particular client. [`ReqwestClient`] implements it
//! with `reqwest`; another environment (for example WebAssembly) can provide its own.
//!
//! ## Features
#![doc = document_features::document_features!()]

mod http;

pub use http::*;
