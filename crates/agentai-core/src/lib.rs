//! # AgentAI Core
//!
//! Shared building blocks of the [AgentAI](https://crates.io/crates/agentai) crates.
//!
//! - [`message`] and [`prompt`]: the conversation types (`Message`, `Part`, `Media`,
//!   `ToolOutput`, `Prompt`, ...), re-exported by `agentai`.
//! - [`storage`]: [`MediaStorage`](storage::MediaStorage) (`MediaRead` + `MediaSave`), a
//!   blob storage for images and other files, with [`MemoryStorage`](storage::MemoryStorage).
//! - HTTP: the [`HttpClient`] trait hides the HTTP stack, so crates such as `agentai-image` send
//! their requests without depending on a particular client. [`ReqwestClient`] implements it
//! with `reqwest`; another environment (for example WebAssembly) can provide its own.
//!
//! ## Features
#![doc = document_features::document_features!()]

mod http;
pub mod message;
pub mod prompt;
pub mod storage;

pub use http::*;
