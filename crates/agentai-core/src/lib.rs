//! # AgentAI Core
//!
//! Shared building blocks of the [AgentAI](https://crates.io/crates/agentai) crates.
//!
//! - [`message`] and [`prompt`]: the conversation types (`Message`, `Part`, `Media`,
//!   `ToolOutput`, `Prompt`, ...), re-exported by `agentai`.
//! - [`history`]: [`HistoryProcessor`](history::HistoryProcessor) changes the messages sent to
//!   the model, [`HistoryPruning`](history::HistoryPruning) removes media and reasoning from
//!   finished turns.
//! - [`mime`]: one table of MIME types and file extensions ([`mime::extension`],
//!   [`mime::from_path`], [`mime::with_extension`]), also behind `Media::extension` and
//!   `Media::file_name`.
//! - [`base64`]: [`base64::encode`] and [`base64::decode`], the encoding of `Media` bytes and
//!   data URLs (`Media::to_url`).
//! - [`storage`]: [`MediaStorage`](storage::MediaStorage) (`MediaRead` + `MediaSave`), a
//!   blob storage for images and other files, with [`MemoryStorage`](storage::MemoryStorage).
//! - HTTP: the [`HttpClient`] trait hides the HTTP stack, so crates such as `agentai-image` send
//! their requests without depending on a particular client. [`ReqwestClient`] implements it
//! with `reqwest`; another environment (for example WebAssembly) can provide its own.
//!   With the `test-support` feature, `mock::MockHttp` records requests and answers with
//!   prepared responses, for tests of providers.
//! - [`tls`]: the TLS backend is chosen with the `rustls-ring` (default), `rustls-aws-lc` and
//!   `native-tls` features.
//!
//! ## Features
#![doc = document_features::document_features!()]

#[cfg(all(
    feature = "native-tls",
    any(feature = "rustls-ring", feature = "rustls-aws-lc")
))]
compile_error!(
    "agentai: `native-tls` cannot be combined with `rustls-ring` or `rustls-aws-lc`. \
     To use native-tls, set `default-features = false` and enable `native-tls`."
);

pub mod base64;
pub mod history;
mod http;
pub mod message;
pub mod mime;
pub mod prompt;
pub mod storage;
pub mod tls;

pub use http::*;
