//! # AgentAI Video
//!
//! Video generation for [AgentAI](https://crates.io/crates/agentai). Video APIs are
//! asynchronous: [`VideoProvider::submit`] sends a request and returns a [`VideoJob`], which you
//! follow with [`VideoProvider::status`] and, when it is [`JobStatus::Completed`], fetch with
//! [`VideoProvider::download`]. Polling is up to the caller.
//!
//! A [`VideoJob`] can be stored (serde) and given back, possibly to a new provider, after a
//! restart. [`VideoModel`] describes what a model accepts, where the provider says so.
//!
//! ## Features
#![doc = document_features::document_features!()]

#[cfg(all(
    feature = "native-tls",
    any(feature = "rustls-ring", feature = "rustls-aws-lc")
))]
compile_error!(
    "agentai-video: `native-tls` cannot be combined with `rustls-ring` or `rustls-aws-lc`. \
     To use native-tls, set `default-features = false` and enable `native-tls`."
);

pub mod provider;

pub use agentai_core::message::Media;
pub use provider::{
    JobStatus, VideoError, VideoInputKind, VideoInputs, VideoJob, VideoModel, VideoProvider,
    VideoReferences, VideoRequest,
};
