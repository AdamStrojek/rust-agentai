//! # AgentAI Video
//!
//! Video generation for [AgentAI](https://crates.io/crates/agentai). Video APIs are
//! asynchronous: [`VideoProvider::submit`] sends a request and returns a [`VideoJob`], which you
//! follow with [`VideoProvider::status`] and, when it is [`JobStatus::Completed`], fetch with
//! [`VideoProvider::download`]. Polling is up to the caller.
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use agentai_video::{JobStatus, VideoRequest, from_model};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // The provider comes from the prefix of the model name, its settings from environment
//! // variables (`COMFY_SERVE_BASE_URL` here)
//! let provider = from_model("comfy-serve::wan-2.2")?;
//! let job = provider
//!     .submit(VideoRequest::new("A lighthouse at dawn").with_duration(5))
//!     .await?;
//! loop {
//!     match provider.status(&job).await? {
//!         JobStatus::Completed => break,
//!         JobStatus::Failed(message) => return Err(message.into()),
//!         JobStatus::Gone => return Err("the job is gone".into()),
//!         _ => tokio::time::sleep(Duration::from_secs(5)).await,
//!     }
//! }
//! let video = provider.download(&job).await?; // a `Media`, base64 or a URL
//! # Ok(())
//! # }
//! ```
//!
//! A [`VideoJob`] can be stored (serde) and given back, possibly to a new provider, after a
//! restart: `status`, `download` and `cancel` need no model. A job does not record which provider
//! made it, so store the provider and model name next to it.
//!
//! ## Inputs
//!
//! [`VideoRequest::with_inputs`] takes [`VideoInputs`], one of three modes:
//!
//! - `None` (the default): text-to-video.
//! - `Frames { first, last }`: image-to-video, from a first frame, a last frame or both.
//! - `References(`[`VideoReferences`]`)`: reference-to-video, from reference images, videos and
//!   audio.
//!
//! Frames and references exclude each other. A provider that cannot send some input returns
//! [`VideoError::Unsupported`]. Options you do not set (duration, resolution, aspect ratio, seed)
//! are not sent, and values are not checked here: the API reports the ones it does not accept.
//!
//! ## Models
//!
//! [`list_models`] (or [`VideoProvider::list_models`]) returns [`VideoModel`]s with the durations,
//! resolutions, aspect ratios and input kinds a model accepts, where the provider says so. `None`
//! in one of these lists means the provider does not say, not that the option is unsupported:
//! offer it and let the server answer. `Some(vec![])` means there is nothing to choose.
//!
//! ## Providers
//!
//! | Prefix | Settings | Cancel | Inputs |
//! |--------|----------|--------|--------|
//! | `comfy-serve` | `COMFY_SERVE_BASE_URL` (required), `COMFY_SERVE_API_KEY` (optional) | yes | first and last frames; image, video and audio references; files are uploaded as multipart |
//! | `openrouter` | `OPENROUTER_API_KEY` (required), `OPENROUTER_BASE_URL` (optional) | no | first and last frames and image references, sent as URLs or data URLs |
//!
//! To set the key and base URL in code, or to change timeouts or the proxy, build the provider
//! yourself and give it your own client:
//!
//! ```rust,ignore
//! // `reqwest::Client` needs `reqwest` in your dependencies; `ReqwestClient` and the other HTTP
//! // types are re-exported from this crate
//! use agentai_video::ReqwestClient;
//! use agentai_video::provider::comfy_serve::ComfyServeVideoProvider;
//!
//! let client = reqwest::Client::builder().build().unwrap();
//! let provider = ComfyServeVideoProvider::new("wan-2.2", "", "http://localhost:8000")
//!     .with_http_client(ReqwestClient::from_client(client));
//! ```
//!
//! The HTTP types ([`HttpClient`], [`HttpRequest`], [`HttpResponse`], [`HttpError`],
//! [`ReqwestClient`]), [`Media`] and [`mime`] are re-exported, so this crate is enough to use a
//! provider with your own client or to implement [`VideoProvider`] for another API.
//!
//! ## Files
//!
//! [`mime::from_path`] guesses the type of an input file from its name, and
//! [`Media::extension`] gives the file extension of a downloaded video's type:
//!
//! ```rust,no_run
//! # use agentai_video::{Media, ReqwestClient, VideoInputs, VideoRequest, mime};
//! # async fn run(video: Media) -> Result<(), Box<dyn std::error::Error>> {
//! let path = "first.jpg";
//! let first = Media::from_bytes(mime::from_path(path).ok_or("unknown type")?, std::fs::read(path)?);
//! let request = VideoRequest::new("The scene comes alive").with_inputs(VideoInputs::Frames {
//!     first: Some(first),
//!     last: None,
//! });
//! // ... submit, poll, download ...
//! let bytes = video.bytes(&ReqwestClient::new()).await?;
//! std::fs::write(format!("video.{}", video.extension()), bytes)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Limits
//!
//! - Files travel in memory as [`Media`] (base64 or a URL), so large videos are held whole.
//! - There is no waiting helper (the loop above is yours to write, with your timing and your
//!   runtime), no retries and no timeouts of its own.
//! - There are no agent tools: this crate does not depend on `agentai`.
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

pub use agentai_core::message::{Media, MediaSource};
#[doc(inline)]
pub use agentai_core::mime;
pub use agentai_core::{HttpClient, HttpError, HttpRequest, HttpResponse, ReqwestClient};
pub use provider::{
    JobStatus, VideoError, VideoInputKind, VideoInputs, VideoJob, VideoModel, VideoProvider,
    VideoReferences, VideoRequest, from_model, list_models,
};
