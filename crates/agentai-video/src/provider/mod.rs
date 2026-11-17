//! # Video providers
//!
//! [`VideoProvider`] is the interface to a video generation API. Each submodule implements it
//! for one API. Implement it yourself to use another API.

pub mod comfy_serve;
pub mod openrouter;
pub(crate) mod video_api;

use agentai_core::HttpError;
use agentai_core::message::Media;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// A request to generate a video.
///
/// Unset options are not sent, the server uses its defaults. Values are not checked here: the
/// API reports the ones it does not accept.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct VideoRequest {
    /// What the video shows.
    pub prompt: String,
    /// Length in whole seconds.
    pub duration: Option<u32>,
    /// Resolution, in the format of the provider (for example `720p`).
    pub resolution: Option<String>,
    /// Aspect ratio, in the format of the provider (for example `16:9`).
    pub aspect_ratio: Option<String>,
    /// Seed of the random generator.
    pub seed: Option<u32>,
    /// Images, videos and audio the video is made from.
    pub inputs: VideoInputs,
}

impl VideoRequest {
    /// Creates a text-to-video request.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            duration: None,
            resolution: None,
            aspect_ratio: None,
            seed: None,
            inputs: VideoInputs::None,
        }
    }

    /// Sets the length in seconds.
    pub fn with_duration(mut self, seconds: u32) -> Self {
        self.duration = Some(seconds);
        self
    }

    /// Sets the resolution.
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Sets the aspect ratio.
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Sets the seed.
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the inputs.
    pub fn with_inputs(mut self, inputs: VideoInputs) -> Self {
        self.inputs = inputs;
        self
    }
}

/// What a video is made from. Frames and references exclude each other.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VideoInputs {
    /// Text-to-video.
    None,
    /// Image-to-video: at least one of the two frames.
    Frames {
        /// The image the video starts with.
        first: Option<Media>,
        /// The image the video ends with.
        last: Option<Media>,
    },
    /// Reference-to-video.
    References(VideoReferences),
}

/// The files of a reference-to-video request.
///
/// A provider that cannot send some of them returns [`VideoError::Unsupported`].
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct VideoReferences {
    /// Reference images.
    pub images: Vec<Media>,
    /// Reference videos.
    pub videos: Vec<Media>,
    /// Reference audio.
    pub audio: Vec<Media>,
    /// Indices into `videos` whose sound is also an audio input.
    pub video_sound: Vec<u32>,
}

impl VideoReferences {
    /// Creates an empty set of references.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a reference image.
    pub fn with_image(mut self, image: Media) -> Self {
        self.images.push(image);
        self
    }

    /// Adds a reference video.
    pub fn with_video(mut self, video: Media) -> Self {
        self.videos.push(video);
        self
    }

    /// Adds a reference audio.
    pub fn with_audio(mut self, audio: Media) -> Self {
        self.audio.push(audio);
        self
    }

    /// Marks the sound of the reference video with this index as an audio input.
    pub fn with_video_sound(mut self, index: u32) -> Self {
        self.video_sound.push(index);
        self
    }
}

/// A video job at a provider. Store it and give it back to follow the job, also after a
/// restart and with a new provider instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VideoJob {
    /// ID of the job at the provider.
    pub id: String,
    /// Whatever the provider needs to find the job again (for example a polling URL).
    pub data: Value,
}

impl VideoJob {
    /// Creates a job with an ID and no data.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            data: Value::Null,
        }
    }

    /// Sets the data the provider needs to find the job.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = data;
        self
    }
}

/// Where a job is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum JobStatus {
    /// Waiting to start.
    Queued,
    /// Being generated.
    Running,
    /// Done: [`VideoProvider::download`] returns the video.
    Completed,
    /// The job failed, with the provider's message.
    Failed(String),
    /// The provider no longer knows the job: it was cancelled or expired.
    Gone,
}

/// A kind of input a model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum VideoInputKind {
    /// An image the video starts with.
    FirstFrame,
    /// An image the video ends with.
    LastFrame,
    /// A reference image.
    Image,
    /// A reference video.
    Video,
    /// A reference audio.
    Audio,
}

/// A video model and what it accepts, as the provider reports it.
///
/// `None` in a list means the provider does not say, never "not supported": offer the option and
/// let the server answer. An empty list means the provider says there is nothing to choose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VideoModel {
    /// The name to use with `from_model`, with the provider prefix.
    pub model: String,
    /// The ID of the model at the provider.
    pub id: String,
    /// A display name.
    pub name: Option<String>,
    /// Lengths in seconds, ascending.
    pub durations: Option<Vec<u32>>,
    /// Resolutions, in the format of the provider.
    pub resolutions: Option<Vec<String>>,
    /// Aspect ratios, in the format of the provider.
    pub aspect_ratios: Option<Vec<String>>,
    /// The kinds of input the model accepts.
    pub inputs: Option<Vec<VideoInputKind>>,
}

impl VideoModel {
    /// Creates a model that says nothing about its capabilities.
    pub fn new(model: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            id: id.into(),
            name: None,
            durations: None,
            resolutions: None,
            aspect_ratios: None,
            inputs: None,
        }
    }
}

/// Errors of video providers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VideoError {
    /// The request could not be sent or the response could not be read.
    #[error(transparent)]
    Http(HttpError),
    /// The API answered with an error status.
    #[error("video API error (status {status}): {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Message of the API.
        message: String,
    },
    /// The API answered with something the provider cannot read.
    #[error("invalid video API response: {0}")]
    InvalidResponse(String),
    /// The provider cannot do this, for example cancel a job.
    #[error("not supported by the video provider: {0}")]
    Unsupported(&'static str),
    /// No provider matches the model name.
    #[error("no video provider for model '{0}'")]
    UnknownProvider(String),
    /// The environment variable with the API key is not set.
    #[error("the API key is missing, set {0}")]
    MissingApiKey(&'static str),
    /// The environment variable with the base URL is not set.
    #[error("the base URL is missing, set {0}")]
    MissingBaseUrl(&'static str),
}

impl From<HttpError> for VideoError {
    fn from(err: HttpError) -> Self {
        match err {
            HttpError::Status { status, message } => Self::Api { status, message },
            err => Self::Http(err),
        }
    }
}

impl VideoError {
    /// Returns `true` when the API rejected the credentials (status 401 or 403).
    pub fn is_auth(&self) -> bool {
        matches!(
            self,
            Self::Api {
                status: 401 | 403,
                ..
            }
        )
    }
}

/// A video generation API.
///
/// `status`, `download` and `cancel` need no model, so a provider built with an empty model
/// name can follow a stored [`VideoJob`].
#[async_trait]
pub trait VideoProvider: Send + Sync + 'static {
    /// Name of the model, if the provider has one.
    fn model(&self) -> Option<&str> {
        None
    }

    /// Sends the request; the job is queued or running (or already complete).
    async fn submit(&self, request: VideoRequest) -> Result<VideoJob, VideoError>;

    /// Asks where the job is.
    async fn status(&self, job: &VideoJob) -> Result<JobStatus, VideoError>;

    /// Downloads the finished video.
    async fn download(&self, job: &VideoJob) -> Result<Media, VideoError>;

    /// Returns `true` when [`cancel`](VideoProvider::cancel) works.
    fn supports_cancel(&self) -> bool {
        false
    }

    /// Cancels the job. Default: [`VideoError::Unsupported`].
    async fn cancel(&self, _job: &VideoJob) -> Result<(), VideoError> {
        Err(VideoError::Unsupported("cancelling a job"))
    }

    /// Lists the models. Default: [`VideoError::Unsupported`].
    async fn list_models(&self) -> Result<Vec<VideoModel>, VideoError> {
        Err(VideoError::Unsupported("listing models"))
    }
}

/// The APIs [`from_model`] knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    ComfyServe,
    OpenRouter,
}

impl ProviderKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "comfy-serve" => Some(Self::ComfyServe),
            "openrouter" => Some(Self::OpenRouter),
            _ => None,
        }
    }

    fn build(self, model: &str) -> Result<Box<dyn VideoProvider>, VideoError> {
        Ok(match self {
            Self::ComfyServe => Box::new(comfy_serve::ComfyServeVideoProvider::from_env(model)?),
            Self::OpenRouter => Box::new(openrouter::OpenRouterVideoProvider::from_env(model)?),
        })
    }
}

/// Splits `provider::model` and finds the provider. The prefix is required.
fn resolve(model: &str) -> Result<(ProviderKind, &str), VideoError> {
    model
        .split_once("::")
        .and_then(|(provider, name)| Some((ProviderKind::from_name(provider)?, name)))
        .ok_or_else(|| VideoError::UnknownProvider(model.to_string()))
}

/// Creates the provider for a `provider::model` name, with its settings from environment
/// variables, for example `comfy-serve::wan` (see
/// [`ComfyServeVideoProvider::from_env`](comfy_serve::ComfyServeVideoProvider::from_env)).
pub fn from_model(model: &str) -> Result<Box<dyn VideoProvider>, VideoError> {
    let (kind, name) = resolve(model)?;
    kind.build(name)
}

/// Lists the video models of a provider (`comfy-serve`, `openrouter`), with its settings from
/// environment variables as in [`from_model`].
///
/// ```rust,no_run
/// # async fn run() -> Result<(), agentai_video::VideoError> {
/// for model in agentai_video::provider::list_models("comfy-serve").await? {
///     println!("{} {:?}", model.model, model.durations);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn list_models(provider: &str) -> Result<Vec<VideoModel>, VideoError> {
    let kind = ProviderKind::from_name(provider)
        .ok_or_else(|| VideoError::UnknownProvider(provider.to_string()))?;
    kind.build("")?.list_models().await
}

#[cfg(test)]
pub(crate) mod tests {
    use agentai_core::{HttpClient, HttpError, HttpRequest, HttpResponse};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Default)]
    struct Recorded {
        requests: Mutex<Vec<HttpRequest>>,
        responses: Mutex<VecDeque<HttpResponse>>,
    }

    /// Records the requests and answers with the prepared responses, in order. Clones share
    /// the record, so one can go to a provider (`with_http_client`) while the test keeps another.
    #[derive(Clone, Default)]
    pub(crate) struct MockHttp(Arc<Recorded>);

    impl MockHttp {
        pub(crate) fn new(responses: Vec<HttpResponse>) -> Self {
            Self(Arc::new(Recorded {
                requests: Mutex::default(),
                responses: Mutex::new(responses.into()),
            }))
        }

        pub(crate) fn request(&self, index: usize) -> HttpRequest {
            self.0.requests.lock().unwrap()[index].clone()
        }

        pub(crate) fn request_count(&self) -> usize {
            self.0.requests.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.0.requests.lock().unwrap().push(request);
            Ok(self
                .0
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("no response left"))
        }
    }

    pub(crate) fn json_response(status: u16, body: Value) -> HttpResponse {
        HttpResponse::new(status, body.to_string())
    }

    pub(crate) fn body_json(request: &HttpRequest) -> Value {
        serde_json::from_slice(request.body.as_ref().unwrap()).unwrap()
    }

    #[test]
    fn a_prefix_selects_the_provider_and_is_required() {
        assert_eq!(
            resolve("comfy-serve::wan/2.2").unwrap(),
            (ProviderKind::ComfyServe, "wan/2.2")
        );
        assert_eq!(
            resolve("openrouter::google/veo-3.1").unwrap(),
            (ProviderKind::OpenRouter, "google/veo-3.1")
        );
        for model in ["wan", "comfy::wan", "other::wan"] {
            assert!(matches!(resolve(model), Err(VideoError::UnknownProvider(m)) if m == model));
        }
    }

    #[test]
    fn jobs_and_statuses_keep_dreamers_stored_format() {
        assert_eq!(
            serde_json::to_value(JobStatus::Failed("x".into())).unwrap(),
            json!({"failed": "x"})
        );
        assert_eq!(
            serde_json::to_value(JobStatus::Queued).unwrap(),
            json!("queued")
        );
        assert_eq!(
            serde_json::to_value(VideoJob::new("j").with_data(json!({"polling_url": null})))
                .unwrap(),
            json!({"id": "j", "data": {"polling_url": null}})
        );
    }
}
