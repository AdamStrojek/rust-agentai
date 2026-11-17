//! OpenRouter video generation (`/api/v1/videos`).
//!
//! The request format follows the OpenRouter video generation guide and its cookbook pages
//! (`frame_images` with a `frame_type`, `input_references` with images); it was not checked
//! against a live key. OpenRouter documents no way to cancel a job and no video or audio
//! references, so [`supports_cancel`](VideoProvider::supports_cancel) is `false` and such
//! inputs are [`VideoError::Unsupported`].

use std::sync::Arc;

use agentai_core::message::{Media, MediaSource};
use agentai_core::{HttpClient, HttpRequest, HttpResponse, ReqwestClient};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{
    JobStatus, VideoError, VideoInputs, VideoJob, VideoModel, VideoProvider, VideoRequest,
    video_api,
};

/// The official API.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Environment variable with the API key.
pub const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
/// Environment variable with the base URL, optional.
pub const BASE_URL_ENV: &str = "OPENROUTER_BASE_URL";

/// Provider for OpenRouter.
///
/// ```rust,no_run
/// # use agentai_video::provider::openrouter::OpenRouterVideoProvider;
/// # fn main() -> Result<(), agentai_video::VideoError> {
/// // From `OPENROUTER_API_KEY` and, if set, `OPENROUTER_BASE_URL`; the model name is only an
/// // example, list the real ones with `list_models`
/// let provider = OpenRouterVideoProvider::from_env("example-vendor/example-video-model")?;
/// // Or everything in code
/// let provider = OpenRouterVideoProvider::new(
///     "example-vendor/example-video-model",
///     "sk-or-...",
///     agentai_video::provider::openrouter::DEFAULT_BASE_URL,
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenRouterVideoProvider {
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenRouterVideoProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterVideoProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenRouterVideoProvider {
    /// Creates a provider for `model` with the API key and base URL given in code. The model
    /// can be empty to follow stored jobs.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Arc::new(ReqwestClient::new()),
        }
    }

    /// Creates a provider for `model` with the key from `OPENROUTER_API_KEY` and the base URL
    /// from `OPENROUTER_BASE_URL` (default: the official API).
    pub fn from_env(model: impl Into<String>) -> Result<Self, VideoError> {
        let api_key =
            std::env::var(API_KEY_ENV).map_err(|_| VideoError::MissingApiKey(API_KEY_ENV))?;
        let base_url = std::env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        Ok(Self::new(model, api_key, base_url))
    }

    /// Sends the requests with another HTTP client, for example with other timeouts or a proxy.
    pub fn with_http_client(mut self, client: impl HttpClient) -> Self {
        self.client = Arc::new(client);
        self
    }

    fn url(&self, path: &str) -> String {
        format!("{}/videos{path}", self.base_url)
    }

    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, VideoError> {
        Ok(self
            .client
            .send(request.with_bearer_token(&self.api_key))
            .await?)
    }

    fn body(&self, video: VideoRequest) -> Result<Value, VideoError> {
        let mut json = video_api::request_options(&self.model, &video);
        match &video.inputs {
            VideoInputs::None => {}
            VideoInputs::Frames { first, last } => {
                let frames: Vec<Value> = [(first, "first_frame"), (last, "last_frame")]
                    .into_iter()
                    .filter_map(|(media, frame_type)| Some((media.as_ref()?, frame_type)))
                    .map(|(media, frame_type)| {
                        let mut entry = image_entry(media);
                        entry["frame_type"] = json!(frame_type);
                        entry
                    })
                    .collect();
                if !frames.is_empty() {
                    json.insert("frame_images".into(), Value::Array(frames));
                }
            }
            VideoInputs::References(refs) => {
                if !refs.videos.is_empty() {
                    return Err(VideoError::Unsupported("reference videos"));
                }
                if !refs.audio.is_empty() {
                    return Err(VideoError::Unsupported("reference audio"));
                }
                if !refs.video_sound.is_empty() {
                    return Err(VideoError::Unsupported("video sound"));
                }
                if !refs.images.is_empty() {
                    let images: Vec<Value> = refs.images.iter().map(image_entry).collect();
                    json.insert("input_references".into(), Value::Array(images));
                }
            }
        }
        Ok(Value::Object(json))
    }
}

/// An image as OpenRouter takes it: a data URL for bytes, the URL itself otherwise.
fn image_entry(media: &Media) -> Value {
    let url = match &media.source {
        MediaSource::Base64(data) => format!("data:{};base64,{data}", media.content_type),
        MediaSource::Url(url) => url.clone(),
    };
    json!({"type": "image_url", "image_url": {"url": url}})
}

#[async_trait]
impl VideoProvider for OpenRouterVideoProvider {
    fn model(&self) -> Option<&str> {
        Some(self.model.as_str()).filter(|model| !model.is_empty())
    }

    async fn submit(&self, request: VideoRequest) -> Result<VideoJob, VideoError> {
        let body = self.body(request)?;
        let request = HttpRequest::post(self.url("")).with_json(&body)?;
        let response = self.send(request).await?.error_for_status()?;
        video_api::parse_job(&response)
    }

    async fn status(&self, job: &VideoJob) -> Result<JobStatus, VideoError> {
        let response = self
            .send(HttpRequest::get(self.url(&format!("/{}", job.id))))
            .await?;
        video_api::parse_status(response)
    }

    async fn download(&self, job: &VideoJob) -> Result<Media, VideoError> {
        let response = self
            .send(HttpRequest::get(
                self.url(&format!("/{}/content?index=0", job.id)),
            ))
            .await?;
        video_api::parse_content(response)
    }

    async fn list_models(&self) -> Result<Vec<VideoModel>, VideoError> {
        let response = self
            .send(HttpRequest::get(self.url("/models")))
            .await?
            .error_for_status()?;
        video_api::parse_models("openrouter", &response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::VideoReferences;
    use crate::provider::tests::{MockHttp, body_json, json_response};

    fn provider(http: &MockHttp) -> OpenRouterVideoProvider {
        OpenRouterVideoProvider::new("google/veo-3.1", "key", "https://or/api/v1")
            .with_http_client(http.clone())
    }

    #[tokio::test]
    async fn frames_are_sent_as_frame_images() {
        let http = MockHttp::new(vec![json_response(
            202,
            json!({"id": "j", "polling_url": "https://openrouter.ai/api/v1/videos/j"}),
        )]);
        let request =
            VideoRequest::new("p")
                .with_resolution("720p")
                .with_inputs(VideoInputs::Frames {
                    first: Some(Media::from_bytes("image/png", b"hi")),
                    last: Some(Media::from_url("image/jpeg", "https://x/last.jpg")),
                });
        provider(&http).submit(request).await.unwrap();
        let sent = http.request(0);
        assert_eq!(sent.url, "https://or/api/v1/videos");
        assert_eq!(sent.header("Authorization"), Some("Bearer key"));
        assert_eq!(
            body_json(&sent),
            json!({
                "model": "google/veo-3.1", "prompt": "p", "resolution": "720p",
                "frame_images": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}, "frame_type": "first_frame"},
                    {"type": "image_url", "image_url": {"url": "https://x/last.jpg"}, "frame_type": "last_frame"},
                ],
            })
        );
    }

    #[tokio::test]
    async fn image_references_are_sent_and_other_kinds_are_unsupported() {
        let http = MockHttp::new(vec![json_response(202, json!({"id": "j"}))]);
        let provider = provider(&http);
        let image = Media::from_bytes("image/png", b"hi");
        provider
            .submit(VideoRequest::new("p").with_inputs(VideoInputs::References(
                VideoReferences::new().with_image(image.clone()),
            )))
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0))["input_references"],
            json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}])
        );
        for refs in [
            VideoReferences::new().with_video(image.clone()),
            VideoReferences::new().with_audio(image.clone()),
            VideoReferences::new()
                .with_image(image.clone())
                .with_video_sound(0),
        ] {
            assert!(matches!(
                provider
                    .submit(VideoRequest::new("p").with_inputs(VideoInputs::References(refs)))
                    .await,
                Err(VideoError::Unsupported(_))
            ));
        }
        // Nothing is sent for unsupported inputs
        assert_eq!(http.request_count(), 1);
    }

    #[tokio::test]
    async fn content_is_downloaded_with_the_key() {
        let http = MockHttp::new(vec![HttpResponse::new(200, b"MP4".to_vec())]);
        provider(&http).download(&VideoJob::new("j")).await.unwrap();
        let sent = http.request(0);
        assert_eq!(sent.url, "https://or/api/v1/videos/j/content?index=0");
        assert_eq!(sent.header("Authorization"), Some("Bearer key"));
    }
}
