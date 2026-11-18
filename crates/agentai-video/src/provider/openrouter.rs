//! OpenRouter video generation (`/api/v1/videos`).
//!
//! The request format follows the OpenRouter video generation guide and its cookbook pages
//! (`frame_images` with a `frame_type`, `input_references` with images). OpenRouter documents
//! no way to cancel a job and no video or audio references, so
//! [`supports_cancel`](VideoProvider::supports_cancel) is `false` and such inputs are
//! [`VideoError::Unsupported`].

use std::sync::Arc;

use agentai_core::message::Media;
use agentai_core::{HttpClient, HttpRequest, HttpResponse, ReqwestClient, percent_encode};
use async_trait::async_trait;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Deserialize, Serialize};

use super::{
    JobStatus, VideoError, VideoInputs, VideoJob, VideoModel, VideoProvider, VideoRequest,
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

    fn body<'a>(&'a self, video: &'a VideoRequest) -> Result<RequestBody<'a>, VideoError> {
        let mut body = RequestBody {
            model: &self.model,
            prompt: &video.prompt,
            duration: video.duration,
            resolution: video.resolution.as_deref(),
            aspect_ratio: video.aspect_ratio.as_deref(),
            seed: video.seed,
            frame_images: Vec::new(),
            input_references: Vec::new(),
        };
        match &video.inputs {
            VideoInputs::None => {}
            VideoInputs::Frames { first, last } => {
                body.frame_images = [(first, "first_frame"), (last, "last_frame")]
                    .into_iter()
                    .filter_map(|(media, frame_type)| {
                        Some(ImagePart::new(media.as_ref()?, Some(frame_type)))
                    })
                    .collect();
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
                body.input_references = refs
                    .images
                    .iter()
                    .map(|image| ImagePart::new(image, None))
                    .collect();
            }
        }
        Ok(body)
    }
}

/// The JSON of `POST /videos`.
#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    frame_images: Vec<ImagePart>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_references: Vec<ImagePart>,
}

/// An image as OpenRouter takes it: a data URL for bytes, the URL itself otherwise.
#[derive(Serialize)]
struct ImagePart {
    r#type: &'static str,
    image_url: ImageUrl,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame_type: Option<&'static str>,
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

impl ImagePart {
    fn new(media: &Media, frame_type: Option<&'static str>) -> Self {
        Self {
            r#type: "image_url",
            image_url: ImageUrl {
                url: media.to_url(),
            },
            frame_type,
        }
    }
}

#[derive(Deserialize)]
struct JobReply {
    id: String,
}

#[derive(Deserialize)]
struct StatusReply {
    status: Status,
    error: Option<ErrorReply>,
}

/// The job statuses OpenRouter documents.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Expired,
}

/// `{"message": ...}`, a plain string, or anything else (no message).
#[derive(Deserialize)]
#[serde(untagged)]
enum ErrorReply {
    Object { message: Option<String> },
    Text(String),
    Other(IgnoredAny),
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

/// OpenRouter lists no input kinds, so [`VideoModel::inputs`] stays `None`.
#[derive(Deserialize)]
struct ModelEntry {
    id: Option<String>,
    name: Option<String>,
    supported_durations: Option<Vec<u32>>,
    supported_resolutions: Option<Vec<String>>,
    supported_aspect_ratios: Option<Vec<String>>,
}

fn json_reply<T: DeserializeOwned>(response: &HttpResponse) -> Result<T, VideoError> {
    response
        .json()
        .map_err(|err| VideoError::InvalidResponse(err.to_string()))
}

/// The job of a submit reply: its `id` is all that is needed to follow it.
fn parse_job(response: &HttpResponse) -> Result<VideoJob, VideoError> {
    let reply: JobReply = json_reply(response)?;
    Ok(VideoJob::new(reply.id))
}

/// 404 is `Gone`, so are cancelled and expired jobs.
fn parse_status(response: HttpResponse) -> Result<JobStatus, VideoError> {
    if response.status == 404 {
        return Ok(JobStatus::Gone);
    }
    let reply: StatusReply = json_reply(&response.error_for_status()?)?;
    Ok(match reply.status {
        Status::Pending => JobStatus::Queued,
        Status::InProgress => JobStatus::Running,
        Status::Completed => JobStatus::Completed,
        Status::Cancelled | Status::Expired => JobStatus::Gone,
        Status::Failed => JobStatus::Failed(match reply.error {
            Some(ErrorReply::Object {
                message: Some(message),
            })
            | Some(ErrorReply::Text(message)) => message,
            _ => "the job failed".to_string(),
        }),
    })
}

/// The models of `data`, named `openrouter::{id}`; entries without an ID are skipped.
fn parse_models(response: &HttpResponse) -> Result<Vec<VideoModel>, VideoError> {
    let list: ModelList = json_reply(response)?;
    Ok(list
        .data
        .into_iter()
        .filter_map(|entry| {
            let id = entry.id?;
            let mut model = VideoModel::new(format!("openrouter::{id}"), id);
            model.name = entry.name;
            model.durations = entry.supported_durations.map(|mut durations| {
                durations.sort_unstable();
                durations
            });
            model.resolutions = entry.supported_resolutions;
            model.aspect_ratios = entry.supported_aspect_ratios;
            Some(model)
        })
        .collect())
}

/// The video bytes, typed by the `Content-Type` header (`video/mp4` without one).
fn parse_content(response: HttpResponse) -> Result<Media, VideoError> {
    let response = response.error_for_status()?;
    let content_type = response.content_type().unwrap_or("video/mp4").to_string();
    Ok(Media::from_bytes(content_type, response.body))
}

#[async_trait]
impl VideoProvider for OpenRouterVideoProvider {
    fn model(&self) -> Option<&str> {
        Some(self.model.as_str()).filter(|model| !model.is_empty())
    }

    async fn submit(&self, request: VideoRequest) -> Result<VideoJob, VideoError> {
        let request = HttpRequest::post(self.url("")).with_json(&self.body(&request)?)?;
        let response = self.send(request).await?.error_for_status()?;
        parse_job(&response)
    }

    async fn status(&self, job: &VideoJob) -> Result<JobStatus, VideoError> {
        let response = self
            .send(HttpRequest::get(
                self.url(&format!("/{}", percent_encode(&job.id))),
            ))
            .await?;
        parse_status(response)
    }

    async fn download(&self, job: &VideoJob) -> Result<Media, VideoError> {
        let response =
            self.send(HttpRequest::get(
                self.url(&format!("/{}/content?index=0", percent_encode(&job.id))),
            ))
            .await?;
        parse_content(response)
    }

    async fn list_models(&self) -> Result<Vec<VideoModel>, VideoError> {
        let response = self
            .send(HttpRequest::get(self.url("/models")))
            .await?
            .error_for_status()?;
        parse_models(&response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::VideoReferences;
    use agentai_core::mock::{MockHttp, body_json, json_response};
    use serde_json::json;

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

    #[test]
    fn statuses_follow_the_documented_values() {
        let status = |code, body| parse_status(json_response(code, body));
        for (reply, expected) in [
            ("pending", JobStatus::Queued),
            ("in_progress", JobStatus::Running),
            ("completed", JobStatus::Completed),
            ("cancelled", JobStatus::Gone),
            ("expired", JobStatus::Gone),
        ] {
            assert_eq!(status(200, json!({"status": reply})).unwrap(), expected);
        }
        assert_eq!(
            status(200, json!({"status": "failed", "error": "content policy"})).unwrap(),
            JobStatus::Failed("content policy".into())
        );
        assert_eq!(status(404, json!({})).unwrap(), JobStatus::Gone);
        assert!(matches!(
            status(200, json!({"status": "queued"})),
            Err(VideoError::InvalidResponse(_))
        ));
    }

    #[tokio::test]
    async fn models_have_no_input_kinds() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"data": [
                {"id": "vendor/video", "name": "Video", "supported_durations": [8, 4],
                 "supported_resolutions": ["720p", "1080p"], "supported_aspect_ratios": ["16:9"]},
                {"name": "no id"},
            ]}),
        )]);
        let models = provider(&http).list_models().await.unwrap();
        assert_eq!(http.request(0).url, "https://or/api/v1/videos/models");
        let mut expected = VideoModel::new("openrouter::vendor/video", "vendor/video");
        expected.name = Some("Video".into());
        expected.durations = Some(vec![4, 8]);
        expected.resolutions = Some(vec!["720p".into(), "1080p".into()]);
        expected.aspect_ratios = Some(vec!["16:9".into()]);
        assert_eq!(models, [expected]);
    }
}
