//! Comfy Serve, a server for video generation models (`/api/v1/videos`).

use std::sync::Arc;

use agentai_core::message::Media;
use agentai_core::{
    HttpClient, HttpRequest, HttpResponse, Multipart, ReqwestClient, percent_encode,
};
use async_trait::async_trait;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Deserialize, Serialize};

use super::{
    JobStatus, VideoError, VideoInputKind, VideoInputs, VideoJob, VideoModel, VideoProvider,
    VideoRequest,
};

/// Environment variable with the base URL of the server.
pub const BASE_URL_ENV: &str = "COMFY_SERVE_BASE_URL";
/// Environment variable with the API key, optional.
pub const API_KEY_ENV: &str = "COMFY_SERVE_API_KEY";

/// Provider for a Comfy Serve server.
///
/// ```rust,no_run
/// # use agentai_video::provider::comfy_serve::ComfyServeVideoProvider;
/// # fn main() -> Result<(), agentai_video::VideoError> {
/// // From `COMFY_SERVE_BASE_URL` and, if set, `COMFY_SERVE_API_KEY`
/// let provider = ComfyServeVideoProvider::from_env("wan")?;
/// // Or everything in code; an empty key sends no `Authorization`
/// let provider = ComfyServeVideoProvider::new("wan", "", "http://localhost:8000");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ComfyServeVideoProvider {
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for ComfyServeVideoProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfyServeVideoProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl ComfyServeVideoProvider {
    /// Creates a provider for `model` with the API key and base URL given in code. An empty
    /// key sends no `Authorization` header. The model can be empty to follow stored jobs.
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

    /// Creates a provider for `model` with the base URL from `COMFY_SERVE_BASE_URL` and the API
    /// key from `COMFY_SERVE_API_KEY`, which may be unset.
    pub fn from_env(model: impl Into<String>) -> Result<Self, VideoError> {
        let base_url =
            std::env::var(BASE_URL_ENV).map_err(|_| VideoError::MissingBaseUrl(BASE_URL_ENV))?;
        let api_key = std::env::var(API_KEY_ENV).unwrap_or_default();
        Ok(Self::new(model, api_key, base_url))
    }

    /// Sends the requests with another HTTP client, for example with other timeouts or a proxy.
    pub fn with_http_client(mut self, client: impl HttpClient) -> Self {
        self.client = Arc::new(client);
        self
    }

    fn url(&self, path: &str) -> String {
        format!("{}/api/v1/videos{path}", self.base_url)
    }

    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, VideoError> {
        Ok(self
            .client
            .send(request.with_bearer_token(&self.api_key))
            .await?)
    }

    /// The JSON body of a request, or the form with the files.
    async fn body(
        &self,
        request: HttpRequest,
        video: VideoRequest,
    ) -> Result<HttpRequest, VideoError> {
        let mut files: Vec<(&str, &Media)> = Vec::new();
        let mut video_sound: &[u32] = &[];
        match &video.inputs {
            VideoInputs::Frames { first, last } => {
                files.extend(first.iter().map(|m| ("first_frame", m)));
                files.extend(last.iter().map(|m| ("last_frame", m)));
            }
            VideoInputs::References(refs) => {
                files.extend(refs.images.iter().map(|m| ("image", m)));
                files.extend(refs.videos.iter().map(|m| ("video", m)));
                files.extend(refs.audio.iter().map(|m| ("audio", m)));
                video_sound = &refs.video_sound;
            }
            VideoInputs::None => {}
        }
        let body = RequestBody {
            model: &self.model,
            prompt: &video.prompt,
            duration: video.duration,
            resolution: video.resolution.as_deref(),
            aspect_ratio: video.aspect_ratio.as_deref(),
            seed: video.seed,
            video_sound,
        };

        if files.is_empty() {
            return Ok(request.with_json(&body)?);
        }
        let mut form = Multipart::new().with_json("request", &body)?;
        for (part, media) in files {
            let bytes = media.bytes(&*self.client).await?;
            form = form.with_file(
                part,
                media.file_name(part),
                media.content_type.clone(),
                bytes,
            );
        }
        Ok(request.with_multipart(form))
    }
}

/// The JSON of `POST /api/v1/videos`, the whole body or the `request` part of the form.
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
    #[serde(skip_serializing_if = "<[u32]>::is_empty")]
    video_sound: &'a [u32],
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

#[derive(Deserialize)]
struct ModelEntry {
    id: Option<String>,
    name: Option<String>,
    supported_durations: Option<Vec<u32>>,
    supported_resolutions: Option<Vec<String>>,
    supported_aspect_ratios: Option<Vec<String>>,
    supported_inputs: Option<Vec<InputKind>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputKind {
    FirstFrame,
    LastFrame,
    Image,
    Video,
    Audio,
    #[serde(other)]
    Unknown,
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
            Some(
                ErrorReply::Object {
                    message: Some(message),
                }
                | ErrorReply::Text(message),
            ) => message,
            _ => "the job failed".to_string(),
        }),
    })
}

/// The models of `data`, named `comfy-serve::{id}`; entries without an ID are skipped.
fn parse_models(response: &HttpResponse) -> Result<Vec<VideoModel>, VideoError> {
    let list: ModelList = json_reply(response)?;
    Ok(list
        .data
        .into_iter()
        .filter_map(|entry| {
            let id = entry.id?;
            let mut model = VideoModel::new(format!("comfy-serve::{id}"), id);
            model.name = entry.name;
            model.durations = entry.supported_durations.map(|mut durations| {
                durations.sort_unstable();
                durations
            });
            model.resolutions = entry.supported_resolutions;
            model.aspect_ratios = entry.supported_aspect_ratios;
            model.inputs = entry.supported_inputs.and_then(|kinds| {
                let known: Vec<VideoInputKind> = kinds
                    .iter()
                    .filter_map(|kind| match kind {
                        InputKind::FirstFrame => Some(VideoInputKind::FirstFrame),
                        InputKind::LastFrame => Some(VideoInputKind::LastFrame),
                        InputKind::Image => Some(VideoInputKind::Image),
                        InputKind::Video => Some(VideoInputKind::Video),
                        InputKind::Audio => Some(VideoInputKind::Audio),
                        InputKind::Unknown => None,
                    })
                    .collect();
                // A list of kinds we do not know says nothing about what the model accepts
                (!known.is_empty() || kinds.is_empty()).then_some(known)
            });
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
impl VideoProvider for ComfyServeVideoProvider {
    fn model(&self) -> Option<&str> {
        Some(self.model.as_str()).filter(|model| !model.is_empty())
    }

    async fn submit(&self, request: VideoRequest) -> Result<VideoJob, VideoError> {
        let request = self.body(HttpRequest::post(self.url("")), request).await?;
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
        let response = self
            .send(HttpRequest::get(
                self.url(&format!("/{}/content", percent_encode(&job.id))),
            ))
            .await?;
        parse_content(response)
    }

    fn supports_cancel(&self) -> bool {
        true
    }

    /// A job that is gone (404) or already finished (409) counts as cancelled.
    async fn cancel(&self, job: &VideoJob) -> Result<(), VideoError> {
        let response = self
            .send(HttpRequest::delete(
                self.url(&format!("/{}", percent_encode(&job.id))),
            ))
            .await?;
        if matches!(response.status, 404 | 409) {
            return Ok(());
        }
        response.error_for_status()?;
        Ok(())
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
    use agentai_core::Method;
    use agentai_core::mock::{MockHttp, body_json, json_response};
    use serde_json::{Value, json};

    fn provider(http: &MockHttp) -> ComfyServeVideoProvider {
        ComfyServeVideoProvider::new("wan", "", "http://host/").with_http_client(http.clone())
    }

    #[tokio::test]
    async fn text_to_video_is_json_with_only_the_set_options() {
        let http = MockHttp::new(vec![json_response(
            202,
            json!({"id": "j1", "polling_url": "/p"}),
        )]);
        let job = provider(&http)
            .submit(VideoRequest::new("a cat").with_duration(5))
            .await
            .unwrap();
        assert_eq!(job.id, "j1");
        let request = http.request(0);
        assert_eq!(request.method, Method::Post);
        assert_eq!(request.url, "http://host/api/v1/videos");
        assert_eq!(
            body_json(&request),
            json!({"model": "wan", "prompt": "a cat", "duration": 5})
        );
        assert_eq!(request.header("Authorization"), None);
    }

    #[tokio::test]
    async fn empty_frames_are_a_text_to_video_request() {
        let http = MockHttp::new(vec![json_response(202, json!({"id": "j"}))]);
        provider(&http)
            .submit(VideoRequest::new("p").with_inputs(VideoInputs::Frames {
                first: None,
                last: None,
            }))
            .await
            .unwrap();
        assert_eq!(
            http.request(0).header("Content-Type"),
            Some("application/json")
        );
    }

    #[tokio::test]
    async fn frames_go_as_multipart_after_the_request_part() {
        let http = MockHttp::new(vec![json_response(202, json!({"id": "j"}))]);
        let first = Media::from_bytes("image/png", b"FIRST").with_name("start.png");
        let last = Media::from_bytes("image/jpeg; q=1", b"LAST");
        provider(&http)
            .submit(
                VideoRequest::new("p")
                    .with_seed(7)
                    .with_inputs(VideoInputs::Frames {
                        first: Some(first),
                        last: Some(last),
                    }),
            )
            .await
            .unwrap();
        let request = http.request(0);
        assert!(
            request
                .header("Content-Type")
                .unwrap()
                .starts_with("multipart/form-data; boundary=")
        );
        let body = String::from_utf8_lossy(request.body.as_ref().unwrap()).into_owned();
        let at = |needle: &str| {
            body.find(needle)
                .unwrap_or_else(|| panic!("missing {needle}"))
        };
        let request_part = at("name=\"request\"\r\nContent-Type: application/json\r\n\r\n{");
        let first_part = at(
            "name=\"first_frame\"; filename=\"start.png\"\r\nContent-Type: image/png\r\n\r\nFIRST",
        );
        let last_part = at(
            "name=\"last_frame\"; filename=\"last_frame.jpeg\"\r\nContent-Type: image/jpeg; q=1\r\n\r\nLAST",
        );
        assert!(request_part < first_part && first_part < last_part);
        assert!(body.contains("\"seed\":7") && body.contains("\"model\":\"wan\""));
    }

    #[tokio::test]
    async fn references_keep_their_order_and_video_sound() {
        let http = MockHttp::new(vec![json_response(202, json!({"id": "j"}))]);
        let refs = VideoReferences::new()
            .with_image(Media::from_bytes("image/png", b"I1").with_name("a.png"))
            .with_image(Media::from_bytes("image/png", b"I2").with_name("b.png"))
            .with_video(Media::from_bytes("video/mp4", b"V1").with_name("v.mp4"))
            .with_audio(Media::from_bytes("audio/wav", b"A1").with_name("s.wav"))
            .with_video_sound(0);
        provider(&http)
            .submit(VideoRequest::new("p").with_inputs(VideoInputs::References(refs)))
            .await
            .unwrap();
        let body = String::from_utf8_lossy(http.request(0).body.as_ref().unwrap()).into_owned();
        let order: Vec<usize> = [
            "name=\"image\"; filename=\"a.png\"",
            "name=\"image\"; filename=\"b.png\"",
            "name=\"video\"; filename=\"v.mp4\"",
            "name=\"audio\"; filename=\"s.wav\"",
        ]
        .iter()
        .map(|n| body.find(n).unwrap())
        .collect();
        assert!(order.is_sorted());
        assert!(body.contains("\"video_sound\":[0]"));
    }

    #[tokio::test]
    async fn a_stored_job_is_followed_without_a_model() {
        let http = MockHttp::new(vec![json_response(200, json!({"status": "in_progress"}))]);
        let provider =
            ComfyServeVideoProvider::new("", "key", "http://host").with_http_client(http.clone());
        let job: VideoJob =
            serde_json::from_value(json!({"id": "j9", "data": {"polling_url": null}})).unwrap();
        assert_eq!(provider.status(&job).await.unwrap(), JobStatus::Running);
        let request = http.request(0);
        assert_eq!(
            (request.method, request.url.as_str()),
            (Method::Get, "http://host/api/v1/videos/j9")
        );
        assert_eq!(request.header("Authorization"), Some("Bearer key"));
    }

    #[tokio::test]
    async fn download_of_missing_content_is_an_api_error() {
        let http = MockHttp::new(vec![
            json_response(404, json!({"error": {"message": "expired"}})),
            HttpResponse::new(200, b"MP4".to_vec()).with_header("Content-Type", "video/mp4"),
        ]);
        let provider = provider(&http);
        let job = VideoJob::new("j");
        assert!(
            matches!(provider.download(&job).await, Err(VideoError::Api { status: 404, message }) if message == "expired")
        );
        let media = provider.download(&job).await.unwrap();
        assert_eq!(http.request(1).url, "http://host/api/v1/videos/j/content");
        assert_eq!(media.byte_len(), Some(3));
    }

    #[tokio::test]
    async fn cancel_deletes_and_accepts_gone_or_finished_jobs() {
        let http = MockHttp::new(vec![
            json_response(200, json!({})),
            json_response(404, json!({})),
            json_response(409, json!({})),
            json_response(500, json!({"error": {"message": "boom"}})),
        ]);
        let provider = provider(&http);
        // An ID is one path segment, whatever characters a stored job holds
        for id in ["a/b?c", "j", "j"] {
            provider.cancel(&VideoJob::new(id)).await.unwrap();
        }
        assert!(matches!(
            provider.cancel(&VideoJob::new("j")).await,
            Err(VideoError::Api { status: 500, .. })
        ));
        assert_eq!(
            (http.request(0).method, http.request(0).url.as_str()),
            (Method::Delete, "http://host/api/v1/videos/a%2Fb%3Fc")
        );
    }

    #[tokio::test]
    async fn models_come_from_the_models_endpoint() {
        let http = MockHttp::new(vec![json_response(200, json!({"data": [{"id": "wan"}]}))]);
        let models = provider(&http).list_models().await.unwrap();
        assert_eq!(http.request(0).url, "http://host/api/v1/videos/models");
        assert_eq!(models[0].model, "comfy-serve::wan");
    }

    #[test]
    fn statuses_map_to_job_status() {
        let status = |code, body: Value| parse_status(json_response(code, body));
        assert_eq!(
            status(200, json!({"status": "pending"})).unwrap(),
            JobStatus::Queued
        );
        assert_eq!(
            status(200, json!({"status": "in_progress"})).unwrap(),
            JobStatus::Running
        );
        assert_eq!(
            status(200, json!({"status": "completed"})).unwrap(),
            JobStatus::Completed
        );
        for gone in ["cancelled", "expired"] {
            assert_eq!(
                status(200, json!({"status": gone})).unwrap(),
                JobStatus::Gone
            );
        }
        assert_eq!(
            status(404, json!({"error": {"message": "no job"}})).unwrap(),
            JobStatus::Gone
        );
        assert_eq!(
            status(
                200,
                json!({"status": "failed", "error": {"message": "oom"}})
            )
            .unwrap(),
            JobStatus::Failed("oom".into())
        );
        assert_eq!(
            status(200, json!({"status": "failed", "error": "bad prompt"})).unwrap(),
            JobStatus::Failed("bad prompt".into())
        );
        assert!(
            matches!(status(200, json!({"status": "failed"})).unwrap(), JobStatus::Failed(m) if !m.is_empty())
        );
        assert!(matches!(
            status(200, json!({"status": "dreaming"})),
            Err(VideoError::InvalidResponse(_))
        ));
        assert!(matches!(
            status(500, json!({"error": {"message": "boom"}})),
            Err(VideoError::Api { status: 500, message }) if message == "boom"
        ));
    }

    #[test]
    fn a_job_without_an_id_is_an_invalid_response() {
        assert!(matches!(
            parse_job(&json_response(202, json!({"polling_url": "/x"}))),
            Err(VideoError::InvalidResponse(_))
        ));
        let job = parse_job(&json_response(
            202,
            json!({"id": "j1", "polling_url": "/api/v1/videos/j1"}),
        ))
        .unwrap();
        assert_eq!(job, VideoJob::new("j1"));
    }

    #[test]
    fn models_keep_absent_capabilities_unknown() {
        let models = parse_models(&json_response(
            200,
            json!({"data": [
                {"id": "wan", "name": "Wan", "supported_durations": [10, 5],
                 "supported_resolutions": ["720p"], "supported_aspect_ratios": ["16:9"],
                 "supported_inputs": ["first_frame", "audio", "hologram"]},
                {"id": "bare", "supported_durations": []},
                {"id": "odd", "supported_inputs": ["hologram"]},
                {"id": "none", "supported_inputs": []},
            ]}),
        ))
        .unwrap();
        assert_eq!(models[0].model, "comfy-serve::wan");
        assert_eq!(models[0].id, "wan");
        assert_eq!(models[0].durations, Some(vec![5, 10]));
        assert_eq!(
            models[0].inputs,
            Some(vec![VideoInputKind::FirstFrame, VideoInputKind::Audio])
        );
        assert_eq!(models[1].durations, Some(vec![]));
        // Only unknown kinds say nothing; an empty list from the server accepts nothing
        assert_eq!(models[2].inputs, None);
        assert_eq!(models[3].inputs, Some(vec![]));
        assert_eq!(
            (
                models[1].resolutions.clone(),
                models[1].inputs.clone(),
                models[1].name.clone()
            ),
            (None, None, None)
        );
        assert!(matches!(
            parse_models(&json_response(200, json!({}))),
            Err(VideoError::InvalidResponse(_))
        ));
    }

    #[test]
    fn content_type_comes_from_the_header_or_defaults_to_mp4() {
        let media = parse_content(
            HttpResponse::new(200, vec![1u8, 2])
                .with_header("Content-Type", "video/webm; codecs=vp9"),
        )
        .unwrap();
        assert_eq!(media.content_type, "video/webm");
        assert_eq!(
            parse_content(HttpResponse::new(200, vec![1u8]))
                .unwrap()
                .content_type,
            "video/mp4"
        );
    }
}
