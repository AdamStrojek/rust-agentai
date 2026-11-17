//! Comfy Serve, a server for video generation models that speaks the "Video API v1" protocol
//! (`/api/v1/videos`).

use std::sync::Arc;

use agentai_core::message::Media;
use agentai_core::{HttpClient, HttpRequest, HttpResponse, Multipart, ReqwestClient};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::{
    JobStatus, VideoError, VideoInputs, VideoJob, VideoModel, VideoProvider, VideoRequest,
    video_api,
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
        let mut json = Map::new();
        json.insert("model".into(), json!(self.model));
        json.insert("prompt".into(), json!(video.prompt));
        if let Some(duration) = video.duration {
            json.insert("duration".into(), json!(duration));
        }
        if let Some(resolution) = video.resolution {
            json.insert("resolution".into(), json!(resolution));
        }
        if let Some(aspect_ratio) = video.aspect_ratio {
            json.insert("aspect_ratio".into(), json!(aspect_ratio));
        }
        if let Some(seed) = video.seed {
            json.insert("seed".into(), json!(seed));
        }

        let mut files: Vec<(&str, &Media)> = Vec::new();
        match &video.inputs {
            VideoInputs::Frames { first, last } => {
                files.extend(first.iter().map(|m| ("first_frame", m)));
                files.extend(last.iter().map(|m| ("last_frame", m)));
            }
            VideoInputs::References(refs) => {
                files.extend(refs.images.iter().map(|m| ("image", m)));
                files.extend(refs.videos.iter().map(|m| ("video", m)));
                files.extend(refs.audio.iter().map(|m| ("audio", m)));
                if !refs.video_sound.is_empty() {
                    json.insert("video_sound".into(), json!(refs.video_sound));
                }
            }
            _ => {}
        }

        if files.is_empty() {
            return Ok(request.with_json(&Value::Object(json))?);
        }
        let mut form = Multipart::new().with_json("request", &Value::Object(json))?;
        for (part, media) in files {
            let filename = media
                .name
                .clone()
                .unwrap_or_else(|| format!("{part}.{}", extension(&media.content_type)));
            let bytes = media.bytes(&*self.client).await?;
            form = form.with_file(part, filename, media.content_type.clone(), bytes);
        }
        Ok(request.with_multipart(form))
    }
}

/// File extension for a MIME type: its subtype, with the two common names that differ.
fn extension(content_type: &str) -> &str {
    match content_type {
        "audio/mpeg" => "mp3",
        "video/quicktime" => "mov",
        other => other
            .split(';')
            .next()
            .and_then(|mime| mime.split_once('/'))
            .map(|(_, subtype)| subtype.trim())
            .filter(|subtype| !subtype.is_empty())
            .unwrap_or("bin"),
    }
}

#[async_trait]
impl VideoProvider for ComfyServeVideoProvider {
    fn model(&self) -> Option<&str> {
        Some(self.model.as_str()).filter(|model| !model.is_empty())
    }

    async fn submit(&self, request: VideoRequest) -> Result<VideoJob, VideoError> {
        let request = self.body(HttpRequest::post(self.url("")), request).await?;
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
            .send(HttpRequest::get(self.url(&format!("/{}/content", job.id))))
            .await?;
        video_api::parse_content(response)
    }

    fn supports_cancel(&self) -> bool {
        true
    }

    /// A job that is gone (404) or already finished (409) counts as cancelled.
    async fn cancel(&self, job: &VideoJob) -> Result<(), VideoError> {
        let response = self
            .send(HttpRequest::delete(self.url(&format!("/{}", job.id))))
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
        video_api::parse_models("comfy-serve", &response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::VideoReferences;
    use crate::provider::tests::{MockHttp, body_json, json_response};
    use agentai_core::Method;

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
        let last = Media::from_bytes("image/jpeg", b"LAST");
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
            "name=\"last_frame\"; filename=\"last_frame.jpeg\"\r\nContent-Type: image/jpeg\r\n\r\nLAST",
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
        for _ in 0..3 {
            provider.cancel(&VideoJob::new("j")).await.unwrap();
        }
        assert!(matches!(
            provider.cancel(&VideoJob::new("j")).await,
            Err(VideoError::Api { status: 500, .. })
        ));
        assert_eq!(
            (http.request(0).method, http.request(0).url.as_str()),
            (Method::Delete, "http://host/api/v1/videos/j")
        );
    }

    #[tokio::test]
    async fn models_come_from_the_models_endpoint() {
        let http = MockHttp::new(vec![json_response(200, json!({"data": [{"id": "wan"}]}))]);
        let models = provider(&http).list_models().await.unwrap();
        assert_eq!(http.request(0).url, "http://host/api/v1/videos/models");
        assert_eq!(models[0].model, "comfy-serve::wan");
    }
}
