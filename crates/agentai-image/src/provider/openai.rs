//! # OpenAI image API
//!
//! [`OpenAiImageProvider`] generates images with `POST /images/generations` and edits them
//! with `POST /images/edits`. It works with OpenAI (`gpt-image-*`, `dall-e-*`) and with other
//! APIs that follow the same format.

use super::{ImageEditRequest, ImageError, ImageProvider, ImageRequest, ImageResponse};
use agentai_core::message::{Media, MediaSource};
use agentai_core::{HttpClient, HttpRequest, HttpResponse, Multipart, ReqwestClient};
use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The OpenAI API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Environment variable with the API key.
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";
/// Environment variable with the base URL, for OpenAI-compatible APIs.
pub const BASE_URL_ENV: &str = "OPENAI_BASE_URL";

/// Provider for the OpenAI image API and APIs compatible with it.
///
/// ```rust,no_run
/// # use agentai_image::provider::openai::OpenAiImageProvider;
/// # fn main() -> Result<(), agentai_image::ImageError> {
/// // From `OPENAI_API_KEY` and `OPENAI_BASE_URL`
/// let provider = OpenAiImageProvider::from_env("gpt-image-1")?;
/// // Or everything in code
/// let provider = OpenAiImageProvider::new("gpt-image-1", "sk-...", "https://api.openai.com/v1");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenAiImageProvider {
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenAiImageProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiImageProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenAiImageProvider {
    /// Creates a provider for `model` with the API key and base URL given in code, for example
    /// [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header.
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

    /// Creates a provider for `model` with the API key from `OPENAI_API_KEY` and the base URL
    /// from `OPENAI_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set. The key may be left
    /// out when `OPENAI_BASE_URL` is set, for local servers without authentication.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ImageError> {
        Self::from_lookup(model, |name| std::env::var(name).ok())
    }

    fn from_lookup(
        model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ImageError> {
        let base_url = lookup(BASE_URL_ENV);
        let api_key = match (lookup(API_KEY_ENV), &base_url) {
            (Some(key), _) => key,
            (None, Some(_)) => String::new(),
            (None, None) => return Err(ImageError::MissingApiKey(API_KEY_ENV)),
        };
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(model, api_key, base_url))
    }

    /// Model name sent to the API.
    pub fn model(&self) -> &str {
        &self.model
    }

    #[cfg(test)]
    fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    async fn send(&self, request: HttpRequest) -> Result<ImageResponse, ImageError> {
        let request = if self.api_key.is_empty() {
            request
        } else {
            request.with_header("Authorization", format!("Bearer {}", self.api_key))
        };
        let response = self.client.send(request).await?;
        if !response.is_success() {
            return Err(api_error(&response));
        }
        parse_response(&response)
    }

    /// Bytes of an image to upload: decoded base64, or downloaded from its URL.
    async fn image_bytes(&self, media: &Media) -> Result<Vec<u8>, ImageError> {
        match &media.source {
            MediaSource::Base64(data) => base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| ImageError::InvalidResponse(format!("invalid base64 image: {e}"))),
            MediaSource::Url(url) => {
                let response = self.client.send(HttpRequest::get(url)).await?;
                if !response.is_success() {
                    return Err(api_error(&response));
                }
                Ok(response.body)
            }
        }
    }
}

#[derive(Serialize)]
struct GenerationBody<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<&'a str>,
}

#[async_trait]
impl ImageProvider for OpenAiImageProvider {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn generate(&self, request: ImageRequest) -> Result<ImageResponse, ImageError> {
        // No `response_format`: gpt-image models reject it and always answer with base64
        let body = GenerationBody {
            model: &self.model,
            prompt: &request.prompt,
            size: request.size.as_deref(),
            quality: request.quality.as_deref(),
        };
        let http =
            HttpRequest::post(format!("{}/images/generations", self.base_url)).with_json(&body)?;
        self.send(http).await
    }

    fn supports_edit(&self) -> bool {
        true
    }

    async fn edit(&self, request: ImageEditRequest) -> Result<ImageResponse, ImageError> {
        let mut form = Multipart::new()
            .with_text("model", self.model.as_str())
            .with_text("prompt", request.prompt);
        if let Some(size) = request.size {
            form = form.with_text("size", size);
        }
        if let Some(quality) = request.quality {
            form = form.with_text("quality", quality);
        }
        for (index, image) in request.images.iter().enumerate() {
            let bytes = self.image_bytes(image).await?;
            let filename = image.name.clone().unwrap_or_else(|| {
                let extension = image.content_type.rsplit('/').next().unwrap_or("png");
                format!("image-{}.{extension}", index + 1)
            });
            form = form.with_file("image[]", filename, image.content_type.as_str(), bytes);
        }
        let http =
            HttpRequest::post(format!("{}/images/edits", self.base_url)).with_multipart(form);
        self.send(http).await
    }
}

#[derive(Deserialize)]
struct ResponseBody {
    #[serde(default)]
    data: Vec<ImageData>,
    output_format: Option<String>,
}

#[derive(Deserialize)]
struct ImageData {
    b64_json: Option<String>,
    url: Option<String>,
    revised_prompt: Option<String>,
}

fn parse_response(response: &HttpResponse) -> Result<ImageResponse, ImageError> {
    let body: ResponseBody = response
        .json()
        .map_err(|e| ImageError::InvalidResponse(e.to_string()))?;
    let data = body
        .data
        .into_iter()
        .next()
        .ok_or_else(|| ImageError::InvalidResponse("no image in the response".to_string()))?;
    let image = match (data.b64_json, data.url) {
        (Some(b64), _) => {
            // Compatible servers may leave out `output_format`; the bytes tell the type then
            let content_type = match body.output_format.as_deref() {
                Some("jpg") => "image/jpeg".to_string(),
                Some(format) => format!("image/{format}"),
                None => sniff_image_type(&b64).to_string(),
            };
            Media::from_base64(content_type, b64)
        }
        (None, Some(url)) => Media::image_url(url),
        (None, None) => {
            return Err(ImageError::InvalidResponse(
                "the image has neither b64_json nor url".to_string(),
            ));
        }
    };
    let mut response = ImageResponse::new(image);
    response.revised_prompt = data.revised_prompt;
    Ok(response)
}

/// Image type from the first bytes of base64 data, `image/png` when unknown.
fn sniff_image_type(b64: &str) -> &'static str {
    let prefix: String = b64.chars().take(16).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(prefix)
        .unwrap_or_default();
    match bytes.as_slice() {
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [b'G', b'I', b'F', b'8', ..] => "image/gif",
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => "image/webp",
        _ => "image/png",
    }
}

/// Reads `error.message` of an OpenAI error, or falls back to the body.
fn api_error(response: &HttpResponse) -> ImageError {
    let message = response
        .json::<serde_json::Value>()
        .ok()
        .and_then(|json| json["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| response.text());
    ImageError::Api {
        status: response.status,
        message,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use agentai_core::HttpError;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// Records the requests and answers with the prepared responses, in order.
    #[derive(Default)]
    pub(crate) struct MockHttp {
        pub(crate) requests: Mutex<Vec<HttpRequest>>,
        responses: Mutex<VecDeque<HttpResponse>>,
    }

    impl MockHttp {
        pub(crate) fn new(responses: Vec<HttpResponse>) -> Arc<Self> {
            Arc::new(Self {
                requests: Mutex::default(),
                responses: Mutex::new(responses.into()),
            })
        }

        pub(crate) fn request(&self, index: usize) -> HttpRequest {
            self.requests.lock().unwrap()[index].clone()
        }
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.requests.lock().unwrap().push(request);
            Ok(self
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

    pub(crate) fn provider(http: &Arc<MockHttp>) -> OpenAiImageProvider {
        OpenAiImageProvider::new("gpt-image-1", "key", "https://api.test/v1/")
            .with_client(http.clone())
    }

    fn body_json(request: &HttpRequest) -> Value {
        serde_json::from_slice(request.body.as_ref().unwrap()).unwrap()
    }

    #[tokio::test]
    async fn generate_sends_only_the_set_parameters_and_reads_base64() {
        let http = MockHttp::new(vec![
            json_response(
                200,
                json!({"data": [{"b64_json": "aGk=", "revised_prompt": "a red cat"}],
                       "output_format": "webp"}),
            ),
            json_response(200, json!({"data": [{"b64_json": "aGk="}]})),
        ]);
        let provider = provider(&http);

        let response = provider
            .generate(ImageRequest::new("a cat").with_size("1024x1024"))
            .await
            .unwrap();
        provider.generate(ImageRequest::new("a dog")).await.unwrap();

        let request = http.request(0);
        assert_eq!(request.url, "https://api.test/v1/images/generations");
        assert_eq!(request.header("authorization"), Some("Bearer key"));
        assert_eq!(
            body_json(&request),
            json!({"model": "gpt-image-1", "prompt": "a cat", "size": "1024x1024"})
        );
        assert_eq!(
            body_json(&http.request(1)),
            json!({"model": "gpt-image-1", "prompt": "a dog"})
        );
        assert_eq!(
            response,
            ImageResponse::new(Media::from_base64("image/webp", "aGk="))
                .with_revised_prompt("a red cat")
        );
    }

    #[tokio::test]
    async fn without_output_format_the_type_comes_from_the_bytes_and_no_key_sends_no_header() {
        use base64::Engine;
        let jpeg = base64::engine::general_purpose::STANDARD.encode([0xFF, 0xD8, 0xFF, 0xE0, 0, 1]);
        // As a local OpenAI-compatible server answers: `url: null`, no `output_format`
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"created": 1, "data": [{"b64_json": jpeg, "url": null}]}),
        )]);
        let provider =
            OpenAiImageProvider::new("m", "", "http://local/v1").with_client(http.clone());

        let response = provider.generate(ImageRequest::new("x")).await.unwrap();

        assert_eq!(response.image.content_type, "image/jpeg");
        assert_eq!(http.request(0).header("authorization"), None);
    }

    #[tokio::test]
    async fn generate_keeps_a_url_without_downloading_it() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"data": [{"url": "https://cdn.test/a.png"}]}),
        )]);

        let response = provider(&http)
            .generate(ImageRequest::new("a cat"))
            .await
            .unwrap();

        assert_eq!(response.image, Media::image_url("https://cdn.test/a.png"));
        assert_eq!(http.requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn api_errors_carry_the_message() {
        let http = MockHttp::new(vec![
            json_response(
                400,
                json!({"error": {"message": "rejected by the safety system"}}),
            ),
            HttpResponse::new(502, "bad gateway"),
        ]);
        let provider = provider(&http);

        let err = provider.generate(ImageRequest::new("x")).await.unwrap_err();
        assert!(
            matches!(err, ImageError::Api { status: 400, ref message } if message == "rejected by the safety system")
        );
        let err = provider.generate(ImageRequest::new("x")).await.unwrap_err();
        assert!(
            matches!(err, ImageError::Api { status: 502, ref message } if message == "bad gateway")
        );
    }

    #[tokio::test]
    async fn edit_uploads_every_image_as_multipart() {
        let http = MockHttp::new(vec![
            // The image given by URL is downloaded first
            HttpResponse::new(200, "JPEGBYTES"),
            json_response(200, json!({"data": [{"b64_json": "aGk="}]})),
        ]);
        let images = vec![
            Media::from_bytes("image/png", b"PNGBYTES").with_name("cat.png"),
            Media::image_url("https://cdn.test/dog.jpg"),
        ];

        provider(&http)
            .edit(ImageEditRequest::new("add a hat", images).with_quality("low"))
            .await
            .unwrap();

        assert_eq!(http.request(0).url, "https://cdn.test/dog.jpg");
        assert_eq!(http.request(0).header("authorization"), None);
        let request = http.request(1);
        assert_eq!(request.url, "https://api.test/v1/images/edits");
        let content_type = request.header("content-type").unwrap();
        let boundary = content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        let body = String::from_utf8(request.body.clone().unwrap()).unwrap();
        let field = |name: &str, value: &str| {
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
        };
        let file = |filename: &str, content_type: &str, data: &str| {
            format!(
                "Content-Disposition: form-data; name=\"image[]\"; filename=\"{filename}\"\r\n\
                 Content-Type: {content_type}\r\n\r\n{data}\r\n"
            )
        };
        let parts: Vec<&str> = body
            .split(&format!("--{boundary}"))
            .map(|part| part.trim_start_matches("\r\n"))
            .collect();
        assert_eq!(
            parts,
            [
                "",
                &field("model", "gpt-image-1"),
                &field("prompt", "add a hat"),
                &field("quality", "low"),
                &file("cat.png", "image/png", "PNGBYTES"),
                &file("image-2.jpeg", "image/jpeg", "JPEGBYTES"),
                "--\r\n",
            ]
        );
    }

    #[test]
    fn from_env_reads_the_key_and_the_base_url() {
        let vars = |key: &'static str, base: Option<&'static str>| {
            move |name: &str| match name {
                API_KEY_ENV => Some(key.to_string()),
                BASE_URL_ENV => base.map(str::to_string),
                _ => None,
            }
        };
        let provider = OpenAiImageProvider::from_lookup("m", vars("k", None)).unwrap();
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
        assert_eq!(provider.api_key, "k");
        let provider =
            OpenAiImageProvider::from_lookup("m", vars("k", Some("http://local:8000/v1"))).unwrap();
        assert_eq!(provider.base_url, "http://local:8000/v1");
        // Local servers often need no key
        let provider = OpenAiImageProvider::from_lookup("m", |name| {
            (name == BASE_URL_ENV).then(|| "http://local:8000/v1".to_string())
        })
        .unwrap();
        assert_eq!(provider.api_key, "");
        assert!(matches!(
            OpenAiImageProvider::from_lookup("m", |_| None),
            Err(ImageError::MissingApiKey(API_KEY_ENV))
        ));
    }
}
