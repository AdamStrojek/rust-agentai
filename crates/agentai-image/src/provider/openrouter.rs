//! # OpenRouter image API
//!
//! [`OpenRouterImageProvider`] generates images with `POST /images`, the dedicated image API of
//! [OpenRouter](https://openrouter.ai), and lists image models with
//! `GET /models?output_modalities=image`. Images to edit are sent as `input_references`, so
//! editing works with every model that takes input images.

use super::openai::{parse_response, send_request};
use super::{ImageEditRequest, ImageError, ImageModel, ImageProvider, ImageRequest, ImageResponse};
use agentai_core::message::{Media, MediaSource};
use agentai_core::{HttpClient, HttpRequest, ReqwestClient};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The OpenRouter API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Environment variable with the API key.
pub const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
/// Environment variable with the base URL.
pub const BASE_URL_ENV: &str = "OPENROUTER_BASE_URL";

/// Provider for the OpenRouter image API.
///
/// ```rust,no_run
/// # use agentai_image::provider::openrouter::{DEFAULT_BASE_URL, OpenRouterImageProvider};
/// # fn main() -> Result<(), agentai_image::ImageError> {
/// // From `OPENROUTER_API_KEY` and, if set, `OPENROUTER_BASE_URL`
/// let provider = OpenRouterImageProvider::from_env("bytedance-seed/seedream-4.5")?;
/// // Or everything in code
/// let provider =
///     OpenRouterImageProvider::new("bytedance-seed/seedream-4.5", "sk-or-...", DEFAULT_BASE_URL);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenRouterImageProvider {
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenRouterImageProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterImageProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenRouterImageProvider {
    /// Creates a provider for `model` with the API key and base URL given in code, for example
    /// [`DEFAULT_BASE_URL`].
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

    /// Creates a provider for `model` with the API key from `OPENROUTER_API_KEY` and the base
    /// URL from `OPENROUTER_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ImageError> {
        let api_key =
            std::env::var(API_KEY_ENV).map_err(|_| ImageError::MissingApiKey(API_KEY_ENV))?;
        let base_url = std::env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
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

    async fn send(&self, body: &ImageBody<'_>) -> Result<ImageResponse, ImageError> {
        let request = HttpRequest::post(format!("{}/images", self.base_url)).with_json(body)?;
        parse_response(&send_request(&*self.client, &self.api_key, request).await?)
    }

    /// Origin of the base URL, to resolve the relative `links.next` of a model list.
    fn origin(&self) -> &str {
        let after_scheme = self.base_url.find("://").map_or(0, |i| i + 3);
        match self.base_url[after_scheme..].find('/') {
            Some(i) => &self.base_url[..after_scheme + i],
            None => &self.base_url,
        }
    }
}

#[derive(Serialize)]
struct ImageBody<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<&'a str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_references: Vec<ImageReference>,
}

#[derive(Serialize)]
struct ImageReference {
    r#type: &'static str,
    image_url: ImageUrl,
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

impl ImageReference {
    /// A data URL for bytes, the URL itself otherwise.
    fn new(media: &Media) -> Self {
        let url = match &media.source {
            MediaSource::Base64(data) => format!("data:{};base64,{data}", media.content_type),
            MediaSource::Url(url) => url.clone(),
        };
        Self {
            r#type: "image_url",
            image_url: ImageUrl { url },
        }
    }
}

#[derive(Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
    links: Option<Links>,
}

#[derive(Deserialize)]
struct Links {
    next: Option<String>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    name: Option<String>,
    description: Option<String>,
    architecture: Option<Architecture>,
}

#[derive(Deserialize)]
struct Architecture {
    input_modalities: Option<Vec<String>>,
}

#[async_trait]
impl ImageProvider for OpenRouterImageProvider {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn generate(&self, request: ImageRequest) -> Result<ImageResponse, ImageError> {
        self.send(&ImageBody {
            model: &self.model,
            prompt: &request.prompt,
            size: request.size.as_deref(),
            quality: request.quality.as_deref(),
            input_references: Vec::new(),
        })
        .await
    }

    fn supports_edit(&self) -> bool {
        true
    }

    /// Sends the images as `input_references`; models that do not take input images answer
    /// with an API error.
    async fn edit(&self, request: ImageEditRequest) -> Result<ImageResponse, ImageError> {
        self.send(&ImageBody {
            model: &self.model,
            prompt: &request.prompt,
            size: request.size.as_deref(),
            quality: request.quality.as_deref(),
            input_references: request.images.iter().map(ImageReference::new).collect(),
        })
        .await
    }

    /// Lists the models that output images, following every page of the list.
    async fn list_models(&self) -> Result<Vec<ImageModel>, ImageError> {
        let mut url = format!("{}/models?output_modalities=image", self.base_url);
        let mut models = Vec::new();
        loop {
            let response =
                send_request(&*self.client, &self.api_key, HttpRequest::get(&url)).await?;
            let list: ModelList = response
                .json()
                .map_err(|e| ImageError::InvalidResponse(e.to_string()))?;
            models.extend(list.data.into_iter().map(|entry| {
                let mut model = ImageModel::new(format!("openrouter::{}", entry.id), entry.id);
                model.name = entry.name;
                model.description = entry.description;
                model.accepts_images = entry
                    .architecture
                    .and_then(|a| a.input_modalities)
                    .map(|inputs| inputs.iter().any(|input| input == "image"));
                model
            }));
            match list.links.and_then(|links| links.next) {
                Some(next) if next.starts_with('/') => url = format!("{}{next}", self.origin()),
                Some(next) => url = next,
                None => return Ok(models),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::tests::{MockHttp, json_response};
    use agentai_core::HttpRequest;
    use serde_json::{Value, json};

    fn provider(http: &Arc<MockHttp>) -> OpenRouterImageProvider {
        OpenRouterImageProvider::new("vendor/model", "key", "https://router.test/api/v1/")
            .with_client(http.clone())
    }

    fn body_json(request: &HttpRequest) -> Value {
        serde_json::from_slice(request.body.as_ref().unwrap()).unwrap()
    }

    #[tokio::test]
    async fn generate_posts_to_images_and_reads_the_media_type() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"created": 1, "data": [{"b64_json": "aGk=", "media_type": "image/webp"}]}),
        )]);

        let response = provider(&http)
            .generate(ImageRequest::new("a cat").with_size("1K"))
            .await
            .unwrap();

        let request = http.request(0);
        assert_eq!(request.url, "https://router.test/api/v1/images");
        assert_eq!(request.header("Authorization"), Some("Bearer key"));
        assert_eq!(
            body_json(&request),
            json!({"model": "vendor/model", "prompt": "a cat", "size": "1K"})
        );
        assert_eq!(response.image, Media::from_base64("image/webp", "aGk="));
    }

    #[tokio::test]
    async fn edit_sends_bytes_as_data_urls_and_urls_as_they_are() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"data": [{"b64_json": "aGk="}]}),
        )]);
        let images = vec![
            Media::from_base64("image/png", "aGk="),
            Media::image_url("https://example.com/cat.jpg"),
        ];

        provider(&http)
            .edit(ImageEditRequest::new("add a hat", images))
            .await
            .unwrap();

        assert_eq!(
            body_json(&http.request(0))["input_references"],
            json!([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
            ])
        );
    }

    #[tokio::test]
    async fn listing_asks_for_image_models_and_follows_every_page() {
        let http = MockHttp::new(vec![
            json_response(
                200,
                json!({
                    "data": [{
                        "id": "a/edit",
                        "name": "Edit",
                        "architecture": {"input_modalities": ["text", "image"]},
                    }],
                    "links": {"next": "/api/v1/models?output_modalities=image&offset=1"},
                }),
            ),
            json_response(
                200,
                json!({
                    "data": [{"id": "b/draw", "architecture": {"input_modalities": ["text"]}}],
                    "links": {"next": null},
                }),
            ),
        ]);

        let models = provider(&http).list_models().await.unwrap();

        assert_eq!(
            http.request(0).url,
            "https://router.test/api/v1/models?output_modalities=image"
        );
        assert_eq!(
            http.request(1).url,
            "https://router.test/api/v1/models?output_modalities=image&offset=1"
        );
        let mut edit = ImageModel::new("openrouter::a/edit", "a/edit");
        edit.name = Some("Edit".to_string());
        edit.accepts_images = Some(true);
        let mut draw = ImageModel::new("openrouter::b/draw", "b/draw");
        draw.accepts_images = Some(false);
        assert_eq!(models, [edit, draw]);
    }
}
