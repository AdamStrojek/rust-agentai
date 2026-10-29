//! [`ImageGenerator`]: generate and edit images from your code.

use crate::provider::{
    self, ImageEditRequest, ImageError, ImageProvider, ImageRequest, ImageResponse,
};
use agentai_core::message::Media;
use agentai_core::storage::MediaSave;
use std::sync::Arc;

/// Generates and edits images with an [`ImageProvider`].
///
/// It is cheap to clone: clones share the provider and the storage.
///
/// ```rust,no_run
/// # use agentai_image::ImageGenerator;
/// # async fn run() -> Result<(), agentai_image::ImageError> {
/// let generator = ImageGenerator::new("openai::gpt-image-1")?.with_size("1024x1024");
/// let response = generator.generate("A lighthouse at dawn").await?;
/// println!("{} image", response.image.content_type);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ImageGenerator {
    provider: Arc<dyn ImageProvider>,
    storage: Option<Arc<dyn MediaSave>>,
    size: Option<String>,
    quality: Option<String>,
}

impl std::fmt::Debug for ImageGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageGenerator")
            .field("model", &self.provider.model())
            .field("storage", &self.storage.is_some())
            .field("size", &self.size)
            .field("quality", &self.quality)
            .finish()
    }
}

impl ImageGenerator {
    /// Creates a generator for a model, see [`provider::from_model`]: the provider comes from
    /// the name (`openai::gpt-image-1`) and its settings from environment variables.
    pub fn new(model: &str) -> Result<Self, ImageError> {
        Ok(Self::with_provider_arc(provider::from_model(model)?.into()))
    }

    /// Creates a generator with a provider you built, for example with the API key set in
    /// code.
    pub fn from_provider(provider: impl ImageProvider) -> Self {
        Self::with_provider_arc(Arc::new(provider))
    }

    fn with_provider_arc(provider: Arc<dyn ImageProvider>) -> Self {
        Self {
            provider,
            storage: None,
            size: None,
            quality: None,
        }
    }

    /// Saves every image in `storage` and sets [`ImageResponse::id`].
    pub fn with_storage(mut self, storage: impl MediaSave) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    /// Size of every image that does not set its own, in the provider's format (for example
    /// `1024x1024`).
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Quality of every image that does not set its own, in the provider's format (for
    /// example `high`).
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    /// Name of the model, if the provider has one.
    pub fn model(&self) -> Option<&str> {
        self.provider.model()
    }

    /// The default size.
    pub fn size(&self) -> Option<&str> {
        self.size.as_deref()
    }

    /// The default quality.
    pub fn quality(&self) -> Option<&str> {
        self.quality.as_deref()
    }

    /// Returns `true` when the provider can edit images.
    pub fn supports_edit(&self) -> bool {
        self.provider.supports_edit()
    }

    /// Generates an image from a prompt.
    pub async fn generate(&self, prompt: impl Into<String>) -> Result<ImageResponse, ImageError> {
        self.generate_with(ImageRequest::new(prompt)).await
    }

    /// Generates an image from a full request. Unset size and quality take the defaults.
    pub async fn generate_with(
        &self,
        mut request: ImageRequest,
    ) -> Result<ImageResponse, ImageError> {
        request.size = request.size.or_else(|| self.size.clone());
        request.quality = request.quality.or_else(|| self.quality.clone());
        let response = self.provider.generate(request).await?;
        self.store(response).await
    }

    /// Edits images following a prompt.
    pub async fn edit(
        &self,
        prompt: impl Into<String>,
        images: Vec<Media>,
    ) -> Result<ImageResponse, ImageError> {
        self.edit_with(ImageEditRequest::new(prompt, images)).await
    }

    /// Edits images with a full request. Unset size and quality take the defaults.
    pub async fn edit_with(
        &self,
        mut request: ImageEditRequest,
    ) -> Result<ImageResponse, ImageError> {
        request.size = request.size.or_else(|| self.size.clone());
        request.quality = request.quality.or_else(|| self.quality.clone());
        let response = self.provider.edit(request).await?;
        self.store(response).await
    }

    /// Names the image `image.<ext>` when the provider did not, and saves it.
    async fn store(&self, mut response: ImageResponse) -> Result<ImageResponse, ImageError> {
        if response.image.name.is_none() {
            let extension = response
                .image
                .content_type
                .rsplit('/')
                .next()
                .unwrap_or("png");
            response.image.name = Some(format!("image.{extension}"));
        }
        if let Some(storage) = &self.storage {
            response.id = Some(storage.save(&response.image).await?);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::tests::{MockHttp, json_response, provider};
    use agentai_core::storage::{MediaRead, MemoryStorage};
    use serde_json::{Value, json};

    #[tokio::test]
    async fn defaults_fill_unset_parameters_and_the_storage_sets_the_id() {
        let http = MockHttp::new(vec![
            json_response(200, json!({"data": [{"b64_json": "aGk="}]})),
            json_response(200, json!({"data": [{"b64_json": "aGk="}]})),
        ]);
        let storage = MemoryStorage::new();
        let generator = ImageGenerator::from_provider(provider(&http))
            .with_size("512x512")
            .with_quality("low")
            .with_storage(storage.clone());

        let response = generator.generate("a cat").await.unwrap();
        generator
            .generate_with(ImageRequest::new("a dog").with_size("1024x1024"))
            .await
            .unwrap();

        let body =
            |i: usize| -> Value { serde_json::from_slice(&http.request(i).body.unwrap()).unwrap() };
        assert_eq!(body(0)["size"], json!("512x512"));
        assert_eq!(body(0)["quality"], json!("low"));
        assert_eq!(body(1)["size"], json!("1024x1024"));
        let id = response.id.clone().unwrap();
        assert_eq!(storage.read(&id).await.unwrap(), response.image);
        assert_eq!(response.image.name.as_deref(), Some("image.png"));
    }
}
