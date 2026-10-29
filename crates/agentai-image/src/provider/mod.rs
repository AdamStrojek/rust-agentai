//! # Image providers
//!
//! [`ImageProvider`] is the interface between [`ImageGenerator`](crate::ImageGenerator) and an
//! image generation API. Each submodule implements it for one API:
//!
//! - [`openai`]: the OpenAI image API and APIs compatible with it.
//!
//! Usually you do not pick a provider yourself: [`from_model`] (used by
//! [`ImageGenerator::new`](crate::ImageGenerator::new)) chooses it from the model name. Build one
//! yourself to set the API key or base URL in code, or implement [`ImageProvider`] for another
//! API.

pub mod openai;

use agentai_core::HttpError;
use agentai_core::message::Media;
use agentai_core::storage::StorageError;
use async_trait::async_trait;
use thiserror::Error;

/// A request to generate an image.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImageRequest {
    /// What to draw.
    pub prompt: String,
    /// Size, in the format of the provider (for example `1024x1024`). Not sent when `None`.
    pub size: Option<String>,
    /// Quality, in the format of the provider (for example `high`). Not sent when `None`.
    pub quality: Option<String>,
}

impl ImageRequest {
    /// Creates a request with a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            size: None,
            quality: None,
        }
    }

    /// Sets the size.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the quality.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }
}

/// A request to edit images, or to make a new image from them.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImageEditRequest {
    /// What to change.
    pub prompt: String,
    /// The images to edit.
    pub images: Vec<Media>,
    /// Size, see [`ImageRequest::size`].
    pub size: Option<String>,
    /// Quality, see [`ImageRequest::quality`].
    pub quality: Option<String>,
}

impl ImageEditRequest {
    /// Creates a request with a prompt and the images to edit.
    pub fn new(prompt: impl Into<String>, images: Vec<Media>) -> Self {
        Self {
            prompt: prompt.into(),
            images,
            size: None,
            quality: None,
        }
    }

    /// Sets the size.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the quality.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }
}

/// An image made by a provider.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImageResponse {
    /// The image, as bytes (base64) or as a URL, depending on the provider.
    pub image: Media,
    /// The prompt the provider actually used, if it rewrote yours.
    pub revised_prompt: Option<String>,
    /// ID of the image in the storage of the [`ImageGenerator`](crate::ImageGenerator), if
    /// it has one.
    pub id: Option<String>,
}

impl ImageResponse {
    /// Creates a response with an image.
    pub fn new(image: Media) -> Self {
        Self {
            image,
            revised_prompt: None,
            id: None,
        }
    }

    /// Sets the revised prompt.
    pub fn with_revised_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.revised_prompt = Some(prompt.into());
        self
    }
}

/// Errors of image providers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ImageError {
    /// The request could not be sent or the response could not be read.
    #[error(transparent)]
    Http(#[from] HttpError),
    /// The API answered with an error status.
    #[error("image API error (status {status}): {message}")]
    Api { status: u16, message: String },
    /// The API answered with something the provider cannot read.
    #[error("invalid image API response: {0}")]
    InvalidResponse(String),
    /// The provider cannot do this, for example edit images.
    #[error("not supported by the image provider: {0}")]
    Unsupported(&'static str),
    /// No provider matches the model name.
    #[error("no image provider for model '{0}'")]
    UnknownProvider(String),
    /// The environment variable with the API key is not set.
    #[error("the API key is missing, set {0}")]
    MissingApiKey(&'static str),
    /// The image was made but could not be saved in the storage.
    #[error("cannot save the image: {0}")]
    Storage(#[from] StorageError),
}

impl ImageError {
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

/// An image generation API.
///
/// Implement [`generate`](ImageProvider::generate); a provider that can edit images also
/// overrides [`supports_edit`](ImageProvider::supports_edit) and
/// [`edit`](ImageProvider::edit).
#[async_trait]
pub trait ImageProvider: Send + Sync + 'static {
    /// Name of the model, if the provider has one.
    fn model(&self) -> Option<&str> {
        None
    }

    /// Generates an image.
    async fn generate(&self, request: ImageRequest) -> Result<ImageResponse, ImageError>;

    /// Returns `true` when the provider implements [`edit`](ImageProvider::edit).
    fn supports_edit(&self) -> bool {
        false
    }

    /// Edits images. The default implementation returns [`ImageError::Unsupported`].
    async fn edit(&self, request: ImageEditRequest) -> Result<ImageResponse, ImageError> {
        let _ = request;
        Err(ImageError::Unsupported("image editing"))
    }
}

/// The APIs [`from_model`] knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    OpenAi,
}

/// Splits `provider::model` and finds the provider, like model names in `agentai`.
fn resolve(model: &str) -> Result<(ProviderKind, &str), ImageError> {
    let unknown = || ImageError::UnknownProvider(model.to_string());
    match model.split_once("::") {
        Some(("openai", name)) => Ok((ProviderKind::OpenAi, name)),
        Some(_) => Err(unknown()),
        None if model.starts_with("gpt-image") || model.starts_with("dall-e") => {
            Ok((ProviderKind::OpenAi, model))
        }
        None => Err(unknown()),
    }
}

/// Creates the provider for a model name, with its settings from environment variables.
///
/// The provider comes from a `provider::` prefix, for example `openai::gpt-image-1`, or, for
/// well-known names (`gpt-image-*`, `dall-e-*`), from the name alone. Use the prefix for other
/// models behind an OpenAI-compatible API. See each provider's `from_env` for its variables,
/// for example [`OpenAiImageProvider::from_env`](openai::OpenAiImageProvider::from_env).
pub fn from_model(model: &str) -> Result<Box<dyn ImageProvider>, ImageError> {
    match resolve(model)? {
        (ProviderKind::OpenAi, name) => Ok(Box::new(openai::OpenAiImageProvider::from_env(name)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_select_the_provider() {
        assert_eq!(
            resolve("openai::my-model").unwrap(),
            (ProviderKind::OpenAi, "my-model")
        );
        assert_eq!(
            resolve("gpt-image-1").unwrap(),
            (ProviderKind::OpenAi, "gpt-image-1")
        );
        assert_eq!(
            resolve("dall-e-3").unwrap(),
            (ProviderKind::OpenAi, "dall-e-3")
        );
        for model in ["flux-1", "other::gpt-image-1"] {
            assert!(matches!(resolve(model), Err(ImageError::UnknownProvider(m)) if m == model));
        }
    }
}
