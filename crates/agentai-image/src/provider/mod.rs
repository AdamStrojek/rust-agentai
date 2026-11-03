//! # Image providers
//!
//! [`ImageProvider`] is the interface between [`ImageGenerator`](crate::ImageGenerator) and an
//! image generation API. Each submodule implements it for one API:
//!
//! - [`openai`]: the OpenAI image API and APIs compatible with it.
//! - [`openrouter`]: the OpenRouter image API, with models of many vendors.
//!
//! [`list_models`] lists the image models a provider offers, with names ready for
//! [`ImageGenerator::new`](crate::ImageGenerator::new).
//!
//! Usually you do not pick a provider yourself: [`from_model`] (used by
//! [`ImageGenerator::new`](crate::ImageGenerator::new)) chooses it from the model name. Build one
//! yourself to set the API key or base URL in code, or implement [`ImageProvider`] for another
//! API.

pub mod openai;
pub mod openrouter;

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
    /// File name of the image, set as [`Media::name`] and so passed to the storage. Not sent
    /// to the provider. Without an extension, the one of the image type is added.
    pub name: Option<String>,
}

impl ImageRequest {
    /// Creates a request with a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            size: None,
            quality: None,
            name: None,
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

    /// Sets the file name of the image, see [`ImageRequest::name`].
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
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
    /// File name of the new image, see [`ImageRequest::name`].
    pub name: Option<String>,
}

impl ImageEditRequest {
    /// Creates a request with a prompt and the images to edit.
    pub fn new(prompt: impl Into<String>, images: Vec<Media>) -> Self {
        Self {
            prompt: prompt.into(),
            images,
            size: None,
            quality: None,
            name: None,
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

    /// Sets the file name of the image, see [`ImageRequest::name`].
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
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

/// An image model offered by a provider, see [`list_models`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImageModel {
    /// Name for [`ImageGenerator::new`](crate::ImageGenerator::new) and [`from_model`], with
    /// the provider prefix, for example `openrouter::bytedance-seed/seedream-4.5`.
    pub model: String,
    /// Name of the model in the API.
    pub id: String,
    /// Display name.
    pub name: Option<String>,
    /// Description.
    pub description: Option<String>,
    /// Whether the model takes input images (needed to edit), when the API says so.
    pub accepts_images: Option<bool>,
}

impl ImageModel {
    /// Creates a model with its full name and its name in the API.
    pub fn new(model: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            id: id.into(),
            name: None,
            description: None,
            accepts_images: None,
        }
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

    /// Lists the image models of the API. The default implementation returns
    /// [`ImageError::Unsupported`].
    async fn list_models(&self) -> Result<Vec<ImageModel>, ImageError> {
        Err(ImageError::Unsupported("listing models"))
    }
}

/// The APIs [`from_model`] knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    OpenAi,
    OpenRouter,
}

impl ProviderKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "openai" => Some(Self::OpenAi),
            "openrouter" => Some(Self::OpenRouter),
            _ => None,
        }
    }

    fn build(self, model: &str) -> Result<Box<dyn ImageProvider>, ImageError> {
        Ok(match self {
            Self::OpenAi => Box::new(openai::OpenAiImageProvider::from_env(model)?),
            Self::OpenRouter => Box::new(openrouter::OpenRouterImageProvider::from_env(model)?),
        })
    }
}

/// Splits `provider::model` and finds the provider, like model names in `agentai`.
fn resolve(model: &str) -> Result<(ProviderKind, &str), ImageError> {
    let unknown = || ImageError::UnknownProvider(model.to_string());
    match model.split_once("::") {
        Some((provider, name)) => ProviderKind::from_name(provider)
            .map(|kind| (kind, name))
            .ok_or_else(unknown),
        None if model.starts_with("gpt-image") || model.starts_with("dall-e") => {
            Ok((ProviderKind::OpenAi, model))
        }
        None => Err(unknown()),
    }
}

/// Creates the provider for a model name, with its settings from environment variables.
///
/// The provider comes from a `provider::` prefix (`openai::`, `openrouter::`), for example
/// `openai::gpt-image-1` or `openrouter::bytedance-seed/seedream-4.5`, or, for
/// well-known names (`gpt-image-*`, `dall-e-*`), from the name alone. Use the prefix for other
/// models behind an OpenAI-compatible API. See each provider's `from_env` for its variables,
/// for example [`OpenAiImageProvider::from_env`](openai::OpenAiImageProvider::from_env).
pub fn from_model(model: &str) -> Result<Box<dyn ImageProvider>, ImageError> {
    let (kind, name) = resolve(model)?;
    kind.build(name)
}

/// Lists the image models of a provider (`openai` or `openrouter`), with its settings from
/// environment variables as in [`from_model`].
///
/// ```rust,no_run
/// # async fn run() -> Result<(), agentai_image::ImageError> {
/// for model in agentai_image::provider::list_models("openrouter").await? {
///     println!("{} {:?}", model.model, model.name);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn list_models(provider: &str) -> Result<Vec<ImageModel>, ImageError> {
    let kind = ProviderKind::from_name(provider)
        .ok_or_else(|| ImageError::UnknownProvider(provider.to_string()))?;
    kind.build("")?.list_models().await
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
            resolve("openrouter::black-forest-labs/flux.2").unwrap(),
            (ProviderKind::OpenRouter, "black-forest-labs/flux.2")
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
