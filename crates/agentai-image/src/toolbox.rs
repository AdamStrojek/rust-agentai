//! Agent tools (feature `toolbox`): [`GenerateImageToolBox`] and [`ReadMediaToolBox`].

use crate::generator::ImageGenerator;
use crate::provider::{ImageEditRequest, ImageError, ImageRequest, ImageResponse};
use agentai::tool::{Tool, ToolBox, ToolError, ToolOutput, ToolResult, toolbox};
use agentai_core::message::MediaSource;
use agentai_core::storage::{MediaRead, MediaStorage, StorageError};
use serde::Serialize;
use std::sync::Arc;

/// What [`GenerateImageToolBox`] sends back to the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageToolOutput {
    /// The image itself and a short text with its ID. The model must accept images.
    #[default]
    Media,
    /// Only a JSON text with what was made (ID, type, size in bytes, model, prompts, ...),
    /// without the image. Saves tokens, and the model can still look at the image with
    /// [`ReadMediaToolBox`].
    Metadata,
}

/// [`ToolBox`] that lets an agent generate and edit images with an [`ImageGenerator`].
///
/// Tools:
/// - `generate_image`: draws an image from a prompt.
/// - `edit_image`: changes stored images the model names by ID. It needs a storage
///   ([`with_storage`](GenerateImageToolBox::with_storage)) and a provider that can edit;
///   without them the model gets an error.
///
/// Both take a `name` from the model (for example `lighthouse`), set as
/// [`Media::name`](agentai_core::message::Media::name) and so passed to
/// [`MediaSave`](agentai_core::storage::MediaSave); the storage chooses the ID and may use it.
///
/// With a storage every image is saved and the model gets its ID. What else the model gets
/// is set with [`with_output`](GenerateImageToolBox::with_output).
///
/// ```rust,no_run
/// # use agentai_image::{GenerateImageToolBox, ImageGenerator};
/// # use agentai_image::storage::MemoryStorage;
/// # fn main() -> Result<(), agentai_image::ImageError> {
/// let storage = MemoryStorage::new();
/// let images = GenerateImageToolBox::new(ImageGenerator::new("openai::gpt-image-1")?)
///     .with_storage(storage.clone());
/// # Ok(())
/// # }
/// ```
pub struct GenerateImageToolBox {
    generator: ImageGenerator,
    storage: Option<Arc<dyn MediaStorage>>,
    output: ImageToolOutput,
}

impl GenerateImageToolBox {
    /// Creates the toolbox. A storage set on `generator` is used to save the images, but
    /// `edit_image` needs one set with [`with_storage`](GenerateImageToolBox::with_storage).
    pub fn new(generator: ImageGenerator) -> Self {
        Self {
            generator,
            storage: None,
            output: ImageToolOutput::default(),
        }
    }

    /// Saves every image in `storage` and loads the images to edit from it.
    pub fn with_storage(mut self, storage: impl MediaStorage) -> Self {
        let storage: Arc<dyn MediaStorage> = Arc::new(storage);
        self.generator = self.generator.with_storage(storage.clone());
        self.storage = Some(storage);
        self
    }

    /// Sets what the model gets back, see [`ImageToolOutput`].
    pub fn with_output(mut self, output: ImageToolOutput) -> Self {
        self.output = output;
        self
    }

    fn output(&self, response: ImageResponse, prompt: &str) -> ToolResult<ToolOutput> {
        match self.output {
            ImageToolOutput::Metadata => {
                let metadata = ImageMetadata {
                    id: response.id.as_deref(),
                    content_type: &response.image.content_type,
                    url: match &response.image.source {
                        MediaSource::Url(url) => Some(url),
                        MediaSource::Base64(_) => None,
                    },
                    bytes: response.image.byte_len(),
                    model: self.generator.model(),
                    prompt,
                    revised_prompt: response.revised_prompt.as_deref(),
                    size: self.generator.size(),
                    quality: self.generator.quality(),
                };
                let json = serde_json::to_string(&metadata).map_err(ToolError::fatal)?;
                Ok(ToolOutput::from_text(json))
            }
            _ => {
                let mut text = match &response.id {
                    Some(id) => format!("Created the image with ID {id}."),
                    None => "Created the image.".to_string(),
                };
                if let Some(revised) = &response.revised_prompt {
                    text.push_str(&format!(" Prompt used: {revised}"));
                }
                Ok(ToolOutput::from_text(text).with_media(response.image))
            }
        }
    }
}

#[toolbox]
impl GenerateImageToolBox {
    /// Generates an image from a text prompt.
    #[tool]
    async fn generate_image(
        &self,
        /// Detailed description of the image: subject, style, composition, colors, text in it.
        prompt: String,
        /// Short file name for the new image, for example `lighthouse`. The storage may add to
        /// it to make the ID unique.
        name: String,
    ) -> ToolResult<ToolOutput> {
        let response = self
            .generator
            .generate_with(ImageRequest::new(&prompt).with_name(name))
            .await
            .map_err(tool_error)?;
        self.output(response, &prompt)
    }

    /// Edits existing images, given by ID, following a text prompt, and creates a new image.
    #[tool]
    async fn edit_image(
        &self,
        /// What to change, or how to combine the images.
        prompt: String,
        /// IDs of the images to edit, as returned by `generate_image` or `edit_image`.
        images: Vec<String>,
        /// Short file name for the new image, for example `lighthouse-at-night`. The storage
        /// may add to it to make the ID unique.
        name: String,
    ) -> ToolResult<ToolOutput> {
        let Some(storage) = &self.storage else {
            return Err(ToolError::Failed(
                "editing is not available: there is no storage for the images".to_string(),
            ));
        };
        if images.is_empty() {
            return Err(ToolError::InvalidArguments(
                "give at least one image ID in `images`".to_string(),
            ));
        }
        let mut media = Vec::with_capacity(images.len());
        for id in &images {
            media.push(storage.read(id).await.map_err(read_error)?);
        }
        let response = self
            .generator
            .edit_with(ImageEditRequest::new(&prompt, media).with_name(name))
            .await
            .map_err(tool_error)?;
        self.output(response, &prompt)
    }
}

/// What [`ImageToolOutput::Metadata`] sends; `None` fields are left out.
#[derive(Serialize)]
struct ImageMetadata<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    content_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<&'a str>,
}

/// Rejected credentials cannot be fixed by the model; anything else (a refused prompt, a
/// provider that cannot edit, a failed request or save) is reported to it.
fn tool_error(err: ImageError) -> ToolError {
    if err.is_auth() || matches!(err, ImageError::MissingApiKey(_)) {
        ToolError::Configuration(err.to_string())
    } else {
        ToolError::Failed(err.to_string())
    }
}

/// An unknown ID is the model's mistake, so it can correct it.
fn read_error(err: StorageError) -> ToolError {
    match err {
        StorageError::NotFound(id) => {
            ToolError::InvalidArguments(format!("there is no file with ID '{id}'"))
        }
        err => ToolError::Failed(err.to_string()),
    }
}

/// [`ToolBox`] with the tool `read_media`, which gives the model a stored file (an image, a
/// document, ...) by its ID, read from any [`MediaRead`].
///
/// Share one storage with [`GenerateImageToolBox`] to let the model look at images it made
/// with [`ImageToolOutput::Metadata`]:
///
/// ```rust
/// # use agentai_image::ReadMediaToolBox;
/// # use agentai_image::storage::MemoryStorage;
/// let storage = MemoryStorage::new();
/// let files = ReadMediaToolBox::new(storage.clone());
/// ```
pub struct ReadMediaToolBox {
    storage: Arc<dyn MediaRead>,
}

impl ReadMediaToolBox {
    /// Creates the toolbox for files in `storage`.
    pub fn new(storage: impl MediaRead) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }
}

#[toolbox]
impl ReadMediaToolBox {
    /// Returns a stored file, for example an image, by its ID, so you can look at it.
    #[tool]
    async fn read_media(
        &self,
        /// ID of the file.
        id: String,
    ) -> ToolResult<ToolOutput> {
        let media = self.storage.read(&id).await.map_err(read_error)?;
        let text = format!("File {id} ({})", media.content_type);
        Ok(ToolOutput::from_text(text).with_media(media))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::tests::{MockHttp, json_response, provider};
    use agentai_core::message::{Media, Part};
    use agentai_core::storage::{MediaSave, MemoryStorage};
    use serde_json::{Value, json};

    fn tool_call(name: &str, arguments: Value) -> agentai::ToolCall {
        agentai::ToolCall::new("call_1", name, arguments)
    }

    fn image_reply() -> agentai_core::HttpResponse {
        json_response(
            200,
            json!({"data": [{"b64_json": "aGk=", "revised_prompt": "a red cat"}]}),
        )
    }

    fn toolbox(http: &Arc<MockHttp>) -> GenerateImageToolBox {
        GenerateImageToolBox::new(
            ImageGenerator::from_provider(provider(http)).with_size("512x512"),
        )
    }

    #[tokio::test]
    async fn generate_image_returns_the_image_with_its_storage_id() {
        let http = MockHttp::new(vec![image_reply()]);
        let storage = MemoryStorage::new();
        let toolbox = toolbox(&http).with_storage(storage.clone());

        let output = toolbox
            .call_tool(&tool_call(
                "generate_image",
                json!({"prompt": "a cat", "name": "image"}),
            ))
            .await
            .unwrap();

        let image = Media::from_base64("image/png", "aGk=").with_name("image.png");
        assert_eq!(
            output.parts,
            vec![
                Part::Text("Created the image with ID 1-image.png. Prompt used: a red cat".into()),
                Part::Media(image.clone()),
            ]
        );
        assert_eq!(storage.read("1-image.png").await.unwrap(), image);
    }

    #[tokio::test]
    async fn the_name_given_by_the_model_reaches_the_storage() {
        let http = MockHttp::new(vec![image_reply(), image_reply()]);
        let storage = MemoryStorage::new();
        let toolbox = toolbox(&http).with_storage(storage.clone());

        toolbox
            .call_tool(&tool_call(
                "generate_image",
                json!({"prompt": "a cat", "name": "cat"}),
            ))
            .await
            .unwrap();
        toolbox
            .call_tool(&tool_call(
                "edit_image",
                json!({"prompt": "add a hat", "images": ["1-cat.png"], "name": "cat-with-hat"}),
            ))
            .await
            .unwrap();

        let mut ids = storage.ids();
        ids.sort();
        assert_eq!(ids, ["1-cat.png", "2-cat-with-hat.png"]);
    }

    #[tokio::test]
    async fn metadata_output_describes_the_image_without_it() {
        let http = MockHttp::new(vec![image_reply()]);
        let toolbox = toolbox(&http)
            .with_storage(MemoryStorage::new())
            .with_output(ImageToolOutput::Metadata);

        let output = toolbox
            .call_tool(&tool_call(
                "generate_image",
                json!({"prompt": "a cat", "name": "image"}),
            ))
            .await
            .unwrap();

        assert!(output.media().is_empty());
        let metadata: Value = serde_json::from_str(&output.text().unwrap()).unwrap();
        assert_eq!(
            metadata,
            json!({
                "id": "1-image.png",
                "content_type": "image/png",
                "bytes": 2,
                "model": "gpt-image-1",
                "prompt": "a cat",
                "revised_prompt": "a red cat",
                "size": "512x512",
            })
        );
    }

    #[tokio::test]
    async fn edit_image_needs_a_storage_and_loads_the_images_from_it() {
        let http = MockHttp::new(vec![image_reply()]);
        let without = toolbox(&http);
        assert!(matches!(
            without
                .call_tool(&tool_call(
                    "edit_image",
                    json!({"prompt": "add a hat", "images": ["1"], "name": "hat"})
                ))
                .await,
            Err(ToolError::Failed(_))
        ));

        let storage = MemoryStorage::new();
        let id = storage
            .save(&Media::from_bytes("image/png", b"PNG"))
            .await
            .unwrap();
        let toolbox = toolbox(&http).with_storage(storage);
        let unknown = toolbox
            .call_tool(&tool_call(
                "edit_image",
                json!({"prompt": "add a hat", "images": ["nope"], "name": "hat"}),
            ))
            .await;
        assert!(matches!(unknown, Err(ToolError::InvalidArguments(_))));
        let output = toolbox
            .call_tool(&tool_call(
                "edit_image",
                json!({"prompt": "add a hat", "images": [id], "name": "hat"}),
            ))
            .await
            .unwrap();

        assert_eq!(output.media().len(), 1);
        let request = http.request(0);
        assert!(request.url.ends_with("/images/edits"));
        assert!(String::from_utf8_lossy(&request.body.unwrap()).contains("\r\n\r\nPNG\r\n"));
    }

    #[tokio::test]
    async fn rejected_keys_stop_the_run_and_other_api_errors_reach_the_model() {
        let http = MockHttp::new(vec![
            json_response(401, json!({"error": {"message": "bad key"}})),
            json_response(400, json!({"error": {"message": "refused"}})),
        ]);
        let toolbox = toolbox(&http);
        let request = tool_call("generate_image", json!({"prompt": "x", "name": "x"}));
        let call = || toolbox.call_tool(&request);

        assert!(matches!(call().await, Err(ToolError::Configuration(_))));
        assert!(matches!(call().await, Err(ToolError::Failed(m)) if m.contains("refused")));
    }

    #[tokio::test]
    async fn read_media_returns_the_stored_file() {
        let storage = MemoryStorage::new();
        let pdf = Media::from_bytes("application/pdf", b"%PDF");
        let id = storage.save(&pdf).await.unwrap();
        let toolbox = ReadMediaToolBox::new(storage);

        let output = toolbox
            .call_tool(&tool_call("read_media", json!({ "id": id })))
            .await
            .unwrap();
        assert_eq!(output.media(), vec![&pdf]);
        assert!(matches!(
            toolbox
                .call_tool(&tool_call("read_media", json!({"id": "nope"})))
                .await,
            Err(ToolError::InvalidArguments(_))
        ));
    }
}
