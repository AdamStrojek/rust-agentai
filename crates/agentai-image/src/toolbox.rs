//! Agent tools (feature `toolbox`): [`GenerateImageToolBox`] and [`ReadMediaToolBox`].

use crate::generator::ImageGenerator;
use crate::provider::{ImageEditRequest, ImageError, ImageRequest, ImageResponse};
use agentai::tool::{Tool, ToolBox, ToolError, ToolOutput, ToolResult};
use agentai_core::message::{Media, MediaSource};
use agentai_core::storage::{MediaRead, MediaStorage, StorageError};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

const GENERATE_IMAGE: &str = "generate_image";
const EDIT_IMAGE: &str = "edit_image";
const READ_MEDIA: &str = "read_media";

/// What [`GenerateImageToolBox`] sends back to the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageToolOutput {
    /// The image itself and a short text with its ID. Providers that do not accept images in
    /// tool results get only the text.
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
/// - `edit_image`: changes stored images the model names by ID. Offered only with a storage
///   ([`with_storage`](GenerateImageToolBox::with_storage)) and a provider that can edit.
///
/// With a storage every image is saved and the model gets its ID. What else the model gets
/// is set with [`with_output`](GenerateImageToolBox::with_output).
///
/// ```rust,no_run
/// # use agentai_image::{GenerateImageToolBox, ImageGenerator};
/// # use agentai_core::storage::MemoryStorage;
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

    /// Saves every image in `storage` and loads the images to edit from it, which enables
    /// `edit_image`.
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

    fn can_edit(&self) -> bool {
        self.storage.is_some() && self.generator.supports_edit()
    }

    async fn generate(&self, args: GenerateArgs) -> ToolResult<ToolOutput> {
        let prompt = args.prompt.clone();
        let response = self
            .generator
            .generate_with(ImageRequest::new(args.prompt))
            .await
            .map_err(tool_error)?;
        self.output(response, prompt)
    }

    async fn edit(&self, args: EditArgs) -> ToolResult<ToolOutput> {
        let Some(storage) = self.storage.as_ref().filter(|_| self.can_edit()) else {
            return Err(ToolError::NotFound(EDIT_IMAGE.to_string()));
        };
        if args.images.is_empty() {
            return Err(ToolError::InvalidArguments(
                "give at least one image ID in `images`".to_string(),
            ));
        }
        let mut images = Vec::with_capacity(args.images.len());
        for id in &args.images {
            images.push(storage.read(id).await.map_err(read_error)?);
        }
        let prompt = args.prompt.clone();
        let response = self
            .generator
            .edit_with(ImageEditRequest::new(args.prompt, images))
            .await
            .map_err(tool_error)?;
        self.output(response, prompt)
    }

    fn output(&self, response: ImageResponse, prompt: String) -> ToolResult<ToolOutput> {
        match self.output {
            ImageToolOutput::Metadata => {
                let metadata = ImageMetadata {
                    id: response.id.as_deref(),
                    content_type: &response.image.content_type,
                    url: match &response.image.source {
                        MediaSource::Url(url) => Some(url),
                        MediaSource::Base64(_) => None,
                    },
                    bytes: match &response.image.source {
                        MediaSource::Base64(data) => Some(base64_len(data)),
                        MediaSource::Url(_) => None,
                    },
                    model: self.generator.model(),
                    prompt: &prompt,
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

/// Size of base64 data once decoded, without decoding it.
fn base64_len(data: &str) -> usize {
    let padding = data.bytes().rev().take_while(|&b| b == b'=').count();
    data.len() / 4 * 3 - padding
}

#[derive(Deserialize, JsonSchema)]
struct GenerateArgs {
    /// Detailed description of the image: subject, style, composition, colors, text in it.
    prompt: String,
}

#[derive(Deserialize, JsonSchema)]
struct EditArgs {
    /// What to change, or how to combine the images.
    prompt: String,
    /// IDs of the images to edit, as returned by `generate_image` or `edit_image`.
    images: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// ID of the file.
    id: String,
}

fn schema<T: JsonSchema>() -> Value {
    let generator = schemars::generate::SchemaSettings::draft2020_12()
        .with(|s| s.meta_schema = None)
        .into_generator();
    let mut schema: Value = generator.into_root_schema_for::<T>().into();
    // Gemini rejects `title`
    if let Some(object) = schema.as_object_mut() {
        object.remove("title");
    }
    schema
}

fn parse<T: for<'de> Deserialize<'de>>(tool: &str, arguments: Value) -> Result<T, ToolError> {
    serde_json::from_value(arguments).map_err(|e| {
        ToolError::InvalidArguments(format!("invalid arguments for tool '{tool}': {e}"))
    })
}

/// Rejected credentials cannot be fixed by the model; anything else (a refused prompt, a
/// failed request or save) is reported to it.
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

#[async_trait]
impl ToolBox for GenerateImageToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        let mut tools = vec![
            Tool::new(GENERATE_IMAGE)
                .with_description("Generates an image from a text prompt.")
                .with_schema(schema::<GenerateArgs>()),
        ];
        if self.can_edit() {
            tools.push(
                Tool::new(EDIT_IMAGE)
                    .with_description(
                        "Edits existing images, given by ID, following a text prompt, and \
                         creates a new image.",
                    )
                    .with_schema(schema::<EditArgs>()),
            );
        }
        Ok(tools)
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult<ToolOutput> {
        match tool_name.as_str() {
            GENERATE_IMAGE => self.generate(parse(GENERATE_IMAGE, arguments)?).await,
            EDIT_IMAGE if self.can_edit() => self.edit(parse(EDIT_IMAGE, arguments)?).await,
            _ => Err(ToolError::NotFound(tool_name)),
        }
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
/// # use agentai_core::storage::MemoryStorage;
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

#[async_trait]
impl ToolBox for ReadMediaToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        Ok(vec![
            Tool::new(READ_MEDIA)
                .with_description(
                    "Returns a stored file, for example an image, by its ID, so you can look at it.",
                )
                .with_schema(schema::<ReadArgs>()),
        ])
    }

    async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult<ToolOutput> {
        if tool_name != READ_MEDIA {
            return Err(ToolError::NotFound(tool_name));
        }
        let args: ReadArgs = parse(READ_MEDIA, arguments)?;
        let media: Media = self.storage.read(&args.id).await.map_err(read_error)?;
        let text = format!("File {} ({})", args.id, media.content_type);
        Ok(ToolOutput::from_text(text).with_media(media))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::tests::{MockHttp, json_response, provider};
    use agentai_core::message::Part;
    use agentai_core::storage::{MediaSave, MemoryStorage};
    use serde_json::json;

    fn image_reply() -> agentai_core::HttpResponse {
        json_response(
            200,
            json!({"data": [{"b64_json": "aGk=", "revised_prompt": "a red cat"}]}),
        )
    }

    fn tool_names(toolbox: &dyn ToolBox) -> Vec<String> {
        toolbox
            .tools_definitions()
            .unwrap()
            .into_iter()
            .map(|tool| tool.name)
            .collect()
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
            .call_tool(GENERATE_IMAGE.into(), json!({"prompt": "a cat"}))
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
    async fn metadata_output_describes_the_image_without_it() {
        let http = MockHttp::new(vec![image_reply()]);
        let toolbox = toolbox(&http)
            .with_storage(MemoryStorage::new())
            .with_output(ImageToolOutput::Metadata);

        let output = toolbox
            .call_tool(GENERATE_IMAGE.into(), json!({"prompt": "a cat"}))
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
        assert_eq!(tool_names(&without), [GENERATE_IMAGE]);
        assert!(matches!(
            without.call_tool(EDIT_IMAGE.into(), json!({})).await,
            Err(ToolError::NotFound(_))
        ));

        let storage = MemoryStorage::new();
        let id = storage
            .save(&Media::from_bytes("image/png", b"PNG"))
            .await
            .unwrap();
        let toolbox = toolbox(&http).with_storage(storage);
        assert_eq!(tool_names(&toolbox), [GENERATE_IMAGE, EDIT_IMAGE]);
        let unknown = toolbox
            .call_tool(
                EDIT_IMAGE.into(),
                json!({"prompt": "add a hat", "images": ["nope"]}),
            )
            .await;
        assert!(matches!(unknown, Err(ToolError::InvalidArguments(_))));
        let output = toolbox
            .call_tool(
                EDIT_IMAGE.into(),
                json!({"prompt": "add a hat", "images": [id]}),
            )
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
        let call = || toolbox.call_tool(GENERATE_IMAGE.into(), json!({"prompt": "x"}));

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
            .call_tool(READ_MEDIA.into(), json!({ "id": id }))
            .await
            .unwrap();
        assert_eq!(output.media(), vec![&pdf]);
        assert!(matches!(
            toolbox
                .call_tool(READ_MEDIA.into(), json!({"id": "nope"}))
                .await,
            Err(ToolError::InvalidArguments(_))
        ));
    }
}
