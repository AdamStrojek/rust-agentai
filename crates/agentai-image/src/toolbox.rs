//! The image tools for agents.

use crate::provider::{
    self, ImageEditRequest, ImageError, ImageProvider, ImageRequest, ImageResponse,
};
use agentai::Media;
use agentai::tool::{Tool, ToolBox, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

const GENERATE_IMAGE: &str = "generate_image";
const EDIT_IMAGE: &str = "edit_image";

/// Saves the images the tools make: to a directory, a database, your UI, ...
///
/// It only stores them. The image goes back to the model as the tool result whatever the
/// sink does, so handle (or log) its errors inside `save`.
#[async_trait]
pub trait ImageSink: Send + Sync + 'static {
    /// Called with every image, after it got its name ([`Media::name`]).
    async fn save(&self, image: &ImageResponse);
}

/// Loads the images the model wants to edit.
///
/// The model refers to an image by name: one returned by `generate_image` or `edit_image`
/// (for example `image-1.png`), or a name you told it, for example a file the user uploaded.
#[async_trait]
pub trait ImageSource: Send + Sync + 'static {
    /// Returns the image called `reference`. Return [`ToolError::InvalidArguments`] for a
    /// name you do not know, so the model can correct it.
    async fn load(&self, reference: &str) -> Result<Media, ToolError>;
}

/// [`ToolBox`] that lets an agent generate and edit images.
///
/// Tools:
/// - `generate_image`: draws an image from a prompt.
/// - `edit_image`: changes images the model names. Offered only with an [`ImageSource`]
///   ([`with_source`](ImageToolBox::with_source)) and a provider that can edit.
///
/// The result of both is the image (the model sees it when its provider accepts images in
/// tool results; others get only the text) and a text with the image's name, which the
/// model can pass to `edit_image` later.
///
/// ```rust,no_run
/// # use agentai_image::ImageToolBox;
/// # fn main() -> Result<(), agentai_image::ImageError> {
/// let images = ImageToolBox::new("openai::gpt-image-1")?.with_size("1024x1024");
/// # Ok(())
/// # }
/// ```
pub struct ImageToolBox {
    provider: Box<dyn ImageProvider>,
    sink: Option<Box<dyn ImageSink>>,
    source: Option<Box<dyn ImageSource>>,
    size: Option<String>,
    quality: Option<String>,
    count: AtomicUsize,
}

impl ImageToolBox {
    /// Creates the toolbox for a model, see [`provider::from_model`]: the provider comes
    /// from the name (`openai::gpt-image-1`) and its API key from the environment.
    pub fn new(model: &str) -> Result<Self, ImageError> {
        Ok(Self::with_boxed_provider(provider::from_model(model)?))
    }

    /// Creates the toolbox with a provider you built, for example with an API key set in
    /// code.
    pub fn from_provider(provider: impl ImageProvider) -> Self {
        Self::with_boxed_provider(Box::new(provider))
    }

    fn with_boxed_provider(provider: Box<dyn ImageProvider>) -> Self {
        Self {
            provider,
            sink: None,
            source: None,
            size: None,
            quality: None,
            count: AtomicUsize::new(0),
        }
    }

    /// Saves every image with `sink`.
    pub fn with_sink(mut self, sink: impl ImageSink) -> Self {
        self.sink = Some(Box::new(sink));
        self
    }

    /// Loads the images to edit with `source`, which enables `edit_image`.
    pub fn with_source(mut self, source: impl ImageSource) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Size of every image, in the provider's format (for example `1024x1024`).
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Quality of every image, in the provider's format (for example `high`).
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.quality = Some(quality.into());
        self
    }

    fn can_edit(&self) -> bool {
        self.source.is_some() && self.provider.supports_edit()
    }

    async fn generate(&self, args: GenerateArgs) -> ToolResult<ToolOutput> {
        let mut request = ImageRequest::new(args.prompt);
        request.size = self.size.clone();
        request.quality = self.quality.clone();
        let response = self.provider.generate(request).await.map_err(tool_error)?;
        Ok(self.finish(response).await)
    }

    async fn edit(&self, args: EditArgs) -> ToolResult<ToolOutput> {
        let Some(source) = self.source.as_ref().filter(|_| self.can_edit()) else {
            return Err(ToolError::NotFound(EDIT_IMAGE.to_string()));
        };
        if args.images.is_empty() {
            return Err(ToolError::InvalidArguments(
                "give at least one image name in `images`".to_string(),
            ));
        }
        let mut images = Vec::with_capacity(args.images.len());
        for reference in &args.images {
            images.push(source.load(reference).await?);
        }
        let mut request = ImageEditRequest::new(args.prompt, images);
        request.size = self.size.clone();
        request.quality = self.quality.clone();
        let response = self.provider.edit(request).await.map_err(tool_error)?;
        Ok(self.finish(response).await)
    }

    /// Names the image, hands it to the sink and builds the tool result.
    async fn finish(&self, mut response: ImageResponse) -> ToolOutput {
        let number = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        let extension = response
            .image
            .content_type
            .rsplit('/')
            .next()
            .unwrap_or("png")
            .to_string();
        let name = format!("image-{number}.{extension}");
        response.image.name = Some(name.clone());
        if let Some(sink) = &self.sink {
            sink.save(&response).await;
        }
        let mut text = format!("Created the image {name}.");
        if let Some(revised) = &response.revised_prompt {
            text.push_str(&format!(" Prompt used: {revised}"));
        }
        ToolOutput::from_text(text).with_media(response.image)
    }
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
    /// Names of the images to edit, for example `image-1.png`.
    images: Vec<String>,
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
/// failed request) is reported to it.
fn tool_error(err: ImageError) -> ToolError {
    if err.is_auth() || matches!(err, ImageError::MissingApiKey(_)) {
        ToolError::Configuration(err.to_string())
    } else {
        ToolError::Failed(err.to_string())
    }
}

#[async_trait]
impl ToolBox for ImageToolBox {
    fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        let mut tools = vec![
            Tool::new(GENERATE_IMAGE)
                .with_description(
                    "Generates an image from a text prompt and returns it with its name.",
                )
                .with_schema(schema::<GenerateArgs>()),
        ];
        if self.can_edit() {
            tools.push(
                Tool::new(EDIT_IMAGE)
                    .with_description(
                        "Edits existing images, given by name, following a text prompt, and \
                         returns the new image with its name.",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::tests::{MockHttp, json_response, provider};
    use agentai::Part;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Saved(Arc<Mutex<Vec<ImageResponse>>>);

    #[async_trait]
    impl ImageSink for Saved {
        async fn save(&self, image: &ImageResponse) {
            self.0.lock().unwrap().push(image.clone());
        }
    }

    struct OneImage;

    #[async_trait]
    impl ImageSource for OneImage {
        async fn load(&self, reference: &str) -> Result<Media, ToolError> {
            match reference {
                "image-1.png" => Ok(Media::from_bytes("image/png", b"PNG")),
                other => Err(ToolError::InvalidArguments(format!("no image {other}"))),
            }
        }
    }

    fn image_reply() -> agentai_core::HttpResponse {
        json_response(
            200,
            json!({"data": [{"b64_json": "aGk=", "revised_prompt": "a red cat"}]}),
        )
    }

    fn tool_names(toolbox: &ImageToolBox) -> Vec<String> {
        toolbox
            .tools_definitions()
            .unwrap()
            .into_iter()
            .map(|tool| tool.name)
            .collect()
    }

    #[tokio::test]
    async fn generate_image_returns_the_named_image_and_saves_it() {
        let http = MockHttp::new(vec![image_reply()]);
        let saved = Saved::default();
        let toolbox = ImageToolBox::from_provider(provider(&http))
            .with_sink(saved.clone())
            .with_size("512x512");

        let output = toolbox
            .call_tool(GENERATE_IMAGE.into(), json!({"prompt": "a cat"}))
            .await
            .unwrap();

        let image = Media::from_base64("image/png", "aGk=").with_name("image-1.png");
        assert_eq!(
            output.parts,
            vec![
                Part::Text("Created the image image-1.png. Prompt used: a red cat".into()),
                Part::Media(image.clone()),
            ]
        );
        assert_eq!(
            *saved.0.lock().unwrap(),
            vec![ImageResponse::new(image).with_revised_prompt("a red cat")]
        );
        let body: Value = serde_json::from_slice(&http.request(0).body.unwrap()).unwrap();
        assert_eq!(body["size"], json!("512x512"));
    }

    #[tokio::test]
    async fn edit_image_is_offered_only_with_a_source() {
        let http = MockHttp::new(vec![image_reply()]);
        let without = ImageToolBox::from_provider(provider(&http));
        assert_eq!(tool_names(&without), [GENERATE_IMAGE]);
        assert!(matches!(
            without.call_tool(EDIT_IMAGE.into(), json!({})).await,
            Err(ToolError::NotFound(_))
        ));

        let toolbox = ImageToolBox::from_provider(provider(&http)).with_source(OneImage);
        assert_eq!(tool_names(&toolbox), [GENERATE_IMAGE, EDIT_IMAGE]);
        let unknown = toolbox
            .call_tool(
                EDIT_IMAGE.into(),
                json!({"prompt": "add a hat", "images": ["image-9.png"]}),
            )
            .await;
        assert!(matches!(unknown, Err(ToolError::InvalidArguments(_))));
        let output = toolbox
            .call_tool(
                EDIT_IMAGE.into(),
                json!({"prompt": "add a hat", "images": ["image-1.png"]}),
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
        let toolbox = ImageToolBox::from_provider(provider(&http));
        let call = || toolbox.call_tool(GENERATE_IMAGE.into(), json!({"prompt": "x"}));

        assert!(matches!(call().await, Err(ToolError::Configuration(_))));
        assert!(matches!(call().await, Err(ToolError::Failed(m)) if m.contains("refused")));
    }
}
