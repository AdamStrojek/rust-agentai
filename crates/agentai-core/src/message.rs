//! # Conversation messages
//!
//! The conversation history is a `Vec<Message>` that you own. Pass it to a run with
//! `AgentRequest::with_history` in `agentai` and append `AgentResult::new_messages` to it to
//! continue the conversation.
//!
//! Messages implement `Serialize` and `Deserialize` with a format defined by this crate, so a
//! stored history keeps working when the LLM client used inside `agentai` changes.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who wrote a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    /// The user, or your code speaking for the user.
    User,
    /// The model.
    Assistant,
    /// Results of tool calls, sent back to the model.
    Tool,
}

/// One message of the conversation history.
///
/// The system prompt is not a message: the `Agent` of `agentai` sends its own with every
/// request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
    /// Who wrote the message.
    pub role: Role,
    /// Content of the message, in order.
    pub parts: Vec<Part>,
}

impl Message {
    /// Creates a message from its parts.
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    /// Creates a user message with text.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![Part::Text(text.into())])
    }

    /// Creates an assistant message with text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![Part::Text(text.into())])
    }

    /// Creates a message with the result of a tool call.
    pub fn tool_response(response: ToolResponse) -> Self {
        Self::new(Role::Tool, vec![Part::ToolResponse(response)])
    }

    /// Returns the text parts joined together, or `None` when the message has no text.
    pub fn text(&self) -> Option<String> {
        joined_text(&self.parts)
    }

    /// Returns the tool calls in the message.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }
}

/// A piece of a [`Message`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Part {
    /// Text.
    Text(String),
    /// A tool call requested by the model.
    ToolCall(ToolCall),
    /// The result of a tool call.
    ToolResponse(ToolResponse),
    /// Reasoning (thinking) text of the model.
    Reasoning(String),
    /// Signature of the model's reasoning. Some providers (Anthropic, Gemini) need it back to
    /// continue a conversation with reasoning; keep it in the history as it is.
    ReasoningSignature(String),
    /// An image, audio or other file.
    Media(Media),
    /// Provider-specific data that has to be sent back unchanged, for example reasoning
    /// details. Keep it in the history as it is.
    ProviderData(ProviderData),
}

/// A tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    /// Id of the call. The matching [`ToolResponse`] has the same id.
    pub id: String,
    /// Name of the tool.
    pub name: String,
    /// Arguments chosen by the model, as JSON.
    pub arguments: Value,
    /// Reasoning signatures some providers attach to the call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    signatures: Vec<String>,
}

impl ToolCall {
    /// Creates a tool call.
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            signatures: Vec::new(),
        }
    }

    /// Reasoning signatures some providers attach to the call. Keep them as they are, the
    /// provider needs them back.
    pub fn signatures(&self) -> &[String] {
        &self.signatures
    }

    /// Sets the reasoning signatures.
    pub fn with_signatures(mut self, signatures: Vec<String>) -> Self {
        self.signatures = signatures;
        self
    }
}

/// The result of a tool call, sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolResponse {
    /// Id of the [`ToolCall`] this result answers.
    pub call_id: String,
    /// Name of the tool.
    pub name: String,
    /// Result sent to the model: [`Part::Text`] and [`Part::Media`] parts, in order.
    ///
    /// Providers that do not accept media in tool results get only the text.
    #[serde(deserialize_with = "deserialize_tool_content")]
    pub content: Vec<Part>,
}

impl ToolResponse {
    /// Creates the result of `call`.
    pub fn new(call: &ToolCall, content: impl Into<ToolOutput>) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: content.into().parts,
        }
    }

    /// Returns the text parts joined together, or `None` when the result has no text.
    pub fn text(&self) -> Option<String> {
        joined_text(&self.content)
    }
}

/// Reads `content` as a list of parts, or as a plain string (the format before 0.5.0).
fn deserialize_tool_content<'de, D>(deserializer: D) -> Result<Vec<Part>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Content {
        Text(String),
        Parts(Vec<Part>),
    }
    Ok(match Content::deserialize(deserializer)? {
        Content::Text(text) => vec![Part::Text(text)],
        Content::Parts(parts) => parts,
    })
}

/// Joins the text parts, `None` when there are none.
fn joined_text(parts: &[Part]) -> Option<String> {
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    (!texts.is_empty()).then(|| texts.concat())
}

/// An image, audio or other file in a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Media {
    /// MIME type, for example `image/png`.
    pub content_type: String,
    /// Where the bytes are.
    pub source: MediaSource,
    /// File name, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Media {
    /// Creates media from its MIME type and source.
    pub fn new(content_type: impl Into<String>, source: MediaSource) -> Self {
        Self {
            content_type: content_type.into(),
            source,
            name: None,
        }
    }

    /// Creates media the provider downloads from `url`. Not every provider accepts URLs; use
    /// [`Media::from_bytes`] when yours does not.
    pub fn from_url(content_type: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(content_type, MediaSource::Url(url.into()))
    }

    /// Creates an image the provider downloads from `url`. The MIME type is guessed from the
    /// file extension (`png`, `jpg`, `jpeg`, `gif`, `webp`); other URLs are sent as
    /// `image/jpeg`. Use [`Media::from_url`] to set it yourself.
    pub fn image_url(url: impl Into<String>) -> Self {
        let url = url.into();
        Self::from_url(image_type_from_url(&url), url)
    }

    /// Creates media from its bytes, for example an image read from a file. The bytes are
    /// sent base64 encoded inside the request.
    pub fn from_bytes(content_type: impl Into<String>, bytes: impl AsRef<[u8]>) -> Self {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.encode(bytes);
        Self::from_base64(content_type, data)
    }

    /// Creates media from bytes that are already base64 encoded.
    pub fn from_base64(content_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::new(content_type, MediaSource::Base64(data.into()))
    }

    /// Sets the file name. Some providers show it to the model.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns `true` when the MIME type is an image type.
    pub fn is_image(&self) -> bool {
        self.content_type.starts_with("image/")
    }
}

/// Guesses the MIME type of an image from the extension of its URL.
fn image_type_from_url(url: &str) -> &'static str {
    let path = url.split(['?', '#']).next().unwrap_or_default();
    let extension = path
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/jpeg",
    }
}

/// Where the bytes of [`Media`] are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaSource {
    /// A URL. Few providers accept it.
    Url(String),
    /// The bytes, base64 encoded.
    Base64(String),
}

/// Provider-specific data in a message. It has no meaning outside the provider that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderData {
    /// Provider that produced the data, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model that produced the data, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The data.
    pub data: Value,
}

impl ProviderData {
    /// Creates provider data without the provider and model.
    pub fn new(data: Value) -> Self {
        Self {
            provider: None,
            model: None,
            data,
        }
    }

    /// Sets the provider.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Sets the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// What a tool sends back to the model: text and media (for example a generated image), in
/// order.
///
/// Text converts into it, so a tool that only returns text can keep returning a `String`.
/// Providers that do not accept media in tool results get only the text parts.
///
/// There is no structured (JSON) output: a tool that produces data converts it to text itself,
/// for example with `serde_json::to_string`, in the format it wants the model to read.
///
/// ```rust
/// # use agentai_core::message::{Media, ToolOutput};
/// let png = [0x89, b'P', b'N', b'G'];
/// let output = ToolOutput::from_text("Here is the chart")
///     .with_media(Media::from_bytes("image/png", png));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ToolOutput {
    /// [`Part::Text`] and [`Part::Media`] parts.
    pub parts: Vec<Part>,
}

impl ToolOutput {
    /// Creates an output with text.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self::default().with_text(text)
    }

    /// Creates an output with one media part.
    pub fn from_media(media: Media) -> Self {
        Self::default().with_media(media)
    }

    /// Appends text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(Part::Text(text.into()));
        self
    }

    /// Appends media.
    pub fn with_media(mut self, media: Media) -> Self {
        self.parts.push(Part::Media(media));
        self
    }

    /// Returns the text parts joined together, or `None` when there is no text.
    pub fn text(&self) -> Option<String> {
        joined_text(&self.parts)
    }

    /// Returns the media parts.
    pub fn media(&self) -> Vec<&Media> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Part::Media(media) => Some(media),
                _ => None,
            })
            .collect()
    }
}

/// Text parts as they are, media as `[content type]`, for example `[image/png]`.
impl std::fmt::Display for ToolOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for part in &self.parts {
            match part {
                Part::Text(text) => f.write_str(text)?,
                Part::Media(media) => write!(f, "[{}]", media.content_type)?,
                _ => {}
            }
        }
        Ok(())
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self::from_text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::from_text(text)
    }
}

impl From<Media> for ToolOutput {
    fn from(media: Media) -> Self {
        Self::from_media(media)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn every_part() -> Message {
        let mut call = ToolCall::new("call_1", "echo", json!({"text": "hi"}));
        call.signatures = vec!["sig".to_string()];
        Message::new(
            Role::Assistant,
            vec![
                Part::ReasoningSignature("signature".into()),
                Part::Reasoning("thinking".into()),
                Part::Text("hello".into()),
                Part::ToolCall(call.clone()),
                Part::ToolResponse(ToolResponse::new(&call, "result")),
                Part::ToolResponse(ToolResponse::new(
                    &call,
                    ToolOutput::from_text("an image")
                        .with_media(Media::new("image/png", MediaSource::Base64("aGk=".into()))),
                )),
                Part::Media(Media::new("image/png", MediaSource::Base64("aGk=".into()))),
                Part::Media(Media::new(
                    "image/png",
                    MediaSource::Url("https://example.com/a.png".into()),
                )),
                Part::ProviderData(ProviderData {
                    provider: Some("OpenRouter".into()),
                    model: Some("m".into()),
                    data: json!({"type": "reasoning.encrypted"}),
                }),
            ],
        )
    }

    #[test]
    fn messages_round_trip_through_json() {
        let message = every_part();
        let json = serde_json::to_string(&message).unwrap();
        assert_eq!(serde_json::from_str::<Message>(&json).unwrap(), message);
    }

    #[test]
    fn tool_responses_stored_before_0_5_still_load() {
        let json = r#"{"role":"tool","parts":[{"type":"tool_response",
            "value":{"call_id":"call_1","name":"echo","content":"result"}}]}"#;
        let message: Message = serde_json::from_str(json).unwrap();
        let call = ToolCall::new("call_1", "echo", json!({}));
        assert_eq!(
            message,
            Message::tool_response(ToolResponse::new(&call, "result"))
        );
    }

    #[test]
    fn image_url_guesses_the_type_from_the_extension() {
        let types: Vec<String> = [
            "https://example.com/a.png",
            "https://example.com/a.PNG?size=2#top",
            "https://example.com/a.webp",
            "https://example.com/a.gif",
            "https://example.com/a.jpeg",
            "https://example.com/image",
        ]
        .into_iter()
        .map(|url| Media::image_url(url).content_type)
        .collect();
        assert_eq!(
            types,
            [
                "image/png",
                "image/png",
                "image/webp",
                "image/gif",
                "image/jpeg",
                "image/jpeg"
            ]
        );
        assert_eq!(
            Media::image_url("https://example.com/a.png").source,
            MediaSource::Url("https://example.com/a.png".into())
        );
    }

    #[test]
    fn json_format_is_stable() {
        let message = Message::user("hi");
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({"role": "user", "parts": [{"type": "text", "value": "hi"}]})
        );
    }
}
