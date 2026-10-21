//! # Conversation messages
//!
//! The conversation history is a `Vec<Message>` that you own. Pass it to a run with
//! [`AgentRequest::with_history`](crate::AgentRequest::with_history) and append
//! [`RunResult::new_messages`](crate::RunResult::new_messages) to it to continue the
//! conversation.
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
/// The system prompt is not a message: the [`Agent`](crate::Agent) sends its own with every
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
        let texts: Vec<&str> = self
            .parts
            .iter()
            .filter_map(|part| match part {
                Part::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        (!texts.is_empty()).then(|| texts.concat())
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
}

/// The result of a tool call, sent back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolResponse {
    /// Id of the [`ToolCall`] this result answers.
    pub call_id: String,
    /// Name of the tool.
    pub name: String,
    /// Result sent to the model.
    pub content: String,
}

impl ToolResponse {
    /// Creates the result of `call`.
    pub fn new(call: &ToolCall, content: impl Into<String>) -> Self {
        Self {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: content.into(),
        }
    }
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
}

/// Where the bytes of [`Media`] are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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

// Conversion to and from genai. Kept private: genai is an implementation detail.

use genai::chat::{
    Binary, BinarySource, ChatMessage, ChatRole, ContentPart, CustomPart, MessageContent,
};

impl ToolCall {
    pub(crate) fn from_genai(call: genai::chat::ToolCall) -> Self {
        Self {
            id: call.call_id,
            name: call.fn_name,
            arguments: call.fn_arguments,
            signatures: call.thought_signatures.unwrap_or_default(),
        }
    }
}

impl ToolCall {
    pub(crate) fn into_genai(self) -> genai::chat::ToolCall {
        let call = self;
        genai::chat::ToolCall {
            call_id: call.id,
            fn_name: call.name,
            fn_arguments: call.arguments,
            thought_signatures: (!call.signatures.is_empty()).then_some(call.signatures),
        }
    }
}

impl ToolResponse {
    pub(crate) fn into_genai(self) -> genai::chat::ToolResponse {
        let response = self;
        let mut out = genai::chat::ToolResponse::new(response.call_id, response.content);
        out.fn_name = Some(response.name);
        out
    }
}

impl ToolResponse {
    pub(crate) fn from_genai(response: genai::chat::ToolResponse) -> Self {
        Self {
            call_id: response.call_id,
            name: response.fn_name.unwrap_or_default(),
            content: response.content,
        }
    }
}

impl Part {
    pub(crate) fn from_genai(part: ContentPart) -> Self {
        match part {
            ContentPart::Text(text) => Part::Text(text),
            ContentPart::ToolCall(call) => Part::ToolCall(ToolCall::from_genai(call)),
            ContentPart::ToolResponse(response) => {
                Part::ToolResponse(ToolResponse::from_genai(response))
            }
            ContentPart::ReasoningContent(text) => Part::Reasoning(text),
            ContentPart::ThoughtSignature(signature) => Part::ReasoningSignature(signature),
            ContentPart::Binary(binary) => Part::Media(Media {
                content_type: binary.content_type,
                source: match binary.source {
                    BinarySource::Url(url) => MediaSource::Url(url),
                    BinarySource::Base64(data) => MediaSource::Base64(data.to_string()),
                },
                name: binary.name,
            }),
            ContentPart::Custom(custom) => Part::ProviderData(ProviderData {
                provider: custom.model_iden.as_ref().and_then(|iden| {
                    serde_json::to_value(iden.adapter_kind)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_string))
                }),
                model: custom
                    .model_iden
                    .as_ref()
                    .map(|iden| iden.model_name.to_string()),
                data: custom.data,
            }),
        }
    }
}

impl Part {
    pub(crate) fn into_genai(self) -> ContentPart {
        let part = self;
        match part {
            Part::Text(text) => ContentPart::Text(text),
            Part::ToolCall(call) => ContentPart::ToolCall(call.into_genai()),
            Part::ToolResponse(response) => ContentPart::ToolResponse(response.into_genai()),
            Part::Reasoning(text) => ContentPart::ReasoningContent(text),
            Part::ReasoningSignature(signature) => ContentPart::ThoughtSignature(signature),
            Part::Media(media) => ContentPart::Binary(Binary {
                content_type: media.content_type,
                source: match media.source {
                    MediaSource::Url(url) => BinarySource::Url(url),
                    MediaSource::Base64(data) => BinarySource::Base64(data.into()),
                },
                name: media.name,
            }),
            Part::ProviderData(data) => {
                let adapter_kind: Option<genai::adapter::AdapterKind> = data
                    .provider
                    .and_then(|provider| serde_json::from_value(Value::String(provider)).ok());
                ContentPart::Custom(CustomPart {
                    model_iden: adapter_kind
                        .map(|kind| genai::ModelIden::new(kind, data.model.unwrap_or_default())),
                    data: data.data,
                })
            }
        }
    }
}

impl Message {
    pub(crate) fn from_genai(message: ChatMessage) -> Self {
        let role = match message.role {
            ChatRole::Assistant => Role::Assistant,
            ChatRole::Tool => Role::Tool,
            // agentai never stores system messages; treat any as user input
            ChatRole::User | ChatRole::System => Role::User,
        };
        Self {
            role,
            parts: message
                .content
                .into_parts()
                .into_iter()
                .map(Part::from_genai)
                .collect(),
        }
    }
}

impl Message {
    pub(crate) fn into_genai(self) -> ChatMessage {
        let message = self;
        let role = match message.role {
            Role::User => ChatRole::User,
            Role::Assistant => ChatRole::Assistant,
            Role::Tool => ChatRole::Tool,
        };
        let parts: Vec<ContentPart> = message.parts.into_iter().map(Part::into_genai).collect();
        ChatMessage::new(role, MessageContent::from_parts(parts))
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
    fn messages_round_trip_through_genai() {
        let message = every_part();
        let back = Message::from_genai(message.clone().into_genai());
        assert_eq!(back, message);
    }

    #[test]
    fn messages_round_trip_through_json() {
        let message = every_part();
        let json = serde_json::to_string(&message).unwrap();
        assert_eq!(serde_json::from_str::<Message>(&json).unwrap(), message);
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
