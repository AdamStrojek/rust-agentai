//! Conversion of the message types (from `agentai-core`) to and from genai. Kept private:
//! genai is an implementation detail.

use crate::message::{
    Media, MediaSource, Message, Part, ProviderData, Role, ToolCall, ToolResponse,
};
use genai::chat::{
    Binary, BinarySource, ChatMessage, ChatRole, ContentPart, CustomPart, MessageContent,
};
use serde_json::Value;

/// Converts a genai value into ours.
pub(crate) trait FromGenai<T> {
    fn from_genai(value: T) -> Self;
}

/// Converts our value into genai's.
pub(crate) trait IntoGenai {
    type Genai;
    fn into_genai(self) -> Self::Genai;
}

impl FromGenai<genai::chat::ToolCall> for ToolCall {
    fn from_genai(call: genai::chat::ToolCall) -> Self {
        ToolCall::new(call.call_id, call.fn_name, call.fn_arguments)
            .with_signatures(call.thought_signatures.unwrap_or_default())
    }
}

impl IntoGenai for ToolCall {
    type Genai = genai::chat::ToolCall;
    fn into_genai(self) -> genai::chat::ToolCall {
        let signatures = self.signatures().to_vec();
        genai::chat::ToolCall {
            call_id: self.id,
            fn_name: self.name,
            fn_arguments: self.arguments,
            thought_signatures: (!signatures.is_empty()).then_some(signatures),
        }
    }
}

impl IntoGenai for ToolResponse {
    type Genai = genai::chat::ToolResponse;
    fn into_genai(self) -> genai::chat::ToolResponse {
        let content: Vec<ContentPart> = self.content.into_iter().map(Part::into_genai).collect();
        let mut out =
            genai::chat::ToolResponse::new(self.call_id, MessageContent::from_parts(content));
        out.fn_name = Some(self.name);
        out
    }
}

impl FromGenai<genai::chat::ToolResponse> for ToolResponse {
    fn from_genai(response: genai::chat::ToolResponse) -> Self {
        let call = ToolCall::new(
            response.call_id,
            response.fn_name.unwrap_or_default(),
            Value::Null,
        );
        let mut out = ToolResponse::new(&call, "");
        out.content = response
            .content
            .into_parts()
            .into_iter()
            .map(Part::from_genai)
            .collect();
        out
    }
}

impl FromGenai<ContentPart> for Part {
    fn from_genai(part: ContentPart) -> Self {
        match part {
            ContentPart::Text(text) => Part::Text(text),
            ContentPart::ToolCall(call) => Part::ToolCall(ToolCall::from_genai(call)),
            ContentPart::ToolResponse(response) => {
                Part::ToolResponse(ToolResponse::from_genai(response))
            }
            ContentPart::ReasoningContent(text) => Part::Reasoning(text),
            ContentPart::ThoughtSignature(signature) => Part::ReasoningSignature(signature),
            ContentPart::Binary(binary) => {
                let source = match binary.source {
                    BinarySource::Url(url) => MediaSource::Url(url),
                    BinarySource::Base64(data) => MediaSource::Base64(data.to_string()),
                };
                let mut media = Media::new(binary.content_type, source);
                media.name = binary.name;
                Part::Media(media)
            }
            ContentPart::Custom(custom) => {
                let mut data = ProviderData::new(custom.data);
                data.provider = custom.model_iden.as_ref().and_then(|iden| {
                    serde_json::to_value(iden.adapter_kind)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_string))
                });
                data.model = custom
                    .model_iden
                    .as_ref()
                    .map(|iden| iden.model_name.to_string());
                Part::ProviderData(data)
            }
        }
    }
}

impl IntoGenai for Part {
    type Genai = ContentPart;
    fn into_genai(self) -> ContentPart {
        match self {
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
            // `Part` is non-exhaustive; both crates are released together, so a new variant
            // gets its mapping here in the same release
            _ => ContentPart::Text(String::new()),
        }
    }
}

impl FromGenai<ChatMessage> for Message {
    fn from_genai(message: ChatMessage) -> Self {
        let role = match message.role {
            ChatRole::Assistant => Role::Assistant,
            ChatRole::Tool => Role::Tool,
            // agentai never stores system messages; treat any as user input
            ChatRole::User | ChatRole::System => Role::User,
        };
        Message::new(
            role,
            message
                .content
                .into_parts()
                .into_iter()
                .map(Part::from_genai)
                .collect(),
        )
    }
}

impl IntoGenai for Message {
    type Genai = ChatMessage;
    fn into_genai(self) -> ChatMessage {
        let role = match self.role {
            Role::User => ChatRole::User,
            Role::Assistant => ChatRole::Assistant,
            Role::Tool => ChatRole::Tool,
            _ => ChatRole::User,
        };
        let parts: Vec<ContentPart> = self.parts.into_iter().map(Part::into_genai).collect();
        ChatMessage::new(role, MessageContent::from_parts(parts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolOutput;
    use serde_json::json;

    #[test]
    fn messages_round_trip_through_genai() {
        let call = ToolCall::new("call_1", "echo", json!({"text": "hi"}))
            .with_signatures(vec!["sig".to_string()]);
        let image = Media::new("image/png", MediaSource::Base64("aGk=".into()));
        let message = Message::new(
            Role::Assistant,
            vec![
                Part::ReasoningSignature("signature".into()),
                Part::Reasoning("thinking".into()),
                Part::Text("hello".into()),
                Part::ToolCall(call.clone()),
                Part::ToolResponse(ToolResponse::new(&call, "result")),
                Part::ToolResponse(ToolResponse::new(
                    &call,
                    ToolOutput::from_text("an image").with_media(image.clone()),
                )),
                Part::Media(image.with_name("a.png")),
                Part::Media(Media::image_url("https://example.com/a.png")),
                Part::ProviderData(
                    ProviderData::new(json!({"type": "reasoning.encrypted"}))
                        .with_provider("OpenRouter")
                        .with_model("m"),
                ),
            ],
        );
        let back = Message::from_genai(message.clone().into_genai());
        assert_eq!(back, message);
    }
}
