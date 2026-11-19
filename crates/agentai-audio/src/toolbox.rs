//! Agent tools (feature `toolbox`): [`SpeechToolBox`].

use crate::generator::SpeechGenerator;
#[cfg(test)]
use crate::provider::VoiceOrigin;
use crate::provider::{AudioError, SpeechResponse, VoiceId, VoiceInfo};
use agentai::tool::{Tool, ToolBox, ToolError, ToolOutput, ToolResult, toolbox};
use agentai_core::storage::MediaStorage;
use serde::Serialize;

/// What [`SpeechToolBox`] sends back to the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpeechToolOutput {
    /// Only a JSON text with what was made (ID, type, size in bytes, model, voice, ...),
    /// without the audio. Most models cannot listen to audio, so this is the default.
    #[default]
    Metadata,
    /// The audio itself and a short text with its ID. The model must accept audio.
    Media,
}

/// [`ToolBox`] that lets an agent speak texts with a [`SpeechGenerator`].
///
/// Tools:
/// - `generate_speech`: speaks a text and saves the audio in the storage.
/// - `list_voices`: lists the voices the model can choose from.
///
/// No tool clones voices: that stays in your code ([`SpeechGenerator::create_voice`]).
///
/// The model names the audio (`name`), and the storage chooses the ID. What else the model
/// gets is set with [`with_output`](SpeechToolBox::with_output). The voices it may use are
/// limited with [`with_voices`](SpeechToolBox::with_voices); without it any voice ID goes to
/// the provider.
///
/// ```rust,no_run
/// # use agentai_audio::provider::openai::OpenAiSpeechProvider;
/// # use agentai_audio::{SpeechGenerator, SpeechToolBox};
/// # use agentai_audio::storage::MemoryStorage;
/// # fn main() -> Result<(), agentai_audio::AudioError> {
/// let generator =
///     SpeechGenerator::from_provider(OpenAiSpeechProvider::from_env("gpt-4o-mini-tts")?);
/// let speech = SpeechToolBox::new(generator, MemoryStorage::new());
/// # Ok(())
/// # }
/// ```
pub struct SpeechToolBox {
    generator: SpeechGenerator,
    voices: Option<Vec<VoiceInfo>>,
    output: SpeechToolOutput,
}

impl SpeechToolBox {
    /// Creates the toolbox. Every audio is saved in `storage`.
    pub fn new(generator: SpeechGenerator, storage: impl MediaStorage) -> Self {
        Self {
            generator: generator.with_storage(storage),
            voices: None,
            output: SpeechToolOutput::default(),
        }
    }

    /// Limits the voices to this list: `list_voices` returns it, and a voice the model gives
    /// that is not in it is rejected. Use it whenever cloned voices exist: without it the model
    /// can use any stored or account voice by its ID.
    pub fn with_voices(mut self, voices: Vec<VoiceInfo>) -> Self {
        self.voices = Some(voices);
        self
    }

    /// Sets what the model gets back, see [`SpeechToolOutput`].
    pub fn with_output(mut self, output: SpeechToolOutput) -> Self {
        self.output = output;
        self
    }

    /// The voice for what the model wrote in `voice`.
    fn voice(&self, voice: Option<&str>) -> ToolResult<Option<VoiceId>> {
        let Some(voice) = voice.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(None);
        };
        let Some(voices) = &self.voices else {
            return Ok(Some(voice.to_string()));
        };
        if voices.iter().any(|v| v.id == voice) {
            Ok(Some(voice.to_string()))
        } else {
            Err(ToolError::Failed(format!(
                "'{voice}' is not a known voice ID; pick one from `list_voices`"
            )))
        }
    }

    fn output(&self, response: SpeechResponse, voice: Option<&str>) -> ToolResult<ToolOutput> {
        match self.output {
            SpeechToolOutput::Metadata => {
                let metadata = SpeechMetadata {
                    id: response.id.as_deref(),
                    content_type: &response.audio.content_type,
                    bytes: response.audio.byte_len(),
                    model: response.model.as_deref(),
                    voice,
                    sample_rate: response.sample_rate,
                };
                let json = serde_json::to_string(&metadata).map_err(ToolError::fatal)?;
                Ok(ToolOutput::from_text(json))
            }
            SpeechToolOutput::Media => {
                let text = match &response.id {
                    Some(id) => format!("Created the audio with ID {id}."),
                    None => "Created the audio.".to_string(),
                };
                Ok(ToolOutput::from_text(text).with_media(response.audio))
            }
        }
    }
}

#[toolbox]
impl SpeechToolBox {
    /// Generates speech from text and saves it as an audio file.
    #[tool]
    async fn generate_speech(
        &self,
        /// The text to speak.
        text: String,
        /// Short file name for the new audio, for example `greeting`. The storage may add to it
        /// to make the ID unique.
        name: String,
        /// A voice ID from `list_voices`. Leave it out for the provider's own voice.
        voice: Option<String>,
        /// How to speak, for example the tone, pace or emotion. Leave it out when not needed.
        instructions: Option<String>,
    ) -> ToolResult<ToolOutput> {
        let mut request = self.generator.generate(text).with_name(name);
        let voice = self.voice(voice.as_deref())?;
        if let Some(voice) = &voice {
            request = request.with_voice(voice.clone());
        }
        if let Some(instructions) = instructions.filter(|i| !i.trim().is_empty()) {
            request = request.with_instructions(instructions);
        }
        let response = request.await.map_err(tool_error)?;
        self.output(response, voice.as_deref())
    }

    /// Lists the voices you can use in `generate_speech`.
    #[tool]
    async fn list_voices(&self) -> ToolResult<String> {
        let voices = match &self.voices {
            Some(voices) => voices.clone(),
            None => match self.generator.list_voices().await {
                Ok(voices) => voices,
                Err(AudioError::Unsupported(_)) => Vec::new(),
                Err(err) => return Err(tool_error(err)),
            },
        };
        let list = VoiceList { voices: &voices };
        serde_json::to_string(&list).map_err(ToolError::fatal)
    }
}

/// What [`SpeechToolOutput::Metadata`] sends; `None` fields are left out.
#[derive(Serialize)]
struct SpeechMetadata<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    content_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
}

/// What `list_voices` sends.
#[derive(Serialize)]
struct VoiceList<'a> {
    voices: &'a [VoiceInfo],
}

/// Rejected credentials cannot be fixed by the model; anything else (a missing voice, a
/// refused text, a failed request or save) is reported to it.
fn tool_error(err: AudioError) -> ToolError {
    if err.is_auth() || matches!(err, AudioError::MissingApiKey(_)) {
        ToolError::Configuration(err.to_string())
    } else {
        ToolError::Failed(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::omlx::OmlxProvider;
    use crate::provider::openai::OpenAiSpeechProvider;
    use agentai::tool::ToolReturn;
    use agentai_core::HttpResponse;
    use agentai_core::message::Part;
    use agentai_core::mock::{MockHttp, body_json, json_response};
    use agentai_core::storage::MemoryStorage;
    use serde_json::{Value, json};
    use std::sync::Arc;

    fn tool_call(name: &str, arguments: Value) -> agentai::ToolCall {
        agentai::ToolCall::new("call_1", name, arguments)
    }

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn openai(http: &MockHttp, base_url: &str) -> OpenAiSpeechProvider {
        OpenAiSpeechProvider::new("gpt-4o-mini-tts", "key", base_url)
            .with_client(Arc::new(http.clone()))
    }

    fn official(http: &MockHttp) -> OpenAiSpeechProvider {
        openai(http, "https://api.openai.com/v1")
    }

    fn toolbox(provider: impl crate::provider::SpeechProvider) -> SpeechToolBox {
        SpeechToolBox::new(
            SpeechGenerator::from_provider(provider),
            MemoryStorage::new(),
        )
    }

    async fn call_done(toolbox: &impl ToolBox, name: &str, arguments: Value) -> ToolOutput {
        match toolbox
            .call_tool(&tool_call(name, arguments))
            .await
            .unwrap()
        {
            ToolReturn::Done(output) => output,
            other => panic!("expected ToolReturn::Done, got {other:?}"),
        }
    }

    fn text_json(output: &ToolOutput) -> Value {
        serde_json::from_str(&output.text().unwrap()).unwrap()
    }

    #[tokio::test]
    async fn generate_speech_returns_metadata_by_default() {
        let http = MockHttp::new(vec![audio()]);
        let toolbox = toolbox(official(&http));

        let output = call_done(
            &toolbox,
            "generate_speech",
            json!({"text": "hi", "name": "greeting", "voice": "alloy"}),
        )
        .await;

        assert!(output.media().is_empty());
        assert_eq!(output.parts.len(), 1);
        assert_eq!(
            text_json(&output),
            json!({
                "id": "1-greeting.mp3",
                "content_type": "audio/mpeg",
                "bytes": 5,
                "model": "gpt-4o-mini-tts",
                "voice": "alloy",
            })
        );
    }

    #[tokio::test]
    async fn media_output_includes_the_audio() {
        let http = MockHttp::new(vec![audio()]);
        let toolbox = toolbox(official(&http)).with_output(SpeechToolOutput::Media);

        let output = call_done(
            &toolbox,
            "generate_speech",
            json!({"text": "hi", "name": "greeting", "voice": "alloy"}),
        )
        .await;

        assert!(output.text().unwrap().contains("1-greeting.mp3"));
        assert!(matches!(output.parts.last(), Some(Part::Media(_))));
        assert_eq!(output.media().len(), 1);
    }

    #[tokio::test]
    async fn a_blank_voice_sends_no_voice() {
        let blanks = [
            json!({"text": "hi", "name": "x", "voice": ""}),
            json!({"text": "hi", "name": "x", "voice": "   "}),
            json!({"text": "hi", "name": "x"}),
            json!({"text": "hi", "name": "x", "voice": null}),
        ];
        for arguments in blanks {
            let http = MockHttp::new(vec![audio()]);
            let provider =
                OmlxProvider::new("m", "", "http://local/v1").with_client(Arc::new(http.clone()));

            toolbox(provider)
                .call_tool(&tool_call("generate_speech", arguments.clone()))
                .await
                .unwrap();

            let body = body_json(&http.request(0));
            assert!(body.get("voice").is_none(), "{arguments}");
            assert!(body.get("ref_audio").is_none(), "{arguments}");
        }
    }

    #[tokio::test]
    async fn a_listed_voice_is_sent() {
        let http = MockHttp::new(vec![audio()]);
        let toolbox = toolbox(official(&http)).with_voices(vec![VoiceInfo::new(
            "voice_ok",
            "OK",
            VoiceOrigin::Preset,
        )]);

        toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x", "voice": "voice_ok"}),
            ))
            .await
            .unwrap();

        assert_eq!(
            body_json(&http.request(0))["voice"],
            json!({"id": "voice_ok"})
        );
    }

    #[tokio::test]
    async fn an_unlisted_voice_fails() {
        let http = MockHttp::new(vec![]);
        let toolbox = toolbox(official(&http)).with_voices(vec![VoiceInfo::new(
            "alloy",
            "Alloy",
            VoiceOrigin::Preset,
        )]);

        let result = toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "a", "voice": "warm low voice"}),
            ))
            .await;

        assert!(matches!(result, Err(ToolError::Failed(m)) if m.contains("list_voices")));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn without_a_list_the_voice_is_an_id() {
        let http = MockHttp::new(vec![audio()]);
        let toolbox = toolbox(openai(&http, "http://local/v1"));

        toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "a", "voice": "af_bella"}),
            ))
            .await
            .unwrap();

        assert_eq!(body_json(&http.request(0))["voice"], json!("af_bella"));
    }

    #[tokio::test]
    async fn instructions_reach_the_provider() {
        let http = MockHttp::new(vec![audio()]);
        let toolbox = toolbox(official(&http));

        toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x", "voice": "alloy", "instructions": "slowly"}),
            ))
            .await
            .unwrap();

        assert_eq!(body_json(&http.request(0))["instructions"], json!("slowly"));
    }

    #[tokio::test]
    async fn provider_errors_are_recoverable_except_auth() {
        let http = MockHttp::new(vec![json_response(
            401,
            json!({"error": {"message": "bad"}}),
        )]);
        let result = toolbox(official(&http))
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x", "voice": "nova"}),
            ))
            .await;
        assert!(matches!(result, Err(ToolError::Configuration(_))));

        let without_voice = toolbox(official(&MockHttp::new(vec![])));
        let result = without_voice
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x"}),
            ))
            .await;
        assert!(matches!(result, Err(ToolError::Failed(_))));
    }

    #[tokio::test]
    async fn list_voices_prefers_the_app_list() {
        let http = MockHttp::new(vec![]);
        let all = toolbox(official(&http));
        let output = call_done(&all, "list_voices", json!({})).await;
        let json = text_json(&output);
        assert_eq!(json["voices"].as_array().unwrap().len(), 13);

        let own = toolbox(official(&http)).with_voices(vec![VoiceInfo::new(
            "a",
            "A",
            VoiceOrigin::Preset,
        )]);
        let output = call_done(&own, "list_voices", json!({})).await;
        let voices = text_json(&output)["voices"].clone();
        assert_eq!(voices.as_array().unwrap().len(), 1);
        assert_eq!(voices[0]["id"], json!("a"));
    }

    #[tokio::test]
    async fn list_voices_without_support_is_empty_not_an_error() {
        let http = MockHttp::new(vec![]);
        let provider =
            OmlxProvider::new("model", "", "http://local/v1").with_client(Arc::new(http));
        let output = call_done(&toolbox(provider), "list_voices", json!({})).await;

        let json = text_json(&output);
        assert_eq!(json["voices"], json!([]));
    }

    #[test]
    fn no_tool_can_clone() {
        let http = MockHttp::new(vec![]);
        let names: Vec<_> = toolbox(official(&http))
            .tools_definitions()
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, ["generate_speech", "list_voices"]);
    }
}
