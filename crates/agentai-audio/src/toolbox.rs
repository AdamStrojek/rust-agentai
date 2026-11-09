//! Agent tools (feature `toolbox`): [`SpeechToolBox`].

use crate::generator::SpeechGenerator;
use crate::provider::{AudioError, SpeechRequest, SpeechResponse, Voice, VoiceInfo};
use agentai::tool::{Tool, ToolBox, ToolError, ToolOutput, ToolResult, toolbox};
use agentai_core::message::MediaSource;
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
/// limited with [`with_voices`](SpeechToolBox::with_voices); without it any voice ID or
/// description goes to the provider.
///
/// ```rust,no_run
/// # use agentai_audio::{SpeechGenerator, SpeechToolBox, Voice};
/// # use agentai_audio::storage::MemoryStorage;
/// # fn main() -> Result<(), agentai_audio::AudioError> {
/// let generator = SpeechGenerator::new("openai::gpt-4o-mini-tts")?
///     .with_default_voice(Voice::id("alloy"));
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
    /// that is not in it is used as a description of the voice, or rejected when the provider
    /// takes none. Use it whenever cloned voices exist: without it the model can use any stored
    /// or account voice by its ID.
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
    fn voice(&self, voice: Option<&str>) -> ToolResult<Option<Voice>> {
        let Some(voice) = voice.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(None);
        };
        let Some(voices) = &self.voices else {
            return Ok(Some(Voice::auto(voice)));
        };
        if voices.iter().any(|v| v.id == voice) {
            Ok(Some(Voice::id(voice)))
        } else if self.generator.supports_voice_description() {
            Ok(Some(Voice::description(voice)))
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
                    bytes: match &response.audio.source {
                        MediaSource::Base64(data) => Some(base64_len(data)),
                        MediaSource::Url(_) => None,
                    },
                    model: self.generator.model(),
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
        /// A voice ID from `list_voices`, or, when `descriptions_supported` is true, a
        /// description of the voice such as `warm low female`. Leave it out for the default
        /// voice.
        voice: Option<String>,
        /// How to speak, for example the tone, pace or emotion. Leave it out when not needed.
        instructions: Option<String>,
    ) -> ToolResult<ToolOutput> {
        let voice_text = voice.as_deref().map(str::trim).filter(|v| !v.is_empty());
        let mut request = SpeechRequest::new(text).with_name(name);
        if let Some(voice) = self.voice(voice.as_deref())? {
            request = request.with_voice(voice);
        }
        if let Some(instructions) = instructions.filter(|i| !i.trim().is_empty()) {
            request = request.with_instructions(instructions);
        }
        let response = self
            .generator
            .generate_with(request)
            .await
            .map_err(tool_error)?;
        self.output(response, voice_text)
    }

    /// Lists the voices you can use in `generate_speech`, and tells whether a description of a
    /// voice works there too.
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
        let list = VoiceList {
            voices: &voices,
            descriptions_supported: self.generator.supports_voice_description(),
        };
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
    descriptions_supported: bool,
}

/// Size of base64 data once decoded, without decoding it.
fn base64_len(data: &str) -> usize {
    let padding = data.bytes().rev().take_while(|&b| b == b'=').count();
    (data.len() / 4 * 3).saturating_sub(padding)
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
    use crate::provider::elevenlabs::ElevenLabsProvider;
    use crate::provider::omlx::OmlxProvider;
    use crate::provider::openai::OpenAiSpeechProvider;
    use crate::provider::test_support::{MockHttp, body_json, json_response};
    use agentai_core::HttpResponse;
    use agentai_core::message::Part;
    use agentai_core::storage::MemoryStorage;
    use serde_json::{Value, json};
    use std::sync::Arc;

    fn tool_call(name: &str, arguments: Value) -> agentai::ToolCall {
        agentai::ToolCall::new("call_1", name, arguments)
    }

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn openai(http: &Arc<MockHttp>, base_url: &str) -> OpenAiSpeechProvider {
        OpenAiSpeechProvider::new("gpt-4o-mini-tts", "key", base_url).with_client(http.clone())
    }

    fn official(http: &Arc<MockHttp>) -> OpenAiSpeechProvider {
        openai(http, "https://api.openai.com/v1")
    }

    fn toolbox(provider: impl crate::provider::SpeechProvider) -> SpeechToolBox {
        SpeechToolBox::new(
            SpeechGenerator::from_provider(provider),
            MemoryStorage::new(),
        )
    }

    fn text_json(output: &ToolOutput) -> Value {
        serde_json::from_str(&output.text().unwrap()).unwrap()
    }

    #[tokio::test]
    async fn generate_speech_returns_metadata_by_default() {
        let http = MockHttp::new(vec![audio()]);
        let toolbox = toolbox(official(&http));

        let output = toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "greeting", "voice": "alloy"}),
            ))
            .await
            .unwrap();

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

        let output = toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "greeting", "voice": "alloy"}),
            ))
            .await
            .unwrap();

        assert!(output.text().unwrap().contains("1-greeting.mp3"));
        assert!(matches!(output.parts.last(), Some(Part::Media(_))));
        assert_eq!(output.media().len(), 1);
    }

    #[tokio::test]
    async fn a_blank_voice_means_the_default() {
        let blanks = [
            json!({"text": "hi", "name": "x", "voice": ""}),
            json!({"text": "hi", "name": "x", "voice": "   "}),
            json!({"text": "hi", "name": "x"}),
            json!({"text": "hi", "name": "x", "voice": null}),
        ];
        for arguments in blanks {
            let http = MockHttp::new(vec![audio()]);
            let generator = SpeechGenerator::from_provider(official(&http))
                .with_default_voice(Voice::id("nova"));
            let toolbox = SpeechToolBox::new(generator, MemoryStorage::new());

            toolbox
                .call_tool(&tool_call("generate_speech", arguments.clone()))
                .await
                .unwrap();

            assert_eq!(
                body_json(&http.request(0))["voice"],
                json!("nova"),
                "{arguments}"
            );
        }
    }

    #[tokio::test]
    async fn the_voice_list_is_an_allowlist() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let generator =
            SpeechGenerator::from_provider(official(&http)).with_default_voice(Voice::id("nova"));
        let toolbox = SpeechToolBox::new(generator, MemoryStorage::new())
            .with_voices(vec![VoiceInfo::new("voice_ok", "OK")]);

        for voice in ["voice_ok", "voice_hidden"] {
            toolbox
                .call_tool(&tool_call(
                    "generate_speech",
                    json!({"text": "hi", "name": "x", "voice": voice}),
                ))
                .await
                .unwrap();
        }

        let listed = body_json(&http.request(0));
        assert_eq!(listed["voice"], json!({"id": "voice_ok"}));
        let hidden = body_json(&http.request(1));
        assert_eq!(hidden["voice"], json!("nova"));
        assert_eq!(hidden["instructions"], json!("voice_hidden"));
    }

    #[tokio::test]
    async fn an_unlisted_id_fails_without_descriptions() {
        let http = MockHttp::new(vec![]);
        let provider = ElevenLabsProvider::new("eleven_v3", "key", "https://api.test")
            .with_client(http.clone());
        let toolbox = toolbox(provider).with_voices(vec![VoiceInfo::new("abc", "Abc")]);

        let result = toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x", "voice": "hidden"}),
            ))
            .await;

        assert!(matches!(result, Err(ToolError::Failed(m)) if m.contains("list_voices")));
        assert!(http.requests.lock().unwrap().is_empty());
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
        let generator =
            SpeechGenerator::from_provider(official(&http)).with_default_voice(Voice::id("nova"));
        let toolbox = SpeechToolBox::new(generator, MemoryStorage::new());
        let result = toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x", "voice": "nova"}),
            ))
            .await;
        assert!(matches!(result, Err(ToolError::Configuration(_))));

        let no_default = toolbox_without_default();
        let result = no_default
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x"}),
            ))
            .await;
        assert!(matches!(result, Err(ToolError::Failed(_))));
    }

    fn toolbox_without_default() -> SpeechToolBox {
        let http = MockHttp::new(vec![]);
        toolbox(official(&http))
    }

    #[tokio::test]
    async fn list_voices_prefers_the_app_list() {
        let http = MockHttp::new(vec![]);
        let all = toolbox(official(&http));
        let output = all
            .call_tool(&tool_call("list_voices", json!({})))
            .await
            .unwrap();
        let json = text_json(&output);
        assert_eq!(json["voices"].as_array().unwrap().len(), 13);
        assert_eq!(json["descriptions_supported"], json!(true));

        let own = toolbox(official(&http)).with_voices(vec![VoiceInfo::new("a", "A")]);
        let output = own
            .call_tool(&tool_call("list_voices", json!({})))
            .await
            .unwrap();
        let voices = text_json(&output)["voices"].clone();
        assert_eq!(voices.as_array().unwrap().len(), 1);
        assert_eq!(voices[0]["id"], json!("a"));
    }

    #[tokio::test]
    async fn list_voices_without_support_is_empty_not_an_error() {
        let http = MockHttp::new(vec![]);
        let provider = OmlxProvider::new("model", "", "http://local/v1").with_client(http);
        let output = toolbox(provider)
            .call_tool(&tool_call("list_voices", json!({})))
            .await
            .unwrap();

        let json = text_json(&output);
        assert_eq!(json["voices"], json!([]));
        assert_eq!(json["descriptions_supported"], json!(true));
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

    #[tokio::test]
    async fn an_unlisted_clone_is_a_description_not_a_reference() {
        use crate::voice_store::{MemoryVoiceStore, StoredVoice, VoiceStore};
        let http = MockHttp::new(vec![audio()]);
        let store = MemoryVoiceStore::new();
        let id = store
            .save(StoredVoice::new(
                "Adam",
                agentai_core::message::Media::from_bytes("audio/wav", b"WAV"),
            ))
            .await
            .unwrap();
        let provider = OmlxProvider::new("model", "", "http://local/v1")
            .with_client(http.clone())
            .with_voice_store(store);
        let toolbox = toolbox(provider).with_voices(vec![VoiceInfo::new("other", "Other")]);
        toolbox
            .call_tool(&tool_call(
                "generate_speech",
                json!({"text": "hi", "name": "x", "voice": id}),
            ))
            .await
            .unwrap();
        let body = body_json(&http.request(0));
        assert!(body.get("ref_audio").is_none());
        assert_eq!(body["instructions"], json!(id));
    }
}
