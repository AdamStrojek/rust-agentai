//! # OpenAI speech API
//!
//! [`OpenAiSpeechProvider`] speaks text with `POST /audio/speech` and lists models with
//! `GET /models`. It works with OpenAI (`gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`) and with other
//! servers that follow the same format.
//!
//! Fields without an equivalent are not sent: `language` is ignored.
//!
//! OpenAI shuts down these models on 2027-01-06.

use super::{
    AudioError, AudioFormat, AudioModel, SpeechProvider, SpeechRequest, SpeechResponse, Voice,
    VoiceInfo, audio_media,
};
use agentai_core::{HttpClient, HttpRequest, ReqwestClient};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The OpenAI API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Environment variable with the API key.
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";
/// Environment variable with the base URL, for OpenAI-compatible APIs.
pub const BASE_URL_ENV: &str = "OPENAI_BASE_URL";

/// Presets of every model.
const PRESETS: [&str; 9] = [
    "alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer",
];
/// Presets only the models after `tts-1` have.
const NEWER_PRESETS: [&str; 4] = ["ballad", "verse", "marin", "cedar"];

/// Provider for the OpenAI speech API and APIs compatible with it.
///
/// ```rust,no_run
/// # use agentai_audio::provider::openai::OpenAiSpeechProvider;
/// # fn main() -> Result<(), agentai_audio::AudioError> {
/// // From `OPENAI_API_KEY` and `OPENAI_BASE_URL`
/// let provider = OpenAiSpeechProvider::from_env("gpt-4o-mini-tts")?;
/// // Or everything in code
/// let provider =
///     OpenAiSpeechProvider::new("gpt-4o-mini-tts", "sk-...", "https://api.openai.com/v1");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenAiSpeechProvider {
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenAiSpeechProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiSpeechProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenAiSpeechProvider {
    /// Creates a provider for `model` with the API key and base URL given in code, for example
    /// [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Arc::new(ReqwestClient::new()),
        }
    }

    /// Creates a provider for `model` with the API key from `OPENAI_API_KEY` and the base URL
    /// from `OPENAI_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set. The key may be left
    /// out when `OPENAI_BASE_URL` is set, for local servers without authentication.
    pub fn from_env(model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(model, |name| std::env::var(name).ok())
    }

    fn from_lookup(
        model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        let base_url = lookup(BASE_URL_ENV);
        let api_key = match (lookup(API_KEY_ENV), &base_url) {
            (Some(key), _) => key,
            (None, Some(_)) => String::new(),
            (None, None) => return Err(AudioError::MissingApiKey(API_KEY_ENV)),
        };
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(model, api_key, base_url))
    }

    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    fn official(&self) -> bool {
        self.base_url == DEFAULT_BASE_URL
    }

    fn is_tts1(&self) -> bool {
        self.model.starts_with("tts-1")
    }

    fn presets(&self) -> impl Iterator<Item = &'static str> {
        let newer: &[&str] = if self.is_tts1() { &[] } else { &NEWER_PRESETS };
        PRESETS.into_iter().chain(newer.iter().copied())
    }

    /// The preset named `name`, in any case.
    fn preset(&self, name: &str) -> Option<&'static str> {
        self.presets().find(|p| p.eq_ignore_ascii_case(name))
    }

    /// The voice field for an ID or a text the provider routes; `None` for a description.
    fn id_voice(&self, voice: &Voice) -> Option<WireVoice> {
        match voice {
            Voice::Description(_) => None,
            Voice::Id(value) | Voice::Auto(value) if !self.official() => {
                Some(WireVoice::Name(value.clone()))
            }
            Voice::Id(value) => Some(if let Some(preset) = self.preset(value) {
                WireVoice::Name(preset.to_string())
            } else if value.starts_with("voice_") {
                WireVoice::Custom { id: value.clone() }
            } else {
                WireVoice::Name(value.clone())
            }),
            Voice::Auto(value) => {
                if let Some(preset) = self.preset(value) {
                    Some(WireVoice::Name(preset.to_string()))
                } else if value.starts_with("voice_") {
                    Some(WireVoice::Custom { id: value.clone() })
                } else {
                    None
                }
            }
        }
    }

    /// The voice field and the instructions of a request.
    fn resolve(&self, request: &SpeechRequest) -> Result<(WireVoice, Option<String>), AudioError> {
        let description = match &request.voice {
            Some(Voice::Description(text)) => Some(text.clone()),
            Some(voice @ Voice::Auto(text)) if self.id_voice(voice).is_none() => Some(text.clone()),
            Some(voice) => {
                let voice = self.id_voice(voice).ok_or(AudioError::MissingVoice)?;
                return Ok((voice, request.instructions.clone()));
            }
            None => None,
        };
        if description.is_some() && self.is_tts1() {
            return Err(AudioError::Unsupported(
                "voice descriptions with tts-1 models",
            ));
        }
        let base = request
            .default_voice
            .as_ref()
            .and_then(|voice| self.id_voice(voice))
            .ok_or(AudioError::MissingVoice)?;
        let Some(description) = description else {
            return Ok((base, request.instructions.clone()));
        };
        let instructions = match &request.instructions {
            Some(extra) => format!("{description}\n{extra}"),
            None => description,
        };
        Ok((base, Some(instructions)))
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireVoice {
    Name(String),
    Custom { id: String },
}

#[derive(Serialize)]
struct SpeechBody<'a> {
    model: &'a str,
    input: &'a str,
    voice: WireVoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
}

fn format_name(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Wav => "wav",
        AudioFormat::Opus => "opus",
        AudioFormat::Flac => "flac",
        AudioFormat::Aac => "aac",
        AudioFormat::Pcm => "pcm",
    }
}

#[async_trait]
impl SpeechProvider for OpenAiSpeechProvider {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse, AudioError> {
        let (voice, instructions) = self.resolve(&request)?;
        let body = SpeechBody {
            model: &self.model,
            input: &request.text,
            voice,
            instructions,
            response_format: request.format.map(format_name),
            speed: request.speed,
        };
        let http = HttpRequest::post(format!("{}/audio/speech", self.base_url))
            .with_json(&body)?
            .with_bearer_token(&self.api_key);
        let response = self.client.send(http).await?.error_for_status()?;
        let format = request.format.unwrap_or(AudioFormat::Mp3);
        let mut audio = audio_media(&response, format.content_type())?;
        audio.name = request.name;
        let mut result = SpeechResponse::new(audio);
        if format == AudioFormat::Pcm {
            result = result.with_sample_rate(24000);
        }
        Ok(result)
    }

    /// Only the OpenAI API with a model after `tts-1` takes a description (sent as
    /// instructions). On compatible servers [`Voice::Auto`] is an ID, so this is `false`.
    fn supports_voice_description(&self) -> bool {
        self.official() && !self.is_tts1()
    }

    /// The presets of the model. Only the OpenAI API has a fixed list; compatible servers
    /// return [`AudioError::Unsupported`].
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        if !self.official() {
            return Err(AudioError::Unsupported(
                "listing voices of a compatible server",
            ));
        }
        Ok(self.presets().map(|p| VoiceInfo::new(p, p)).collect())
    }

    /// Lists the models of `GET /models`. On the OpenAI API only speech models (`tts-1*`,
    /// `gpt-4o-mini-tts*`) are kept; compatible servers keep every model they list.
    async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        let request =
            HttpRequest::get(format!("{}/models", self.base_url)).with_bearer_token(&self.api_key);
        let response = self.client.send(request).await?.error_for_status()?;
        let list: ModelList = response
            .json()
            .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
        let official = self.official();
        Ok(list
            .data
            .into_iter()
            .filter(|entry| {
                !official
                    || entry.id.starts_with("tts-1")
                    || entry.id.starts_with("gpt-4o-mini-tts")
            })
            .map(|entry| AudioModel::new(format!("openai::{}", entry.id), entry.id))
            .collect())
    }
}

#[derive(Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::{MockHttp, body_json, json_response};
    use crate::provider::{AudioFormat, SpeechRequest, Voice};
    use agentai_core::HttpResponse;
    use serde_json::json;
    use std::sync::Arc;

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn build(model: &str, base_url: &str, http: &Arc<MockHttp>) -> OpenAiSpeechProvider {
        OpenAiSpeechProvider::new(model, "key", base_url).with_client(http.clone())
    }

    fn official(http: &Arc<MockHttp>) -> OpenAiSpeechProvider {
        build("gpt-4o-mini-tts", DEFAULT_BASE_URL, http)
    }

    fn tts1(http: &Arc<MockHttp>) -> OpenAiSpeechProvider {
        build("tts-1", DEFAULT_BASE_URL, http)
    }

    fn compatible(http: &Arc<MockHttp>) -> OpenAiSpeechProvider {
        build("kokoro", "http://local/v1", http)
    }

    #[tokio::test]
    async fn generate_sends_only_what_is_set() {
        let http = MockHttp::new(vec![audio()]);
        let r = official(&http)
            .generate(SpeechRequest::new("hi").with_voice(Voice::id("alloy")))
            .await
            .unwrap();
        assert_eq!(
            http.request(0).url,
            "https://api.openai.com/v1/audio/speech"
        );
        assert_eq!(
            body_json(&http.request(0)),
            json!({"model": "gpt-4o-mini-tts", "input": "hi", "voice": "alloy"})
        );
        assert_eq!(r.audio.content_type, "audio/mpeg");
        assert_eq!(r.sample_rate, None);
    }

    #[tokio::test]
    async fn format_and_pcm_rate() {
        let http = MockHttp::new(vec![audio()]);
        let req = SpeechRequest::new("hi").with_voice(Voice::id("alloy"));
        let r = official(&http)
            .generate(req.with_format(AudioFormat::Pcm))
            .await
            .unwrap();
        assert_eq!(body_json(&http.request(0))["response_format"], json!("pcm"));
        assert_eq!(r.audio.content_type, "audio/pcm");
        assert_eq!(r.sample_rate, Some(24000));
    }

    async fn voice_body(
        provider: OpenAiSpeechProvider,
        http: &Arc<MockHttp>,
        request: SpeechRequest,
    ) -> serde_json::Value {
        provider.generate(request).await.unwrap();
        body_json(&http.request(0))
    }

    #[tokio::test]
    async fn custom_voice_id_is_an_object() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi").with_voice(Voice::id("voice_123")),
        )
        .await;
        assert_eq!(body["voice"], json!({"id": "voice_123"}));
    }

    #[tokio::test]
    async fn only_voice_prefixed_ids_are_objects() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi").with_voice(Voice::id("custom-name")),
        )
        .await;
        assert_eq!(body["voice"], json!("custom-name"));
    }

    #[tokio::test]
    async fn auto_routes_presets_ids_and_descriptions() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi").with_voice(Voice::auto("Alloy")),
        )
        .await;
        assert_eq!(body["voice"], json!("alloy"));

        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi").with_voice(Voice::auto("voice_9")),
        )
        .await;
        assert_eq!(body["voice"], json!({"id": "voice_9"}));

        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi")
                .with_voice(Voice::auto("warm low voice"))
                .with_default_voice(Voice::id("onyx")),
        )
        .await;
        assert_eq!(body["voice"], json!("onyx"));
        assert_eq!(body["instructions"], json!("warm low voice"));
    }

    #[tokio::test]
    async fn a_description_is_joined_with_instructions() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi")
                .with_voice(Voice::description("warm"))
                .with_instructions("slowly")
                .with_default_voice(Voice::id("onyx")),
        )
        .await;
        assert_eq!(body["instructions"], json!("warm\nslowly"));
    }

    #[tokio::test]
    async fn missing_voice_cases() {
        let http = MockHttp::new(vec![]);
        let provider = official(&http);
        let err = provider.generate(SpeechRequest::new("hi")).await;
        assert!(matches!(err, Err(AudioError::MissingVoice)));
        let err = provider
            .generate(SpeechRequest::new("hi").with_voice(Voice::description("warm")))
            .await;
        assert!(matches!(err, Err(AudioError::MissingVoice)));
        assert!(http.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn default_voice_is_used_without_a_voice() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            official(&http),
            &http,
            SpeechRequest::new("hi").with_default_voice(Voice::id("nova")),
        )
        .await;
        assert_eq!(body["voice"], json!("nova"));
    }

    #[tokio::test]
    async fn tts1_rejects_descriptions() {
        let http = MockHttp::new(vec![]);
        let err = tts1(&http)
            .generate(
                SpeechRequest::new("hi")
                    .with_voice(Voice::description("warm"))
                    .with_default_voice(Voice::id("onyx")),
            )
            .await;
        assert!(matches!(err, Err(AudioError::Unsupported(_))));
        assert!(http.requests.lock().unwrap().is_empty());
        assert!(!tts1(&http).supports_voice_description());
        assert!(official(&http).supports_voice_description());
        assert!(!compatible(&http).supports_voice_description());
        // Without a default voice it is still Unsupported, not MissingVoice.
        let err = tts1(&http)
            .generate(SpeechRequest::new("hi").with_voice(Voice::description("warm")))
            .await;
        assert!(matches!(err, Err(AudioError::Unsupported(_))));
    }

    #[tokio::test]
    async fn compatible_servers_take_auto_as_an_id() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(
            compatible(&http),
            &http,
            SpeechRequest::new("hi").with_voice(Voice::auto("af_bella")),
        )
        .await;
        assert_eq!(body["voice"], json!("af_bella"));
        assert_eq!(http.request(0).url, "http://local/v1/audio/speech");
    }

    #[tokio::test]
    async fn voices_are_the_presets_of_the_model() {
        let http = MockHttp::new(vec![]);
        assert_eq!(tts1(&http).list_voices().await.unwrap().len(), 9);
        assert_eq!(official(&http).list_voices().await.unwrap().len(), 13);
        assert!(matches!(
            compatible(&http).list_voices().await,
            Err(AudioError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn models_are_filtered_on_the_official_api() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"data": [{"id": "gpt-4o"}, {"id": "tts-1"}, {"id": "gpt-4o-mini-tts"}]}),
        )]);
        let models = official(&http).list_models().await.unwrap();
        let names: Vec<_> = models.iter().map(|m| m.model.as_str()).collect();
        assert_eq!(names, ["openai::tts-1", "openai::gpt-4o-mini-tts"]);
        assert_eq!(http.request(0).url, "https://api.openai.com/v1/models");
    }
}
