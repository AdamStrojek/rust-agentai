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
    AudioError, AudioFormat, AudioModel, SpeechProvider, SpeechProviderKind, SpeechResponse,
    VoiceInfo, VoiceOrigin, audio_media, known_capabilities,
};
use crate::request::SpeechRequest;
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
    speech_model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenAiSpeechProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiSpeechProvider")
            .field("speech_model", &self.speech_model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenAiSpeechProvider {
    /// Creates a provider for `speech_model` with the API key and base URL given in code, for
    /// example [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header.
    pub fn new(
        speech_model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            speech_model: speech_model.into(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Arc::new(ReqwestClient::new()),
        }
    }

    /// Creates a provider for `speech_model` with the API key from `OPENAI_API_KEY` and the
    /// base URL from `OPENAI_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set. The key may
    /// be left out when `OPENAI_BASE_URL` is set, for local servers without authentication.
    pub fn from_env(speech_model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(speech_model, |name| std::env::var(name).ok())
    }

    pub(crate) fn from_lookup(
        speech_model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        let base_url = lookup(BASE_URL_ENV);
        let api_key = match (lookup(API_KEY_ENV), &base_url) {
            (Some(key), _) => key,
            (None, Some(_)) => String::new(),
            (None, None) => return Err(AudioError::MissingApiKey(API_KEY_ENV)),
        };
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(speech_model, api_key, base_url))
    }

    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    fn official(&self) -> bool {
        self.base_url == DEFAULT_BASE_URL
    }

    /// The voice field for an ID, with `model`.
    fn id_voice(&self, model: &str, id: &str) -> WireVoice {
        if !self.official() {
            WireVoice::Name(id.to_string())
        } else if let Some(preset) = presets(model).find(|p| p.eq_ignore_ascii_case(id)) {
            WireVoice::Name(preset.to_string())
        } else if id.starts_with("voice_") {
            WireVoice::Custom { id: id.to_string() }
        } else {
            WireVoice::Name(id.to_string())
        }
    }
}

/// The presets of a model.
fn presets(model: &str) -> impl Iterator<Item = &'static str> {
    let newer: &[&str] = if model.starts_with("tts-1") {
        &[]
    } else {
        &NEWER_PRESETS
    };
    PRESETS.into_iter().chain(newer.iter().copied())
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
    instructions: Option<&'a str>,
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
    async fn generate(&self, request: &SpeechRequest<'_>) -> Result<SpeechResponse, AudioError> {
        request.without_timestamps()?;
        let model = request.model().unwrap_or(&self.speech_model);
        let voice = request.voice().ok_or(AudioError::MissingVoice)?;
        let body = SpeechBody {
            model,
            input: request.text(),
            voice: self.id_voice(model, voice),
            instructions: request.instructions(),
            response_format: request.format().map(format_name),
            speed: request.speed(),
        };
        let http = HttpRequest::post(format!("{}/audio/speech", self.base_url))
            .with_json(&body)?
            .with_bearer_token(&self.api_key);
        let response = self.client.send(http).await?.error_for_status()?;
        let format = request.format().unwrap_or(AudioFormat::Mp3);
        let audio = audio_media(&response, format.content_type())?;
        let mut result = SpeechResponse::new(audio).with_model(model);
        if format == AudioFormat::Pcm {
            result = result.with_sample_rate(24000);
        }
        Ok(result)
    }

    /// The presets of the model. Only the OpenAI API has a fixed list; compatible servers
    /// return [`AudioError::Unsupported`].
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        if !self.official() {
            return Err(AudioError::Unsupported(
                "listing voices of a compatible server",
            ));
        }
        Ok(presets(&self.speech_model)
            .map(|p| VoiceInfo::new(p, p, VoiceOrigin::Preset))
            .collect())
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
            .map(|entry| {
                let name = entry.id.clone();
                known_capabilities(SpeechProviderKind::OpenAi, AudioModel::new(entry.id, name))
            })
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
    use crate::SpeechGenerator;
    use crate::provider::AudioFormat;
    use agentai_core::HttpResponse;
    use agentai_core::mock::{MockHttp, body_json, json_response};
    use serde_json::json;
    use std::sync::Arc;

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn build(model: &str, base_url: &str, http: &MockHttp) -> OpenAiSpeechProvider {
        OpenAiSpeechProvider::new(model, "key", base_url).with_client(Arc::new(http.clone()))
    }

    fn official(http: &MockHttp) -> OpenAiSpeechProvider {
        build("gpt-4o-mini-tts", DEFAULT_BASE_URL, http)
    }

    fn tts1(http: &MockHttp) -> OpenAiSpeechProvider {
        build("tts-1", DEFAULT_BASE_URL, http)
    }

    fn compatible(http: &MockHttp) -> OpenAiSpeechProvider {
        build("kokoro", "http://local/v1", http)
    }

    #[tokio::test]
    async fn generate_sends_only_what_is_set() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(official(&http))
            .generate("hi")
            .with_voice("alloy")
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
        assert_eq!(r.model.as_deref(), Some("gpt-4o-mini-tts"));
    }

    #[tokio::test]
    async fn format_and_pcm_rate() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(official(&http))
            .generate("hi")
            .with_voice("alloy")
            .with_format(AudioFormat::Pcm)
            .await
            .unwrap();
        assert_eq!(body_json(&http.request(0))["response_format"], json!("pcm"));
        assert_eq!(r.audio.content_type, "audio/pcm");
        assert_eq!(r.sample_rate, Some(24000));
    }

    async fn voice_body(
        provider: OpenAiSpeechProvider,
        http: &MockHttp,
        voice: &str,
    ) -> serde_json::Value {
        SpeechGenerator::from_provider(provider)
            .generate("hi")
            .with_voice(voice)
            .await
            .unwrap();
        body_json(&http.request(0))
    }

    #[tokio::test]
    async fn custom_voice_id_is_an_object() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(official(&http), &http, "voice_123").await;
        assert_eq!(body["voice"], json!({"id": "voice_123"}));
    }

    #[tokio::test]
    async fn only_voice_prefixed_ids_are_objects() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(official(&http), &http, "custom-name").await;
        assert_eq!(body["voice"], json!("custom-name"));
    }

    #[tokio::test]
    async fn missing_voice_cases() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(official(&http))
            .generate("hi")
            .await;
        assert!(matches!(err, Err(AudioError::MissingVoice)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn compatible_servers_take_any_id() {
        let http = MockHttp::new(vec![audio()]);
        let body = voice_body(compatible(&http), &http, "af_bella").await;
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
        let names: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(names, ["tts-1", "gpt-4o-mini-tts"]);
        assert_eq!(http.request(0).url, "https://api.openai.com/v1/models");
    }
}
