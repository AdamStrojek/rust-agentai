//! # ElevenLabs speech API
//!
//! [`ElevenLabsProvider`] speaks text with `POST /v1/text-to-speech/{voice_id}`, clones voices
//! with `POST /v1/voices/add` and lists voices and models. It designs voices from a description
//! with `POST /v1/text-to-voice/design` and saves a chosen preview with `POST /v1/text-to-voice`.
//!
//! Voice design needs a design model, set with
//! [`with_design_model`](ElevenLabsProvider::with_design_model), for example `eleven_ttv_v3`.
//!
//! Dialogues go to `POST /v1/text-to-dialogue` and need a dialogue model, set with
//! [`with_dialogue_model`](ElevenLabsProvider::with_dialogue_model), for example `eleven_v3`.
//!
//! Fields without an equivalent are not sent: `instructions` is ignored.

use super::{
    AudioError, AudioFormat, AudioModel, SpeechProvider, SpeechResponse, VoiceId, VoiceInfo,
    VoicePreview, VoiceSource, audio_media,
};
use crate::request::{DesignVoiceRequest, DialogueRequest, SpeechRequest};
use agentai_core::message::Media;
use agentai_core::{HttpClient, HttpRequest, Multipart, ReqwestClient, percent_encode};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The ElevenLabs API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.elevenlabs.io";
/// Environment variable with the API key.
pub const API_KEY_ENV: &str = "ELEVENLABS_API_KEY";
/// Environment variable with the base URL.
pub const BASE_URL_ENV: &str = "ELEVENLABS_BASE_URL";

/// Provider for the ElevenLabs API.
///
/// ```rust,no_run
/// # use agentai_audio::provider::elevenlabs::ElevenLabsProvider;
/// # fn main() -> Result<(), agentai_audio::AudioError> {
/// // From `ELEVENLABS_API_KEY`
/// let provider = ElevenLabsProvider::from_env("eleven_multilingual_v2")?;
/// // Or everything in code
/// let provider = ElevenLabsProvider::new(
///     "eleven_multilingual_v2",
///     "xi-...",
///     "https://api.elevenlabs.io",
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ElevenLabsProvider {
    speech_model: String,
    dialogue_model: Option<String>,
    design_model: Option<String>,
    api_key: String,
    base_url: String,
    output_format: Option<String>,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for ElevenLabsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElevenLabsProvider")
            .field("speech_model", &self.speech_model)
            .field("dialogue_model", &self.dialogue_model)
            .field("design_model", &self.design_model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl ElevenLabsProvider {
    /// Creates a provider for `speech_model` with the API key and base URL given in code, for
    /// example [`DEFAULT_BASE_URL`].
    pub fn new(
        speech_model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            speech_model: speech_model.into(),
            dialogue_model: None,
            design_model: None,
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            output_format: None,
            client: Arc::new(ReqwestClient::new()),
        }
    }

    /// Creates a provider for `speech_model` with the API key from `ELEVENLABS_API_KEY` and the
    /// base URL from `ELEVENLABS_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set.
    pub fn from_env(speech_model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(speech_model, |name| std::env::var(name).ok())
    }

    pub(crate) fn from_lookup(
        speech_model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        let api_key = lookup(API_KEY_ENV).ok_or(AudioError::MissingApiKey(API_KEY_ENV))?;
        let base_url = lookup(BASE_URL_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(speech_model, api_key, base_url))
    }

    /// Sets the model of dialogues, for example `eleven_v3`.
    pub fn with_dialogue_model(mut self, model: impl Into<String>) -> Self {
        self.dialogue_model = Some(model.into());
        self
    }

    /// Sets the model of voice design, for example `eleven_ttv_v3`. Without it voice design is
    /// off.
    pub fn with_design_model(mut self, model: impl Into<String>) -> Self {
        self.design_model = Some(model.into());
        self
    }

    /// Sets the `output_format` of the API, for example `pcm_16000`. It wins over the format of
    /// the request.
    pub fn with_output_format(mut self, output_format: impl Into<String>) -> Self {
        self.output_format = Some(output_format.into());
        self
    }

    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    fn get(&self, path: &str) -> HttpRequest {
        self.authorize(HttpRequest::get(format!("{}{path}", self.base_url)))
    }

    fn authorize(&self, request: HttpRequest) -> HttpRequest {
        request.with_header("xi-api-key", self.api_key.as_str())
    }

    /// The `output_format` to send: the provider's own wins over the format of the request.
    fn output_format_of(&self, format: Option<AudioFormat>) -> Result<Option<&str>, AudioError> {
        match (&self.output_format, format) {
            (Some(over), _) => Ok(Some(over.as_str())),
            (None, Some(format)) => output_format_for(format).map(Some),
            (None, None) => Ok(None),
        }
    }

    async fn send_audio(
        &self,
        url: String,
        body: &impl Serialize,
        output_format: Option<&str>,
        model: &str,
    ) -> Result<SpeechResponse, AudioError> {
        let http = self.authorize(HttpRequest::post(url).with_json(body)?);
        let response = self.client.send(http).await?.error_for_status()?;
        let content_type = output_format.map_or("audio/mpeg", content_type_of);
        let audio = audio_media(&response, content_type)?;
        let mut result = SpeechResponse::new(audio).with_model(model);
        if let Some(rate) = output_format.and_then(pcm_sample_rate) {
            result = result.with_sample_rate(rate);
        }
        Ok(result)
    }

    /// The path segment of the voice of a request.
    fn voice_id<'a>(&self, request: &'a SpeechRequest<'_>) -> Result<&'a str, AudioError> {
        let id = request.voice().ok_or(AudioError::MissingVoice)?;
        let safe = !id.is_empty()
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if safe {
            Ok(id)
        } else {
            Err(AudioError::InvalidRequest(format!(
                "invalid ElevenLabs voice ID '{id}'"
            )))
        }
    }
}

/// The `output_format` for a format of the request.
fn output_format_for(format: AudioFormat) -> Result<&'static str, AudioError> {
    match format {
        AudioFormat::Pcm => Ok("pcm_24000"),
        AudioFormat::Mp3 => Ok("mp3_44100_128"),
        AudioFormat::Wav => Ok("wav_44100"),
        AudioFormat::Opus => Ok("opus_48000_128"),
        AudioFormat::Flac => Err(AudioError::Unsupported("FLAC output with ElevenLabs")),
        AudioFormat::Aac => Err(AudioError::Unsupported("AAC output with ElevenLabs")),
    }
}

/// MIME type of an `output_format` such as `mp3_44100_128`.
fn content_type_of(output_format: &str) -> &'static str {
    match output_format.split('_').next().unwrap_or_default() {
        "mp3" => "audio/mpeg",
        "pcm" => "audio/pcm",
        "wav" => "audio/wav",
        "opus" => "audio/ogg",
        "ulaw" => "audio/basic",
        "alaw" => "audio/x-alaw-basic",
        _ => "application/octet-stream",
    }
}

/// The sample rate of a `pcm_<rate>` output format.
fn pcm_sample_rate(output_format: &str) -> Option<u32> {
    output_format.strip_prefix("pcm_")?.parse().ok()
}

#[derive(Serialize)]
struct SpeechBody<'a> {
    text: &'a str,
    model_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_code: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_settings: Option<VoiceSettings>,
}

#[derive(Serialize)]
struct DialogueBody<'a> {
    inputs: Vec<DialogueInput<'a>>,
    model_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_code: Option<&'a str>,
}

#[derive(Serialize)]
struct DialogueInput<'a> {
    text: &'a str,
    voice_id: &'a str,
}

#[derive(Serialize)]
struct VoiceSettings {
    speed: f64,
}

#[async_trait]
impl SpeechProvider for ElevenLabsProvider {
    async fn generate(&self, request: &SpeechRequest<'_>) -> Result<SpeechResponse, AudioError> {
        request.without_timestamps()?;
        let model = request.model().unwrap_or(&self.speech_model);
        let voice = self.voice_id(request)?;
        let output_format = self.output_format_of(request.format())?;
        let mut url = format!("{}/v1/text-to-speech/{voice}", self.base_url);
        if let Some(output_format) = output_format {
            url.push_str("?output_format=");
            url.push_str(&percent_encode(output_format));
        }
        let body = SpeechBody {
            text: request.text(),
            model_id: model,
            language_code: request.language(),
            voice_settings: request.speed().map(|speed| VoiceSettings { speed }),
        };
        self.send_audio(url, &body, output_format, model).await
    }

    fn supports_dialogue(&self) -> bool {
        self.dialogue_model.is_some()
    }

    /// Speaks the turns with `POST /v1/text-to-dialogue`, one input per turn with the voice
    /// ID of its speaker. Timestamps are not supported.
    async fn generate_dialogue(
        &self,
        request: &DialogueRequest<'_>,
    ) -> Result<SpeechResponse, AudioError> {
        let default_model = self
            .dialogue_model
            .as_deref()
            .ok_or(AudioError::MissingModel("dialogue"))?;
        request.without_timestamps()?;
        let model = request.model().unwrap_or(default_model);
        let output_format = self.output_format_of(request.format())?;
        let mut url = format!("{}/v1/text-to-dialogue", self.base_url);
        if let Some(output_format) = output_format {
            url.push_str("?output_format=");
            url.push_str(&percent_encode(output_format));
        }
        let inputs = request
            .turns()
            .iter()
            .map(|(speaker, text)| {
                let voice_id = request.voices().get(*speaker).ok_or_else(|| {
                    AudioError::InvalidRequest(format!("no voice for speaker {speaker}"))
                })?;
                Ok(DialogueInput { text, voice_id })
            })
            .collect::<Result<Vec<_>, AudioError>>()?;
        let body = DialogueBody {
            inputs,
            model_id: model,
            language_code: request.language(),
        };
        self.send_audio(url, &body, output_format, model).await
    }

    fn supports_voice_cloning(&self) -> bool {
        true
    }

    fn supports_voice_design(&self) -> bool {
        self.design_model.is_some()
    }

    /// Designs voices with `POST /v1/text-to-voice/design`. The API picks the number of
    /// previews, so `count` is not used. Without `text` the API writes one itself. Keep a
    /// preview by passing its `sample_id` to `create_voice`.
    async fn design_voice(
        &self,
        request: &DesignVoiceRequest<'_>,
    ) -> Result<Vec<VoicePreview>, AudioError> {
        let model = self
            .design_model
            .as_deref()
            .ok_or(AudioError::MissingModel("voice design"))?;
        let model = request.model().unwrap_or(model);
        let text = request.text();
        let body = DesignBody {
            voice_description: request.description(),
            model_id: model,
            text,
            auto_generate_text: text.is_none().then_some(true),
        };
        let http = self
            .authorize(HttpRequest::post(format!(
                "{}/v1/text-to-voice/design",
                self.base_url
            )))
            .with_json(&body)?;
        let response = self.client.send(http).await?.error_for_status()?;
        let designed: DesignResponse = response
            .json()
            .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
        designed
            .previews
            .into_iter()
            .map(|preview| {
                let bytes = agentai_core::base64::decode(&preview.audio_base_64)
                    .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
                let audio = Media::from_bytes(preview.media_type, &bytes);
                let result = VoicePreview::new(audio).with_sample_id(preview.generated_voice_id);
                Ok(match designed.text.as_deref().or(text) {
                    Some(text) => result.with_text(text),
                    None => result,
                })
            })
            .collect()
    }

    /// Clones a voice from the samples with `POST /v1/voices/add` and returns its ID. A
    /// [`VoiceSource::SampleId`] saves a designed preview with `POST /v1/text-to-voice`; the
    /// API requires a `description` for it.
    ///
    /// When the API replies with `requires_verification: true`, the voice exists but cannot
    /// speak until it is verified at ElevenLabs. `transcript` is not used.
    async fn create_voice(
        &self,
        name: &str,
        source: VoiceSource,
        description: Option<&str>,
    ) -> Result<VoiceId, AudioError> {
        let samples = match source {
            VoiceSource::Samples { samples, .. } => samples,
            VoiceSource::SampleId(generated_voice_id) => {
                let body = SaveVoiceBody {
                    voice_name: name,
                    voice_description: description,
                    generated_voice_id: &generated_voice_id,
                };
                let http = self
                    .authorize(HttpRequest::post(format!(
                        "{}/v1/text-to-voice",
                        self.base_url
                    )))
                    .with_json(&body)?;
                let response = self.client.send(http).await?.error_for_status()?;
                let created: CreatedVoice = response
                    .json()
                    .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
                return Ok(created.voice_id);
            }
        };
        if samples.is_empty() {
            return Err(AudioError::InvalidRequest(
                "voice cloning needs at least one sample".to_string(),
            ));
        }
        let mut form = Multipart::new().with_text("name", name);
        if let Some(description) = description {
            form = form.with_text("description", description);
        }
        for (index, sample) in samples.iter().enumerate() {
            let bytes = sample.bytes(&*self.client).await?;
            let filename = sample.file_name(&format!("sample-{}", index + 1));
            form = form.with_file("files", filename, sample.content_type.as_str(), bytes);
        }
        let http = self
            .authorize(HttpRequest::post(format!(
                "{}/v1/voices/add",
                self.base_url
            )))
            .with_multipart(form);
        let response = self.client.send(http).await?.error_for_status()?;
        let created: CreatedVoice = response
            .json()
            .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
        Ok(created.voice_id)
    }

    /// Lists the voices of the account with `GET /v2/voices`, following the page tokens.
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        let mut voices = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut path = "/v2/voices?page_size=100".to_string();
            if let Some(token) = &token {
                path.push_str("&next_page_token=");
                path.push_str(&percent_encode(token));
            }
            let response = self
                .client
                .send(self.get(&path))
                .await?
                .error_for_status()?;
            let page: VoicePage = response
                .json()
                .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
            voices.extend(page.voices.into_iter().map(|voice| {
                let info = VoiceInfo::new(voice.voice_id, voice.name)
                    .with_cloned(voice.category.as_deref() == Some("cloned"));
                match voice.description {
                    Some(description) => info.with_description(description),
                    None => info,
                }
            }));
            match page.next_page_token {
                Some(next) if page.has_more && token.as_ref() != Some(&next) => token = Some(next),
                _ => return Ok(voices),
            }
        }
    }

    /// Lists the models of `GET /v1/models` that can speak.
    async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        let response = self
            .client
            .send(self.get("/v1/models"))
            .await?
            .error_for_status()?;
        let models: Vec<ModelEntry> = response
            .json()
            .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
        Ok(models
            .into_iter()
            .filter(|model| model.can_do_text_to_speech)
            .map(|entry| {
                let mut model =
                    AudioModel::new(format!("elevenlabs::{}", entry.model_id), entry.model_id);
                model.name = entry.name;
                model.description = entry.description;
                model
            })
            .collect())
    }
}

#[derive(Serialize)]
struct DesignBody<'a> {
    voice_description: &'a str,
    model_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    auto_generate_text: Option<bool>,
}

#[derive(Deserialize)]
struct DesignResponse {
    previews: Vec<DesignPreview>,
    text: Option<String>,
}

#[derive(Deserialize)]
struct DesignPreview {
    generated_voice_id: String,
    audio_base_64: String,
    media_type: String,
}

#[derive(Serialize)]
struct SaveVoiceBody<'a> {
    voice_name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_description: Option<&'a str>,
    generated_voice_id: &'a str,
}

#[derive(Deserialize)]
struct CreatedVoice {
    voice_id: String,
}

#[derive(Deserialize)]
struct VoicePage {
    #[serde(default)]
    voices: Vec<VoiceEntry>,
    #[serde(default)]
    has_more: bool,
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
struct VoiceEntry {
    voice_id: String,
    #[serde(default)]
    name: String,
    category: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize)]
struct ModelEntry {
    model_id: String,
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    can_do_text_to_speech: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpeechGenerator;
    use crate::request::DialogueTurn;
    use agentai_core::HttpResponse;
    use agentai_core::message::Media;
    use agentai_core::mock::{MockHttp, body_json, json_response};
    use serde_json::json;

    fn provider(http: &MockHttp) -> ElevenLabsProvider {
        ElevenLabsProvider::new("eleven_v4", "key", "https://api.test")
            .with_client(Arc::new(http.clone()))
    }

    fn designer(http: &MockHttp) -> ElevenLabsProvider {
        provider(http).with_design_model("eleven_ttv_v3")
    }

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn dialogue_provider(http: &MockHttp) -> ElevenLabsProvider {
        provider(http).with_dialogue_model("eleven_v3")
    }

    #[tokio::test]
    async fn a_dialogue_posts_inputs_in_turn_order() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(dialogue_provider(&http))
            .generate_dialogue([
                DialogueTurn::new("a", "Hi"),
                DialogueTurn::new("b", "Hello"),
                DialogueTurn::new("a", "Bye"),
            ])
            .with_format(AudioFormat::Pcm)
            .with_language("pl")
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(
            req.url,
            "https://api.test/v1/text-to-dialogue?output_format=pcm_24000"
        );
        assert_eq!(req.header("xi-api-key"), Some("key"));
        assert_eq!(
            body_json(&req),
            json!({
                "inputs": [{"text": "Hi", "voice_id": "a"}, {"text": "Hello", "voice_id": "b"},
                           {"text": "Bye", "voice_id": "a"}],
                "model_id": "eleven_v3", "language_code": "pl"
            })
        );
        assert_eq!(r.sample_rate, Some(24000));
        assert_eq!(r.model.as_deref(), Some("eleven_v3"));
    }

    #[tokio::test]
    async fn the_provider_output_format_wins_for_dialogues() {
        let http = MockHttp::new(vec![audio()]);
        SpeechGenerator::from_provider(dialogue_provider(&http).with_output_format("mp3_22050_32"))
            .generate_dialogue([DialogueTurn::new("a", "Hi")])
            .with_format(AudioFormat::Wav)
            .await
            .unwrap();
        assert!(http.request(0).url.ends_with("?output_format=mp3_22050_32"));
    }

    #[tokio::test]
    async fn a_dialogue_without_a_dialogue_model_or_with_timestamps_sends_nothing() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(provider(&http))
            .generate_dialogue([DialogueTurn::new("a", "Hi")])
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::MissingModel("dialogue")));
        let err = SpeechGenerator::from_provider(dialogue_provider(&http))
            .generate_dialogue([DialogueTurn::new("a", "Hi")])
            .with_timestamps(true)
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn generate_posts_to_the_voice_path() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(provider(&http))
            .generate("hi")
            .with_voice("abc123")
            .with_language("pl")
            .with_speed(1.1)
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/text-to-speech/abc123");
        assert_eq!(req.header("xi-api-key"), Some("key"));
        assert_eq!(
            body_json(&req),
            json!({
                "text": "hi", "model_id": "eleven_v4", "language_code": "pl",
                "voice_settings": {"speed": 1.1}
            })
        );
        assert_eq!(r.audio.content_type, "audio/mpeg");
        assert_eq!(r.model.as_deref(), Some("eleven_v4"));
    }

    #[tokio::test]
    async fn formats_map_to_output_format() {
        let cases = [
            (AudioFormat::Pcm, "pcm_24000", "audio/pcm", Some(24000)),
            (AudioFormat::Mp3, "mp3_44100_128", "audio/mpeg", None),
            (AudioFormat::Wav, "wav_44100", "audio/wav", None),
            (AudioFormat::Opus, "opus_48000_128", "audio/ogg", None),
        ];
        for (format, query, content_type, rate) in cases {
            let http = MockHttp::new(vec![audio()]);
            let r = SpeechGenerator::from_provider(provider(&http))
                .generate("hi")
                .with_voice("v")
                .with_format(format)
                .await
                .unwrap();
            assert!(
                http.request(0)
                    .url
                    .ends_with(&format!("?output_format={query}"))
            );
            assert_eq!(r.audio.content_type, content_type);
            assert_eq!(r.sample_rate, rate);
        }
        for format in [AudioFormat::Flac, AudioFormat::Aac] {
            let http = MockHttp::new(vec![]);
            let err = SpeechGenerator::from_provider(provider(&http))
                .generate("hi")
                .with_voice("v")
                .with_format(format)
                .await
                .unwrap_err();
            assert!(matches!(err, AudioError::Unsupported(_)));
            assert_eq!(http.request_count(), 0);
        }
    }

    #[tokio::test]
    async fn the_output_format_override_wins_and_sets_the_type() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(provider(&http).with_output_format("pcm_16000"))
            .generate("hi")
            .with_voice("v")
            .with_format(AudioFormat::Mp3)
            .await
            .unwrap();
        assert!(http.request(0).url.ends_with("?output_format=pcm_16000"));
        assert_eq!(r.audio.content_type, "audio/pcm");
        assert_eq!(r.sample_rate, Some(16000));
    }

    #[tokio::test]
    async fn unsafe_voice_ids_are_rejected_before_sending() {
        let http = MockHttp::new(vec![]);
        let generator = SpeechGenerator::from_provider(provider(&http));
        for id in ["../../v1/voices/x", "a/b", "a b", "..", ""] {
            let err = generator.generate("hi").with_voice(id).await.unwrap_err();
            assert!(matches!(err, AudioError::InvalidRequest(_)), "{id}");
        }
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn a_missing_voice_is_an_error() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(provider(&http))
            .generate("hi")
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::MissingVoice));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn create_voice_uploads_every_sample() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"voice_id": "new1", "requires_verification": false}),
        )]);
        let samples = vec![
            Media::from_bytes("audio/wav", b"ONE").with_name("a.wav"),
            Media::from_bytes("audio/mpeg", b"TWO"),
        ];
        let id = provider(&http)
            .create_voice(
                "Adam",
                VoiceSource::samples(samples.clone(), None),
                Some("my voice"),
            )
            .await
            .unwrap();
        assert_eq!(id, "new1");

        let request = http.request(0);
        assert_eq!(request.url, "https://api.test/v1/voices/add");
        assert_eq!(request.header("xi-api-key"), Some("key"));
        let boundary = request
            .header("content-type")
            .unwrap()
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        let body = String::from_utf8(request.body.clone().unwrap()).unwrap();
        let field = |name: &str, value: &str| {
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
        };
        let file = |filename: &str, content_type: &str, data: &str| {
            format!(
                "Content-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\n\
                 Content-Type: {content_type}\r\n\r\n{data}\r\n"
            )
        };
        let parts: Vec<&str> = body
            .split(&format!("--{boundary}"))
            .map(|part| part.trim_start_matches("\r\n"))
            .collect();
        assert_eq!(
            parts,
            [
                "",
                &field("name", "Adam"),
                &field("description", "my voice"),
                &file("a.wav", "audio/wav", "ONE"),
                &file("sample-2.mp3", "audio/mpeg", "TWO"),
                "--\r\n",
            ]
        );

        let http = MockHttp::new(vec![]);
        let err = provider(&http)
            .create_voice("Adam", VoiceSource::samples(vec![], None), None)
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRequest(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn list_voices_follows_the_token() {
        let http = MockHttp::new(vec![
            json_response(
                200,
                json!({
                    "voices": [{"voice_id": "v1", "name": "A", "category": "cloned", "description": "d"}],
                    "has_more": true, "next_page_token": "a+b"
                }),
            ),
            json_response(
                200,
                json!({
                    "voices": [{"voice_id": "v2", "name": "B", "category": "premade"}],
                    "has_more": false
                }),
            ),
        ]);
        let voices = provider(&http).list_voices().await.unwrap();
        assert_eq!(
            http.request(0).url,
            "https://api.test/v2/voices?page_size=100"
        );
        assert_eq!(
            http.request(1).url,
            "https://api.test/v2/voices?page_size=100&next_page_token=a%2Bb"
        );
        assert_eq!(
            voices,
            [
                VoiceInfo::new("v1", "A")
                    .with_description("d")
                    .with_cloned(true),
                VoiceInfo::new("v2", "B"),
            ]
        );
    }

    #[tokio::test]
    async fn models_keep_only_text_to_speech() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!([
                {"model_id": "eleven_v4", "name": "V4", "can_do_text_to_speech": true},
                {"model_id": "scribe", "can_do_text_to_speech": false}
            ]),
        )]);
        let models = provider(&http).list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model, "elevenlabs::eleven_v4");
        assert_eq!(models[0].id, "eleven_v4");
        assert_eq!(http.request(0).url, "https://api.test/v1/models");
    }

    #[tokio::test]
    async fn list_voices_stops_on_a_repeated_token() {
        let page = || {
            json_response(
                200,
                json!({
                    "voices": [{"voice_id": "v1", "name": "A"}],
                    "has_more": true, "next_page_token": "same"
                }),
            )
        };
        let http = MockHttp::new(vec![page(), page(), page(), page()]);
        let voices = provider(&http).list_voices().await.unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(http.request_count(), 2);
    }

    #[tokio::test]
    async fn design_voice_returns_the_previews() {
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"previews": [{"generated_voice_id": "g1",
                "audio_base_64": agentai_core::base64::encode(b"MP3"),
                "media_type": "audio/mpeg", "duration_secs": 2.0, "language": "en"}],
                "text": "Hello there"}),
        )]);
        let previews = SpeechGenerator::from_provider(designer(&http))
            .design_voice("warm low voice")
            .with_text("Hello there")
            .with_count(3)
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/text-to-voice/design");
        assert_eq!(req.header("xi-api-key"), Some("key"));
        assert_eq!(
            body_json(&req),
            json!({"voice_description": "warm low voice", "model_id": "eleven_ttv_v3",
                   "text": "Hello there"})
        );
        assert_eq!(
            previews,
            [VoicePreview::new(Media::from_bytes("audio/mpeg", b"MP3"))
                .with_text("Hello there")
                .with_sample_id("g1")]
        );
    }

    #[tokio::test]
    async fn design_voice_without_text_auto_generates_it() {
        let http = MockHttp::new(vec![json_response(200, json!({"previews": []}))]);
        SpeechGenerator::from_provider(designer(&http))
            .design_voice("warm")
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0)),
            json!({"voice_description": "warm", "model_id": "eleven_ttv_v3",
                   "auto_generate_text": true})
        );
    }

    #[tokio::test]
    async fn create_voice_saves_a_designed_preview() {
        let http = MockHttp::new(vec![
            json_response(200, json!({"voice_id": "v9"})),
            json_response(200, json!({"voice_id": "v10"})),
        ]);
        let id = provider(&http)
            .create_voice(
                "Narrator",
                VoiceSource::SampleId("g1".into()),
                Some("warm low voice"),
            )
            .await
            .unwrap();
        assert_eq!(id, "v9");
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/text-to-voice");
        assert_eq!(req.header("xi-api-key"), Some("key"));
        assert_eq!(
            body_json(&req),
            json!({"voice_name": "Narrator", "voice_description": "warm low voice",
                   "generated_voice_id": "g1"})
        );
        provider(&http)
            .create_voice("Narrator", VoiceSource::SampleId("g1".into()), None)
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(1)),
            json!({"voice_name": "Narrator", "generated_voice_id": "g1"})
        );
    }
}
