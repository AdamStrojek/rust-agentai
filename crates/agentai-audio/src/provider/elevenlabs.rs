//! # ElevenLabs speech API
//!
//! [`ElevenLabsProvider`] speaks text with `POST /v1/text-to-speech/{voice_id}`, clones voices
//! with `POST /v1/voices/add` and lists voices and models.

use super::wire::{audio_media, media_bytes, percent_encode, send};
use super::{
    AudioError, AudioFormat, AudioModel, CreateVoice, SpeechProvider, SpeechRequest,
    SpeechResponse, Voice, VoiceInfo, extension_for,
};
use agentai_core::{HttpClient, HttpRequest, Multipart, ReqwestClient};
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
    model: String,
    api_key: String,
    base_url: String,
    output_format: Option<String>,
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for ElevenLabsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElevenLabsProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl ElevenLabsProvider {
    /// Creates a provider for `model` with the API key and base URL given in code, for example
    /// [`DEFAULT_BASE_URL`].
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            output_format: None,
            client: Arc::new(ReqwestClient::new()),
        }
    }

    /// Creates a provider for `model` with the API key from `ELEVENLABS_API_KEY` and the base
    /// URL from `ELEVENLABS_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(model, |name| std::env::var(name).ok())
    }

    fn from_lookup(
        model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        let api_key = lookup(API_KEY_ENV).ok_or(AudioError::MissingApiKey(API_KEY_ENV))?;
        let base_url = lookup(BASE_URL_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(model, api_key, base_url))
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

    /// The path segment of the voice: the request voice, else the default voice.
    fn voice_id<'a>(&self, request: &'a SpeechRequest) -> Result<&'a str, AudioError> {
        let voice = request
            .voice
            .as_ref()
            .or(request.default_voice.as_ref())
            .ok_or(AudioError::MissingVoice)?;
        match voice {
            Voice::Description(_) => Err(AudioError::Unsupported(
                "voice descriptions with ElevenLabs",
            )),
            Voice::Id(id) | Voice::Auto(id) => {
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
struct VoiceSettings {
    speed: f64,
}

#[async_trait]
impl SpeechProvider for ElevenLabsProvider {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse, AudioError> {
        let voice = self.voice_id(&request)?;
        let output_format = match (&self.output_format, request.format) {
            (Some(over), _) => Some(over.as_str()),
            (None, Some(format)) => Some(output_format_for(format)?),
            (None, None) => None,
        };
        let mut url = format!("{}/v1/text-to-speech/{voice}", self.base_url);
        if let Some(output_format) = output_format {
            url.push_str("?output_format=");
            url.push_str(&percent_encode(output_format));
        }
        let body = SpeechBody {
            text: &request.text,
            model_id: &self.model,
            language_code: request.language.as_deref(),
            voice_settings: request.speed.map(|speed| VoiceSettings { speed }),
        };
        let http = self.authorize(HttpRequest::post(url).with_json(&body)?);
        let response = send(&*self.client, http).await?;
        let content_type = output_format.map_or("audio/mpeg", content_type_of);
        let mut audio = audio_media(&response, content_type)?;
        audio.name = request.name;
        let mut result = SpeechResponse::new(audio);
        if let Some(rate) = output_format.and_then(pcm_sample_rate) {
            result = result.with_sample_rate(rate);
        }
        Ok(result)
    }

    fn supports_voice_cloning(&self) -> bool {
        true
    }

    /// Clones a voice from the samples with `POST /v1/voices/add` and returns its ID.
    ///
    /// When the API replies with `requires_verification: true`, the voice exists but cannot
    /// speak until it is verified at ElevenLabs. `transcript` is not used.
    async fn create_voice(&self, request: CreateVoice) -> Result<String, AudioError> {
        if request.samples.is_empty() {
            return Err(AudioError::InvalidRequest(
                "voice cloning needs at least one sample".to_string(),
            ));
        }
        let mut form = Multipart::new().with_text("name", request.name);
        if let Some(description) = request.description {
            form = form.with_text("description", description);
        }
        for (index, sample) in request.samples.iter().enumerate() {
            let bytes = media_bytes(&*self.client, sample).await?;
            let filename = sample.name.clone().unwrap_or_else(|| {
                format!(
                    "sample-{}.{}",
                    index + 1,
                    extension_for(&sample.content_type)
                )
            });
            form = form.with_file("files", filename, sample.content_type.as_str(), bytes);
        }
        let http = self
            .authorize(HttpRequest::post(format!(
                "{}/v1/voices/add",
                self.base_url
            )))
            .with_multipart(form);
        let response = send(&*self.client, http).await?;
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
            let response = send(&*self.client, self.get(&path)).await?;
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
                Some(next) if page.has_more => token = Some(next),
                _ => return Ok(voices),
            }
        }
    }

    /// Lists the models of `GET /v1/models` that can speak.
    async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        let response = send(&*self.client, self.get("/v1/models")).await?;
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
    use crate::provider::test_support::{MockHttp, body_json, json_response};
    use agentai_core::HttpResponse;
    use agentai_core::message::Media;
    use serde_json::json;

    fn provider(http: &Arc<MockHttp>) -> ElevenLabsProvider {
        ElevenLabsProvider::new("eleven_v4", "key", "https://api.test").with_client(http.clone())
    }

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    #[tokio::test]
    async fn generate_posts_to_the_voice_path() {
        let http = MockHttp::new(vec![audio()]);
        let r = provider(&http)
            .generate(
                SpeechRequest::new("hi")
                    .with_voice(Voice::id("abc123"))
                    .with_language("pl")
                    .with_speed(1.1),
            )
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
            let r = provider(&http)
                .generate(
                    SpeechRequest::new("hi")
                        .with_voice(Voice::id("v"))
                        .with_format(format),
                )
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
            let err = provider(&http)
                .generate(
                    SpeechRequest::new("hi")
                        .with_voice(Voice::id("v"))
                        .with_format(format),
                )
                .await
                .unwrap_err();
            assert!(matches!(err, AudioError::Unsupported(_)));
            assert!(http.requests.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn the_output_format_override_wins_and_sets_the_type() {
        let http = MockHttp::new(vec![audio()]);
        let r = provider(&http)
            .with_output_format("pcm_16000")
            .generate(
                SpeechRequest::new("hi")
                    .with_voice(Voice::id("v"))
                    .with_format(AudioFormat::Mp3),
            )
            .await
            .unwrap();
        assert!(http.request(0).url.ends_with("?output_format=pcm_16000"));
        assert_eq!(r.audio.content_type, "audio/pcm");
        assert_eq!(r.sample_rate, Some(16000));
    }

    #[tokio::test]
    async fn unsafe_voice_ids_are_rejected_before_sending() {
        let http = MockHttp::new(vec![]);
        for id in ["../../v1/voices/x", "a/b", "a b", "..", ""] {
            let err = provider(&http)
                .generate(SpeechRequest::new("hi").with_voice(Voice::auto(id)))
                .await
                .unwrap_err();
            assert!(matches!(err, AudioError::InvalidRequest(_)), "{id}");
        }
        assert!(http.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn descriptions_and_missing_voices() {
        let http = MockHttp::new(vec![]);
        let err = provider(&http)
            .generate(SpeechRequest::new("hi").with_voice(Voice::description("warm")))
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
        let err = provider(&http)
            .generate(SpeechRequest::new("hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::MissingVoice));
        assert!(!provider(&http).supports_voice_description());
        assert!(http.requests.lock().unwrap().is_empty());
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
            .create_voice(CreateVoice::new("Adam", samples).with_description("my voice"))
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
            .create_voice(CreateVoice::new("Adam", vec![]))
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRequest(_)));
        assert!(http.requests.lock().unwrap().is_empty());
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
}
