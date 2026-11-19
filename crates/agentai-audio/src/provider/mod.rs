//! # Speech providers
//!
//! [`SpeechProvider`] is the interface between your code and a text-to-speech API. Each
//! submodule implements it for one API.

pub mod elevenlabs;
pub mod fishaudio;
pub mod omlx;
pub mod openai;

use crate::request::{DesignVoiceRequest, DialogueRequest, SpeechRequest};
use agentai_core::message::Media;
use agentai_core::storage::StorageError;
use agentai_core::{HttpError, HttpResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The ID of a voice, as the provider names it: a preset, a voice of the account or a voice
/// of a [`VoiceStore`](crate::VoiceStore).
pub type VoiceId = String;

/// What a voice is created from, see [`create_voice`](SpeechProvider::create_voice).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VoiceSource {
    /// Recordings to clone. `transcript` is what is said in them.
    Samples {
        samples: Vec<Media>,
        transcript: Option<String>,
    },
    /// The ID of a designed voice preview, see [`VoicePreview::sample_id`].
    SampleId(String),
}

impl VoiceSource {
    /// Recordings to clone, with the text said in them if known.
    pub fn samples(samples: Vec<Media>, transcript: Option<&str>) -> Self {
        Self::Samples {
            samples,
            transcript: transcript.map(str::to_string),
        }
    }
}

/// A sample of a designed voice: listen to it, and keep it with
/// [`create_voice`](SpeechProvider::create_voice) if you like it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VoicePreview {
    /// The audio of the preview.
    pub audio: Media,
    /// The text spoken in the preview, if the provider reports it.
    pub text: Option<String>,
    /// The ID that saves the preview as a voice, for providers that keep previews.
    pub sample_id: Option<String>,
    /// Sample rate of the audio in Hz, if known.
    pub sample_rate: Option<u32>,
}

impl VoicePreview {
    /// Creates a preview from its audio; the other fields are empty.
    pub fn new(audio: Media) -> Self {
        Self {
            audio,
            text: None,
            sample_id: None,
            sample_rate: None,
        }
    }

    /// Sets the text spoken in the preview.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Sets the ID that saves the preview as a voice.
    pub fn with_sample_id(mut self, id: impl Into<String>) -> Self {
        self.sample_id = Some(id.into());
        self
    }

    /// Sets the sample rate in Hz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }
}

impl From<VoicePreview> for VoiceSource {
    fn from(preview: VoicePreview) -> Self {
        match preview.sample_id {
            Some(id) => Self::SampleId(id),
            None => Self::Samples {
                samples: vec![preview.audio],
                transcript: preview.text,
            },
        }
    }
}

impl From<&VoicePreview> for VoiceSource {
    fn from(preview: &VoicePreview) -> Self {
        preview.clone().into()
    }
}

/// Audio file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioFormat {
    Mp3,
    Wav,
    Opus,
    Flac,
    Aac,
    /// Headerless PCM; see [`SpeechResponse::sample_rate`].
    Pcm,
}

impl AudioFormat {
    /// MIME type of the format.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Opus => "audio/ogg",
            Self::Flac => "audio/flac",
            Self::Aac => "audio/aac",
            Self::Pcm => "audio/pcm",
        }
    }
}

/// The spoken audio.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeechResponse {
    /// The audio.
    pub audio: Media,
    /// ID of the audio in the storage, when it was saved.
    pub id: Option<String>,
    /// Sample rate in Hz, when known. Matters for headerless PCM.
    pub sample_rate: Option<u32>,
    /// The time of every word, when the request asked for it.
    pub timestamps: Option<Vec<WordTimestamp>>,
    /// The model that was sent, if any.
    pub model: Option<String>,
}

impl SpeechResponse {
    /// Creates a response with the audio.
    pub fn new(audio: Media) -> Self {
        Self {
            audio,
            id: None,
            sample_rate: None,
            timestamps: None,
            model: None,
        }
    }

    /// Sets the model that was sent.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the ID in the storage.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Sets the time of every word.
    pub fn with_timestamps(mut self, timestamps: Vec<WordTimestamp>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }
}

/// When a word is spoken, in seconds from the start of the audio.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct WordTimestamp {
    /// The word, as the provider split the text.
    pub text: String,
    /// Start in seconds.
    pub start: f64,
    /// End in seconds.
    pub end: f64,
}

impl WordTimestamp {
    /// Creates a timestamp of a word.
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

/// A voice a provider offers.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct VoiceInfo {
    /// ID for [`SpeechRequest::with_voice`].
    pub id: VoiceId,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Whether the voice was cloned, not a built-in preset.
    pub cloned: bool,
}

impl VoiceInfo {
    /// Creates a voice with its ID and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            cloned: false,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Marks the voice as cloned.
    pub fn with_cloned(mut self, cloned: bool) -> Self {
        self.cloned = cloned;
        self
    }
}

/// A speech model a provider offers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AudioModel {
    /// Model name with the provider prefix, for example `elevenlabs::eleven_multilingual_v2`.
    pub model: String,
    /// Name of the model in the API.
    pub id: String,
    /// Display name.
    pub name: Option<String>,
    /// Description.
    pub description: Option<String>,
}

impl AudioModel {
    /// Creates a model with its full name and its name in the API.
    pub fn new(model: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            id: id.into(),
            name: None,
            description: None,
        }
    }
}

/// Errors of speech providers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AudioError {
    /// The request could not be sent or the response could not be read.
    #[error(transparent)]
    Http(HttpError),
    /// The API answered with an error status.
    #[error("audio API error (status {status}): {message}")]
    Api { status: u16, message: String },
    /// The API answered with something the provider cannot read.
    #[error("invalid audio API response: {0}")]
    InvalidResponse(String),
    /// The provider cannot do this, for example clone voices.
    #[error("not supported by the audio provider: {0}")]
    Unsupported(&'static str),
    /// No provider matches the model name.
    #[error("no audio provider for model '{0}'")]
    UnknownProvider(String),
    /// The environment variable with the API key is not set.
    #[error("the API key is missing, set {0}")]
    MissingApiKey(&'static str),
    /// The audio was made but could not be saved in the storage.
    #[error("cannot save the audio: {0}")]
    Storage(#[from] StorageError),
    /// The request has no voice and the provider needs one.
    #[error("no voice: the provider needs one")]
    MissingVoice,
    /// The provider has no model for this kind of call, so it is off.
    #[error("no model for {0}")]
    MissingModel(&'static str),
    /// The request cannot be fulfilled as given.
    #[error("invalid audio request: {0}")]
    InvalidRequest(String),
}

impl From<HttpError> for AudioError {
    fn from(err: HttpError) -> Self {
        match err {
            HttpError::Status { status, message } => Self::Api { status, message },
            err => Self::Http(err),
        }
    }
}

/// The body of a response as audio of the given type.
fn audio_media(response: &HttpResponse, content_type: &str) -> Result<Media, AudioError> {
    if response.body.is_empty() {
        return Err(AudioError::InvalidResponse(
            "the response has no audio".to_string(),
        ));
    }
    Ok(Media::from_bytes(content_type, &response.body))
}

impl AudioError {
    /// Returns `true` when the API rejected the credentials (status 401 or 403).
    pub fn is_auth(&self) -> bool {
        matches!(
            self,
            Self::Api {
                status: 401 | 403,
                ..
            }
        )
    }
}

/// A text-to-speech API.
///
/// Implement [`generate`](SpeechProvider::generate); a provider that can clone voices, speak
/// dialogues or design voices also overrides the matching `supports_*` method and the call.
/// The model of a call is [`SpeechRequest::model`] when set, else the provider's own model.
#[async_trait]
pub trait SpeechProvider: Send + Sync + 'static {
    /// Speaks a text.
    async fn generate(&self, request: &SpeechRequest<'_>) -> Result<SpeechResponse, AudioError>;

    /// Returns `true` when the provider implements [`create_voice`](SpeechProvider::create_voice).
    fn supports_voice_cloning(&self) -> bool {
        false
    }

    /// Creates a voice and returns its ID for [`SpeechRequest::with_voice`]: from recordings
    /// to clone ([`VoiceSource::Samples`]; `transcript` is what is said in them, and oMLX
    /// rejects a cloned voice without it when the voice speaks) or by keeping a designed
    /// [`VoicePreview`] ([`VoiceSource::SampleId`]). The default implementation returns
    /// [`AudioError::Unsupported`].
    async fn create_voice(
        &self,
        name: &str,
        source: VoiceSource,
        description: Option<&str>,
    ) -> Result<VoiceId, AudioError> {
        let _ = (name, source, description);
        Err(AudioError::Unsupported("voice cloning"))
    }

    /// Returns `true` when the provider can speak dialogues.
    fn supports_dialogue(&self) -> bool {
        false
    }

    /// Speaks a dialogue of several voices. The default implementation returns
    /// [`AudioError::Unsupported`].
    async fn generate_dialogue(
        &self,
        request: &DialogueRequest<'_>,
    ) -> Result<SpeechResponse, AudioError> {
        let _ = request;
        Err(AudioError::Unsupported("dialogue"))
    }

    /// Returns `true` when the provider can design voices.
    fn supports_voice_design(&self) -> bool {
        false
    }

    /// Designs voices from a text description and returns previews to listen to and choose
    /// from. Keep one with [`create_voice`](SpeechProvider::create_voice) (its
    /// [`VoiceSource::SampleId`] when the preview has one). A provider may return its own
    /// number of previews. The default implementation returns [`AudioError::Unsupported`].
    async fn design_voice(
        &self,
        request: &DesignVoiceRequest<'_>,
    ) -> Result<Vec<VoicePreview>, AudioError> {
        let _ = request;
        Err(AudioError::Unsupported("voice design"))
    }

    /// Lists the voices of the API. The default implementation returns
    /// [`AudioError::Unsupported`].
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        Err(AudioError::Unsupported("listing voices"))
    }

    /// Lists the speech models of the API. The default implementation returns
    /// [`AudioError::Unsupported`].
    async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        Err(AudioError::Unsupported("listing models"))
    }
}

/// The APIs [`from_model`] knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    OpenAi,
    ElevenLabs,
    FishAudio,
    Omlx,
}

impl ProviderKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "openai" => Some(Self::OpenAi),
            "elevenlabs" => Some(Self::ElevenLabs),
            "fishaudio" => Some(Self::FishAudio),
            "omlx" => Some(Self::Omlx),
            _ => None,
        }
    }

    fn build(self, model: &str) -> Result<Box<dyn SpeechProvider>, AudioError> {
        Ok(match self {
            Self::OpenAi => Box::new(openai::OpenAiSpeechProvider::from_env(model)?),
            Self::ElevenLabs => Box::new(elevenlabs::ElevenLabsProvider::from_env(model)?),
            Self::FishAudio => Box::new(fishaudio::FishAudioProvider::from_env(model)?),
            Self::Omlx => Box::new(omlx::OmlxProvider::from_env(model)?),
        })
    }
}

/// Splits `provider::model` and finds the provider.
fn resolve(model: &str) -> Result<(ProviderKind, &str), AudioError> {
    let unknown = || AudioError::UnknownProvider(model.to_string());
    match model.split_once("::") {
        Some((provider, name)) => ProviderKind::from_name(provider)
            .map(|kind| (kind, name))
            .ok_or_else(unknown),
        None if model.starts_with("tts-1") || model.starts_with("gpt-4o-mini-tts") => {
            Ok((ProviderKind::OpenAi, model))
        }
        None if model.starts_with("eleven_") => Ok((ProviderKind::ElevenLabs, model)),
        None => Err(unknown()),
    }
}

/// Creates the provider for a model name, with its settings from environment variables.
///
/// The provider comes from a `provider::` prefix (`openai::`, `elevenlabs::`, `fishaudio::`,
/// `omlx::`), for example `omlx::mlx-community/VoxCPM2-8bit` or `fishaudio::drama-3`, or, for
/// well-known names (`tts-1*`, `gpt-4o-mini-tts*`, `eleven_*`), from the name alone. Use the
/// prefix for other models behind an OpenAI-compatible API. See each provider's `from_env` for its variables.
///
/// An `omlx::` provider made here has no voice store, so it cannot clone voices or speak
/// with cloned voices; a `fishaudio::` one uses only voices of the Fish Audio
/// account. Build it with [`OmlxProvider::with_voice_store`](omlx::OmlxProvider::with_voice_store)
/// and pass it to [`SpeechGenerator::from_provider`](crate::SpeechGenerator::from_provider)
/// instead.
pub fn from_model(model: &str) -> Result<Box<dyn SpeechProvider>, AudioError> {
    let (kind, name) = resolve(model)?;
    kind.build(name)
}

/// Lists the speech models of a provider (`openai` or `elevenlabs`; `fishaudio` and `omlx` return
/// [`AudioError::Unsupported`]), with its settings from environment variables as in
/// [`from_model`].
///
/// ```rust,no_run
/// # async fn run() -> Result<(), agentai_audio::AudioError> {
/// for model in agentai_audio::provider::list_models("elevenlabs").await? {
///     println!("{} {:?}", model.model, model.name);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn list_models(provider: &str) -> Result<Vec<AudioModel>, AudioError> {
    let kind = ProviderKind::from_name(provider)
        .ok_or_else(|| AudioError::UnknownProvider(provider.to_string()))?;
    kind.build("")?.list_models().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preview_with_sample_id_is_a_sample_id() {
        let preview = VoicePreview::new(Media::from_bytes("audio/mpeg", b"A"))
            .with_sample_id("g1")
            .with_text("Hi");
        assert_eq!(
            VoiceSource::from(&preview),
            VoiceSource::SampleId("g1".into())
        );
    }

    #[test]
    fn a_preview_without_sample_id_is_samples() {
        let audio = Media::from_bytes("audio/wav", b"A");
        let preview = VoicePreview::new(audio.clone()).with_text("Hi");
        assert_eq!(
            VoiceSource::from(preview),
            VoiceSource::Samples {
                samples: vec![audio],
                transcript: Some("Hi".into())
            }
        );
    }

    #[test]
    fn model_names_select_the_provider() {
        assert_eq!(
            resolve("omlx::mlx-community/VoxCPM2-8bit").unwrap(),
            (ProviderKind::Omlx, "mlx-community/VoxCPM2-8bit")
        );
        assert_eq!(
            resolve("elevenlabs::x").unwrap(),
            (ProviderKind::ElevenLabs, "x")
        );
        assert_eq!(
            resolve("fishaudio::drama-3").unwrap(),
            (ProviderKind::FishAudio, "drama-3")
        );
        assert_eq!(
            resolve("eleven_v4").unwrap(),
            (ProviderKind::ElevenLabs, "eleven_v4")
        );
        assert_eq!(
            resolve("tts-1-hd").unwrap(),
            (ProviderKind::OpenAi, "tts-1-hd")
        );
        assert_eq!(
            resolve("gpt-4o-mini-tts").unwrap(),
            (ProviderKind::OpenAi, "gpt-4o-mini-tts")
        );
        assert!(matches!(
            resolve("gpt-4o"),
            Err(AudioError::UnknownProvider(_))
        ));
        assert!(matches!(
            resolve("nope::x"),
            Err(AudioError::UnknownProvider(_))
        ));
    }

    #[tokio::test]
    async fn simple_providers_reject_timestamps() {
        use agentai_core::mock::MockHttp;
        use std::sync::Arc;
        let http = MockHttp::new(vec![]);
        let providers: Vec<Arc<dyn SpeechProvider>> = vec![
            Arc::new(
                openai::OpenAiSpeechProvider::new("tts-1", "k", "http://x")
                    .with_client(Arc::new(http.clone())),
            ),
            Arc::new(
                elevenlabs::ElevenLabsProvider::new("m", "k", "http://x")
                    .with_client(Arc::new(http.clone())),
            ),
            Arc::new(
                omlx::OmlxProvider::new("m", "", "http://x").with_client(Arc::new(http.clone())),
            ),
        ];
        for provider in providers {
            let err = crate::SpeechGenerator::new(provider)
                .generate("hi")
                .with_voice("a")
                .with_timestamps(true)
                .await
                .unwrap_err();
            assert!(matches!(err, AudioError::Unsupported(_)), "{err}");
        }
        assert_eq!(http.request_count(), 0);
    }
}
