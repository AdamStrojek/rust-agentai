//! # Speech providers
//!
//! [`SpeechProvider`] is the interface between your code and a text-to-speech API. Each
//! submodule implements it for one API.

pub mod elevenlabs;
pub mod fishaudio;
pub mod omlx;
pub mod openai;

use agentai_core::message::Media;
use agentai_core::storage::StorageError;
use agentai_core::{HttpError, HttpResponse};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

/// The voice that speaks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Voice {
    /// The ID of a preset or cloned voice, as the provider names it.
    Id(String),
}

impl Voice {
    /// Creates a voice from its ID.
    pub fn id(id: impl Into<String>) -> Self {
        Self::Id(id.into())
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

/// A request to speak a text.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeechRequest {
    /// The text to speak.
    pub text: String,
    /// The voices, in the order of the speakers. A provider that speaks with one voice only
    /// returns [`AudioError::Unsupported`] for more than one. With several voices the text marks
    /// who speaks in the provider's syntax, for example `<|speaker:0|>` for Fish Audio. Falls
    /// back to `default_voice` when empty.
    pub voices: Vec<Voice>,
    /// Voice used when `voices` is empty.
    pub default_voice: Option<Voice>,
    /// Output format. Not sent when `None`.
    pub format: Option<AudioFormat>,
    /// Speaking speed. Not sent when `None`.
    pub speed: Option<f64>,
    /// Language of the text. Not sent when `None`; OpenAI has no equivalent and ignores it.
    pub language: Option<String>,
    /// Style instructions. Not sent when `None`; ElevenLabs has no equivalent and ignores them.
    pub instructions: Option<String>,
    /// File name of the audio, set as [`Media::name`]. Never sent to the provider.
    pub name: Option<String>,
    /// Asks for the time of every word in [`SpeechResponse::timestamps`]. A provider that
    /// cannot give them returns [`AudioError::Unsupported`].
    pub timestamps: bool,
}

impl SpeechRequest {
    /// Creates a request with the text to speak.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voices: Vec::new(),
            default_voice: None,
            format: None,
            speed: None,
            language: None,
            instructions: None,
            name: None,
            timestamps: false,
        }
    }

    /// Adds a voice after the voices already set.
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voices.push(voice);
        self
    }

    /// Sets the voices, in the order of the speakers.
    pub fn with_voices(mut self, voices: Vec<Voice>) -> Self {
        self.voices = voices;
        self
    }

    /// Sets the voice used when there is no `voice`.
    pub fn with_default_voice(mut self, voice: Voice) -> Self {
        self.default_voice = Some(voice);
        self
    }

    /// Sets the output format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the speaking speed.
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Sets the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the style instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Sets the file name of the audio.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets whether to ask for the time of every word.
    pub fn with_timestamps(mut self, timestamps: bool) -> Self {
        self.timestamps = timestamps;
        self
    }

    /// Fails for a request with timestamps, for providers that cannot give them.
    pub(crate) fn without_timestamps(&self) -> Result<(), AudioError> {
        if self.timestamps {
            Err(AudioError::Unsupported("word timestamps"))
        } else {
            Ok(())
        }
    }

    /// The voice of the request, for providers that speak with one voice only.
    pub(crate) fn request_voice(&self) -> Result<Option<&Voice>, AudioError> {
        match self.voices.as_slice() {
            [] => Ok(None),
            [voice] => Ok(Some(voice)),
            _ => Err(AudioError::Unsupported("more than one voice")),
        }
    }

    /// The voice of the request, else the default voice, for providers that speak with one
    /// voice only.
    pub(crate) fn single_voice(&self) -> Result<Option<&Voice>, AudioError> {
        Ok(self.request_voice()?.or(self.default_voice.as_ref()))
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
}

impl SpeechResponse {
    /// Creates a response with the audio.
    pub fn new(audio: Media) -> Self {
        Self {
            audio,
            id: None,
            sample_rate: None,
            timestamps: None,
        }
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
    /// ID for [`Voice::Id`].
    pub id: String,
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
    #[error("no voice: set a voice or a default voice")]
    MissingVoice,
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
/// Implement [`generate`](SpeechProvider::generate); a provider that can clone voices also
/// overrides [`supports_voice_cloning`](SpeechProvider::supports_voice_cloning) and
/// [`create_voice`](SpeechProvider::create_voice).
#[async_trait]
pub trait SpeechProvider: Send + Sync + 'static {
    /// Name of the model, if the provider has one.
    fn model(&self) -> Option<&str> {
        None
    }

    /// Speaks a text.
    async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse, AudioError>;

    /// Returns `true` when the provider implements [`create_voice`](SpeechProvider::create_voice).
    fn supports_voice_cloning(&self) -> bool {
        false
    }

    /// Clones a voice from recordings (`samples`) and returns the ID to use with [`Voice::id`].
    /// `transcript` is what is said in the sample; oMLX rejects a cloned voice without it when
    /// the voice speaks. The default implementation returns [`AudioError::Unsupported`].
    async fn create_voice(
        &self,
        name: &str,
        samples: &[Media],
        transcript: Option<&str>,
        description: Option<&str>,
    ) -> Result<String, AudioError> {
        let _ = (name, samples, transcript, description);
        Err(AudioError::Unsupported("voice cloning"))
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
/// An `omlx::` provider made here has no voice store, so it cannot clone voices or use
/// [`Voice::Id`] of cloned voices; a `fishaudio::` one uses only voices of the Fish Audio
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
    async fn simple_providers_reject_several_voices_and_timestamps() {
        use agentai_core::mock::MockHttp;
        use std::sync::Arc;
        let http = MockHttp::new(vec![]);
        let providers: Vec<Box<dyn SpeechProvider>> = vec![
            Box::new(
                openai::OpenAiSpeechProvider::new("tts-1", "k", "http://x")
                    .with_client(Arc::new(http.clone())),
            ),
            Box::new(
                elevenlabs::ElevenLabsProvider::new("m", "k", "http://x")
                    .with_client(Arc::new(http.clone())),
            ),
            Box::new(
                omlx::OmlxProvider::new("m", "", "http://x").with_client(Arc::new(http.clone())),
            ),
        ];
        let several = SpeechRequest::new("hi")
            .with_voice(Voice::id("a"))
            .with_voice(Voice::id("b"));
        let timestamps = SpeechRequest::new("hi")
            .with_voice(Voice::id("a"))
            .with_timestamps(true);
        for provider in providers {
            for request in [several.clone(), timestamps.clone()] {
                let err = provider.generate(request).await.unwrap_err();
                assert!(matches!(err, AudioError::Unsupported(_)), "{err}");
            }
        }
        assert_eq!(http.request_count(), 0);
    }
}
