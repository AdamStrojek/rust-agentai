//! # Speech providers
//!
//! [`SpeechProvider`] is the interface between your code and a text-to-speech API. Each
//! submodule implements it for one API.

mod wire;

pub mod elevenlabs;
pub mod omlx;
pub mod openai;

use agentai_core::HttpError;
use agentai_core::message::Media;
use agentai_core::storage::StorageError;
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

/// The voice that speaks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Voice {
    /// The ID of a preset or cloned voice, as the provider names it.
    Id(String),
    /// A description of the voice to synthesize.
    Description(String),
    /// Let the provider decide whether the text is an ID or a description.
    Auto(String),
}

impl Voice {
    /// Creates a voice from its ID.
    pub fn id(id: impl Into<String>) -> Self {
        Self::Id(id.into())
    }

    /// Creates a voice from a description.
    pub fn description(description: impl Into<String>) -> Self {
        Self::Description(description.into())
    }

    /// Creates a voice the provider resolves to an ID or a description.
    pub fn auto(value: impl Into<String>) -> Self {
        Self::Auto(value.into())
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

/// File extension for an audio MIME type, for names of uploads.
pub(crate) fn extension_for(content_type: &str) -> &str {
    match content_type {
        "audio/mpeg" => "mp3",
        "audio/wav" => "wav",
        "audio/ogg" => "opus",
        "audio/flac" => "flac",
        "audio/aac" => "aac",
        "audio/pcm" => "pcm",
        other => other.rsplit('/').next().unwrap_or(other),
    }
}

/// A request to speak a text.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeechRequest {
    /// The text to speak.
    pub text: String,
    /// The voice. Falls back to `default_voice` when `None`.
    pub voice: Option<Voice>,
    /// Voice used when `voice` is `None`, or when `voice` is a description on a provider that
    /// needs a base voice.
    pub default_voice: Option<Voice>,
    /// Output format. Not sent when `None`.
    pub format: Option<AudioFormat>,
    /// Speaking speed. Not sent when `None`.
    pub speed: Option<f64>,
    /// Language of the text. Not sent when `None`.
    pub language: Option<String>,
    /// Style instructions. Not sent when `None`.
    pub instructions: Option<String>,
    /// File name of the audio, set as [`Media::name`]. Never sent to the provider.
    pub name: Option<String>,
}

impl SpeechRequest {
    /// Creates a request with the text to speak.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: None,
            default_voice: None,
            format: None,
            speed: None,
            language: None,
            instructions: None,
            name: None,
        }
    }

    /// Sets the voice.
    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = Some(voice);
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
}

impl SpeechResponse {
    /// Creates a response with the audio.
    pub fn new(audio: Media) -> Self {
        Self {
            audio,
            id: None,
            sample_rate: None,
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
}

/// A request to clone a voice from samples.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CreateVoice {
    /// Name of the new voice.
    pub name: String,
    /// Recordings of the voice.
    pub samples: Vec<Media>,
    /// What is said in the sample, when the provider needs it.
    pub transcript: Option<String>,
    /// Description of the voice.
    pub description: Option<String>,
}

impl CreateVoice {
    /// Creates a request with the name and the samples.
    pub fn new(name: impl Into<String>, samples: Vec<Media>) -> Self {
        Self {
            name: name.into(),
            samples,
            transcript: None,
            description: None,
        }
    }

    /// Sets the transcript of the sample.
    pub fn with_transcript(mut self, transcript: impl Into<String>) -> Self {
        self.transcript = Some(transcript.into());
        self
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
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
    Http(#[from] HttpError),
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

    /// Clones a voice and returns the ID to use with [`Voice::id`]. The default implementation
    /// returns [`AudioError::Unsupported`].
    async fn create_voice(&self, request: CreateVoice) -> Result<String, AudioError> {
        let _ = request;
        Err(AudioError::Unsupported("voice cloning"))
    }

    /// Returns `true` when [`Voice::Description`] works.
    fn supports_voice_description(&self) -> bool {
        false
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
    Omlx,
}

impl ProviderKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "openai" => Some(Self::OpenAi),
            "elevenlabs" => Some(Self::ElevenLabs),
            "omlx" => Some(Self::Omlx),
            _ => None,
        }
    }

    fn build(self, model: &str) -> Result<Box<dyn SpeechProvider>, AudioError> {
        Ok(match self {
            Self::OpenAi => Box::new(openai::OpenAiSpeechProvider::from_env(model)?),
            Self::ElevenLabs => Box::new(elevenlabs::ElevenLabsProvider::from_env(model)?),
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
/// The provider comes from a `provider::` prefix (`openai::`, `elevenlabs::`, `omlx::`), for
/// example `omlx::mlx-community/VoxCPM2-8bit`, or, for well-known names (`tts-1*`,
/// `gpt-4o-mini-tts*`, `eleven_*`), from the name alone. Use the prefix for other models behind
/// an OpenAI-compatible API. See each provider's `from_env` for its variables.
///
/// An `omlx::` provider made here has no voice store, so it cannot clone voices or use
/// [`Voice::Id`] of cloned voices. Build it with [`OmlxProvider::with_voice_store`](omlx::OmlxProvider::with_voice_store)
/// and pass it to [`SpeechGenerator::from_provider`](crate::SpeechGenerator::from_provider)
/// instead.
pub fn from_model(model: &str) -> Result<Box<dyn SpeechProvider>, AudioError> {
    let (kind, name) = resolve(model)?;
    kind.build(name)
}

/// Lists the speech models of a provider (`openai`, `elevenlabs` or `omlx`), with its settings
/// from environment variables as in [`from_model`].
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
#[allow(dead_code)] // used by the provider tests added later
pub(crate) mod test_support {
    use agentai_core::{HttpClient, HttpError, HttpRequest, HttpResponse};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// Records the requests and answers with the prepared responses, in order.
    #[derive(Default)]
    pub(crate) struct MockHttp {
        pub(crate) requests: Mutex<Vec<HttpRequest>>,
        responses: Mutex<VecDeque<HttpResponse>>,
    }

    impl MockHttp {
        pub(crate) fn new(responses: Vec<HttpResponse>) -> Arc<Self> {
            Arc::new(Self {
                requests: Mutex::default(),
                responses: Mutex::new(responses.into()),
            })
        }

        pub(crate) fn request(&self, index: usize) -> HttpRequest {
            self.requests.lock().unwrap()[index].clone()
        }
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.requests.lock().unwrap().push(request);
            Ok(self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("no response left"))
        }
    }

    pub(crate) fn json_response(status: u16, body: Value) -> HttpResponse {
        HttpResponse::new(status, body.to_string())
    }

    pub(crate) fn body_json(request: &HttpRequest) -> Value {
        serde_json::from_slice(request.body.as_ref().unwrap()).unwrap()
    }
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
}
