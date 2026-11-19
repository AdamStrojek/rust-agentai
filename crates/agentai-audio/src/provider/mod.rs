//! # Speech providers
//!
//! [`SpeechProvider`] is the interface between your code and a text-to-speech API. Each
//! submodule implements it for one API.

pub mod elevenlabs;
pub mod fishaudio;
mod kind;
pub mod omlx;
pub mod openai;

pub(crate) use kind::known_capabilities;
pub use kind::{
    IntoSpeechProviderKind, SpeechProviderBuilder, SpeechProviderKind, speech_provider,
};

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
    /// Where the voice comes from.
    pub origin: VoiceOrigin,
}

/// Where a [`VoiceInfo`] comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum VoiceOrigin {
    /// A built-in voice of the provider, such as an OpenAI preset or an ElevenLabs premade voice.
    Preset,
    /// A voice of your account on the provider's server: cloned, designed or professional.
    Account,
    /// A voice of the [`VoiceStore`](crate::VoiceStore) set on the provider.
    Store,
}

impl VoiceInfo {
    /// Creates a voice with its ID, name and origin.
    pub fn new(id: impl Into<String>, name: impl Into<String>, origin: VoiceOrigin) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            origin,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A model of a speech provider and what it can do.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AudioModel {
    /// Name of the model in the API.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Whether the model speaks single texts.
    pub speech: bool,
    /// Whether the model speaks dialogues.
    pub dialogue: bool,
    /// Whether the model designs voices.
    pub voice_design: bool,
}

impl AudioModel {
    /// Creates a speech model with its ID and display name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            speech: true,
            dialogue: false,
            voice_design: false,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets whether the model speaks single texts.
    pub fn with_speech(mut self, speech: bool) -> Self {
        self.speech = speech;
        self
    }

    /// Sets whether the model speaks dialogues.
    pub fn with_dialogue(mut self, dialogue: bool) -> Self {
        self.dialogue = dialogue;
        self
    }

    /// Sets whether the model designs voices.
    pub fn with_voice_design(mut self, voice_design: bool) -> Self {
        self.voice_design = voice_design;
        self
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
    /// No provider has this name.
    #[error("no audio provider named '{0}'")]
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

    /// Lists the models of the API (see [`AudioModel`] for dialogue and voice design). ElevenLabs
    /// lists only its text-to-speech models, so for design models see
    /// [`SpeechProviderKind::known_models`](crate::SpeechProviderKind::known_models). The default
    /// implementation returns [`AudioError::Unsupported`].
    async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        Err(AudioError::Unsupported("listing models"))
    }
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
