//! Choosing a speech provider by name and building it.

use super::{
    AudioError, AudioModel, SpeechProvider, elevenlabs, elevenlabs::ElevenLabsProvider, fishaudio,
    fishaudio::FishAudioProvider, omlx, omlx::OmlxProvider, openai, openai::OpenAiSpeechProvider,
};
use crate::voice_store::{StoredVoice, VoiceStore};
use agentai_core::storage::StorageError;
use async_trait::async_trait;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

/// The speech APIs this crate implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpeechProviderKind {
    /// OpenAI and compatible servers, name `openai`.
    OpenAi,
    /// ElevenLabs, name `elevenlabs`.
    ElevenLabs,
    /// Fish Audio, name `fishaudio`.
    FishAudio,
    /// oMLX, name `omlx`.
    Omlx,
}

impl fmt::Display for SpeechProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for SpeechProviderKind {
    type Err = AudioError;

    fn from_str(name: &str) -> Result<Self, AudioError> {
        match name {
            "openai" => Ok(Self::OpenAi),
            "elevenlabs" => Ok(Self::ElevenLabs),
            "fishaudio" => Ok(Self::FishAudio),
            "omlx" => Ok(Self::Omlx),
            _ => Err(AudioError::UnknownProvider(name.to_string())),
        }
    }
}

impl SpeechProviderKind {
    fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::ElevenLabs => "elevenlabs",
            Self::FishAudio => "fishaudio",
            Self::Omlx => "omlx",
        }
    }

    /// Starts building a provider of this kind with the given speech model.
    pub fn builder(self, speech_model: impl Into<String>) -> SpeechProviderBuilder {
        speech_provider(self, speech_model)
    }

    /// The models this crate knows of the provider and what they can do. A server may offer
    /// more, see [`SpeechGenerator::list_models`](crate::SpeechGenerator::list_models).
    pub fn known_models(self) -> Vec<AudioModel> {
        let m = |id: &str, name: &str| AudioModel::new(id, name);
        match self {
            Self::FishAudio => vec![
                m("s1", "Fish Audio S1"),
                m("s2-pro", "Fish Audio S2-Pro").with_dialogue(true),
                m("s2.1-pro", "Fish Audio S2.1-Pro").with_dialogue(true),
                m("s2.1-pro-free", "Fish Audio S2.1-Pro Free").with_dialogue(true),
                m("drama-3", "Fish Audio Drama 3").with_dialogue(true),
                m("s3-pro", "Fish Audio S3-Pro"),
                m("voice-design-1", "Fish Audio Voice Design 1")
                    .with_speech(false)
                    .with_voice_design(true),
            ],
            Self::ElevenLabs => vec![
                m("eleven_v4", "Eleven v4").with_dialogue(true),
                m("eleven_v3", "Eleven v3").with_dialogue(true),
                m("eleven_v4_turbo", "Eleven v4 Turbo"),
                m("eleven_v3_conversational", "Eleven v3 Conversational"),
                m("eleven_multilingual_v2", "Eleven Multilingual v2"),
                m("eleven_flash_v2_5", "Eleven Flash v2.5"),
                m("eleven_flash_v2", "Eleven Flash v2"),
                m("eleven_ttv_v3", "Eleven v3 Voice Design")
                    .with_speech(false)
                    .with_voice_design(true),
                m(
                    "eleven_multilingual_ttv_v2",
                    "Eleven Multilingual Voice Design v2",
                )
                .with_speech(false)
                .with_voice_design(true),
            ],
            Self::OpenAi => vec![
                m("gpt-4o-mini-tts", "GPT-4o mini TTS"),
                m("tts-1", "TTS-1"),
                m("tts-1-hd", "TTS-1 HD"),
            ],
            Self::Omlx => vec![],
        }
    }

    fn lacks(self, what: Capability) -> Option<&'static str> {
        match (what, self) {
            (Capability::Dialogue, Self::FishAudio | Self::ElevenLabs) => None,
            (Capability::Dialogue, Self::OpenAi) => Some("dialogue with openai"),
            (Capability::Dialogue, Self::Omlx) => Some("dialogue with omlx"),
            (Capability::Design, Self::FishAudio | Self::ElevenLabs | Self::Omlx) => None,
            (Capability::Design, Self::OpenAi) => Some("voice design with openai"),
            (Capability::Store, Self::FishAudio | Self::Omlx) => None,
            (Capability::Store, Self::OpenAi) => Some("a voice store with openai"),
            (Capability::Store, Self::ElevenLabs) => Some("a voice store with elevenlabs"),
        }
    }
}

#[derive(Clone, Copy)]
enum Capability {
    Dialogue,
    Design,
    Store,
}

/// Copies what the catalogue knows of the model with the same ID (speech, dialogue, voice
/// design) into `model`; an unknown ID stays a speech model.
pub(crate) fn known_capabilities(kind: SpeechProviderKind, model: AudioModel) -> AudioModel {
    match kind.known_models().into_iter().find(|m| m.id == model.id) {
        Some(known) => model
            .with_speech(known.speech)
            .with_dialogue(known.dialogue)
            .with_voice_design(known.voice_design),
        None => model,
    }
}

/// A provider name or [`SpeechProviderKind`], for [`speech_provider`].
pub trait IntoSpeechProviderKind {
    /// Returns the kind, or [`AudioError::UnknownProvider`] for an unknown name.
    fn into_kind(self) -> Result<SpeechProviderKind, AudioError>;
}

impl IntoSpeechProviderKind for SpeechProviderKind {
    fn into_kind(self) -> Result<SpeechProviderKind, AudioError> {
        Ok(self)
    }
}

impl IntoSpeechProviderKind for &str {
    fn into_kind(self) -> Result<SpeechProviderKind, AudioError> {
        self.parse()
    }
}

impl IntoSpeechProviderKind for String {
    fn into_kind(self) -> Result<SpeechProviderKind, AudioError> {
        self.parse()
    }
}

/// Starts building a speech provider by kind or name (`openai`, `elevenlabs`, `fishaudio`,
/// `omlx`) with its speech model. Keys and URLs come from the environment unless set on the
/// builder.
///
/// ```rust,no_run
/// # fn run() -> Result<(), agentai_audio::AudioError> {
/// let provider = agentai_audio::speech_provider("openai", "gpt-4o-mini-tts").build()?;
/// # Ok(())
/// # }
/// ```
pub fn speech_provider(
    provider: impl IntoSpeechProviderKind,
    speech_model: impl Into<String>,
) -> SpeechProviderBuilder {
    SpeechProviderBuilder {
        kind: provider.into_kind(),
        speech_model: speech_model.into(),
        dialogue_model: None,
        design_model: None,
        voice_store: None,
        api_key: None,
        base_url: None,
    }
}

/// Builds a [`SpeechProvider`], see [`speech_provider`].
pub struct SpeechProviderBuilder {
    kind: Result<SpeechProviderKind, AudioError>,
    speech_model: String,
    dialogue_model: Option<String>,
    design_model: Option<String>,
    voice_store: Option<BoxedStore>,
    api_key: Option<String>,
    base_url: Option<String>,
}

/// Lets the builder hold any store and pass it on as `impl VoiceStore`.
struct BoxedStore(Box<dyn VoiceStore>);

#[async_trait]
impl VoiceStore for BoxedStore {
    async fn save(&self, voice: StoredVoice) -> Result<String, StorageError> {
        self.0.save(voice).await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredVoice>, StorageError> {
        self.0.get(id).await
    }

    async fn list(&self) -> Result<Vec<(String, StoredVoice)>, StorageError> {
        self.0.list().await
    }
}

impl SpeechProviderBuilder {
    /// Sets the model of dialogues; without it dialogue is off.
    pub fn with_dialogue_model(mut self, model: impl Into<String>) -> Self {
        self.dialogue_model = Some(model.into());
        self
    }

    /// Sets the model of voice design; without it voice design is off.
    pub fn with_design_model(mut self, model: impl Into<String>) -> Self {
        self.design_model = Some(model.into());
        self
    }

    /// Sets the store of voice samples.
    pub fn with_voice_store(mut self, store: impl VoiceStore) -> Self {
        self.voice_store = Some(BoxedStore(Box::new(store)));
        self
    }

    /// Sets the API key instead of reading it from the environment.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL instead of reading it from the environment.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Builds the provider, reading what is not set from the environment. Fails with
    /// [`AudioError::Unsupported`] when the provider lacks an engine that was set.
    pub fn build(self) -> Result<Arc<dyn SpeechProvider>, AudioError> {
        self.build_with(|name| std::env::var(name).ok())
    }

    pub(crate) fn build_with(
        self,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Arc<dyn SpeechProvider>, AudioError> {
        let kind = self.kind?;
        let checks = [
            (self.dialogue_model.is_some(), Capability::Dialogue),
            (self.design_model.is_some(), Capability::Design),
            (self.voice_store.is_some(), Capability::Store),
        ];
        for (set, what) in checks {
            if let (true, Some(message)) = (set, kind.lacks(what)) {
                return Err(AudioError::Unsupported(message));
            }
        }
        let (key_env, url_env) = match kind {
            SpeechProviderKind::OpenAi => (openai::API_KEY_ENV, openai::BASE_URL_ENV),
            SpeechProviderKind::ElevenLabs => (elevenlabs::API_KEY_ENV, elevenlabs::BASE_URL_ENV),
            SpeechProviderKind::FishAudio => (fishaudio::API_KEY_ENV, fishaudio::BASE_URL_ENV),
            SpeechProviderKind::Omlx => (omlx::API_KEY_ENV, omlx::ENDPOINT_ENV),
        };
        let lookup = |name: &str| {
            if name == key_env && self.api_key.is_some() {
                self.api_key.clone()
            } else if name == url_env && self.base_url.is_some() {
                self.base_url.clone()
            } else {
                env(name)
            }
        };
        let model = self.speech_model.as_str();
        Ok(match kind {
            SpeechProviderKind::OpenAi => {
                Arc::new(OpenAiSpeechProvider::from_lookup(model, lookup)?)
            }
            SpeechProviderKind::ElevenLabs => {
                let mut p = ElevenLabsProvider::from_lookup(model, lookup)?;
                if let Some(m) = self.dialogue_model {
                    p = p.with_dialogue_model(m);
                }
                if let Some(m) = self.design_model {
                    p = p.with_design_model(m);
                }
                Arc::new(p)
            }
            SpeechProviderKind::FishAudio => {
                let mut p = FishAudioProvider::from_lookup(model, lookup)?;
                if let Some(m) = self.dialogue_model {
                    p = p.with_dialogue_model(m);
                }
                if let Some(m) = self.design_model {
                    p = p.with_design_model(m);
                }
                if let Some(store) = self.voice_store {
                    p = p.with_voice_store(store);
                }
                Arc::new(p)
            }
            SpeechProviderKind::Omlx => {
                let mut p = OmlxProvider::from_lookup(model, lookup)?;
                if let Some(m) = self.design_model {
                    p = p.with_design_model(m);
                }
                if let Some(store) = self.voice_store {
                    p = p.with_voice_store(store);
                }
                Arc::new(p)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryVoiceStore;

    #[test]
    fn a_builder_rejects_engines_the_provider_lacks() {
        let env = |_: &str| Some("k".to_string());
        for (builder, expected) in [
            (
                speech_provider("openai", "tts-1").with_dialogue_model("x"),
                "dialogue",
            ),
            (
                speech_provider("omlx", "m").with_dialogue_model("x"),
                "dialogue",
            ),
            (
                speech_provider("openai", "tts-1").with_design_model("x"),
                "voice design",
            ),
            (
                speech_provider("elevenlabs", "m").with_voice_store(MemoryVoiceStore::new()),
                "voice store",
            ),
        ] {
            let err = builder.build_with(env).err().unwrap();
            assert!(
                matches!(err, AudioError::Unsupported(m) if m.contains(expected)),
                "{expected}"
            );
        }
        assert!(matches!(
            speech_provider("nope", "m").build_with(env),
            Err(AudioError::UnknownProvider(n)) if n == "nope"
        ));
    }

    #[test]
    fn builder_keys_come_from_code_or_the_environment() {
        let no_env = |_: &str| None;
        assert!(
            speech_provider("elevenlabs", "m")
                .with_api_key("k")
                .build_with(no_env)
                .is_ok()
        );
        assert!(matches!(
            speech_provider("elevenlabs", "m").build_with(no_env),
            Err(AudioError::MissingApiKey(_))
        ));
        let fish = speech_provider(SpeechProviderKind::FishAudio, "s1")
            .with_base_url("http://localhost:8080")
            .with_dialogue_model("drama-3")
            .build_with(no_env)
            .unwrap();
        assert!(fish.supports_dialogue());
        assert!(!fish.supports_voice_design());
    }

    #[test]
    fn names_round_trip() {
        for kind in [
            SpeechProviderKind::OpenAi,
            SpeechProviderKind::ElevenLabs,
            SpeechProviderKind::Omlx,
            SpeechProviderKind::FishAudio,
        ] {
            assert_eq!(
                kind.to_string().parse::<SpeechProviderKind>().unwrap(),
                kind
            );
        }
    }
}
