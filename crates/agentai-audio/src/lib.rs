//! # AgentAI Audio
//!
//! Text-to-speech and voice cloning for [AgentAI](https://crates.io/crates/agentai): from your
//! code with [`SpeechGenerator`], or by an agent with [`SpeechToolBox`] (feature `toolbox`).
//!
//! ## Speaking texts
//!
//! ```rust,no_run
//! use agentai_audio::{SpeechGenerator, Voice};
//!
//! # async fn run() -> Result<(), agentai_audio::AudioError> {
//! // The provider comes from the model name, the API key from `OPENAI_API_KEY`
//! let generator = SpeechGenerator::new("openai::gpt-4o-mini-tts")?
//!     .with_default_voice(Voice::id("alloy"));
//! let response = generator.generate("Good morning").await?;
//! // response.audio is an agentai `Media`, base64 data or a URL
//! # Ok(())
//! # }
//! ```
//!
//! With [`with_storage`](SpeechGenerator::with_storage) every audio is saved in a
//! [`MediaSave`](storage::MediaSave) and gets an ID. The [`storage`] module (from
//! `agentai-core`) has the storage traits and [`MemoryStorage`](storage::MemoryStorage).
//!
//! OpenAI and ElevenLabs need a voice: set a default one with
//! [`with_default_voice`](SpeechGenerator::with_default_voice) or send `Voice` with each request,
//! or the request fails with `MissingVoice`. oMLX and Fish Audio use their own voice when none
//! is set.
//!
//! ## Several voices
//!
//! [`SpeechRequest::with_voices`] sets several voices, in the order of the speakers, for models
//! that speak a dialogue, such as Fish Audio `drama-3`. Who speaks when is marked in the text in
//! the provider's syntax (Fish Audio: `<|speaker:0|>`, `<|speaker:1|>`, ...); writing it is up to
//! you. Providers that speak with one voice return `Unsupported` for more than one.
//!
//! [`SpeechRequest::with_timestamps`] asks for the time of every word, returned in
//! [`SpeechResponse::timestamps`] as [`WordTimestamp`]s (Fish Audio; other providers return
//! `Unsupported`).
//!
//! ```rust,no_run
//! use agentai_audio::provider::fishaudio::FishAudioProvider;
//! use agentai_audio::{MemoryVoiceStore, SpeechGenerator, SpeechRequest, Voice};
//!
//! # async fn run() -> Result<(), agentai_audio::AudioError> {
//! // Settings from `FISH_API_KEY` and `FISH_BASE_URL`. IDs found in the voice store are sent
//! // as their samples, other IDs are voice models of the Fish Audio account.
//! let provider = FishAudioProvider::from_env("drama-3")?.with_voice_store(MemoryVoiceStore::new());
//! let generator = SpeechGenerator::from_provider(provider);
//! let response = generator
//!     .generate_with(
//!         SpeechRequest::new("<|speaker:0|>Ready?<|speaker:1|>Always.")
//!             .with_voices(vec![Voice::id("fish-model-id"), Voice::id("voice-1")]),
//!     )
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Cloning voices
//!
//! [`SpeechGenerator::create_voice`] clones a voice from samples and returns an ID for
//! [`Voice::id`]. It works with providers that can clone
//! ([`supports_voice_cloning`](SpeechGenerator::supports_voice_cloning)): ElevenLabs, Fish Audio
//! (a voice model in the account) and oMLX.
//! oMLX has no voice library, it clones from a sample sent with every request, so you give the
//! provider a [`VoiceStore`] explicitly (`SpeechGenerator::new` builds it without one):
//!
//! ```rust,no_run
//! use agentai_audio::provider::omlx::OmlxProvider;
//! use agentai_audio::{MemoryVoiceStore, SpeechGenerator, SpeechRequest, Voice};
//! use agentai::Media;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Settings from `OMLX_ENDPOINT` and `OMLX_API_KEY`
//! let provider = OmlxProvider::from_env("mlx-community/VoxCPM2-8bit")?
//!     .with_voice_store(MemoryVoiceStore::new());
//! let generator = SpeechGenerator::from_provider(provider);
//! let sample = Media::from_bytes("audio/wav", std::fs::read("sample.wav")?);
//! let voice = generator
//!     .create_voice("me", &[sample], Some("What is said in the sample."), None)
//!     .await?;
//! let response = generator
//!     .generate_with(SpeechRequest::new("Hello").with_voice(Voice::id(voice)))
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Tools for agents
//!
//! [`SpeechToolBox`] gives the agent two tools:
//!
//! - `generate_speech`: speaks a text (with a required file `name`, an optional `voice` and
//!   `instructions`) and saves it in the storage.
//! - `list_voices`: lists the voices the model can choose from.
//!
//! By default the model gets only the metadata of the audio as JSON ([`SpeechToolOutput::Metadata`]),
//! because most models cannot listen; [`SpeechToolOutput::Media`] returns the audio itself.
//! [`with_voices`](SpeechToolBox::with_voices) limits the voices to an allowlist. No tool clones
//! voices: `create_voice` stays in your code.
//!
//! ```rust,no_run
//! # #[cfg(feature = "toolbox")]
//! # async fn run() -> anyhow::Result<()> {
//! use agentai::Agent;
//! use agentai::tool::ToolBoxSet;
//! use agentai_audio::storage::MemoryStorage;
//! use agentai_audio::{SpeechGenerator, SpeechToolBox, Voice};
//!
//! let generator = SpeechGenerator::new("openai::gpt-4o-mini-tts")?
//!     .with_default_voice(Voice::id("alloy"));
//! let mut tools = ToolBoxSet::new();
//! tools.add_tool(SpeechToolBox::new(generator, MemoryStorage::new()));
//! let agent = Agent::new("claude-haiku-5-5", "You are a voice assistant.")?.with_toolbox(tools);
//! let result = agent.run::<String>("Say good morning and name the file greeting").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Providers
//!
//! The speech APIs live in [`provider`]: [`openai`](provider::openai) (OpenAI and compatible
//! servers; OpenAI shuts its speech models down on 2027-01-06),
//! [`elevenlabs`](provider::elevenlabs), [`fishaudio`](provider::fishaudio) and
//! [`omlx`](provider::omlx). [`SpeechGenerator::new`] picks one from the model name
//! (`openai::gpt-4o-mini-tts`, `elevenlabs::eleven_multilingual_v2`, `fishaudio::drama-3`,
//! `omlx::mlx-community/VoxCPM2-8bit`) with its settings
//! from environment variables. [`provider::list_models`] lists the speech models of a provider.
//! To set the API key or base URL in code, build the provider and pass it to
//! [`SpeechGenerator::from_provider`]. Implement [`provider::SpeechProvider`] to use another API.
//!
//! ## Features
#![doc = document_features::document_features!()]

mod generator;
pub mod provider;
#[cfg(feature = "toolbox")]
mod toolbox;
mod voice_store;

#[doc(inline)]
pub use agentai_core::{base64, mime, storage};
pub use generator::SpeechGenerator;
pub use provider::{
    AudioError, AudioFormat, AudioModel, SpeechRequest, SpeechResponse, Voice, VoiceInfo,
    WordTimestamp,
};
#[cfg(feature = "toolbox")]
pub use toolbox::{SpeechToolBox, SpeechToolOutput};
pub use voice_store::{MemoryVoiceStore, StoredVoice, VoiceStore};
