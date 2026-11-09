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
//! or the request fails with `MissingVoice`. oMLX uses its own voice when none is set.
//!
//! ## Cloning voices
//!
//! [`SpeechGenerator::create_voice`] clones a voice from samples and returns an ID for
//! [`Voice::id`]. It works with providers that can clone
//! ([`supports_voice_cloning`](SpeechGenerator::supports_voice_cloning)): ElevenLabs and oMLX.
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
//! [`elevenlabs`](provider::elevenlabs) and [`omlx`](provider::omlx).
//! [`SpeechGenerator::new`] picks one from the model name (`openai::gpt-4o-mini-tts`,
//! `elevenlabs::eleven_multilingual_v2`, `omlx::mlx-community/VoxCPM2-8bit`) with its settings
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
pub use agentai_core::storage;
pub use generator::SpeechGenerator;
pub use provider::{
    AudioError, AudioFormat, AudioModel, SpeechRequest, SpeechResponse, Voice, VoiceInfo,
};
#[cfg(feature = "toolbox")]
pub use toolbox::{SpeechToolBox, SpeechToolOutput};
pub use voice_store::{MemoryVoiceStore, StoredVoice, VoiceStore};
