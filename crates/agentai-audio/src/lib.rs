//! # AgentAI Audio
//!
//! Text-to-speech, dialogues, voice cloning and voice design for
//! [AgentAI](https://crates.io/crates/agentai): from your code with [`SpeechGenerator`], or by an
//! agent with [`SpeechToolBox`] (feature `toolbox`).
//!
//! ## Speaking texts
//!
//! [`speech_provider`] builds a provider by name (`openai`, `elevenlabs`, `fishaudio`, `omlx`) or
//! [`SpeechProviderKind`] and its speech model; the API key and URL come from the provider's
//! environment variables unless you set them on the builder. Every call of the generator is a
//! builder: set its options with `with_*` and `.await` it.
//!
//! ```rust,no_run
//! use agentai_audio::{SpeechGenerator, speech_provider};
//!
//! # async fn run() -> Result<(), agentai_audio::AudioError> {
//! // The API key from `OPENAI_API_KEY`
//! let generator = SpeechGenerator::new(speech_provider("openai", "gpt-4o-mini-tts").build()?);
//! let response = generator
//!     .generate("Good morning")
//!     .with_voice("alloy")
//!     .with_instructions("Cheerful and fast")
//!     .await?;
//! // response.audio is an agentai `Media`, base64 data or a URL
//! # Ok(())
//! # }
//! ```
//!
//! A voice is a [`VoiceId`]: a preset, a voice of your account or a voice you cloned or designed.
//! OpenAI and ElevenLabs need one, or the request fails with [`AudioError::MissingVoice`]; Fish
//! Audio and oMLX use their own voice when none is set.
//!
//! With [`with_storage`](SpeechGenerator::with_storage) every audio is saved in a
//! [`MediaSave`](storage::MediaSave) and gets an ID. The [`storage`] module (from
//! `agentai-core`) has the storage traits and [`MemoryStorage`](storage::MemoryStorage).
//!
//! [`SpeechRequest::with_timestamps`] asks for the time of every word, returned in
//! [`SpeechResponse::timestamps`] as [`WordTimestamp`]s (Fish Audio; other providers return
//! [`AudioError::Unsupported`]).
//!
//! ## Engines and models
//!
//! A provider has up to three engines, each with its own model: speech (always), dialogue
//! ([`with_dialogue_model`](SpeechProviderBuilder::with_dialogue_model)) and voice design
//! ([`with_design_model`](SpeechProviderBuilder::with_design_model)). An engine without a model is
//! off: calling it returns [`AudioError::MissingModel`] without sending anything, and
//! [`supports_dialogue`](SpeechGenerator::supports_dialogue) and
//! [`supports_voice_design`](SpeechGenerator::supports_voice_design) tell whether it is on.
//! `with_model` on a call overrides the engine's model for that call only. The builder fails with
//! [`AudioError::Unsupported`] when the provider has no such engine.
//!
//! | Provider | Dialogue | Voice design | Voice store |
//! |---|---|---|---|
//! | Fish Audio | yes | yes | yes |
//! | ElevenLabs | yes | yes | no |
//! | oMLX | no | yes | yes |
//! | OpenAI | no | no | no |
//!
//! [`SpeechProviderKind::known_models`] lists the models this crate knows and what each can do
//! ([`AudioModel`]: `speech`, `dialogue`, `voice_design`), for example to fill a model picker;
//! [`SpeechGenerator::list_models`] asks the API, where it can.
//!
//! ```rust,no_run
//! use agentai_audio::SpeechProviderKind;
//!
//! # fn run() -> Result<(), agentai_audio::AudioError> {
//! for model in SpeechProviderKind::FishAudio.known_models() {
//!     if model.dialogue {
//!         println!("{} ({})", model.id, model.name);
//!     }
//! }
//! let provider = SpeechProviderKind::FishAudio
//!     .builder("s2-pro")
//!     .with_dialogue_model("drama-3")
//!     .with_design_model("voice-design-1")
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Dialogues
//!
//! [`SpeechGenerator::generate_dialogue`] speaks a list of [`DialogueTurn`]s, each a voice and
//! its plain text, as one audio, with the dialogue model. Fish Audio (`drama-3`, `s2-pro`, ...)
//! and ElevenLabs (`eleven_v3`, `eleven_v4`) speak dialogues; the provider marks the speakers
//! itself.
//!
//! ```rust,no_run
//! use agentai_audio::{DialogueTurn, SpeechGenerator, speech_provider};
//!
//! # async fn run() -> Result<(), agentai_audio::AudioError> {
//! // The API key from `ELEVENLABS_API_KEY`
//! let provider = speech_provider("elevenlabs", "eleven_multilingual_v2")
//!     .with_dialogue_model("eleven_v3")
//!     .build()?;
//! let generator = SpeechGenerator::new(provider);
//! let response = generator
//!     .generate_dialogue([
//!         DialogueTurn::new("voice-a", "Ready?"),
//!         DialogueTurn::new("voice-b", "Always."),
//!     ])
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Fish Audio takes a voice either as a voice model of the account or as a sample. With a
//! [`VoiceStore`] set, the IDs found in it are sent as their samples: a request whose voices
//! are all in the store sends samples, one whose voices are none in it sends IDs, and a mix
//! fails with [`AudioError::InvalidRequest`] before anything is sent.
//!
//! ## Cloning voices
//!
//! [`SpeechGenerator::create_voice`] clones a voice from samples and returns its [`VoiceId`].
//! It works with providers that can clone
//! ([`supports_voice_cloning`](SpeechGenerator::supports_voice_cloning)): ElevenLabs, Fish Audio
//! (a voice model in the account) and oMLX. oMLX has no voice library: it clones from a sample
//! sent with every request, so it needs a [`VoiceStore`] that keeps the samples:
//!
//! ```rust,no_run
//! use agentai::Media;
//! use agentai_audio::{MemoryVoiceStore, SpeechGenerator, VoiceSource, speech_provider};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Settings from `OMLX_ENDPOINT` and `OMLX_API_KEY`
//! let provider = speech_provider("omlx", "mlx-community/VoxCPM2-8bit")
//!     .with_voice_store(MemoryVoiceStore::new())
//!     .build()?;
//! let generator = SpeechGenerator::new(provider);
//! let sample = Media::from_bytes("audio/wav", std::fs::read("sample.wav")?);
//! let voice = generator
//!     .create_voice(
//!         "me",
//!         VoiceSource::samples(vec![sample], Some("What is said in the sample.")),
//!         None,
//!     )
//!     .await?;
//! let response = generator.generate("Hello").with_voice(voice).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Designing voices
//!
//! [`SpeechGenerator::design_voice`] makes voices from a description with the design model:
//! Fish Audio (`voice-design-1`), ElevenLabs (`eleven_ttv_v3`) and oMLX (a model that turns
//! `instructions` into a voice). It returns [`VoicePreview`]s to listen to;
//! [`create_voice`](SpeechGenerator::create_voice) keeps one as a voice. Fish Audio returns
//! `with_count` previews and ElevenLabs its own number; oMLX needs the text and a [`VoiceStore`]
//! to keep a preview. ElevenLabs needs the description again to save a preview.
//!
//! ```rust,no_run
//! use agentai_audio::{SpeechGenerator, speech_provider};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = speech_provider("fishaudio", "s2-pro")
//!     .with_design_model("voice-design-1")
//!     .build()?;
//! let generator = SpeechGenerator::new(provider);
//! let description = "A warm, calm woman with a low voice";
//! let previews = generator
//!     .design_voice(description)
//!     .with_text("Welcome back.")
//!     .with_count(2)
//!     .await?;
//! // Listen to `previews[n].audio`, then keep the one you like
//! let voice = generator
//!     .create_voice("narrator", &previews[0], Some(description))
//!     .await?;
//! let response = generator.generate("Good evening").with_voice(voice).await?;
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
//! [`with_voices`](SpeechToolBox::with_voices) limits the voices to an allowlist. No tool speaks
//! dialogues, clones or designs voices: those stay in your code.
//!
//! ```rust,no_run
//! # #[cfg(feature = "toolbox")]
//! # async fn run() -> anyhow::Result<()> {
//! use agentai::Agent;
//! use agentai::tool::ToolBoxSet;
//! use agentai_audio::storage::MemoryStorage;
//! use agentai_audio::{SpeechGenerator, SpeechToolBox, speech_provider};
//!
//! let generator = SpeechGenerator::new(speech_provider("openai", "gpt-4o-mini-tts").build()?);
//! let mut tools = ToolBoxSet::new();
//! tools.add_tool(SpeechToolBox::new(generator, MemoryStorage::new()));
//! let agent = Agent::new("claude-haiku-5-5", "You are a voice assistant.")?.with_toolbox(tools);
//! let result = agent
//!     .run::<String>("Say good morning with the voice alloy and name the file greeting")
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Providers
//!
//! The speech APIs live in [`provider`]: [`openai`](provider::openai) (OpenAI and compatible
//! servers; OpenAI shuts its speech models down on 2027-01-06),
//! [`elevenlabs`](provider::elevenlabs), [`fishaudio`](provider::fishaudio) and
//! [`omlx`](provider::omlx). Each is public, with `new`, `from_env` and its own `with_*`
//! settings, for when you build it yourself and pass it to
//! [`SpeechGenerator::from_provider`]. Implement [`provider::SpeechProvider`] to use another API.
//!
//! ## Features
#![doc = document_features::document_features!()]

mod generator;
pub mod provider;
mod request;
#[cfg(feature = "toolbox")]
mod toolbox;
mod voice_store;

#[doc(inline)]
pub use agentai_core::{base64, mime, storage};
pub use generator::SpeechGenerator;
pub use provider::{
    AudioError, AudioFormat, AudioModel, IntoSpeechProviderKind, SpeechProviderBuilder,
    SpeechProviderKind, SpeechResponse, VoiceId, VoiceInfo, VoiceOrigin, VoicePreview, VoiceSource,
    WordTimestamp, speech_provider,
};
pub use request::{DesignVoiceRequest, DialogueRequest, DialogueTurn, SpeechRequest};
#[cfg(feature = "toolbox")]
pub use toolbox::{SpeechToolBox, SpeechToolOutput};
pub use voice_store::{MemoryVoiceStore, StoredVoice, VoiceStore};
