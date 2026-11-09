//! # AgentAI Audio
//!
//! Text-to-speech and voice cloning for [AgentAI](https://crates.io/crates/agentai).
//!
//! The speech APIs live in [`provider`]; [`SpeechProvider`](provider::SpeechProvider) is the
//! interface they implement. Audio comes back as an agentai `Media`, and the [`storage`] module
//! (from `agentai-core`) has the traits to save it.

mod generator;
pub mod provider;
#[cfg(feature = "toolbox")]
mod toolbox;
mod voice_store;

#[doc(inline)]
pub use agentai_core::storage;
pub use generator::SpeechGenerator;
pub use provider::{
    AudioError, AudioFormat, AudioModel, CreateVoice, SpeechRequest, SpeechResponse, Voice,
    VoiceInfo,
};
#[cfg(feature = "toolbox")]
pub use toolbox::{SpeechToolBox, SpeechToolOutput};
pub use voice_store::{MemoryVoiceStore, StoredVoice, VoiceStore};
