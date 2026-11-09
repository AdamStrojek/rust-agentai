//! Storage for cloned voices, for providers that keep the sample themselves.

use agentai_core::message::Media;
use agentai_core::storage::StorageError;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// A cloned voice: a name, a sample and what is said in it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StoredVoice {
    /// Name of the voice.
    pub name: String,
    /// Description of the voice.
    pub description: Option<String>,
    /// Recording of the voice.
    pub sample: Media,
    /// What is said in the sample.
    pub transcript: Option<String>,
}

impl StoredVoice {
    /// Creates a voice with its name and sample.
    pub fn new(name: impl Into<String>, sample: Media) -> Self {
        Self {
            name: name.into(),
            description: None,
            sample,
            transcript: None,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the transcript of the sample.
    pub fn with_transcript(mut self, transcript: impl Into<String>) -> Self {
        self.transcript = Some(transcript.into());
        self
    }
}

/// Saves cloned voices and chooses their IDs.
#[async_trait]
pub trait VoiceStore: Send + Sync + 'static {
    /// Stores the voice and returns the ID to use with [`Voice::id`](crate::Voice::id).
    async fn save(&self, voice: StoredVoice) -> Result<String, StorageError>;

    /// Returns the voice stored under `id`, or `None`.
    ///
    /// `id` may be arbitrary untrusted text, for example what a model wrote as a voice. An
    /// implementation backed by files or a database must return `Ok(None)` for anything that is
    /// not a valid ID (no path traversal, no query injection).
    async fn get(&self, id: &str) -> Result<Option<StoredVoice>, StorageError>;

    /// Returns every voice with its ID, in the order they were saved.
    async fn list(&self) -> Result<Vec<(String, StoredVoice)>, StorageError>;
}

/// [`VoiceStore`] in memory, with the IDs `voice-1`, `voice-2`, ...
///
/// Clones share the same voices.
#[derive(Debug, Clone, Default)]
pub struct MemoryVoiceStore {
    voices: Arc<Mutex<Vec<(String, StoredVoice)>>>,
}

impl MemoryVoiceStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VoiceStore for MemoryVoiceStore {
    async fn save(&self, voice: StoredVoice) -> Result<String, StorageError> {
        let mut voices = self.voices.lock().unwrap();
        let id = format!("voice-{}", voices.len() + 1);
        voices.push((id.clone(), voice));
        Ok(id)
    }

    async fn get(&self, id: &str) -> Result<Option<StoredVoice>, StorageError> {
        let voices = self.voices.lock().unwrap();
        Ok(voices.iter().find(|(i, _)| i == id).map(|(_, v)| v.clone()))
    }

    async fn list(&self) -> Result<Vec<(String, StoredVoice)>, StorageError> {
        Ok(self.voices.lock().unwrap().clone())
    }
}
