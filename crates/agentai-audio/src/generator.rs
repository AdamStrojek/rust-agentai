//! [`SpeechGenerator`]: speak texts and dialogues, clone voices and design voices from your code.

use crate::provider::{
    AudioError, AudioModel, SpeechProvider, SpeechResponse, VoiceId, VoiceInfo, VoicePreview,
    VoiceSource,
};
use crate::request::{DesignVoiceRequest, DialogueRequest, DialogueTurn, SpeechRequest};
use agentai_core::storage::MediaSave;
use std::sync::Arc;

/// Speaks texts and dialogues, clones voices and designs voices with a [`SpeechProvider`].
///
/// Every call is a builder: set its options with `with_*` and `.await` it.
/// It is cheap to clone: clones share the provider and the storage.
#[derive(Clone)]
pub struct SpeechGenerator {
    provider: Arc<dyn SpeechProvider>,
    storage: Option<Arc<dyn MediaSave>>,
}

impl std::fmt::Debug for SpeechGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechGenerator")
            .field("storage", &self.storage.is_some())
            .finish_non_exhaustive()
    }
}

impl SpeechGenerator {
    /// Creates a generator with a shared provider.
    pub fn new(provider: Arc<dyn SpeechProvider>) -> Self {
        Self {
            provider,
            storage: None,
        }
    }

    /// Creates a generator with a provider you built, for example with the API key set in
    /// code.
    pub fn from_provider(provider: impl SpeechProvider) -> Self {
        Self::new(Arc::new(provider))
    }

    /// Saves every audio in `storage` and sets [`SpeechResponse::id`].
    pub fn with_storage(mut self, storage: impl MediaSave) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    /// Returns `true` when the provider can clone voices.
    pub fn supports_voice_cloning(&self) -> bool {
        self.provider.supports_voice_cloning()
    }

    /// Returns `true` when the provider can speak dialogues.
    pub fn supports_dialogue(&self) -> bool {
        self.provider.supports_dialogue()
    }

    /// Returns `true` when the provider can design voices.
    pub fn supports_voice_design(&self) -> bool {
        self.provider.supports_voice_design()
    }

    /// Speaks a text. Set the options on the returned request and `.await` it.
    pub fn generate(&self, text: impl Into<String>) -> SpeechRequest<'_> {
        SpeechRequest::new(self, text.into())
    }

    /// Speaks a dialogue, turn after turn. Set the options on the returned request and
    /// `.await` it. Fails with [`AudioError::Unsupported`] when the provider cannot speak
    /// dialogues, or with [`AudioError::MissingModel`] when its dialogue model is not set.
    pub fn generate_dialogue(
        &self,
        turns: impl IntoIterator<Item = DialogueTurn>,
    ) -> DialogueRequest<'_> {
        DialogueRequest::new(self, turns)
    }

    /// Designs voices from a description and returns previews to choose from. Keep one with
    /// [`create_voice`](Self::create_voice). Fails with [`AudioError::Unsupported`] when the
    /// provider cannot design voices, or with [`AudioError::MissingModel`] when its design
    /// model is not set.
    pub fn design_voice(&self, description: impl Into<String>) -> DesignVoiceRequest<'_> {
        DesignVoiceRequest::new(self, description.into())
    }

    /// Creates a voice and returns its ID for [`SpeechRequest::with_voice`]: from recordings
    /// to clone ([`VoiceSource::samples`]) or by keeping a designed [`VoicePreview`] (passed
    /// as is). Fails with [`AudioError::Unsupported`] when the provider cannot do that.
    pub async fn create_voice(
        &self,
        name: &str,
        source: impl Into<VoiceSource>,
        description: Option<&str>,
    ) -> Result<VoiceId, AudioError> {
        self.provider
            .create_voice(name, source.into(), description)
            .await
    }

    /// Lists the voices of the provider.
    pub async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        self.provider.list_voices().await
    }

    /// Lists the models of the provider (speech, and where the API says so, dialogue and
    /// voice design). ElevenLabs lists only its text-to-speech models, so for design models see
    /// [`SpeechProviderKind::known_models`](crate::SpeechProviderKind::known_models).
    pub async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        self.provider.list_models().await
    }

    pub(crate) async fn speak(
        &self,
        request: &SpeechRequest<'_>,
    ) -> Result<SpeechResponse, AudioError> {
        let response = self.provider.generate(request).await?;
        self.finish(response, request.name(), "speech").await
    }

    pub(crate) async fn speak_dialogue(
        &self,
        request: &DialogueRequest<'_>,
    ) -> Result<SpeechResponse, AudioError> {
        let response = self.provider.generate_dialogue(request).await?;
        self.finish(response, request.name(), "dialogue").await
    }

    pub(crate) async fn design(
        &self,
        request: &DesignVoiceRequest<'_>,
    ) -> Result<Vec<VoicePreview>, AudioError> {
        self.provider.design_voice(request).await
    }

    /// Names the audio and saves it in the storage.
    async fn finish(
        &self,
        mut response: SpeechResponse,
        name: Option<&str>,
        stem: &str,
    ) -> Result<SpeechResponse, AudioError> {
        if let Some(name) = name {
            response.audio.name = Some(name.to_string());
        }
        response.audio.name = Some(response.audio.file_name(stem));
        if let Some(storage) = &self.storage {
            response.id = Some(storage.save(&response.audio).await?);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAiSpeechProvider;
    use agentai_core::HttpResponse;
    use agentai_core::mock::MockHttp;
    use agentai_core::storage::{MediaRead, MemoryStorage};

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn openai(http: &MockHttp) -> OpenAiSpeechProvider {
        OpenAiSpeechProvider::new("gpt-4o-mini-tts", "key", "https://api.openai.com/v1")
            .with_client(Arc::new(http.clone()))
    }

    #[tokio::test]
    async fn a_dialogue_is_unsupported_by_default_and_sends_nothing() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(openai(&http))
            .generate_dialogue([DialogueTurn::new("a", "hi")])
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn the_storage_sets_the_id() {
        let http = MockHttp::new(vec![audio()]);
        let storage = MemoryStorage::new();
        let generator = SpeechGenerator::from_provider(openai(&http)).with_storage(storage.clone());

        let response = generator.generate("hi").with_voice("nova").await.unwrap();

        assert_eq!(response.id.as_deref(), Some("1-speech.mp3"));
        assert_eq!(storage.read("1-speech.mp3").await.unwrap(), response.audio);
    }

    #[tokio::test]
    async fn an_empty_reply_saves_nothing() {
        let http = MockHttp::new(vec![HttpResponse::new(200, Vec::new())]);
        let storage = MemoryStorage::new();
        let generator = SpeechGenerator::from_provider(openai(&http)).with_storage(storage.clone());

        let result = generator.generate("hi").with_voice("nova").await;

        assert!(matches!(result, Err(AudioError::InvalidResponse(_))));
        assert!(storage.ids().is_empty());
    }
}
