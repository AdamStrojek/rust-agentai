//! [`SpeechGenerator`]: speak texts and clone voices from your code.

use crate::provider::{
    self, AudioError, AudioFormat, AudioModel, CreateVoice, SpeechProvider, SpeechRequest,
    SpeechResponse, Voice, VoiceInfo, extension_for,
};
use agentai_core::storage::MediaSave;
use std::sync::Arc;

/// File extensions that count as audio extensions of a name.
const AUDIO_EXTENSIONS: [&str; 7] = ["mp3", "wav", "opus", "ogg", "flac", "aac", "pcm"];

/// Speaks texts and clones voices with a [`SpeechProvider`].
///
/// It is cheap to clone: clones share the provider and the storage.
///
/// ```rust,no_run
/// # use agentai_audio::{SpeechGenerator, Voice};
/// # async fn run() -> Result<(), agentai_audio::AudioError> {
/// let generator = SpeechGenerator::new("openai::gpt-4o-mini-tts")?
///     .with_default_voice(Voice::id("alloy"));
/// let response = generator.generate("Good morning").await?;
/// println!("{} audio", response.audio.content_type);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct SpeechGenerator {
    provider: Arc<dyn SpeechProvider>,
    storage: Option<Arc<dyn MediaSave>>,
    default_voice: Option<Voice>,
    format: Option<AudioFormat>,
    speed: Option<f64>,
    language: Option<String>,
}

impl std::fmt::Debug for SpeechGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechGenerator")
            .field("model", &self.provider.model())
            .field("storage", &self.storage.is_some())
            .field("default_voice", &self.default_voice)
            .field("format", &self.format)
            .field("speed", &self.speed)
            .field("language", &self.language)
            .finish()
    }
}

impl SpeechGenerator {
    /// Creates a generator for a model, see [`provider::from_model`]: the provider comes from
    /// the name (`openai::gpt-4o-mini-tts`) and its settings from environment variables.
    pub fn new(model: &str) -> Result<Self, AudioError> {
        Ok(Self::with_provider_arc(provider::from_model(model)?.into()))
    }

    /// Creates a generator with a provider you built, for example with the API key set in
    /// code.
    pub fn from_provider(provider: impl SpeechProvider) -> Self {
        Self::with_provider_arc(Arc::new(provider))
    }

    fn with_provider_arc(provider: Arc<dyn SpeechProvider>) -> Self {
        Self {
            provider,
            storage: None,
            default_voice: None,
            format: None,
            speed: None,
            language: None,
        }
    }

    /// Saves every audio in `storage` and sets [`SpeechResponse::id`].
    pub fn with_storage(mut self, storage: impl MediaSave) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    /// Voice of every request that sets no `default_voice` of its own.
    pub fn with_default_voice(mut self, voice: Voice) -> Self {
        self.default_voice = Some(voice);
        self
    }

    /// Format of every request that sets none.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Speaking speed of every request that sets none.
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Language of every request that sets none.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Name of the model, if the provider has one.
    pub fn model(&self) -> Option<&str> {
        self.provider.model()
    }

    /// Returns `true` when the provider can clone voices.
    pub fn supports_voice_cloning(&self) -> bool {
        self.provider.supports_voice_cloning()
    }

    /// Returns `true` when the provider accepts [`Voice::Description`].
    pub fn supports_voice_description(&self) -> bool {
        self.provider.supports_voice_description()
    }

    /// Speaks a text.
    pub async fn generate(&self, text: impl Into<String>) -> Result<SpeechResponse, AudioError> {
        self.generate_with(SpeechRequest::new(text)).await
    }

    /// Speaks a text from a full request. Unset fields take the defaults of the generator.
    pub async fn generate_with(
        &self,
        mut request: SpeechRequest,
    ) -> Result<SpeechResponse, AudioError> {
        request.default_voice = request.default_voice.or_else(|| self.default_voice.clone());
        request.format = request.format.or(self.format);
        request.speed = request.speed.or(self.speed);
        request.language = request.language.or_else(|| self.language.clone());
        let name = request.name.take();
        let mut response = self.provider.generate(request).await?;
        let name = name.or_else(|| response.audio.name.take());
        response.audio.name = Some(audio_name(name, &response.audio.content_type));
        if let Some(storage) = &self.storage {
            response.id = Some(storage.save(&response.audio).await?);
        }
        Ok(response)
    }

    /// Clones a voice and returns the ID to use with [`Voice::id`]. Fails with
    /// [`AudioError::Unsupported`] when the provider cannot clone voices.
    pub async fn create_voice(&self, request: CreateVoice) -> Result<String, AudioError> {
        self.provider.create_voice(request).await
    }

    /// Lists the voices of the provider.
    pub async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        self.provider.list_voices().await
    }

    /// Lists the speech models of the provider, see [`provider::list_models`].
    pub async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError> {
        self.provider.list_models().await
    }
}

/// The name of the audio: the requested one (else `speech`), with the extension of the content
/// type unless it already ends in an audio extension.
fn audio_name(name: Option<String>, content_type: &str) -> String {
    let name = name.unwrap_or_else(|| "speech".to_string());
    let has_extension = name
        .rsplit_once('.')
        .is_some_and(|(_, ext)| AUDIO_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()));
    if has_extension {
        name
    } else {
        format!("{name}.{}", extension_for(content_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAiSpeechProvider;
    use crate::provider::test_support::{MockHttp, body_json};
    use crate::provider::{AudioFormat, SpeechRequest, Voice};
    use agentai_core::HttpResponse;
    use agentai_core::storage::{MediaRead, MemoryStorage};
    use serde_json::json;
    use std::sync::Arc;

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn openai(http: &Arc<MockHttp>) -> OpenAiSpeechProvider {
        OpenAiSpeechProvider::new("gpt-4o-mini-tts", "key", "https://api.openai.com/v1")
            .with_client(http.clone())
    }

    #[tokio::test]
    async fn defaults_fill_unset_fields_and_the_request_wins() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let generator = SpeechGenerator::from_provider(openai(&http))
            .with_default_voice(Voice::id("nova"))
            .with_format(AudioFormat::Wav)
            .with_speed(1.5)
            .with_language("pl");

        generator.generate("a").await.unwrap();
        generator
            .generate_with(
                SpeechRequest::new("b")
                    .with_voice(Voice::id("onyx"))
                    .with_format(AudioFormat::Mp3),
            )
            .await
            .unwrap();

        let first = body_json(&http.request(0));
        assert_eq!(first["voice"], json!("nova"));
        assert_eq!(first["response_format"], json!("wav"));
        assert_eq!(first["speed"], json!(1.5));
        let second = body_json(&http.request(1));
        assert_eq!(second["voice"], json!("onyx"));
        assert_eq!(second["response_format"], json!("mp3"));
        assert_eq!(second["speed"], json!(1.5));
    }

    #[test]
    fn the_name_gets_a_real_extension() {
        for (name, content_type, expected) in [
            (Some("intro"), "audio/mpeg", "intro.mp3"),
            (Some("intro.v2"), "audio/mpeg", "intro.v2.mp3"),
            (Some("intro.wav"), "audio/mpeg", "intro.wav"),
            (None, "audio/mpeg", "speech.mp3"),
        ] {
            assert_eq!(audio_name(name.map(String::from), content_type), expected);
        }
    }

    #[tokio::test]
    async fn the_storage_sets_the_id() {
        let http = MockHttp::new(vec![audio()]);
        let storage = MemoryStorage::new();
        let generator = SpeechGenerator::from_provider(openai(&http))
            .with_default_voice(Voice::id("nova"))
            .with_storage(storage.clone());

        let response = generator.generate("hi").await.unwrap();

        assert_eq!(response.id.as_deref(), Some("1-speech.mp3"));
        assert_eq!(storage.read("1-speech.mp3").await.unwrap(), response.audio);
    }

    #[tokio::test]
    async fn an_empty_reply_saves_nothing() {
        let http = MockHttp::new(vec![HttpResponse::new(200, Vec::new())]);
        let storage = MemoryStorage::new();
        let generator = SpeechGenerator::from_provider(openai(&http))
            .with_default_voice(Voice::id("nova"))
            .with_storage(storage.clone());

        let result = generator.generate("hi").await;

        assert!(matches!(result, Err(AudioError::InvalidResponse(_))));
        assert!(storage.ids().is_empty());
    }
}
