//! # oMLX speech API
//!
//! [`OmlxProvider`] speaks text with `POST /audio/speech` of an oMLX
//! server, for example with `mlx-community/VoxCPM2-8bit`. The server has no voice list: a voice
//! is a text description (sent as `instructions`) or a cloned voice. oMLX clones from a sample
//! sent with every request (`ref_audio` and `ref_text`), so the provider keeps the samples in a
//! [`VoiceStore`] set with [`with_voice_store`](OmlxProvider::with_voice_store).
//!
//! The `speed` and `language` fields, and the sampler settings, appear to have no effect on
//! VoxCPM2. They are sent when set, and the server decides.

use super::wire::{audio_media, media_bytes, send, with_bearer};
use super::{
    AudioError, AudioFormat, CreateVoice, SpeechProvider, SpeechRequest, SpeechResponse, Voice,
    VoiceInfo,
};
use crate::voice_store::{StoredVoice, VoiceStore};
use agentai_core::message::{Media, MediaSource};
use agentai_core::{HttpClient, HttpRequest, ReqwestClient};
use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;

/// The default endpoint of a local oMLX server.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";
/// Environment variable with the endpoint.
pub const ENDPOINT_ENV: &str = "OMLX_ENDPOINT";
/// Environment variable with the API key. It is optional.
pub const API_KEY_ENV: &str = "OMLX_API_KEY";

/// Largest base64 sample the provider stores, in encoded bytes.
const MAX_REF_AUDIO_LEN: usize = 20 * 1024 * 1024;
/// `ref_text` when the voice has no transcript. The server requires the field, and accepted
/// this value with `mlx-community/VoxCPM2-8bit`.
const REF_TEXT_PLACEHOLDER: &str = ".";

/// Provider for the speech API of an oMLX server.
///
/// ```rust,no_run
/// # use agentai_audio::provider::omlx::OmlxProvider;
/// # use agentai_audio::MemoryVoiceStore;
/// # fn main() -> Result<(), agentai_audio::AudioError> {
/// let provider = OmlxProvider::from_env("mlx-community/VoxCPM2-8bit")?
///     .with_voice_store(MemoryVoiceStore::new());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OmlxProvider {
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
    store: Option<Arc<dyn VoiceStore>>,
}

impl std::fmt::Debug for OmlxProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmlxProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OmlxProvider {
    /// Creates a provider for `model` with the API key and base URL given in code, for example
    /// [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Arc::new(ReqwestClient::new()),
            store: None,
        }
    }

    /// Creates a provider for `model` with the endpoint from `OMLX_ENDPOINT`, or
    /// [`DEFAULT_BASE_URL`] when it is not set, and the optional API key from `OMLX_API_KEY`.
    ///
    /// No voice store is created, so cloning needs [`with_voice_store`](Self::with_voice_store).
    pub fn from_env(model: impl Into<String>) -> Result<Self, AudioError> {
        Ok(Self::from_lookup(model, |name| std::env::var(name).ok()))
    }

    fn from_lookup(model: impl Into<String>, lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self::new(
            model,
            lookup(API_KEY_ENV).unwrap_or_default(),
            lookup(ENDPOINT_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
        )
    }

    /// Sets the store of cloned voices. It enables
    /// [`create_voice`](SpeechProvider::create_voice) and [`Voice::Id`].
    pub fn with_voice_store(mut self, store: impl VoiceStore) -> Self {
        self.store = Some(Arc::new(store));
        self
    }

    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// The cloned voice and the instructions of a request.
    async fn resolve(
        &self,
        request: &SpeechRequest,
    ) -> Result<(Option<StoredVoice>, Option<String>), AudioError> {
        let mut description = None;
        let mut cloned = None;
        match request.voice.as_ref().or(request.default_voice.as_ref()) {
            None => {}
            Some(Voice::Description(text)) => description = Some(text.clone()),
            Some(Voice::Id(id)) => {
                let store = self
                    .store
                    .as_ref()
                    .ok_or(AudioError::Unsupported("voice IDs without a voice store"))?;
                cloned = Some(store.get(id).await?.ok_or_else(|| {
                    AudioError::InvalidRequest(format!("no cloned voice with id '{id}'"))
                })?);
            }
            Some(Voice::Auto(text)) => {
                if let Some(store) = &self.store {
                    cloned = store.get(text).await?;
                }
                if cloned.is_none() {
                    description = Some(text.clone());
                }
            }
        }
        let instructions = match (description, &request.instructions) {
            (Some(description), Some(extra)) => Some(format!("{description}\n{extra}")),
            (Some(description), None) => Some(description),
            (None, extra) => extra.clone(),
        };
        Ok((cloned, instructions))
    }
}

#[derive(Serialize)]
struct SpeechBody<'a> {
    model: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ref_audio: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ref_text: Option<&'a str>,
}

fn format_name(format: AudioFormat) -> Result<&'static str, AudioError> {
    match format {
        AudioFormat::Mp3 => Ok("mp3"),
        AudioFormat::Wav => Ok("wav"),
        AudioFormat::Opus => Ok("opus"),
        AudioFormat::Flac => Ok("flac"),
        AudioFormat::Pcm => Ok("pcm"),
        _ => Err(AudioError::Unsupported("this format with oMLX")),
    }
}

#[async_trait]
impl SpeechProvider for OmlxProvider {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse, AudioError> {
        let response_format = request.format.map(format_name).transpose()?;
        let (cloned, instructions) = self.resolve(&request).await?;
        let ref_audio = match cloned.as_ref().map(|voice| &voice.sample.source) {
            Some(MediaSource::Base64(data)) => Some(data.as_str()),
            // `create_voice` stores base64, but a custom store may hold anything
            Some(MediaSource::Url(_)) => {
                return Err(AudioError::InvalidRequest(
                    "the stored voice sample is not base64".to_string(),
                ));
            }
            None => None,
        };
        let ref_text = cloned
            .as_ref()
            .map(|voice| voice.transcript.as_deref().unwrap_or(REF_TEXT_PLACEHOLDER));
        let body = SpeechBody {
            model: &self.model,
            input: &request.text,
            response_format,
            speed: request.speed,
            language: request.language.as_deref(),
            instructions,
            ref_audio,
            ref_text,
        };
        let http = with_bearer(
            HttpRequest::post(format!("{}/audio/speech", self.base_url)).with_json(&body)?,
            &self.api_key,
        );
        let response = send(&*self.client, http).await?;
        let format = request.format.unwrap_or(AudioFormat::Wav);
        let mut audio = audio_media(&response, format.content_type())?;
        audio.name = request.name;
        Ok(SpeechResponse::new(audio))
    }

    fn supports_voice_cloning(&self) -> bool {
        self.store.is_some()
    }

    /// Stores the sample in the voice store; nothing is sent to the server until a request
    /// uses the voice. Needs exactly one sample.
    async fn create_voice(&self, request: CreateVoice) -> Result<String, AudioError> {
        let store = self.store.as_ref().ok_or(AudioError::Unsupported(
            "voice cloning without a voice store",
        ))?;
        let [sample] = <[Media; 1]>::try_from(request.samples).map_err(|samples| {
            AudioError::InvalidRequest(format!(
                "oMLX clones from exactly one sample, got {}",
                samples.len()
            ))
        })?;
        let sample = match &sample.source {
            MediaSource::Base64(_) => sample,
            MediaSource::Url(_) => {
                let bytes = media_bytes(&*self.client, &sample).await?;
                Media::from_bytes(sample.content_type, bytes)
            }
        };
        if matches!(&sample.source, MediaSource::Base64(data) if data.len() > MAX_REF_AUDIO_LEN) {
            return Err(AudioError::InvalidRequest(format!(
                "the sample is larger than {} MiB",
                MAX_REF_AUDIO_LEN / 1024 / 1024
            )));
        }
        let mut voice = StoredVoice::new(request.name, sample);
        voice.description = request.description;
        voice.transcript = request.transcript;
        Ok(store.save(voice).await?)
    }

    fn supports_voice_description(&self) -> bool {
        true
    }

    /// The cloned voices of the voice store.
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        let store = self.store.as_ref().ok_or(AudioError::Unsupported(
            "listing voices without a voice store",
        ))?;
        Ok(store
            .list()
            .await?
            .into_iter()
            .map(|(id, voice)| {
                let mut info = VoiceInfo::new(id, voice.name).with_cloned(true);
                info.description = voice.description;
                info
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::{MockHttp, body_json};
    use crate::provider::{
        AudioError, AudioFormat, CreateVoice, SpeechProvider, SpeechRequest, Voice, VoiceInfo,
    };
    use crate::{MemoryVoiceStore, VoiceStore};
    use agentai_core::HttpResponse;
    use agentai_core::message::{Media, MediaSource};
    use serde_json::json;
    use std::sync::Arc;

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn provider(http: &Arc<MockHttp>) -> OmlxProvider {
        OmlxProvider::new("m", "", "http://local/v1").with_client(http.clone())
    }

    fn sample() -> Media {
        Media::from_bytes("audio/wav", b"WAV")
    }

    async fn cloned(http: &Arc<MockHttp>, transcript: Option<&str>) -> OmlxProvider {
        let p = provider(http).with_voice_store(MemoryVoiceStore::new());
        let mut request = CreateVoice::new("Adam", vec![sample()]);
        if let Some(t) = transcript {
            request = request.with_transcript(t);
        }
        assert_eq!(p.create_voice(request).await.unwrap(), "voice-1");
        p
    }

    #[tokio::test]
    async fn generate_without_a_voice_sends_no_voice_field() {
        let http = MockHttp::new(vec![audio()]);
        let r = provider(&http)
            .generate(SpeechRequest::new("hi"))
            .await
            .unwrap();
        assert_eq!(http.request(0).url, "http://local/v1/audio/speech");
        assert_eq!(
            body_json(&http.request(0)),
            json!({"model": "m", "input": "hi"})
        );
        assert_eq!(r.audio.content_type, "audio/wav");
        assert_eq!(r.sample_rate, None);
    }

    #[test]
    fn endpoint_keeps_v1() {
        let p = OmlxProvider::from_lookup("m", |name| {
            (name == ENDPOINT_ENV).then(|| "http://mini:8000/v1/".to_string())
        })
        .with_client(MockHttp::new(vec![audio()]));
        assert_eq!(p.base_url, "http://mini:8000/v1");
        assert_eq!(
            OmlxProvider::from_lookup("m", |_| None).base_url,
            DEFAULT_BASE_URL
        );
    }

    #[tokio::test]
    async fn create_voice_then_speak_with_it() {
        let http = MockHttp::new(vec![audio()]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        assert!(p.supports_voice_cloning());
        let id = p
            .create_voice(CreateVoice::new("Adam", vec![sample()]).with_transcript("hello"))
            .await
            .unwrap();
        assert_eq!(id, "voice-1");
        assert!(http.requests.lock().unwrap().is_empty());
        p.generate(SpeechRequest::new("hi").with_voice(Voice::id(&id)))
            .await
            .unwrap();
        let body = body_json(&http.request(0));
        assert_eq!(body["ref_audio"], json!("V0FW"));
        assert_eq!(body["ref_text"], json!("hello"));
        assert!(body.get("voice").is_none());
    }

    #[tokio::test]
    async fn a_missing_transcript_sends_the_placeholder() {
        let http = MockHttp::new(vec![audio()]);
        let p = cloned(&http, None).await;
        p.generate(SpeechRequest::new("hi").with_voice(Voice::id("voice-1")))
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0))["ref_text"],
            json!(REF_TEXT_PLACEHOLDER)
        );
    }

    #[tokio::test]
    async fn auto_is_a_clone_only_when_stored() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let p = cloned(&http, Some("hello")).await;
        p.generate(SpeechRequest::new("hi").with_voice(Voice::auto("voice-1")))
            .await
            .unwrap();
        let body = body_json(&http.request(0));
        assert_eq!(body["ref_audio"], json!("V0FW"));
        assert!(body.get("instructions").is_none());
        p.generate(SpeechRequest::new("hi").with_voice(Voice::auto("deep male voice")))
            .await
            .unwrap();
        let body = body_json(&http.request(1));
        assert_eq!(body["instructions"], json!("deep male voice"));
        assert!(body.get("ref_audio").is_none());
    }

    #[tokio::test]
    async fn description_joins_instructions() {
        let http = MockHttp::new(vec![audio()]);
        provider(&http)
            .generate(
                SpeechRequest::new("hi")
                    .with_voice(Voice::description("warm"))
                    .with_instructions("slowly"),
            )
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0))["instructions"],
            json!("warm\nslowly")
        );
    }

    #[tokio::test]
    async fn the_default_voice_resolves_like_a_voice() {
        let http = MockHttp::new(vec![audio()]);
        let p = cloned(&http, Some("hello")).await;
        p.generate(SpeechRequest::new("hi").with_default_voice(Voice::id("voice-1")))
            .await
            .unwrap();
        assert_eq!(body_json(&http.request(0))["ref_audio"], json!("V0FW"));
    }

    #[tokio::test]
    async fn without_a_store() {
        let http = MockHttp::new(vec![audio()]);
        let p = provider(&http);
        assert!(!p.supports_voice_cloning());
        assert!(matches!(
            p.create_voice(CreateVoice::new("A", vec![sample()])).await,
            Err(AudioError::Unsupported(_))
        ));
        assert!(matches!(
            p.generate(SpeechRequest::new("hi").with_voice(Voice::id("x")))
                .await,
            Err(AudioError::Unsupported(_))
        ));
        assert!(matches!(
            p.list_voices().await,
            Err(AudioError::Unsupported(_))
        ));
        assert!(http.requests.lock().unwrap().is_empty());
        p.generate(SpeechRequest::new("hi").with_voice(Voice::auto("x")))
            .await
            .unwrap();
        assert_eq!(body_json(&http.request(0))["instructions"], json!("x"));
    }

    #[tokio::test]
    async fn unknown_id_with_a_store() {
        let http = MockHttp::new(vec![]);
        let p = provider(&http).with_voice_store(MemoryVoiceStore::new());
        assert!(matches!(
            p.generate(SpeechRequest::new("hi").with_voice(Voice::id("nope")))
                .await,
            Err(AudioError::InvalidRequest(_))
        ));
        assert!(http.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn exactly_one_sample() {
        let http = MockHttp::new(vec![]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        for samples in [vec![], vec![sample(), sample()]] {
            assert!(matches!(
                p.create_voice(CreateVoice::new("A", samples)).await,
                Err(AudioError::InvalidRequest(_))
            ));
        }
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn oversized_sample() {
        let http = MockHttp::new(vec![]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        let big = Media::from_base64("audio/wav", "A".repeat(20 * 1024 * 1024 + 1));
        assert!(matches!(
            p.create_voice(CreateVoice::new("A", vec![big])).await,
            Err(AudioError::InvalidRequest(_))
        ));
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn url_samples_are_downloaded_once() {
        let http = MockHttp::new(vec![HttpResponse::new(200, b"WAV".to_vec()), audio()]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        let id = p
            .create_voice(CreateVoice::new(
                "A",
                vec![Media::from_url("audio/wav", "http://x/a.wav")],
            ))
            .await
            .unwrap();
        let stored = store.get(&id).await.unwrap().unwrap();
        assert_eq!(
            stored.sample.source,
            MediaSource::Base64("V0FW".to_string())
        );
        p.generate(SpeechRequest::new("hi").with_voice(Voice::id(&id)))
            .await
            .unwrap();
        let requests = http.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].url, "http://x/a.wav");
        assert_eq!(requests[1].url, "http://local/v1/audio/speech");
    }

    #[tokio::test]
    async fn failed_download_stores_nothing() {
        let http = MockHttp::new(vec![HttpResponse::new(404, b"gone".to_vec())]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        let result = p
            .create_voice(CreateVoice::new(
                "A",
                vec![Media::from_url("audio/wav", "http://x/a.wav")],
            ))
            .await;
        assert!(matches!(result, Err(AudioError::Api { status: 404, .. })));
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn aac_is_unsupported() {
        let http = MockHttp::new(vec![]);
        let result = provider(&http)
            .generate(SpeechRequest::new("hi").with_format(AudioFormat::Aac))
            .await;
        assert!(matches!(result, Err(AudioError::Unsupported(_))));
        assert!(http.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_voices_reads_the_store() {
        let http = MockHttp::new(vec![]);
        let p = provider(&http).with_voice_store(MemoryVoiceStore::new());
        assert!(p.supports_voice_description());
        p.create_voice(CreateVoice::new("Adam", vec![sample()]).with_description("calm"))
            .await
            .unwrap();
        assert_eq!(
            p.list_voices().await.unwrap(),
            vec![
                VoiceInfo::new("voice-1", "Adam")
                    .with_description("calm")
                    .with_cloned(true)
            ]
        );
    }
}
