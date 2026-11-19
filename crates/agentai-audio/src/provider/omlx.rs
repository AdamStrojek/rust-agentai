//! # oMLX speech API
//!
//! [`OmlxProvider`] speaks text with `POST /audio/speech` of an oMLX server, for example with
//! `mlx-community/VoxCPM2-8bit`. The server has no voice list: a voice is a cloned voice; without
//! one the model's own voice is used, styled by `instructions`. oMLX clones from a sample
//! sent with every request (`ref_audio` and `ref_text`), so the provider keeps the samples in a
//! [`VoiceStore`] set with [`with_voice_store`](OmlxProvider::with_voice_store).
//!
//! There is no design endpoint: [`design_voice`](SpeechProvider::design_voice) speaks the text
//! once per candidate with the description as `instructions`, with the design model set by
//! [`with_design_model`](OmlxProvider::with_design_model), which models with a designable voice
//! (VoxCPM2 and alike) turn into a voice. Keep a preview with `create_voice`, which needs a
//! `VoiceStore` like any cloned voice.
//!
//! Every field of a request has an equivalent, so none is left out. What the server does not
//! accept comes back as [`AudioError::Api`].

use super::{
    AudioError, AudioFormat, SpeechProvider, SpeechResponse, VoiceId, VoiceInfo, VoicePreview,
    VoiceSource, audio_media,
};
use crate::request::{DesignVoiceRequest, SpeechRequest};
use crate::voice_store::{StoredVoice, VoiceStore};
use agentai_core::message::{Media, MediaSource};
use agentai_core::{HttpClient, HttpError, HttpRequest, ReqwestClient};
use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;

/// The default endpoint of a local oMLX server.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";
/// Environment variable with the endpoint.
pub const ENDPOINT_ENV: &str = "OMLX_ENDPOINT";
/// Environment variable with the API key. It is optional.
pub const API_KEY_ENV: &str = "OMLX_API_KEY";

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
    speech_model: String,
    design_model: Option<String>,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
    store: Option<Arc<dyn VoiceStore>>,
}

impl std::fmt::Debug for OmlxProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmlxProvider")
            .field("speech_model", &self.speech_model)
            .field("design_model", &self.design_model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OmlxProvider {
    /// Creates a provider for `speech_model` with the API key and base URL given in code, for
    /// example [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header.
    pub fn new(
        speech_model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            speech_model: speech_model.into(),
            design_model: None,
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Arc::new(ReqwestClient::new()),
            store: None,
        }
    }

    /// Creates a provider for `speech_model` with the endpoint from `OMLX_ENDPOINT`, or
    /// [`DEFAULT_BASE_URL`] when it is not set, and the optional API key from `OMLX_API_KEY`.
    ///
    /// No voice store is created, so cloning needs [`with_voice_store`](Self::with_voice_store).
    pub fn from_env(speech_model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(speech_model, |name| std::env::var(name).ok())
    }

    pub(crate) fn from_lookup(
        speech_model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        Ok(Self::new(
            speech_model,
            lookup(API_KEY_ENV).unwrap_or_default(),
            lookup(ENDPOINT_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
        ))
    }

    /// Sets the model of voice design, a model that turns `instructions` into a voice. Without
    /// it voice design is off.
    pub fn with_design_model(mut self, model: impl Into<String>) -> Self {
        self.design_model = Some(model.into());
        self
    }

    /// Sets the store of cloned voices. It enables
    /// [`create_voice`](SpeechProvider::create_voice) and speaking with cloned voices.
    pub fn with_voice_store(mut self, store: impl VoiceStore) -> Self {
        self.store = Some(Arc::new(store));
        self
    }

    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    /// The cloned voice with an ID.
    async fn stored_voice(&self, id: &str) -> Result<StoredVoice, AudioError> {
        let store = self
            .store
            .as_ref()
            .ok_or(AudioError::Unsupported("voice IDs without a voice store"))?;
        let voice = store
            .get(id)
            .await?
            .ok_or_else(|| AudioError::InvalidRequest(format!("no cloned voice with id '{id}'")))?;
        Ok(voice)
    }

    /// Posts `/audio/speech`, with the sample of a stored voice when there is one.
    async fn speak(&self, speech: Speech<'_>) -> Result<SpeechResponse, AudioError> {
        let cloned = match speech.voice {
            Some(id) => Some(self.stored_voice(id).await?),
            None => None,
        };
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
            .and_then(|voice| voice.transcript.as_deref());
        let body = SpeechBody {
            model: speech.model,
            input: speech.text,
            response_format: speech.format.map(format_name),
            speed: speech.speed,
            language: speech.language,
            instructions: speech.instructions,
            ref_audio,
            ref_text,
        };
        let http = HttpRequest::post(format!("{}/audio/speech", self.base_url))
            .with_json(&body)?
            .with_bearer_token(&self.api_key);
        let response = self.client.send(http).await?.error_for_status()?;
        let format = speech.format.unwrap_or(AudioFormat::Wav);
        let audio = audio_media(&response, format.content_type())?;
        Ok(SpeechResponse::new(audio).with_model(speech.model))
    }
}

/// What [`OmlxProvider::speak`] sends.
struct Speech<'a> {
    model: &'a str,
    text: &'a str,
    voice: Option<&'a str>,
    instructions: Option<&'a str>,
    format: Option<AudioFormat>,
    speed: Option<f64>,
    language: Option<&'a str>,
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
    instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ref_audio: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ref_text: Option<&'a str>,
}

fn format_name(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Wav => "wav",
        AudioFormat::Opus => "opus",
        AudioFormat::Flac => "flac",
        AudioFormat::Aac => "aac",
        AudioFormat::Pcm => "pcm",
    }
}

#[async_trait]
impl SpeechProvider for OmlxProvider {
    async fn generate(&self, request: &SpeechRequest<'_>) -> Result<SpeechResponse, AudioError> {
        request.without_timestamps()?;
        self.speak(Speech {
            model: request.model().unwrap_or(&self.speech_model),
            text: request.text(),
            voice: request.voice(),
            instructions: request.instructions(),
            format: request.format(),
            speed: request.speed(),
            language: request.language(),
        })
        .await
    }

    fn supports_voice_cloning(&self) -> bool {
        self.store.is_some()
    }

    /// Stores the sample in the voice store; nothing is sent to the server until a request
    /// uses the voice. Needs exactly one sample. Without `transcript` the server answers such a
    /// request with an error, which is returned as [`AudioError::Api`].
    async fn create_voice(
        &self,
        name: &str,
        source: VoiceSource,
        description: Option<&str>,
    ) -> Result<VoiceId, AudioError> {
        let VoiceSource::Samples {
            samples,
            transcript,
        } = source
        else {
            return Err(AudioError::Unsupported("saving a designed voice"));
        };
        let transcript = transcript.as_deref();
        let store = self.store.as_ref().ok_or(AudioError::Unsupported(
            "voice cloning without a voice store",
        ))?;
        let [sample] = samples.as_slice() else {
            return Err(AudioError::InvalidRequest(format!(
                "oMLX clones from exactly one sample, got {}",
                samples.len()
            )));
        };
        // Decoding also rejects what oMLX would, for example a `data:` URL
        let bytes = sample.bytes(&*self.client).await.map_err(|err| match err {
            HttpError::InvalidMedia(message) => AudioError::InvalidRequest(message),
            err => err.into(),
        })?;
        let mut stored = Media::from_bytes(sample.content_type.as_str(), bytes);
        stored.name = sample.name.clone();
        let mut voice = StoredVoice::new(name, stored);
        voice.description = description.map(str::to_string);
        voice.transcript = transcript.map(str::to_string);
        Ok(store.save(voice).await?)
    }

    fn supports_voice_design(&self) -> bool {
        self.design_model.is_some()
    }

    /// Emulates voice design: speaks `text` `count` times with `description` as `instructions`,
    /// one request after another. `text` is required. The previews have no ID: keep one by
    /// sending it to `create_voice` (needs a voice store).
    async fn design_voice(
        &self,
        request: &DesignVoiceRequest<'_>,
    ) -> Result<Vec<VoicePreview>, AudioError> {
        let model = self
            .design_model
            .as_deref()
            .ok_or(AudioError::MissingModel("voice design"))?;
        let model = request.model().unwrap_or(model);
        let text = request.text().ok_or_else(|| {
            AudioError::InvalidRequest(
                "oMLX designs a voice by speaking a text; give one".to_string(),
            )
        })?;
        let mut previews = Vec::with_capacity(request.count());
        for _ in 0..request.count() {
            let response = self
                .speak(Speech {
                    model,
                    text,
                    voice: None,
                    instructions: Some(request.description()),
                    format: None,
                    speed: None,
                    language: None,
                })
                .await?;
            let mut preview = VoicePreview::new(response.audio).with_text(text);
            if let Some(rate) = response.sample_rate {
                preview = preview.with_sample_rate(rate);
            }
            previews.push(preview);
        }
        Ok(previews)
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
    use crate::provider::{AudioError, SpeechProvider, VoiceInfo};
    use crate::{MemoryVoiceStore, SpeechGenerator, VoiceStore};
    use agentai_core::HttpResponse;
    use agentai_core::message::{Media, MediaSource};
    use agentai_core::mock::{MockHttp, body_json};
    use serde_json::json;
    use std::sync::Arc;

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    fn provider(http: &MockHttp) -> OmlxProvider {
        OmlxProvider::new("m", "", "http://local/v1").with_client(Arc::new(http.clone()))
    }

    async fn speak(p: &OmlxProvider, voice: &str) -> Result<SpeechResponse, AudioError> {
        SpeechGenerator::from_provider(p.clone())
            .generate("hi")
            .with_voice(voice)
            .await
    }

    fn designer(http: &MockHttp) -> OmlxProvider {
        provider(http).with_design_model("design-model")
    }

    fn sample() -> Media {
        Media::from_bytes("audio/wav", b"WAV")
    }

    async fn cloned(http: &MockHttp, transcript: Option<&str>) -> OmlxProvider {
        let p = provider(http).with_voice_store(MemoryVoiceStore::new());
        let id = p
            .create_voice(
                "Adam",
                VoiceSource::samples(vec![sample()], transcript),
                None,
            )
            .await
            .unwrap();
        assert_eq!(id, "voice-1");
        p
    }

    #[tokio::test]
    async fn a_sample_id_is_unsupported() {
        let http = MockHttp::new(vec![]);
        let err = cloned(&http, None)
            .await
            .create_voice("a", VoiceSource::SampleId("g1".into()), None)
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn generate_without_a_voice_sends_no_voice_field() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(provider(&http))
            .generate("hi")
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
        .unwrap();
        assert_eq!(p.base_url, "http://mini:8000/v1");
        assert_eq!(
            OmlxProvider::from_lookup("m", |_| None).unwrap().base_url,
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
            .create_voice(
                "Adam",
                VoiceSource::samples(vec![sample()], Some("hello")),
                None,
            )
            .await
            .unwrap();
        assert_eq!(id, "voice-1");
        assert_eq!(http.request_count(), 0);
        speak(&p, &id).await.unwrap();
        let body = body_json(&http.request(0));
        assert_eq!(body["ref_audio"], json!("V0FW"));
        assert_eq!(body["ref_text"], json!("hello"));
        assert!(body.get("voice").is_none());
    }

    #[tokio::test]
    async fn a_missing_transcript_sends_no_ref_text() {
        let http = MockHttp::new(vec![audio()]);
        let p = cloned(&http, None).await;
        speak(&p, "voice-1").await.unwrap();
        let body = body_json(&http.request(0));
        assert_eq!(body["ref_audio"], json!("V0FW"));
        assert!(body.get("ref_text").is_none());
    }

    #[tokio::test]
    async fn without_a_store() {
        let http = MockHttp::new(vec![audio()]);
        let p = provider(&http);
        assert!(!p.supports_voice_cloning());
        assert!(matches!(
            p.create_voice("A", VoiceSource::samples(vec![sample()], None), None)
                .await,
            Err(AudioError::Unsupported(_))
        ));
        assert!(matches!(
            speak(&p, "x").await,
            Err(AudioError::Unsupported(_))
        ));
        assert!(matches!(
            p.list_voices().await,
            Err(AudioError::Unsupported(_))
        ));
        assert_eq!(http.request_count(), 0);
        SpeechGenerator::from_provider(p)
            .generate("hi")
            .with_instructions("x")
            .await
            .unwrap();
        assert_eq!(body_json(&http.request(0))["instructions"], json!("x"));
    }

    #[tokio::test]
    async fn unknown_id_with_a_store() {
        let http = MockHttp::new(vec![]);
        let p = provider(&http).with_voice_store(MemoryVoiceStore::new());
        assert!(matches!(
            speak(&p, "nope").await,
            Err(AudioError::InvalidRequest(_))
        ));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn exactly_one_sample() {
        let http = MockHttp::new(vec![]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        for samples in [vec![], vec![sample(), sample()]] {
            assert!(matches!(
                p.create_voice("A", VoiceSource::samples(samples.clone(), None), None)
                    .await,
                Err(AudioError::InvalidRequest(_))
            ));
        }
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn url_samples_are_downloaded_once() {
        let http = MockHttp::new(vec![HttpResponse::new(200, b"WAV".to_vec()), audio()]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        let url_sample = Media::from_url("audio/wav", "http://x/a.wav");
        let id = p
            .create_voice("A", VoiceSource::samples(vec![url_sample], None), None)
            .await
            .unwrap();
        let stored = store.get(&id).await.unwrap().unwrap();
        assert_eq!(
            stored.sample.source,
            MediaSource::Base64("V0FW".to_string())
        );
        speak(&p, &id).await.unwrap();
        assert_eq!(http.request_count(), 2);
        assert_eq!(http.request(0).url, "http://x/a.wav");
        assert_eq!(http.request(1).url, "http://local/v1/audio/speech");
    }

    #[tokio::test]
    async fn failed_download_stores_nothing() {
        let http = MockHttp::new(vec![HttpResponse::new(404, b"gone".to_vec())]);
        let store = MemoryVoiceStore::new();
        let p = provider(&http).with_voice_store(store.clone());
        let url_sample = Media::from_url("audio/wav", "http://x/a.wav");
        let result = p
            .create_voice("A", VoiceSource::samples(vec![url_sample], None), None)
            .await;
        assert!(matches!(result, Err(AudioError::Api { status: 404, .. })));
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn list_voices_reads_the_store() {
        let http = MockHttp::new(vec![]);
        let p = provider(&http).with_voice_store(MemoryVoiceStore::new());
        p.create_voice(
            "Adam",
            VoiceSource::samples(vec![sample()], None),
            Some("calm"),
        )
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

    #[tokio::test]
    async fn create_voice_rejects_a_data_url_sample() {
        let p = provider(&MockHttp::new(vec![])).with_voice_store(MemoryVoiceStore::new());
        let sample = Media::from_base64("audio/wav", "data:audio/wav;base64,V0FW");
        let err = p
            .create_voice("A", VoiceSource::samples(vec![sample], None), None)
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn design_voice_speaks_the_text_once_per_candidate() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let previews = SpeechGenerator::from_provider(designer(&http))
            .design_voice("deep calm voice")
            .with_text("Hello")
            .with_count(2)
            .await
            .unwrap();
        assert_eq!(previews.len(), 2);
        for (i, preview) in previews.iter().enumerate() {
            let req = http.request(i);
            assert_eq!(req.url, "http://local/v1/audio/speech");
            assert_eq!(
                body_json(&req),
                json!({"model": "design-model", "input": "Hello", "instructions": "deep calm voice"})
            );
            assert_eq!(preview.audio.content_type, "audio/wav");
            assert_eq!(preview.audio.bytes(&http).await.unwrap(), b"audio");
            assert_eq!(preview.text.as_deref(), Some("Hello"));
            assert_eq!(preview.sample_id, None);
        }
    }

    #[tokio::test]
    async fn design_voice_needs_a_text() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(designer(&http))
            .design_voice("x")
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRequest(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn a_designed_voice_can_be_saved_and_spoken() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let p = designer(&http).with_voice_store(MemoryVoiceStore::new());
        let previews = SpeechGenerator::from_provider(p.clone())
            .design_voice("calm")
            .with_text("Hello")
            .await
            .unwrap();
        let id = p
            .create_voice("calm", VoiceSource::from(&previews[0]), None)
            .await
            .unwrap();
        speak(&p, &id).await.unwrap();
        let body = body_json(&http.request(1));
        assert_eq!(body["ref_text"], json!("Hello"));
        assert_eq!(body["ref_audio"], json!("YXVkaW8="));
    }
}
