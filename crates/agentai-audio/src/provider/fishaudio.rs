//! # Fish Audio speech API
//!
//! [`FishAudioProvider`] speaks text with `POST /v1/tts`, creates voices in the Fish Audio
//! account with `POST /model` and lists them with `GET /model`. The model (`s1`, `s2-pro`,
//! `drama-3`, ...) goes in the `model` header. It designs voices from a description with
//! `POST /v1/voice-design` with the design model set by
//! [`with_design_model`](FishAudioProvider::with_design_model) (for example `voice-design-1`);
//! a preview is kept with `create_voice`, which saves it as a sample.
//!
//! The provider has three engines, each with its own model: speech
//! ([`new`](FishAudioProvider::new)), dialogue
//! ([`with_dialogue_model`](FishAudioProvider::with_dialogue_model)) and voice design
//! ([`with_design_model`](FishAudioProvider::with_design_model)). A dialogue is one request
//! whose text marks the speakers, `<|speaker:0|>Hi<|speaker:1|>Hello`, numbered like the
//! voices of the request.
//!
//! A voice is the ID of a Fish Audio voice model or, with a [`VoiceStore`] set with
//! [`with_voice_store`](FishAudioProvider::with_voice_store), the ID of a voice in that store:
//! its sample and transcript are then sent with the request (`references`, as MessagePack).
//! A request takes either IDs or samples for all its voices: when a store is set, all voices
//! must be in it or none, and a mix is [`AudioError::InvalidRequest`].
//!
//! With [`SpeechRequest::with_timestamps`] the request goes to
//! `POST /v1/tts/stream/with-timestamp`: the streamed audio chunks are joined and the time of
//! every word is returned in [`SpeechResponse::timestamps`].
//!
//! Fields without an equivalent are not sent: `instructions` is ignored.

use super::{
    AudioError, AudioFormat, SpeechProvider, SpeechResponse, VoiceId, VoiceInfo, VoiceOrigin,
    VoicePreview, VoiceSource, WordTimestamp, audio_media,
};
use crate::request::{DesignVoiceRequest, DialogueRequest, SpeechRequest};
use crate::voice_store::{StoredVoice, VoiceStore};
use agentai_core::message::Media;
use agentai_core::{HttpClient, HttpRequest, Multipart, ReqwestClient};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The Fish Audio API endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.fish.audio";
/// Environment variable with the API key.
pub const API_KEY_ENV: &str = "FISH_API_KEY";
/// Environment variable with the base URL.
pub const BASE_URL_ENV: &str = "FISH_BASE_URL";

/// Sample rate of PCM audio from Fish Audio.
const PCM_SAMPLE_RATE: u32 = 44_100;

/// Provider for the Fish Audio API.
///
/// ```rust,no_run
/// # use agentai_audio::provider::fishaudio::FishAudioProvider;
/// # use agentai_audio::MemoryVoiceStore;
/// # fn main() -> Result<(), agentai_audio::AudioError> {
/// // From `FISH_API_KEY` and `FISH_BASE_URL`
/// let provider = FishAudioProvider::from_env("drama-3")?;
/// // Or everything in code, with local voice samples
/// let provider = FishAudioProvider::new("drama-3", "key", "https://api.fish.audio")
///     .with_voice_store(MemoryVoiceStore::new());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct FishAudioProvider {
    speech_model: String,
    dialogue_model: Option<String>,
    design_model: Option<String>,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
    store: Option<Arc<dyn VoiceStore>>,
}

impl std::fmt::Debug for FishAudioProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishAudioProvider")
            .field("speech_model", &self.speech_model)
            .field("dialogue_model", &self.dialogue_model)
            .field("design_model", &self.design_model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl FishAudioProvider {
    /// Creates a provider for `speech_model` with the API key and base URL given in code, for
    /// example [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header, for
    /// self-hosted servers without authentication.
    pub fn new(
        speech_model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            speech_model: speech_model.into(),
            dialogue_model: None,
            design_model: None,
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Arc::new(ReqwestClient::new()),
            store: None,
        }
    }

    /// Creates a provider for `speech_model` with the API key from `FISH_API_KEY` and the base
    /// URL from `FISH_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set. The key is
    /// optional when `FISH_BASE_URL` is set, for self-hosted servers.
    pub fn from_env(speech_model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(speech_model, |name| std::env::var(name).ok())
    }

    pub(crate) fn from_lookup(
        speech_model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        let (api_key, base_url) = match (lookup(API_KEY_ENV), lookup(BASE_URL_ENV)) {
            (Some(key), base_url) => (key, base_url),
            (None, Some(base_url)) => (String::new(), Some(base_url)),
            (None, None) => return Err(AudioError::MissingApiKey(API_KEY_ENV)),
        };
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(speech_model, api_key, base_url))
    }

    /// Sets the model of dialogues, for example `drama-3`.
    pub fn with_dialogue_model(mut self, model: impl Into<String>) -> Self {
        self.dialogue_model = Some(model.into());
        self
    }

    /// Sets the model of voice design, for example `voice-design-1`. Without it voice design is
    /// off.
    pub fn with_design_model(mut self, model: impl Into<String>) -> Self {
        self.design_model = Some(model.into());
        self
    }

    /// Sets the store of voice samples. A voice ID found in it is sent as its sample and
    /// transcript; other IDs are Fish Audio voice models.
    pub fn with_voice_store(mut self, store: impl VoiceStore) -> Self {
        self.store = Some(Arc::new(store));
        self
    }

    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: Arc<dyn HttpClient>) -> Self {
        self.client = client;
        self
    }

    fn request(&self, request: HttpRequest) -> HttpRequest {
        request.with_bearer_token(&self.api_key)
    }

    /// The samples of `voices`, one per voice, when all of them are in the voice store;
    /// `None` when none is (or no store is set), so they are voice model IDs. A mix is an
    /// error. Every voice is looked up before a sample is downloaded.
    async fn samples(&self, voices: &[VoiceId]) -> Result<Option<Vec<Reference>>, AudioError> {
        let Some(store) = &self.store else {
            return Ok(None);
        };
        let mut found = Vec::new();
        for id in voices {
            found.push(store.get(id).await?);
        }
        let stored = found.iter().flatten().count();
        if stored == 0 {
            return Ok(None);
        }
        if stored < found.len() {
            return Err(AudioError::InvalidRequest(
                "Fish Audio takes either voice IDs or samples for all voices of a request, not both"
                    .to_string(),
            ));
        }
        let mut samples = Vec::new();
        for voice in found.into_iter().flatten() {
            samples.push(self.reference(voice).await?);
        }
        Ok(Some(samples))
    }

    async fn reference(&self, voice: StoredVoice) -> Result<Reference, AudioError> {
        Ok(Reference {
            audio: voice.sample.bytes(&*self.client).await?,
            text: voice.transcript,
        })
    }

    /// Sends a body to `/v1/tts` (or the timestamp stream) and reads the audio. Samples are
    /// binary, which only the MessagePack body takes.
    async fn send<Id: Serialize, Refs: Serialize>(
        &self,
        model: &str,
        body: &SpeechBody<'_, Id, Refs>,
        format: Option<AudioFormat>,
        timestamps: bool,
    ) -> Result<SpeechResponse, AudioError> {
        let path = if timestamps {
            "/v1/tts/stream/with-timestamp"
        } else {
            "/v1/tts"
        };
        let mut http = HttpRequest::post(format!("{}{path}", self.base_url));
        if !model.is_empty() {
            http = http.with_header("model", model);
        }
        http = if body.references.is_some() {
            let bytes = rmp_serde::to_vec_named(body)
                .map_err(|e| AudioError::InvalidRequest(e.to_string()))?;
            http.with_header("Content-Type", "application/msgpack")
                .with_body(bytes)
        } else {
            http.with_json(body)?
        };
        let response = self
            .client
            .send(self.request(http))
            .await?
            .error_for_status()?;
        let format = format.unwrap_or(AudioFormat::Mp3);
        let (audio, timestamps) = if timestamps {
            let (bytes, timestamps) = parse_timestamp_stream(&response.body)?;
            if bytes.is_empty() {
                return Err(AudioError::InvalidResponse(
                    "the response has no audio".to_string(),
                ));
            }
            (
                Media::from_bytes(format.content_type(), bytes),
                Some(timestamps),
            )
        } else {
            (audio_media(&response, format.content_type())?, None)
        };
        let mut result = SpeechResponse::new(audio);
        if !model.is_empty() {
            result = result.with_model(model);
        }
        if format == AudioFormat::Pcm {
            result = result.with_sample_rate(PCM_SAMPLE_RATE);
        }
        if let Some(timestamps) = timestamps {
            result = result.with_timestamps(timestamps);
        }
        Ok(result)
    }
}

#[derive(Serialize)]
struct Reference {
    #[serde(with = "serde_bytes")]
    audio: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[derive(Serialize)]
struct Prosody {
    speed: f64,
}

#[derive(Serialize)]
struct SpeechBody<'a, Id, Refs> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_id: Option<Id>,
    #[serde(skip_serializing_if = "Option::is_none")]
    references: Option<Refs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prosody: Option<Prosody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

fn format_name(format: AudioFormat) -> Result<&'static str, AudioError> {
    match format {
        AudioFormat::Mp3 => Ok("mp3"),
        AudioFormat::Wav => Ok("wav"),
        AudioFormat::Opus => Ok("opus"),
        AudioFormat::Pcm => Ok("pcm"),
        AudioFormat::Flac => Err(AudioError::Unsupported("FLAC output with Fish Audio")),
        AudioFormat::Aac => Err(AudioError::Unsupported("AAC output with Fish Audio")),
    }
}

#[async_trait]
impl SpeechProvider for FishAudioProvider {
    async fn generate(&self, request: &SpeechRequest<'_>) -> Result<SpeechResponse, AudioError> {
        let model = request.model().unwrap_or(&self.speech_model);
        let format = request.format().map(format_name).transpose()?;
        let voice: Vec<VoiceId> = request.voice().map(String::from).into_iter().collect();
        let (reference_id, references) = match self.samples(&voice).await? {
            Some(samples) => (None, Some(samples)),
            None => (request.voice(), None),
        };
        let body = SpeechBody {
            text: request.text(),
            reference_id,
            references,
            format,
            prosody: request.speed().map(|speed| Prosody { speed }),
            language: request.language(),
        };
        self.send(model, &body, request.format(), request.timestamps())
            .await
    }

    fn supports_dialogue(&self) -> bool {
        self.dialogue_model.is_some()
    }

    /// Speaks a dialogue with the dialogue model. The text marks the speakers as
    /// `<|speaker:0|>Hi<|speaker:1|>Hello`, numbered like the voices of the request. Voices
    /// are IDs of voice models (`reference_id` list) or, when all of them are in the voice
    /// store, their samples (one list of references per speaker).
    async fn generate_dialogue(
        &self,
        request: &DialogueRequest<'_>,
    ) -> Result<SpeechResponse, AudioError> {
        let default_model = self
            .dialogue_model
            .as_deref()
            .ok_or(AudioError::MissingModel("dialogue"))?;
        let model = request.model().unwrap_or(default_model);
        let format = request.format().map(format_name).transpose()?;
        let text: String = request
            .turns()
            .iter()
            .map(|(speaker, text)| format!("<|speaker:{speaker}|>{text}"))
            .collect();
        let (reference_id, references) = match self.samples(request.voices()).await? {
            Some(samples) => (
                None,
                Some(
                    samples
                        .into_iter()
                        .map(|sample| vec![sample])
                        .collect::<Vec<_>>(),
                ),
            ),
            None => (Some(request.voices()).filter(|v| !v.is_empty()), None),
        };
        let body = SpeechBody {
            text: &text,
            reference_id,
            references,
            format,
            prosody: request.speed().map(|speed| Prosody { speed }),
            language: request.language(),
        };
        self.send(model, &body, request.format(), request.timestamps())
            .await
    }

    fn supports_voice_cloning(&self) -> bool {
        true
    }

    /// Creates a voice model in the Fish Audio account with `POST /model` (fast training, so
    /// it can speak at once) and returns its ID. Without `transcript` Fish Audio transcribes
    /// the samples itself.
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
        if samples.is_empty() {
            return Err(AudioError::InvalidRequest(
                "voice cloning needs at least one sample".to_string(),
            ));
        }
        let mut form = Multipart::new()
            .with_text("type", "tts")
            .with_text("train_mode", "fast")
            .with_text("title", name);
        if let Some(description) = description {
            form = form.with_text("description", description);
        }
        if let Some(transcript) = transcript {
            form = form.with_text("texts", transcript);
        }
        for (index, sample) in samples.iter().enumerate() {
            let bytes = sample.bytes(&*self.client).await?;
            let filename = sample.file_name(&format!("sample-{}", index + 1));
            form = form.with_file("voices", filename, sample.content_type.as_str(), bytes);
        }
        let http = HttpRequest::post(format!("{}/model", self.base_url)).with_multipart(form);
        let response = self
            .client
            .send(self.request(http))
            .await?
            .error_for_status()?;
        let created: ModelEntry = response
            .json()
            .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
        Ok(created.id)
    }

    fn supports_voice_design(&self) -> bool {
        self.design_model.is_some()
    }

    /// Designs voices with `POST /v1/voice-design`, `count` of them. The previews have no ID:
    /// keep one by sending its audio to `create_voice`.
    async fn design_voice(
        &self,
        request: &DesignVoiceRequest<'_>,
    ) -> Result<Vec<VoicePreview>, AudioError> {
        let model = self
            .design_model
            .as_deref()
            .ok_or(AudioError::MissingModel("voice design"))?;
        let model = request.model().unwrap_or(model);
        let text = request.text();
        let body = VoiceDesignBody {
            instruction: request.description(),
            reference_text: text,
            n: request.count(),
        };
        let http = HttpRequest::post(format!("{}/v1/voice-design", self.base_url))
            .with_header("model", model)
            .with_json(&body)?;
        let response = self
            .client
            .send(self.request(http))
            .await?
            .error_for_status()?;
        let designed: VoiceDesignResponse = response
            .json()
            .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
        designed
            .candidates
            .into_iter()
            .map(|candidate| {
                let bytes = agentai_core::base64::decode(&candidate.audio_base64)
                    .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
                let audio = Media::from_bytes(audio_type(&bytes), &bytes);
                let preview = VoicePreview::new(audio).with_sample_rate(candidate.sample_rate);
                Ok(match candidate.text.as_deref().or(text) {
                    Some(text) => preview.with_text(text),
                    None => preview,
                })
            })
            .collect()
    }

    /// Lists the voice models of the account with `GET /model?self=true`, page by page, and
    /// then the voices of the voice store, if one is set.
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        let mut voices = Vec::new();
        for page_number in 1.. {
            let url = format!(
                "{}/model?self=true&page_size=100&page_number={page_number}",
                self.base_url
            );
            let response = self
                .client
                .send(self.request(HttpRequest::get(url)))
                .await?
                .error_for_status()?;
            let page: ModelPage = response
                .json()
                .map_err(|e| AudioError::InvalidResponse(e.to_string()))?;
            let empty = page.items.is_empty();
            voices.extend(page.items.into_iter().map(|model| {
                let mut info = VoiceInfo::new(model.id, model.title, VoiceOrigin::Account);
                info.description = model.description.filter(|d| !d.is_empty());
                info.preview = model_preview(model.samples);
                info
            }));
            let done = empty
                || page.has_more == Some(false)
                || page.total.is_some_and(|total| voices.len() >= total);
            if done {
                break;
            }
        }
        if let Some(store) = &self.store {
            voices.extend(
                store
                    .list()
                    .await?
                    .into_iter()
                    .map(|(id, voice)| voice.into_info(id)),
            );
        }
        Ok(voices)
    }
}

/// The body of `POST /v1/voice-design`.
#[derive(Serialize)]
struct VoiceDesignBody<'a> {
    instruction: &'a str,
    reference_text: Option<&'a str>,
    n: usize,
}

#[derive(Deserialize)]
struct VoiceDesignResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    audio_base64: String,
    sample_rate: u32,
    text: Option<String>,
}

/// The response names no audio type, so it is read from the first bytes (WAV or MP3).
fn audio_type(bytes: &[u8]) -> &'static str {
    let mp3 = bytes.starts_with(b"ID3")
        || (bytes.first() == Some(&0xFF) && bytes.get(1).is_some_and(|b| b & 0xE0 == 0xE0));
    if mp3 { "audio/mpeg" } else { "audio/wav" }
}

/// An event of `/v1/tts/stream/with-timestamp`.
#[derive(Deserialize)]
struct TimestampEvent {
    #[serde(default)]
    audio_base64: String,
    alignment: Option<Alignment>,
    #[serde(default)]
    chunk_seq: u64,
    #[serde(default)]
    chunk_audio_offset_sec: f64,
}

#[derive(Deserialize)]
struct Alignment {
    #[serde(default)]
    segments: Vec<Segment>,
}

#[derive(Deserialize)]
struct Segment {
    text: String,
    start: f64,
    end: f64,
}

/// The audio and word times of a `/v1/tts/stream/with-timestamp` response (server-sent
/// events). Audio chunks are joined in order; the alignment of a text chunk is a snapshot that
/// a later one replaces, with times relative to the chunk's offset.
fn parse_timestamp_stream(body: &[u8]) -> Result<(Vec<u8>, Vec<WordTimestamp>), AudioError> {
    let invalid = |message: String| AudioError::InvalidResponse(message);
    let body = std::str::from_utf8(body).map_err(|e| invalid(e.to_string()))?;
    let mut audio = Vec::new();
    let mut alignments = BTreeMap::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        // Only JSON objects are events; skip anything else, such as a `[DONE]` sentinel
        let Ok(serde_json::Value::Object(object)) = serde_json::from_str(data.trim()) else {
            continue;
        };
        if !object.contains_key("audio_base64")
            && let Some(message) = object.get("message").or(object.get("error"))
        {
            let message = message
                .as_str()
                .map_or_else(|| message.to_string(), String::from);
            return Err(invalid(format!("error event in the stream: {message}")));
        }
        let event: TimestampEvent =
            serde_json::from_value(object.into()).map_err(|e| invalid(e.to_string()))?;
        audio.extend(
            agentai_core::base64::decode(&event.audio_base64)
                .map_err(|e| invalid(e.to_string()))?,
        );
        if let Some(alignment) = event.alignment {
            alignments.insert(event.chunk_seq, (event.chunk_audio_offset_sec, alignment));
        }
    }
    let timestamps = alignments
        .into_values()
        .flat_map(|(offset, alignment)| {
            alignment.segments.into_iter().map(move |segment| {
                WordTimestamp::new(segment.text, segment.start + offset, segment.end + offset)
            })
        })
        .collect();
    Ok((audio, timestamps))
}

#[derive(Deserialize)]
struct ModelPage {
    #[serde(default)]
    items: Vec<ModelEntry>,
    total: Option<usize>,
    has_more: Option<bool>,
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    description: Option<String>,
    #[serde(default)]
    samples: Vec<ModelSample>,
}

#[derive(Deserialize)]
struct ModelSample {
    audio: String,
}

/// The first sample of a voice model that can be played: a full URL. Self-hosted servers may
/// give a path that is not served.
fn model_preview(samples: Vec<ModelSample>) -> Option<Media> {
    samples
        .into_iter()
        .map(|sample| sample.audio)
        .find(|audio| audio.starts_with("http://") || audio.starts_with("https://"))
        .map(|url| {
            Media::from_url(
                agentai_core::mime::from_path(&url).unwrap_or("audio/mpeg"),
                url,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::DialogueTurn;
    use crate::{MemoryVoiceStore, SpeechGenerator};
    use agentai_core::HttpResponse;
    use agentai_core::mock::{MockHttp, body_json, json_response};
    use serde_json::{Value, json};

    fn provider(http: &MockHttp) -> FishAudioProvider {
        FishAudioProvider::new("drama-3", "key", "https://api.test/")
            .with_client(Arc::new(http.clone()))
    }

    fn designer(http: &MockHttp) -> FishAudioProvider {
        provider(http).with_design_model("voice-design-1")
    }

    fn audio() -> HttpResponse {
        HttpResponse::new(200, b"audio".to_vec())
    }

    async fn store_with_sample() -> MemoryVoiceStore {
        let store = MemoryVoiceStore::new();
        let voice = StoredVoice::new("Ala", Media::from_bytes("audio/wav", b"WAV"))
            .with_transcript("Hello there.");
        assert_eq!(store.save(voice).await.unwrap(), "voice-1");
        store
    }

    fn dialogue_provider(http: &MockHttp) -> FishAudioProvider {
        provider(http).with_dialogue_model("drama-3")
    }

    async fn store_with_two() -> MemoryVoiceStore {
        let store = MemoryVoiceStore::new();
        for (name, sample, transcript) in [("Ala", b"WAV1", "One."), ("Bo", b"WAV2", "Two.")] {
            let voice = StoredVoice::new(name, Media::from_bytes("audio/wav", sample))
                .with_transcript(transcript);
            store.save(voice).await.unwrap();
        }
        store
    }

    /// The MessagePack body as JSON, with `bin` as `{"bin": [bytes]}`.
    fn msgpack(request: &HttpRequest) -> Value {
        fn to_json(value: rmpv::Value) -> Value {
            match value {
                rmpv::Value::Binary(data) => json!({ "bin": data }),
                rmpv::Value::Array(items) => items.into_iter().map(to_json).collect(),
                rmpv::Value::Map(entries) => entries
                    .into_iter()
                    .map(|(k, v)| (k.as_str().unwrap().to_string(), to_json(v)))
                    .collect::<serde_json::Map<_, _>>()
                    .into(),
                rmpv::Value::String(s) => json!(s.as_str().unwrap()),
                other => panic!("unexpected MessagePack value {other}"),
            }
        }
        assert_eq!(request.header("content-type"), Some("application/msgpack"));
        let mut body = request.body.as_deref().unwrap();
        to_json(rmpv::decode::read_value(&mut body).unwrap())
    }

    fn bytes(data: &[u8]) -> Value {
        json!({ "bin": data })
    }

    #[tokio::test]
    async fn design_without_a_design_model_sends_nothing() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(provider(&http))
            .design_voice("calm")
            .with_model("voice-design-1")
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::MissingModel("voice design")));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn the_call_model_wins_over_the_speech_model() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(provider(&http))
            .generate("hi")
            .with_model("s2-pro")
            .await
            .unwrap();
        assert_eq!(http.request(0).header("model"), Some("s2-pro"));
        assert_eq!(r.model.as_deref(), Some("s2-pro"));
    }

    #[tokio::test]
    async fn a_sample_id_is_unsupported() {
        let http = MockHttp::new(vec![]);
        let err = provider(&http)
            .create_voice("a", VoiceSource::SampleId("g1".into()), None)
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn generate_sends_one_model_voice_as_json() {
        let http = MockHttp::new(vec![audio()]);
        let r = SpeechGenerator::from_provider(provider(&http))
            .generate("hi")
            .with_voice("abc")
            .with_format(AudioFormat::Pcm)
            .with_speed(1.2)
            .with_language("pl")
            .with_instructions("ignored")
            .with_name("hi.pcm")
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/tts");
        assert_eq!(req.header("authorization"), Some("Bearer key"));
        assert_eq!(req.header("model"), Some("drama-3"));
        assert_eq!(
            body_json(&req),
            json!({
                "text": "hi", "reference_id": "abc", "format": "pcm",
                "prosody": {"speed": 1.2}, "language": "pl"
            })
        );
        assert_eq!(r.audio.content_type, "audio/pcm");
        assert_eq!(r.audio.name.as_deref(), Some("hi.pcm"));
        assert_eq!(r.sample_rate, Some(44_100));
        assert_eq!(r.model.as_deref(), Some("drama-3"));
    }

    #[tokio::test]
    async fn timestamps_join_the_audio_and_offset_the_words() {
        let event = |audio: &[u8], seq: u64, offset: f64, alignment: Value| {
            let audio = agentai_core::base64::encode(audio);
            let event = json!({
                "audio_base64": audio, "content": "x", "alignment": alignment,
                "chunk_seq": seq, "chunk_audio_offset_sec": offset
            });
            format!("data: {event}\n\n")
        };
        let words = |list: &[(&str, f64, f64)]| {
            let segments: Vec<Value> = list
                .iter()
                .map(|(text, start, end)| json!({"text": text, "start": start, "end": end}))
                .collect();
            json!({"audio_duration": 1.0, "segments": segments})
        };
        let body = [
            ": keep-alive\n\n".to_string(),
            event(b"AB", 0, 0.0, Value::Null),
            event(b"CD", 0, 0.0, words(&[("Hi", 0.0, 0.25)])),
            event(b"", 0, 0.0, words(&[("Hi", 0.0, 0.5), ("there", 0.5, 1.0)])),
            event(b"EF", 1, 1.5, Value::Null),
            event(b"", 1, 1.5, words(&[("Bye", 0.25, 0.75)])),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat();
        let http = MockHttp::new(vec![HttpResponse::new(200, body)]);
        let r = SpeechGenerator::from_provider(provider(&http))
            .generate("Hi there. Bye")
            .with_voice("abc")
            .with_format(AudioFormat::Opus)
            .with_timestamps(true)
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/tts/stream/with-timestamp");
        assert_eq!(body_json(&req)["reference_id"], "abc");
        assert_eq!(
            r.audio,
            Media::from_bytes("audio/ogg", b"ABCDEF").with_name("speech.opus")
        );
        assert_eq!(
            r.timestamps.unwrap(),
            [
                WordTimestamp::new("Hi", 0.0, 0.5),
                WordTimestamp::new("there", 0.5, 1.0),
                WordTimestamp::new("Bye", 1.75, 2.25),
            ]
        );
    }

    #[tokio::test]
    async fn an_error_event_fails_the_timestamp_stream() {
        let body = "data: {\"audio_base64\": \"QUI=\", \"chunk_seq\": 0}\n\n\
                    data: {\"status\": 503, \"message\": \"server overloaded\"}\n\n";
        let http = MockHttp::new(vec![HttpResponse::new(200, body)]);
        let err = SpeechGenerator::from_provider(provider(&http))
            .generate("hi")
            .with_timestamps(true)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, AudioError::InvalidResponse(m) if m.contains("server overloaded")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_model_voice_is_sent_as_its_id_and_no_voice_sends_none() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let generator = SpeechGenerator::from_provider(provider(&http));
        generator.generate("hi").with_voice("d").await.unwrap();
        assert_eq!(body_json(&http.request(0))["reference_id"], "d");
        generator.generate("hi").await.unwrap();
        assert_eq!(body_json(&http.request(1)), json!({"text": "hi"}));
    }

    #[tokio::test]
    async fn one_stored_voice_is_sent_as_its_sample() {
        let http = MockHttp::new(vec![audio()]);
        SpeechGenerator::from_provider(provider(&http).with_voice_store(store_with_sample().await))
            .generate("hi")
            .with_voice("voice-1")
            .await
            .unwrap();
        assert_eq!(
            msgpack(&http.request(0)),
            json!({
                "text": "hi",
                "references": [{"audio": bytes(b"WAV"), "text": "Hello there."}]
            })
        );
    }

    #[tokio::test]
    async fn unsupported_formats_send_nothing() {
        let http = MockHttp::new(vec![]);
        let generator = SpeechGenerator::from_provider(provider(&http));
        for format in [AudioFormat::Flac, AudioFormat::Aac] {
            let err = generator
                .generate("hi")
                .with_format(format)
                .await
                .unwrap_err();
            assert!(matches!(err, AudioError::Unsupported(_)));
        }
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn create_voice_creates_a_model() {
        let http = MockHttp::new(vec![json_response(
            201,
            json!({"_id": "m1", "title": "Adam", "state": "trained"}),
        )]);
        let samples = vec![
            Media::from_bytes("audio/wav", b"ONE").with_name("a.wav"),
            Media::from_bytes("audio/mpeg", b"TWO"),
        ];
        let id = provider(&http)
            .create_voice(
                "Adam",
                VoiceSource::samples(samples.clone(), Some("Said text.")),
                Some("my voice"),
            )
            .await
            .unwrap();
        assert_eq!(id, "m1");

        let request = http.request(0);
        assert_eq!(request.url, "https://api.test/model");
        assert_eq!(request.header("authorization"), Some("Bearer key"));
        let boundary = request
            .header("content-type")
            .unwrap()
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap();
        let body = String::from_utf8(request.body.clone().unwrap()).unwrap();
        let field = |name: &str, value: &str| {
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
        };
        let file = |filename: &str, content_type: &str, data: &str| {
            format!(
                "Content-Disposition: form-data; name=\"voices\"; filename=\"{filename}\"\r\n\
                 Content-Type: {content_type}\r\n\r\n{data}\r\n"
            )
        };
        let parts: Vec<&str> = body
            .split(&format!("--{boundary}"))
            .map(|part| part.trim_start_matches("\r\n"))
            .collect();
        assert_eq!(
            parts,
            [
                "",
                &field("type", "tts"),
                &field("train_mode", "fast"),
                &field("title", "Adam"),
                &field("description", "my voice"),
                &field("texts", "Said text."),
                &file("a.wav", "audio/wav", "ONE"),
                &file("sample-2.mp3", "audio/mpeg", "TWO"),
                "--\r\n",
            ]
        );
    }

    #[tokio::test]
    async fn list_voices_reads_every_page_and_the_store() {
        let http = MockHttp::new(vec![
            json_response(
                200,
                json!({"total": 2, "has_more": true,
                       "items": [{"_id": "m1", "title": "A", "description": "",
                                  "samples": [{"title": "s", "text": "t", "task_id": "x",
                                               "audio": "https://cdn.test/m1.wav"}]}]}),
            ),
            json_response(
                200,
                json!({"total": 2, "items": [{"_id": "m2", "title": "B", "description": "d",
                                              "samples": [{"title": "s", "text": "t", "task_id": "y",
                                                           "audio": "samples/0.wav"}]}]}),
            ),
        ]);
        let voices = provider(&http)
            .with_voice_store(store_with_sample().await)
            .list_voices()
            .await
            .unwrap();
        assert_eq!(
            http.request(0).url,
            "https://api.test/model?self=true&page_size=100&page_number=1"
        );
        assert!(http.request(1).url.ends_with("&page_number=2"));
        assert_eq!(
            voices,
            [
                VoiceInfo::new("m1", "A", VoiceOrigin::Account)
                    .with_preview(Media::from_url("audio/wav", "https://cdn.test/m1.wav")),
                // A path a self-hosted server does not serve is no preview
                VoiceInfo::new("m2", "B", VoiceOrigin::Account).with_description("d"),
                VoiceInfo::new("voice-1", "Ala", VoiceOrigin::Store)
                    .with_preview(Media::from_bytes("audio/wav", b"WAV")),
            ]
        );
    }

    #[tokio::test]
    async fn design_voice_posts_the_instruction() {
        let enc = |b: &[u8]| agentai_core::base64::encode(b);
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"candidates": [
                {"id": "c1", "index": 0, "audio_base64": enc(b"RIFF\0\0\0\0WAVEfmt "),
                 "sample_rate": 44100, "duration_ms": 900, "text": "Hello"},
                {"id": "c2", "index": 1, "audio_base64": enc(b"ID3\x04rest"),
                 "sample_rate": 44100, "duration_ms": 800}
            ]}),
        )]);
        let previews = SpeechGenerator::from_provider(designer(&http))
            .design_voice("warm low voice")
            .with_text("Hello")
            .with_count(2)
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/voice-design");
        assert_eq!(req.header("model"), Some("voice-design-1"));
        assert_eq!(req.header("authorization"), Some("Bearer key"));
        assert_eq!(
            body_json(&req),
            json!({"instruction": "warm low voice", "reference_text": "Hello", "n": 2})
        );
        assert_eq!(previews[0].audio.content_type, "audio/wav");
        assert_eq!(
            previews[0].audio.bytes(&http).await.unwrap(),
            b"RIFF\0\0\0\0WAVEfmt "
        );
        assert_eq!(previews[0].text.as_deref(), Some("Hello"));
        assert_eq!(previews[0].sample_rate, Some(44_100));
        assert_eq!(previews[0].sample_id, None);
        assert_eq!(previews[1].audio.content_type, "audio/mpeg");
        assert_eq!(previews[1].text.as_deref(), Some("Hello"));
    }

    #[tokio::test]
    async fn design_voice_detects_the_audio_type() {
        let enc = |b: &[u8]| agentai_core::base64::encode(b);
        let http = MockHttp::new(vec![json_response(
            200,
            json!({"candidates": [
                {"audio_base64": enc(b"\xff\xfb\x90\x00"), "sample_rate": 24000},
                {"audio_base64": enc(b"OggS\0\0\0\0\0\0\0\0"), "sample_rate": 24000}
            ]}),
        )]);
        let previews = SpeechGenerator::from_provider(designer(&http))
            .design_voice("x")
            .await
            .unwrap();
        assert_eq!(http.request(0).header("model"), Some("voice-design-1"));
        assert_eq!(
            body_json(&http.request(0)),
            json!({"instruction": "x", "reference_text": null, "n": 1})
        );
        assert_eq!(previews[0].audio.content_type, "audio/mpeg");
        assert_eq!(previews[1].audio.content_type, "audio/wav");
        assert_eq!(previews[1].sample_rate, Some(24_000));
        assert_eq!(previews[1].text, None);
    }

    #[tokio::test]
    async fn a_dialogue_of_stored_voices_sends_only_their_samples() {
        let http = MockHttp::new(vec![audio()]);
        SpeechGenerator::from_provider(
            dialogue_provider(&http).with_voice_store(store_with_two().await),
        )
        .generate_dialogue([
            DialogueTurn::new("voice-1", "Hi"),
            DialogueTurn::new("voice-2", "Hello"),
            DialogueTurn::new("voice-1", "Bye"),
        ])
        .await
        .unwrap();
        assert_eq!(
            msgpack(&http.request(0)),
            json!({
                "text": "<|speaker:0|>Hi<|speaker:1|>Hello<|speaker:0|>Bye",
                "references": [[{"audio": bytes(b"WAV1"), "text": "One."}],
                               [{"audio": bytes(b"WAV2"), "text": "Two."}]]
            })
        );
        assert_eq!(http.request(0).header("model"), Some("drama-3"));
    }

    #[tokio::test]
    async fn a_dialogue_of_model_ids_sends_only_reference_id() {
        let http = MockHttp::new(vec![audio()]);
        SpeechGenerator::from_provider(dialogue_provider(&http))
            .generate_dialogue([
                DialogueTurn::new("m-a", "Hi"),
                DialogueTurn::new("m-b", "Hello"),
            ])
            .with_speed(1.5)
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0)),
            json!({
                "text": "<|speaker:0|>Hi<|speaker:1|>Hello", "reference_id": ["m-a", "m-b"],
                "prosody": {"speed": 1.5}
            })
        );
    }

    #[tokio::test]
    async fn stored_and_model_voices_cannot_be_mixed() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(
            dialogue_provider(&http).with_voice_store(store_with_sample().await),
        )
        .generate_dialogue([
            DialogueTurn::new("voice-1", "Hi"),
            DialogueTurn::new("m-a", "Hello"),
        ])
        .await
        .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRequest(_)));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn a_dialogue_without_a_dialogue_model_sends_nothing() {
        let http = MockHttp::new(vec![]);
        let err = SpeechGenerator::from_provider(provider(&http))
            .generate_dialogue([DialogueTurn::new("m-a", "Hi")])
            .with_model("drama-3")
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::MissingModel("dialogue")));
        assert_eq!(http.request_count(), 0);
    }

    #[tokio::test]
    async fn one_stored_voice_with_timestamps_streams_its_sample() {
        let audio = agentai_core::base64::encode(b"AB");
        let body = format!(
            "data: {{\"audio_base64\": \"{audio}\", \"chunk_seq\": 0}}\n\ndata: [DONE]\n\n"
        );
        let http = MockHttp::new(vec![HttpResponse::new(200, body)]);
        SpeechGenerator::from_provider(provider(&http).with_voice_store(store_with_sample().await))
            .generate("hi")
            .with_voice("voice-1")
            .with_timestamps(true)
            .await
            .unwrap();
        assert_eq!(
            http.request(0).url,
            "https://api.test/v1/tts/stream/with-timestamp"
        );
        assert_eq!(
            msgpack(&http.request(0))["references"],
            json!([{"audio": bytes(b"WAV"), "text": "Hello there."}])
        );
    }

    #[tokio::test]
    async fn a_voice_missing_from_a_set_store_is_sent_as_its_id() {
        let http = MockHttp::new(vec![audio()]);
        SpeechGenerator::from_provider(provider(&http).with_voice_store(store_with_sample().await))
            .generate("hi")
            .with_voice("m-a")
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0)),
            json!({"text": "hi", "reference_id": "m-a"})
        );
    }
}
