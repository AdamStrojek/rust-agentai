//! # Fish Audio speech API
//!
//! [`FishAudioProvider`] speaks text with `POST /v1/tts`, creates voices in the Fish Audio
//! account with `POST /model` and lists them with `GET /model`. The model (`s1`, `s2-pro`,
//! `drama-3`, ...) goes in the `model` header.
//!
//! A request may have several voices ([`SpeechRequest::with_voices`]), for models that speak a
//! dialogue (`drama-3` and the S2 family). Their order is the speaker index, and the text marks
//! who speaks with `<|speaker:0|>`, `<|speaker:1|>`, ...; writing those tags is up to you.
//!
//! A voice is the ID of a Fish Audio voice model or, with a [`VoiceStore`] set with
//! [`with_voice_store`](FishAudioProvider::with_voice_store), the ID of a voice in that store:
//! its sample and transcript are then sent with the request (`references`, as MessagePack).
//!
//! With [`SpeechRequest::with_timestamps`] the request goes to
//! `POST /v1/tts/stream/with-timestamp`: the streamed audio chunks are joined and the time of
//! every word is returned in [`SpeechResponse::timestamps`].
//!
//! Fields without an equivalent are not sent: `instructions` is ignored.

use super::{
    AudioError, AudioFormat, SpeechProvider, SpeechRequest, SpeechResponse, Voice, VoiceInfo,
    WordTimestamp, audio_media, extension_for,
};
use crate::voice_store::{StoredVoice, VoiceStore};
use agentai_core::message::Media;
use agentai_core::{HttpClient, HttpRequest, Multipart, ReqwestClient};
use async_trait::async_trait;
use base64::Engine;
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
    model: String,
    api_key: String,
    base_url: String,
    client: Arc<dyn HttpClient>,
    store: Option<Arc<dyn VoiceStore>>,
}

impl std::fmt::Debug for FishAudioProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishAudioProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl FishAudioProvider {
    /// Creates a provider for `model` with the API key and base URL given in code, for example
    /// [`DEFAULT_BASE_URL`]. An empty key sends no `Authorization` header, for self-hosted
    /// servers without authentication.
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

    /// Creates a provider for `model` with the API key from `FISH_API_KEY` and the base URL
    /// from `FISH_BASE_URL`, or [`DEFAULT_BASE_URL`] when it is not set. The key is optional
    /// when `FISH_BASE_URL` is set, for self-hosted servers.
    pub fn from_env(model: impl Into<String>) -> Result<Self, AudioError> {
        Self::from_lookup(model, |name| std::env::var(name).ok())
    }

    fn from_lookup(
        model: impl Into<String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, AudioError> {
        let (api_key, base_url) = match (lookup(API_KEY_ENV), lookup(BASE_URL_ENV)) {
            (Some(key), base_url) => (key, base_url),
            (None, Some(base_url)) => (String::new(), Some(base_url)),
            (None, None) => return Err(AudioError::MissingApiKey(API_KEY_ENV)),
        };
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(model, api_key, base_url))
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

    /// The voices of a request: its voices, else the default voice.
    async fn speakers(&self, request: &SpeechRequest) -> Result<Vec<Speaker>, AudioError> {
        let voices = match (&request.voices[..], &request.default_voice) {
            ([], Some(voice)) => std::slice::from_ref(voice),
            (voices, _) => voices,
        };
        let mut speakers = Vec::with_capacity(voices.len());
        for voice in voices {
            let id = match voice {
                Voice::Id(id) | Voice::Auto(id) => id,
                Voice::Description(_) => {
                    return Err(AudioError::Unsupported(
                        "voice descriptions with Fish Audio",
                    ));
                }
            };
            let stored = match &self.store {
                Some(store) => store.get(id).await?,
                None => None,
            };
            let reference = match stored {
                Some(voice) => Some(self.reference(voice).await?),
                None => None,
            };
            speakers.push(Speaker {
                id: id.clone(),
                reference,
            });
        }
        Ok(speakers)
    }

    async fn reference(&self, voice: StoredVoice) -> Result<Reference, AudioError> {
        Ok(Reference {
            audio: voice.sample.bytes(&*self.client).await?,
            text: voice.transcript,
        })
    }
}

/// A voice of a request: the ID and, for a voice of the store, its sample.
struct Speaker {
    id: String,
    reference: Option<Reference>,
}

#[derive(Serialize)]
struct Reference {
    #[serde(with = "serde_bytes")]
    audio: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ReferenceId {
    One(String),
    Many(Vec<String>),
}

#[derive(Serialize)]
#[serde(untagged)]
enum References {
    One(Vec<Reference>),
    Many(Vec<Vec<Reference>>),
}

#[derive(Serialize)]
struct Prosody {
    speed: f64,
}

#[derive(Serialize)]
struct SpeechBody<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_id: Option<ReferenceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    references: Option<References>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prosody: Option<Prosody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

/// `reference_id` and `references` for the voices of a request.
fn voice_fields(mut speakers: Vec<Speaker>) -> (Option<ReferenceId>, Option<References>) {
    match speakers.len() {
        0 => (None, None),
        1 => {
            let speaker = speakers.remove(0);
            match speaker.reference {
                Some(reference) => (None, Some(References::One(vec![reference]))),
                None => (Some(ReferenceId::One(speaker.id)), None),
            }
        }
        _ => {
            let has_samples = speakers.iter().any(|s| s.reference.is_some());
            let mut ids = Vec::with_capacity(speakers.len());
            let mut references = Vec::with_capacity(speakers.len());
            for speaker in speakers {
                ids.push(speaker.id);
                references.push(speaker.reference.into_iter().collect());
            }
            let references = has_samples.then_some(References::Many(references));
            (Some(ReferenceId::Many(ids)), references)
        }
    }
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
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse, AudioError> {
        let format = request.format.map(format_name).transpose()?;
        let (reference_id, references) = voice_fields(self.speakers(&request).await?);
        let inline = references.is_some();
        let body = SpeechBody {
            text: &request.text,
            reference_id,
            references,
            format,
            prosody: request.speed.map(|speed| Prosody { speed }),
            language: request.language.as_deref(),
        };
        let path = if request.timestamps {
            "/v1/tts/stream/with-timestamp"
        } else {
            "/v1/tts"
        };
        let mut http = HttpRequest::post(format!("{}{path}", self.base_url));
        if !self.model.is_empty() {
            http = http.with_header("model", self.model.as_str());
        }
        // Samples are sent as binary, which only the MessagePack body takes
        http = if inline {
            let bytes = rmp_serde::to_vec_named(&body)
                .map_err(|e| AudioError::InvalidRequest(e.to_string()))?;
            http.with_header("Content-Type", "application/msgpack")
                .with_body(bytes)
        } else {
            http.with_json(&body)?
        };
        let response = self
            .client
            .send(self.request(http))
            .await?
            .error_for_status()?;
        let format = request.format.unwrap_or(AudioFormat::Mp3);
        let (mut audio, timestamps) = if request.timestamps {
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
        audio.name = request.name;
        let mut result = SpeechResponse::new(audio);
        if format == AudioFormat::Pcm {
            result = result.with_sample_rate(PCM_SAMPLE_RATE);
        }
        if let Some(timestamps) = timestamps {
            result = result.with_timestamps(timestamps);
        }
        Ok(result)
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
        samples: &[Media],
        transcript: Option<&str>,
        description: Option<&str>,
    ) -> Result<String, AudioError> {
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
            let filename = sample.name.clone().unwrap_or_else(|| {
                format!(
                    "sample-{}.{}",
                    index + 1,
                    extension_for(&sample.content_type)
                )
            });
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
                let info = VoiceInfo::new(model.id, model.title).with_cloned(true);
                match model.description.filter(|d| !d.is_empty()) {
                    Some(description) => info.with_description(description),
                    None => info,
                }
            }));
            let done = empty
                || page.has_more == Some(false)
                || page.total.is_some_and(|total| voices.len() >= total);
            if done {
                break;
            }
        }
        if let Some(store) = &self.store {
            voices.extend(store.list().await?.into_iter().map(|(id, voice)| {
                let mut info = VoiceInfo::new(id, voice.name).with_cloned(true);
                info.description = voice.description;
                info
            }));
        }
        Ok(voices)
    }
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
            base64::engine::general_purpose::STANDARD
                .decode(&event.audio_base64)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryVoiceStore;
    use crate::provider::test_support::{MockHttp, body_json, json_response};
    use agentai_core::HttpResponse;
    use serde_json::{Value, json};

    fn provider(http: &Arc<MockHttp>) -> FishAudioProvider {
        FishAudioProvider::new("drama-3", "key", "https://api.test/").with_client(http.clone())
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
    async fn generate_sends_one_model_voice_as_json() {
        let http = MockHttp::new(vec![audio()]);
        let r = provider(&http)
            .generate(
                SpeechRequest::new("hi")
                    .with_voice(Voice::id("abc"))
                    .with_format(AudioFormat::Pcm)
                    .with_speed(1.2)
                    .with_language("pl")
                    .with_instructions("ignored")
                    .with_name("hi.pcm"),
            )
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
    }

    #[tokio::test]
    async fn timestamps_join_the_audio_and_offset_the_words() {
        let event = |audio: &[u8], seq: u64, offset: f64, alignment: Value| {
            let audio = base64::engine::general_purpose::STANDARD.encode(audio);
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
        let r = provider(&http)
            .generate(
                SpeechRequest::new("Hi there. Bye")
                    .with_voice(Voice::id("abc"))
                    .with_format(AudioFormat::Opus)
                    .with_timestamps(true),
            )
            .await
            .unwrap();
        let req = http.request(0);
        assert_eq!(req.url, "https://api.test/v1/tts/stream/with-timestamp");
        assert_eq!(body_json(&req)["reference_id"], "abc");
        assert_eq!(r.audio, Media::from_bytes("audio/ogg", b"ABCDEF"));
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
        let err = provider(&http)
            .generate(SpeechRequest::new("hi").with_timestamps(true))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, AudioError::InvalidResponse(m) if m.contains("server overloaded")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn the_default_voice_is_used_without_voices() {
        let http = MockHttp::new(vec![audio(), audio()]);
        let p = provider(&http);
        p.generate(SpeechRequest::new("hi").with_default_voice(Voice::id("d")))
            .await
            .unwrap();
        assert_eq!(body_json(&http.request(0))["reference_id"], "d");
        p.generate(SpeechRequest::new("hi")).await.unwrap();
        assert_eq!(body_json(&http.request(1)), json!({"text": "hi"}));
    }

    #[tokio::test]
    async fn several_model_voices_are_a_list_in_order() {
        let http = MockHttp::new(vec![audio()]);
        provider(&http)
            .generate(
                SpeechRequest::new("<|speaker:0|>Hi<|speaker:1|>Hello")
                    .with_default_voice(Voice::id("unused"))
                    .with_voices(vec![Voice::id("b"), Voice::auto("a")]),
            )
            .await
            .unwrap();
        assert_eq!(
            body_json(&http.request(0)),
            json!({"text": "<|speaker:0|>Hi<|speaker:1|>Hello", "reference_id": ["b", "a"]})
        );
    }

    #[tokio::test]
    async fn one_stored_voice_is_sent_as_its_sample() {
        let http = MockHttp::new(vec![audio()]);
        provider(&http)
            .with_voice_store(store_with_sample().await)
            .generate(SpeechRequest::new("hi").with_voice(Voice::id("voice-1")))
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
    async fn stored_and_model_voices_are_one_speaker_each() {
        let http = MockHttp::new(vec![audio()]);
        provider(&http)
            .with_voice_store(store_with_sample().await)
            .generate(
                SpeechRequest::new("dialogue")
                    .with_voice(Voice::id("model-a"))
                    .with_voice(Voice::auto("voice-1")),
            )
            .await
            .unwrap();
        assert_eq!(
            msgpack(&http.request(0)),
            json!({
                "text": "dialogue",
                "reference_id": ["model-a", "voice-1"],
                "references": [[], [{"audio": bytes(b"WAV"), "text": "Hello there."}]]
            })
        );
    }

    #[tokio::test]
    async fn unsupported_voices_and_formats_send_nothing() {
        let http = MockHttp::new(vec![]);
        let p = provider(&http);
        let err = p
            .generate(SpeechRequest::new("hi").with_voice(Voice::description("warm")))
            .await
            .unwrap_err();
        assert!(matches!(err, AudioError::Unsupported(_)));
        for format in [AudioFormat::Flac, AudioFormat::Aac] {
            let err = p
                .generate(SpeechRequest::new("hi").with_format(format))
                .await
                .unwrap_err();
            assert!(matches!(err, AudioError::Unsupported(_)));
        }
        assert!(http.requests.lock().unwrap().is_empty());
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
            .create_voice("Adam", &samples, Some("Said text."), Some("my voice"))
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
                       "items": [{"_id": "m1", "title": "A", "description": ""}]}),
            ),
            json_response(
                200,
                json!({"total": 2, "items": [{"_id": "m2", "title": "B", "description": "d"}]}),
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
                VoiceInfo::new("m1", "A").with_cloned(true),
                VoiceInfo::new("m2", "B")
                    .with_description("d")
                    .with_cloned(true),
                VoiceInfo::new("voice-1", "Ala").with_cloned(true),
            ]
        );
    }
}
