# agentai-audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `agentai-audio` crate: text-to-speech and voice cloning on OpenAI, ElevenLabs and oMLX, usable from application code and as agent tools.

**Architecture:** Mirrors `crates/agentai-image`: a `SpeechProvider` trait with one module per API, a `SpeechGenerator` that applies defaults and saves results, and a `SpeechToolBox` on `#[toolbox]`. oMLX has no server-side voices, so its adapter keeps samples in an explicit `VoiceStore`.

**Tech Stack:** Rust (edition 2024, rust-version 1.88, as `agentai-image`), `agentai-core` HTTP and storage, `async-trait`, `serde`, `thiserror`, `base64`, `schemars` and `agentai` behind the `toolbox` feature.

**Spec:** `docs/superpowers/specs/2026-11-09-agentai-audio-design.md`

## Global Constraints

- `cargo fmt --check`, `cargo clippy -- -D warnings` and `cargo test -p agentai-audio` pass after every task.
- `cargo tree -i aws-lc-rs` finds nothing. No new TLS-enabling dependency.
- Runtime-neutral outside tests and examples: no tokio, clock or randomness.
- No genai types. `thiserror` enums only; `anyhow` is a dev-dependency.
- Every setter is `with_*`, getters are bare names. Public structs and enums are `#[non_exhaustive]`.
- The library invents no model parameters: unset optional fields are not sent.
- No tool may reach `create_voice`.
- Tests follow the "Tests" section of `CLAUDE.md`: no tests of setters, constants, serde derives or other crates.
- Files, comments and commits in English. Commit format `<type>(audio): <message>`, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never print values from `.env`; show variable names only.

## Review Focus

- An LLM-chosen `voice` such as `../../v1/voices/x` on ElevenLabs goes into a URL path: nothing is sent and the model gets an error (Task 4).
- An LLM sends `voice: ""` or only spaces: treated as no voice, so the default applies (Task 7).
- A server answers 200 with an empty body: `InvalidResponse`, and nothing is saved in storage (Task 2 helper, Task 6 test).
- A name with a dot that is not an extension (`intro.v2`) still gets one (`intro.v2.mp3`), while `intro.wav` is left alone (Task 6).
- A clone sample given as a URL on oMLX is downloaded once, at `create_voice`; a failed download stores nothing (Task 5).

---

### Task 1: Prove oMLX can do this

No code. Everything oMLX-specific rests on this, and spec Verification says to test it first.

**Files:** none. Scratch files go in the session scratchpad.

**Interfaces:**
- Produces: the value of `REF_TEXT_PLACEHOLDER` for Task 5.

- [ ] **Step 1: Generate plain speech**

From the repo root that holds `.env` (`/Users/adam/Projects/ai-agents/rust-agentai`): `set -a && source .env && set +a`, then `curl -sS -o plain.wav -w '%{http_code}\n' -H "Authorization: Bearer $OMLX_API_KEY" -H 'Content-Type: application/json' -d '{"model":"mlx-community/VoxCPM2-8bit","input":"Testing one two three."}' "${OMLX_ENDPOINT%/}/audio/speech"`.
Expected: `200` and `file plain.wav` reports WAV audio. The first call may take minutes while the model downloads.

- [ ] **Step 2: Clone with a placeholder transcript**

Make a sample with `say -o sample.aiff "This is a short reference recording of my voice" && ffmpeg -y -i sample.aiff sample.wav`. POST the same body plus `"ref_audio": "<base64 -i sample.wav>"` and `"ref_text": "."`.
Expected: `200` and WAV audio. If `"."` gets a 400, try `"-"`, then a sentence (`"Reference voice sample."`). The first accepted value is `REF_TEXT_PLACEHOLDER`.

- [ ] **Step 3: Stop or continue**

If Step 1 or 2 fails with a model-loading error (jundot/omlx#1305) or any other server fault, stop and report it to your human partner with the status and response body. Do not start Task 2. Otherwise write the accepted placeholder in your task report.

---

### Task 2: Crate, core types and HTTP helpers

**Files:**
- Create: `crates/agentai-audio/{Cargo.toml,README.md,src/lib.rs,src/provider/mod.rs,src/provider/wire.rs}`
- Modify: `release-plz.toml` (add `agentai-audio` to `changelog_include`; add a `[[package]]` block with `version_group = "agentai"`, `changelog_update = false`)
- Test: inline in `src/provider/wire.rs`

**Interfaces:**
- Produces, in `provider/mod.rs` (each struct `new` + `with_*`):
  - `Voice::{Id(String), Description(String), Auto(String)}`, constructors `Voice::id`, `Voice::description`, `Voice::auto` taking `impl Into<String>`.
  - `AudioFormat::{Mp3, Wav, Opus, Flac, Aac, Pcm}`, `fn content_type(self) -> &'static str`: `audio/mpeg`, `audio/wav`, `audio/ogg`, `audio/flac`, `audio/aac`, `audio/pcm`. `Copy`.
  - `SpeechRequest { text, voice: Option<Voice>, default_voice: Option<Voice>, format: Option<AudioFormat>, speed: Option<f64>, language: Option<String>, instructions: Option<String>, name: Option<String> }`, `new(text)`.
  - `SpeechResponse { audio: Media, id: Option<String>, sample_rate: Option<u32> }`, `new(audio)`.
  - `CreateVoice { name, samples: Vec<Media>, transcript: Option<String>, description: Option<String> }`, `new(name, samples)`.
  - `VoiceInfo { id, name, description: Option<String>, cloned: bool }` (`Serialize`), `new(id, name)`.
  - `AudioModel` with the fields of `ImageModel`.
  - `AudioError`: the variants of `ImageError` (`crates/agentai-image/src/provider/mod.rs:181`), message text saying "audio" instead of "image", plus `MissingVoice` and `InvalidRequest(String)`; `is_auth()`.
  - `SpeechProvider` exactly as in the spec. Default `create_voice` returns `Unsupported("voice cloning")`, `list_voices` `Unsupported("listing voices")`, `list_models` `Unsupported("listing models")`.
- Produces, in `provider/wire.rs` (`pub(super)`):
  - `fn with_bearer(request: HttpRequest, api_key: &str) -> HttpRequest` (no header for an empty key)
  - `async fn send(client: &dyn HttpClient, request: HttpRequest) -> Result<HttpResponse, AudioError>` (non-2xx becomes `AudioError::Api`)
  - `fn audio_media(response: &HttpResponse, content_type: &str) -> Result<Media, AudioError>`
  - `async fn media_bytes(client: &dyn HttpClient, media: &Media) -> Result<Vec<u8>, AudioError>` (decode base64, or GET a URL)
  - `fn percent_encode(value: &str) -> String`
- Produces, `#[cfg(test)] pub(crate) mod test_support` in `provider/mod.rs`: `MockHttp` and `json_response`, copied from `crates/agentai-image/src/provider/openai.rs:325-358`, plus `body_json(&HttpRequest) -> Value`.

- [ ] **Step 1: Create `Cargo.toml` and the type skeleton so the crate builds**

Copy `crates/agentai-image/Cargo.toml`: name, description ("Text-to-speech and voice cloning for AgentAI agents."), categories `["multimedia::audio"]`, keywords `["generative-ai", "agent", "text-to-speech", "voice-cloning"]`. Same `[features]` and comments, naming `SpeechToolBox` and `SpeechGenerator`. No `[[example]]` yet. Root `lib.rs` declares `pub mod provider;`, re-exports `agentai_core::storage` (`#[doc(inline)]`) and `AudioError`, `SpeechResponse`, `Voice`, `AudioFormat`, `CreateVoice`, `VoiceInfo`, `AudioModel`, `SpeechRequest`. Declare the types and trait from Interfaces.

Run: `cargo build -p agentai-audio`. Expected: success.

- [ ] **Step 2: Write the failing tests in `wire.rs`**

```rust
#[test]
fn api_error_reads_every_error_shape() {
    let message = |body: Value| match api_error(&json_response(400, body)) {
        AudioError::Api { message, .. } => message,
        other => panic!("{other:?}"),
    };
    assert_eq!(message(json!({"error": {"message": "openai"}})), "openai");
    assert_eq!(message(json!({"detail": "omlx"})), "omlx");
    assert_eq!(message(json!({"detail": {"message": "eleven"}})), "eleven");
    assert_eq!(message(json!({"detail": [{"msg": "validation"}]})), "validation");
    assert_eq!(message(json!({"other": 1})), r#"{"other":1}"#);
}

#[test]
fn empty_audio_body_is_invalid() {
    let empty = HttpResponse::new(200, Vec::new());
    assert!(matches!(audio_media(&empty, "audio/mpeg"), Err(AudioError::InvalidResponse(_))));
    let full = HttpResponse::new(200, b"hi".to_vec());
    assert_eq!(audio_media(&full, "audio/mpeg").unwrap(), Media::from_bytes("audio/mpeg", b"hi"));
}

#[test]
fn percent_encode_escapes_everything_but_unreserved() {
    assert_eq!(percent_encode("a-b_c.d~1"), "a-b_c.d~1");
    assert_eq!(percent_encode("a+b/c=d e"), "a%2Bb%2Fc%3Dd%20e");
}
```

- [ ] **Step 3: Run them**

Run: `cargo test -p agentai-audio wire`. Expected: FAIL, functions not implemented.

- [ ] **Step 4: Implement `wire.rs`**

`api_error` (private) tries, in order: `error.message`, `detail` as string, `detail.message`, `detail[0].msg`, then `response.text()`.

- [ ] **Step 5: Run and commit**

Run: `cargo test -p agentai-audio && cargo clippy -p agentai-audio -- -D warnings`. Expected: PASS.

```bash
git add crates/agentai-audio release-plz.toml Cargo.lock
git commit -m "feat(audio): add the agentai-audio crate with its provider types"
```

---

### Task 3: OpenAI provider

**Files:**
- Create: `crates/agentai-audio/src/provider/openai.rs`
- Modify: `crates/agentai-audio/src/provider/mod.rs` (`pub mod openai;`)

**Interfaces:**
- Consumes: Task 2.
- Produces: `openai::OpenAiSpeechProvider` with `new(model, api_key, base_url)`, `from_env(model) -> Result<Self, AudioError>`; env rules identical to `OpenAiImageProvider::from_lookup` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, key optional when the base URL is set); `#[cfg(test)] with_client`; constants `DEFAULT_BASE_URL`, `API_KEY_ENV`, `BASE_URL_ENV`.

- [ ] **Step 1: Write the failing tests**

Helpers: `official(&http)` builds with `DEFAULT_BASE_URL` and model `gpt-4o-mini-tts`; `tts1(&http)` with model `tts-1`; `compatible(&http)` with base URL `http://local/v1`. Each reply is `HttpResponse::new(200, b"audio".to_vec())`.

```rust
// generate_sends_only_what_is_set
let r = official(&http).generate(SpeechRequest::new("hi").with_voice(Voice::id("alloy"))).await.unwrap();
assert_eq!(http.request(0).url, "https://api.openai.com/v1/audio/speech");
assert_eq!(body_json(&http.request(0)), json!({"model": "gpt-4o-mini-tts", "input": "hi", "voice": "alloy"}));
assert_eq!(r.audio.content_type, "audio/mpeg");
assert_eq!(r.sample_rate, None);

// format_and_pcm_rate
let r = official(&http).generate(req.with_format(AudioFormat::Pcm)).await.unwrap();
assert_eq!(body_json(&http.request(0))["response_format"], json!("pcm"));
assert_eq!(r.audio.content_type, "audio/pcm");
assert_eq!(r.sample_rate, Some(24000));

// custom_voice_id_is_an_object
// Voice::id("voice_123") → body["voice"] == json!({"id": "voice_123"})

// auto_routes_presets_ids_and_descriptions
// Voice::auto("Alloy") → body["voice"] == json!("alloy")          (presets match in any case)
// Voice::auto("voice_9") → body["voice"] == json!({"id": "voice_9"})
// Voice::auto("warm low voice") + with_default_voice(Voice::id("onyx"))
//   → body["voice"] == json!("onyx"), body["instructions"] == json!("warm low voice")

// a_description_is_joined_with_instructions
// Voice::description("warm") + with_instructions("slowly") + default onyx
//   → body["instructions"] == json!("warm\nslowly")

// missing_voice_cases, nothing sent (http.requests is empty)
// no voice, no default → AudioError::MissingVoice
// Voice::description("warm"), no default → AudioError::MissingVoice

// default_voice_is_used_without_a_voice
// with_default_voice(Voice::id("nova")) → body["voice"] == json!("nova")

// tts1_rejects_descriptions
// tts1: Voice::description("warm") + default → AudioError::Unsupported(_), nothing sent
assert!(!tts1(&http).supports_voice_description());
assert!(official(&http).supports_voice_description());

// compatible_servers_take_auto_as_an_id
// compatible: Voice::auto("af_bella") → body["voice"] == json!("af_bella"), url "http://local/v1/audio/speech"

// voices_are_the_presets_of_the_model
assert_eq!(tts1(&http).list_voices().await.unwrap().len(), 9);
assert_eq!(official(&http).list_voices().await.unwrap().len(), 13);
// compatible: list_voices → AudioError::Unsupported(_)

// models_are_filtered_on_the_official_api
// GET /models with ids gpt-4o, tts-1, gpt-4o-mini-tts → models "openai::tts-1", "openai::gpt-4o-mini-tts"
```

- [ ] **Step 2: Run them**

Run: `cargo test -p agentai-audio openai`. Expected: FAIL (module missing items).

- [ ] **Step 3: Implement `OpenAiSpeechProvider`**

Body struct `{model, input, voice (untagged enum: Name(String) | Custom{id}), instructions?, response_format?, speed?}`; `language` and `name` are not sent. Format names are lowercase variant names. Default format Mp3 (`response_format` omitted, content type `audio/mpeg`). `sample_rate` is `Some(24000)` only for Pcm.

Presets, in `tts-1*` order: `alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer`; other models add `ballad, verse, marin, cedar`. `list_voices` is `Unsupported` off the official base URL.

Voice resolution: `Auto` on a non-official base URL is an Id. On the official URL, a case-insensitive preset match is sent lowercase, a `voice_` prefix is an Id, anything else is a Description. A description needs `default_voice` resolving to an Id or preset, otherwise `MissingVoice`. Plain `instructions` on `tts-1*` are sent as given: the API reports what the model does not accept.

`create_voice` keeps the trait default. `list_models` filters to `tts-1*` and `gpt-4o-mini-tts*` on the official URL, as in `crates/agentai-image/src/provider/openai.rs:178`.

- [ ] **Step 4: Run and commit**

Run: `cargo test -p agentai-audio openai`. Expected: PASS.

```bash
git add crates/agentai-audio
git commit -m "feat(audio): generate speech with the OpenAI API"
```

---

### Task 4: ElevenLabs provider

**Files:**
- Create: `crates/agentai-audio/src/provider/elevenlabs.rs`
- Modify: `crates/agentai-audio/src/provider/mod.rs` (`pub mod elevenlabs;`)

**Interfaces:**
- Consumes: Task 2.
- Produces: `elevenlabs::ElevenLabsProvider` with `new(model, api_key, base_url)`, `from_env(model)` (requires `ELEVENLABS_API_KEY`; `ELEVENLABS_BASE_URL` optional), `with_output_format(impl Into<String>)`; `DEFAULT_BASE_URL = "https://api.elevenlabs.io"`.

- [ ] **Step 1: Write the failing tests**

```rust
// generate_posts_to_the_voice_path
let r = provider(&http).generate(
    SpeechRequest::new("hi").with_voice(Voice::id("abc123")).with_language("pl").with_speed(1.1)
).await.unwrap();
let req = http.request(0);
assert_eq!(req.url, "https://api.test/v1/text-to-speech/abc123");
assert_eq!(req.header("xi-api-key"), Some("key"));
assert_eq!(body_json(&req), json!({
    "text": "hi", "model_id": "eleven_v4", "language_code": "pl", "voice_settings": {"speed": 1.1}
}));
assert_eq!(r.audio.content_type, "audio/mpeg");

// formats_map_to_output_format
// Pcm  → url ends "?output_format=pcm_24000", content type audio/pcm, sample_rate Some(24000)
// Mp3  → "?output_format=mp3_44100_128"
// Wav  → "?output_format=wav_44100"
// Opus → "?output_format=opus_48000_128", content type audio/ogg
// Flac → AudioError::Unsupported(_), nothing sent
// Aac  → AudioError::Unsupported(_), nothing sent

// the_output_format_override_wins_and_sets_the_type
// with_output_format("pcm_16000") + AudioFormat::Mp3 → "?output_format=pcm_16000", audio/pcm, Some(16000)

// unsafe_voice_ids_are_rejected_before_sending
for id in ["../../v1/voices/x", "a/b", "a b", "..", ""] {
    let err = provider(&http).generate(SpeechRequest::new("hi").with_voice(Voice::auto(id))).await.unwrap_err();
    assert!(matches!(err, AudioError::InvalidRequest(_)));
}
assert!(http.requests.lock().unwrap().is_empty());

// descriptions_and_missing_voices
// Voice::description("warm") → AudioError::Unsupported(_); no voice, no default → MissingVoice
assert!(!provider(&http).supports_voice_description());

// create_voice_uploads_every_sample
// two samples: Media::from_bytes("audio/wav", b"ONE").with_name("a.wav"), Media::from_bytes("audio/mpeg", b"TWO")
// reply {"voice_id": "new1", "requires_verification": false} → returns "new1"
// url "https://api.test/v1/voices/add", header xi-api-key present
// multipart has field name=Adam, description when set, and two file parts named "files":
//   filename a.wav / audio/wav / ONE, and sample-2.mp3 / audio/mpeg / TWO
// no samples → AudioError::InvalidRequest(_), nothing sent

// list_voices_follows_the_token
// page 1 {"voices": [{"voice_id": "v1", "name": "A", "category": "cloned", "description": "d"}],
//         "has_more": true, "next_page_token": "a+b"}
// page 2 {"voices": [{"voice_id": "v2", "name": "B", "category": "premade"}], "has_more": false}
// request 0 url "https://api.test/v2/voices?page_size=100"
// request 1 url "https://api.test/v2/voices?page_size=100&next_page_token=a%2Bb"
// result: [VoiceInfo{v1, A, Some("d"), cloned: true}, VoiceInfo{v2, B, None, cloned: false}]

// models_keep_only_text_to_speech
// [{"model_id":"eleven_v4","name":"V4","can_do_text_to_speech":true},
//  {"model_id":"scribe","can_do_text_to_speech":false}] → one model "elevenlabs::eleven_v4", id "eleven_v4"
```

Parse a multipart body as the `edit_uploads_every_image_as_multipart` test does.

- [ ] **Step 2: Run them**

Run: `cargo test -p agentai-audio elevenlabs`. Expected: FAIL.

- [ ] **Step 3: Implement `ElevenLabsProvider`**

Voice IDs must match `^[A-Za-z0-9_-]+$` (check by hand, no regex dependency), else `InvalidRequest`. `Id` and `Auto` are both the path segment; with no voice, use `default_voice` when it is Id or Auto, otherwise `MissingVoice`. No `output_format` query when neither the request nor the override sets one (content type `audio/mpeg`).

Content type of an override comes from its prefix: `mp3` → `audio/mpeg`, `pcm` → `audio/pcm`, `wav` → `audio/wav`, `opus` → `audio/ogg`, `ulaw` → `audio/basic`, `alaw` → `audio/x-alaw-basic`, otherwise `application/octet-stream`. `sample_rate` is the number after `pcm_`.

The upload field is `files`, repeated once per sample: ElevenLabs' own SDK sends that name. The spec's `files[]` was notation for "a list". Filename is `Media::name`, or `sample-{n}.{ext}` using a new `pub(crate) fn extension_for(content_type: &str) -> &str` in `provider/mod.rs` (`audio/mpeg` → `mp3`, `audio/wav` → `wav`, `audio/ogg` → `opus`, `audio/flac` → `flac`, `audio/aac` → `aac`, `audio/pcm` → `pcm`, else the MIME subtype).

Document on `create_voice` that `requires_verification: true` means the voice cannot speak until verified at ElevenLabs. `instructions` is not sent. `supports_voice_cloning` is true.

- [ ] **Step 4: Run and commit**

Run: `cargo test -p agentai-audio elevenlabs`. Expected: PASS.

```bash
git add crates/agentai-audio
git commit -m "feat(audio): generate speech and clone voices with ElevenLabs"
```

---

### Task 5: VoiceStore and oMLX provider

**Files:**
- Create: `crates/agentai-audio/src/voice_store.rs`, `crates/agentai-audio/src/provider/omlx.rs`
- Modify: `src/lib.rs` (`mod voice_store; pub use voice_store::{MemoryVoiceStore, StoredVoice, VoiceStore};`), `src/provider/mod.rs` (`pub mod omlx;`)

**Interfaces:**
- Consumes: Task 2; `REF_TEXT_PLACEHOLDER` value from Task 1.
- Produces:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StoredVoice { pub name: String, pub description: Option<String>, pub sample: Media, pub transcript: Option<String> }

#[async_trait]
pub trait VoiceStore: Send + Sync + 'static {
    async fn save(&self, voice: StoredVoice) -> Result<String, StorageError>;
    async fn get(&self, id: &str) -> Result<Option<StoredVoice>, StorageError>;
    async fn list(&self) -> Result<Vec<(String, StoredVoice)>, StorageError>;
}
```

`MemoryVoiceStore` (`Clone`, clones share data, IDs `voice-1`, `voice-2`, ...). `omlx::OmlxProvider` with `new(model, api_key, base_url)`, `from_env(model)` (`OMLX_ENDPOINT`, default `DEFAULT_BASE_URL = "http://localhost:8000/v1"`; `OMLX_API_KEY` optional), `with_voice_store(impl VoiceStore)`.

- [ ] **Step 1: Write the failing tests**

```rust
// generate_without_a_voice_sends_no_voice_field
// body == json!({"model": "m", "input": "hi"}), url "http://local/v1/audio/speech"
// response content type audio/wav, sample_rate None

// endpoint_keeps_v1 : from_lookup with OMLX_ENDPOINT "http://mini:8000/v1/" → url "http://mini:8000/v1/audio/speech"

// create_voice_then_speak_with_it
let store = MemoryVoiceStore::new();
let p = provider(&http).with_voice_store(store.clone());
assert!(p.supports_voice_cloning());
let id = p.create_voice(CreateVoice::new("Adam", vec![Media::from_bytes("audio/wav", b"WAV")]).with_transcript("hello")).await.unwrap();
assert_eq!(id, "voice-1");
assert!(http.requests.lock().unwrap().is_empty());
p.generate(SpeechRequest::new("hi").with_voice(Voice::id(&id))).await.unwrap();
let body = body_json(&http.request(0));
assert_eq!(body["ref_audio"], json!("V0FW"));
assert_eq!(body["ref_text"], json!("hello"));
assert!(body.get("voice").is_none());

// a_missing_transcript_sends_the_placeholder
// body["ref_text"] == json!(REF_TEXT_PLACEHOLDER)

// auto_is_a_clone_only_when_stored
// Voice::auto(stored id) → has ref_audio; Voice::auto("deep male voice") → body["instructions"] == json!("deep male voice"), no ref_audio

// description_joins_instructions : Voice::description("warm") + with_instructions("slowly") → "warm\nslowly"

// without_a_store
let p = provider(&http);
assert!(!p.supports_voice_cloning());
// create_voice, Voice::id("x"), list_voices → AudioError::Unsupported(_); Voice::auto("x") → body["instructions"] == json!("x")

// unknown_id_with_a_store : Voice::id("nope") → AudioError::InvalidRequest(_), nothing sent

// exactly_one_sample : 0 or 2 samples → AudioError::InvalidRequest(_), store still empty
// oversized_sample : Media::from_base64("audio/wav", "A".repeat(20 * 1024 * 1024 + 1)) → AudioError::InvalidRequest(_)

// url_samples_are_downloaded_once
// Media::from_url sample; replies: bytes "WAV" for the GET; store holds MediaSource::Base64("V0FW");
// generate does not repeat the GET (http.requests has GET then POST only)
// failed_download_stores_nothing : GET answers 404 → AudioError::Api{..}, store.list() is empty

// aac_is_unsupported : AudioFormat::Aac → AudioError::Unsupported(_), nothing sent
// list_voices_reads_the_store : [VoiceInfo{"voice-1","Adam",desc,cloned:true}]
assert!(provider(&http).supports_voice_description());
```

- [ ] **Step 2: Run them**

Run: `cargo test -p agentai-audio omlx`. Expected: FAIL.

- [ ] **Step 3: Implement `MemoryVoiceStore` and `OmlxProvider`**

`create_voice`: require exactly one sample, fetch bytes through `wire::media_bytes` for URLs, normalize to `Media::from_bytes`, reject over `MAX_REF_AUDIO_LEN = 20 * 1024 * 1024` encoded bytes with `InvalidRequest`, then `store.save`. `StorageError` converts to `AudioError::Storage`.

Body: `{model, input, response_format?, speed?, language?, instructions?, ref_audio?, ref_text?}`; `voice` is never sent. Default format Wav (`response_format` omitted). The `default_voice` is used when `voice` is `None`, through the same resolution. Module docs state that `speed`, `language` and samplers appear to have no effect on VoxCPM2, and next to `from_env` that no store is created, so cloning requires `with_voice_store`.

`const REF_TEXT_PLACEHOLDER: &str = "<value from Task 1>";`

`list_models` keeps the trait default.

- [ ] **Step 4: Run and commit**

Run: `cargo test -p agentai-audio omlx`. Expected: PASS.

```bash
git add crates/agentai-audio
git commit -m "feat(audio): generate speech and clone voices with oMLX"
```

---

### Task 6: Model routing and SpeechGenerator

**Files:**
- Create: `crates/agentai-audio/src/generator.rs`
- Modify: `src/provider/mod.rs` (`ProviderKind`, `resolve`, `from_model`, `list_models`), `src/lib.rs` (`pub use generator::SpeechGenerator;`)

**Interfaces:**
- Consumes: Tasks 2-5.
- Produces:
  - `provider::from_model(model: &str) -> Result<Box<dyn SpeechProvider>, AudioError>`, `provider::list_models(provider: &str) -> Result<Vec<AudioModel>, AudioError>`.
  - `SpeechGenerator` (`Clone`): `new(model: &str) -> Result<Self, AudioError>`, `from_provider(impl SpeechProvider)`, `with_storage(impl MediaSave)`, `with_default_voice(Voice)`, `with_format(AudioFormat)`, `with_speed(f64)`, `with_language(impl Into<String>)`, `model()`, `supports_voice_cloning()`, `supports_voice_description()`, `generate(impl Into<String>)`, `generate_with(SpeechRequest)`, `create_voice(CreateVoice) -> Result<String, AudioError>`, `list_voices()`, `list_models()`.

- [ ] **Step 1: Write the failing tests**

```rust
// provider/mod.rs: model_names_select_the_provider
assert_eq!(resolve("omlx::mlx-community/VoxCPM2-8bit").unwrap(), (ProviderKind::Omlx, "mlx-community/VoxCPM2-8bit"));
assert_eq!(resolve("elevenlabs::x").unwrap(), (ProviderKind::ElevenLabs, "x"));
assert_eq!(resolve("eleven_v4").unwrap(), (ProviderKind::ElevenLabs, "eleven_v4"));
assert_eq!(resolve("tts-1-hd").unwrap(), (ProviderKind::OpenAi, "tts-1-hd"));
assert_eq!(resolve("gpt-4o-mini-tts").unwrap(), (ProviderKind::OpenAi, "gpt-4o-mini-tts"));
assert!(matches!(resolve("gpt-4o"), Err(AudioError::UnknownProvider(_))));
assert!(matches!(resolve("nope::x"), Err(AudioError::UnknownProvider(_))));

// generator.rs: defaults_fill_unset_fields_and_the_request_wins
// generator with_default_voice(Voice::id("nova")), with_format(Wav), with_speed(1.5), with_language("pl")
// generate("a"): OpenAI body has voice "nova", response_format "wav", speed 1.5
// generate_with(SpeechRequest::new("b").with_voice(Voice::id("onyx")).with_format(AudioFormat::Mp3)):
//   body voice "onyx", response_format "mp3", speed 1.5

// the_name_gets_a_real_extension
// (name, content type) → stored Media::name
// ("intro",    audio/mpeg) → "intro.mp3"
// ("intro.v2", audio/mpeg) → "intro.v2.mp3"
// ("intro.wav",audio/mpeg) → "intro.wav"
// (none,       audio/mpeg) → "speech.mp3"

// the_storage_sets_the_id : MemoryStorage; response.id == Some("1-speech.mp3"), storage.read(id) == response.audio
// an_empty_reply_saves_nothing : 200 with empty body → AudioError::InvalidResponse(_); storage has no entries
```

- [ ] **Step 2: Run them**

Run: `cargo test -p agentai-audio`. Expected: FAIL (`resolve` and `SpeechGenerator` missing).

- [ ] **Step 3: Implement**

`ProviderKind::{OpenAi, ElevenLabs, Omlx}`, names `openai`, `elevenlabs`, `omlx`. Well-known names: `tts-1*`, `gpt-4o-mini-tts*`, `eleven_*`. `from_model` docs say an `omlx::` provider has no voice store and point to `OmlxProvider::with_voice_store`.

`generate_with` copies `default_voice`, `format`, `speed`, `language` into unset request fields (`request.default_voice` keeps its own value when set), takes `name`, calls the provider, then stores. A name keeps its text when its last `.suffix` is a known audio extension (`mp3`, `wav`, `opus`, `ogg`, `flac`, `aac`, `pcm`), otherwise `.{extension_for(content_type)}` is appended.

- [ ] **Step 4: Run and commit**

Run: `cargo test -p agentai-audio`. Expected: PASS.

```bash
git add crates/agentai-audio
git commit -m "feat(audio): pick providers from model names and add SpeechGenerator"
```

---

### Task 7: SpeechToolBox

**Files:**
- Create: `crates/agentai-audio/src/toolbox.rs`
- Modify: `src/lib.rs` (`#[cfg(feature = "toolbox")] mod toolbox;` and `pub use toolbox::{SpeechToolBox, SpeechToolOutput};`)

**Interfaces:**
- Consumes: `SpeechGenerator` (Task 6).
- Produces: `SpeechToolOutput::{Metadata (default), Media}`; `SpeechToolBox::new(generator: SpeechGenerator, storage: impl MediaStorage)`, `with_voices(Vec<VoiceInfo>)`, `with_output(SpeechToolOutput)`. Tools `generate_speech` and `list_voices`.

- [ ] **Step 1: Write the failing tests**

Use `call_tool(&ToolCall::new("call_1", name, args))` as `crates/agentai-image/src/toolbox.rs` tests do, with the OpenAI provider on `MockHttp`.

```rust
// generate_speech_returns_metadata_by_default
// args {"text":"hi","name":"greeting","voice":"alloy"}; output is one text part, JSON:
//   id == "1-greeting.mp3", content_type == "audio/mpeg", bytes == 5, model == "gpt-4o-mini-tts", voice == "alloy"
// no Part::Media in output.parts

// media_output_includes_the_audio : with_output(Media) → text contains "1-greeting.mp3" and a Part::Media

// a_blank_voice_means_the_default
// generator with default Voice::id("nova"); for voice in ["", "   "] and voice absent and null: body["voice"] == json!("nova")

// the_voice_list_is_an_allowlist
// with_voices(vec![VoiceInfo::new("voice_ok", "OK")]) on the official OpenAI provider, default Voice::id("nova")
// "voice_ok"     → body["voice"] == json!({"id": "voice_ok"})
// "voice_hidden" → body["voice"] == json!("nova"), body["instructions"] == json!("voice_hidden")

// an_unlisted_id_fails_without_descriptions
// ElevenLabs provider with with_voices([abc]); "hidden" → Err(ToolError::Failed(_)), no HTTP request

// instructions_reach_the_provider : "instructions":"slowly" → body["instructions"] == json!("slowly")

// provider_errors_are_recoverable_except_auth
// no voice, no default → ToolError::Failed(_); 401 reply → ToolError::Configuration(_)

// list_voices_prefers_the_app_list
// without with_voices: JSON voices has 13 entries, descriptions_supported == true
// with_voices([a]): voices has only a

// list_voices_without_support_is_empty_not_an_error
// OmlxProvider without a store: voices == [], descriptions_supported == true

// no_tool_can_clone
let names: Vec<_> = toolbox.tools_definitions()?.into_iter().map(|t| t.name).collect();
assert_eq!(names, ["generate_speech", "list_voices"]);
```

Omlx test needs `provider::omlx` `with_client` visible: mark it `pub(crate)` under `#[cfg(test)]`; same for the other providers.

- [ ] **Step 2: Run them**

Run: `cargo test -p agentai-audio toolbox`. Expected: FAIL.

- [ ] **Step 3: Implement on `#[toolbox]`**

`generate_speech(&self, text: String, name: String, voice: Option<String>, instructions: Option<String>) -> ToolResult<ToolOutput>`, `list_voices(&self) -> ToolResult<String>`. Parameter docs: `voice` is "A voice ID from `list_voices`, or, when `descriptions_supported` is true, a description of the voice such as `warm low female`."; the first line of `generate_speech` docs is "Generates speech from text and saves it as an audio file."

Voice mapping, after `trim`: empty → no voice. Without `with_voices` → `Voice::auto`. With the list: exact match on `VoiceInfo::id` → `Voice::id`; otherwise `Voice::description` if `supports_voice_description()`, else `ToolError::Failed` naming `list_voices`.

`tool_error` as in `crates/agentai-image/src/toolbox.rs:200`. Metadata struct: `id`, `content_type`, `bytes` (use `base64_len` logic from the image toolbox), `model`, `voice`, `sample_rate`, skipping `None`. `list_voices` turns `AudioError::Unsupported` into an empty list.

`#[toolbox]` supports `Option<T>` parameters (documented at `crates/agentai-macros/src/lib.rs:91`). If that proves false, stop and report the blocker.

- [ ] **Step 4: Run and commit**

Run: `cargo test -p agentai-audio && cargo clippy -p agentai-audio --all-features -- -D warnings`. Expected: PASS.

```bash
git add crates/agentai-audio
git commit -m "feat(audio): add SpeechToolBox with generate_speech and list_voices"
```

---

### Task 8: Examples, docs and live verification

**Files:**
- Create: `crates/agentai-audio/examples/{speech_generator,speech_tools,clone_voice}.rs`
- Modify: `crates/agentai-audio/Cargo.toml` (`[[example]] name = "speech_tools" required-features = ["toolbox"]`), `crates/agentai-audio/src/lib.rs` and `README.md` (crate docs modelled on `crates/agentai-image/src/lib.rs`), `crates/agentai/src/lib.rs` (next to lines 22, 29, 55), `crates/agentai/src/tool/mod.rs:22`, root `README.md` (regenerated), `llms.txt` (lines 19, 32, 44-45), `llms-full.txt` (next to the image section at lines 459-498, and line 34), `CLAUDE.md` (lines 13, 47; Commands if needed)

**Interfaces:**
- Consumes: all public API above.

- [ ] **Step 1: Write the examples**

All call `agentai_core`'s `install_crypto_provider` the way the image examples do, read `AGENTAI_SPEECH_MODEL`, and write into `audio/`.
- `speech_generator`: `SpeechGenerator::new(&model)?.with_default_voice(...)`, saves one file. Voice from `AGENTAI_SPEECH_VOICE` when set.
- `speech_tools`: an agent on `AGENTAI_MODEL` with `ToolBoxSet` holding `SpeechToolBox`.
- `clone_voice`: arg 1 is a WAV path. `OmlxProvider::from_env(model)?.with_voice_store(MemoryVoiceStore::new())`, then `create_voice` and `generate_with` through `Voice::id`.

Add `audio/` to `.gitignore` if the image examples' `images/` is listed there.

- [ ] **Step 2: Live check on oMLX**

From the worktree, with the main checkout's `.env` sourced:
- `AGENTAI_SPEECH_MODEL=omlx::mlx-community/VoxCPM2-8bit cargo run -p agentai-audio --example speech_generator`. Expected: a WAV file in `audio/` that `afplay` plays.
- `cargo run -p agentai-audio --example clone_voice -- <sample.wav from Task 1>` with the same model. Expected: a WAV file. Ask your human partner to listen and confirm it resembles the sample if a human check is available; otherwise report that likeness was not checked.
- `AGENTAI_MODEL=omlx::Qwen3.8-27B-oQ4e-mtp` and `cargo run -p agentai-audio --example speech_tools`. Expected: the agent calls `generate_speech` and a file ID is printed.

- [ ] **Step 3: Write the documentation**

State in `llms*.txt` and crate docs: tool names, `Metadata` default, `with_voices` as allowlist, the explicit oMLX store, the OpenAI shutdown on 2027-01-06 in `provider::openai` docs. No `MIGRATION.md` entry. Update the version named in `llms*.txt` only if it differs. CLAUDE.md gets a bullet like the `agentai-image` one and the crate added to the release note on TLS features.

- [ ] **Step 4: Regenerate README and check everything**

Run: `cargo rdme --workspace-project agentai --readme-path README.md` (check `git diff --stat README.md` touches only the generated section).
Run: `cargo fmt --check && cargo clippy -- -D warnings && cargo test -p agentai-audio && cargo test -p agentai --lib && cargo tree -i aws-lc-rs`.
Expected: all pass, and `cargo tree -i aws-lc-rs` reports the package was not found.

- [ ] **Step 5: Commit**

```bash
git add crates/agentai-audio crates/agentai/src README.md llms.txt llms-full.txt CLAUDE.md .gitignore
git commit -m "docs(audio): add speech examples and document agentai-audio"
```

Report plainly that ElevenLabs and OpenAI were not run live unless keys were available and they were.
